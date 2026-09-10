#!/usr/bin/env python3
"""Extract Stage 1 chart anchors from generated chart images.

This script links each chart image back to its cleaned CBS time-series CSV by
parsing the public chart filename convention:

    cbs_<dataset>_<slice_id>_<index>_<start_yyyymm>_<end_yyyymm>.png

It then reconstructs the exact plotted data window and extracts structured
anchors used by the Stage 1 narrative generation script. The extraction logic
preserves the original research implementation: rolling sums are used for flow
series, moving averages are used for other series, and raw values are selected
for short windows.

Inputs:
    data/images/*.png
    data/cleaned_csv_data/*.csv

Outputs:
    data/stage1/chart_anchors_stage1.json

Run from the repository root:
    python scripts/stage1/extract_anchors.py
    python scripts/stage1/extract_anchors.py --no-progress
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
CSV_DIR = PROJECT_ROOT / "data" / "cleaned_csv_data"
OUTPUT_FILE = PROJECT_ROOT / "data" / "stage1" / "chart_anchors_stage1.json"

ROLLING_SUM_FILES = ['cbs_Bankruptcies_Key_Figures_Companies.csv', 'cbs_Bankruptcies_Key_Figures_Individuals.csv', 'cbs_Bankruptcies_Key_Figures_Total.csv', 'cbs_Existing_Homes_Prices_Sales_AvgPrice_Sold_Dwellings.csv', 'cbs_Population_Dynamics_Monthly_Deaths.csv', 'cbs_Population_Dynamics_Monthly_Live_Births.csv', 'cbs_Population_Dynamics_Monthly_Net_Migration.csv']

IMAGE_FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<idx>\d{3})_(?P<start>\d{6})_(?P<end>\d{6})\.png$"
)


def normalize_csv_stem(stem: str) -> str:
    """Normalize duplicated cbs_ prefixes left by older chart-generation runs."""
    while stem.startswith("cbs_cbs_"):
        stem = "cbs_" + stem[len("cbs_cbs_") :]
    return stem


def parse_image_filename(filename: str) -> Optional[dict[str, str]]:
    """Parse a ChartNarrator chart filename into CSV and date-window metadata."""
    match = IMAGE_FILENAME_RE.match(filename)
    if match is None:
        return None

    prefix = normalize_csv_stem(match.group("prefix"))
    return {
        "csv_filename": f"{prefix}.csv",
        "start_yyyymm": match.group("start"),
        "end_yyyymm": match.group("end"),
        "index": match.group("idx"),
    }


def parse_date_from_yyyymm(yyyymm: str) -> Optional[datetime]:
    """Parse a YYYYMM string as the first day of the represented month."""
    try:
        return datetime(int(yyyymm[:4]), int(yyyymm[4:6]), 1)
    except (TypeError, ValueError):
        return None


def apply_financial_processing_full(
    df: pd.DataFrame,
    csv_filename: str,
) -> tuple[pd.DataFrame, str]:
    """Compute the trend column on the full series before window slicing.

    The logic matches chart_utils.py: selected flow series use a 12-month
    rolling sum, while all other series use a 12-month moving average. Both
    use min_periods=12 to avoid artificial early-window ramps.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    if csv_filename in ROLLING_SUM_FILES:
        df["trend_value"] = df["value"].rolling(window=12, min_periods=12).sum()
        return df, "rolling_sum"

    df["trend_value"] = df["value"].rolling(window=12, min_periods=12).mean()
    return df, "moving_average"


def infer_value_type(csv_filename: str) -> str:
    """Infer the minimal value semantics used by Stage 1 generation."""
    return "count" if csv_filename in ROLLING_SUM_FILES else "index"


def select_plot_column(duration_months: int) -> str:
    """Select raw values for short windows and trend values otherwise."""
    return "value" if duration_months < 24 else "trend_value"


def extract_anchors_for_window(
    csv_path: Path,
    start_date: datetime,
    end_date: datetime,
    csv_filename: str,
) -> Optional[dict[str, Any]]:
    """Extract structured anchors for a chart's encoded time window.

    Args:
        csv_path: Path to the cleaned CSV file referenced by the chart image.
        start_date: Requested start month parsed from the image filename.
        end_date: Requested end month parsed from the image filename.
        csv_filename: Cleaned CSV filename, used for series-specific processing.

    Returns:
        A nested dictionary of Stage 1 anchors, or None if the window cannot
        produce at least two valid observations.
    """
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "value" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df_full, processing_type = apply_financial_processing_full(df, csv_filename)

    mask = (df_full["date"] >= start_date) & (df_full["date"] <= end_date)
    df_slice = df_full.loc[mask].copy()
    if df_slice.empty:
        return None

    df_slice = df_slice.sort_values("date").reset_index(drop=True)

    requested_start = start_date
    requested_end = end_date
    requested_duration_months = (
        (requested_end.year - requested_start.year) * 12
        + (requested_end.month - requested_start.month)
        + 1
    )

    slice_data_points = int(len(df_slice))
    column = select_plot_column(slice_data_points)

    effective_points = slice_data_points
    fallback_to_value = False
    nan_months_dropped = 0

    if column == "trend_value":
        df_tv = df_slice.dropna(subset=["trend_value"]).copy()
        nan_months_dropped = len(df_slice) - len(df_tv)

        if len(df_tv) >= 3:
            df_use = df_tv
            effective_points = int(len(df_tv))
        else:
            df_use = df_slice
            column = "value"
            fallback_to_value = True
            effective_points = int(len(df_slice))
            nan_months_dropped = 0
    else:
        df_use = df_slice

    actual_start = df_use["date"].min()
    actual_end = df_use["date"].max()
    actual_duration_months = (
        (actual_end.year - actual_start.year) * 12
        + (actual_end.month - actual_start.month)
        + 1
    )
    data_completeness = (
        effective_points / requested_duration_months
        if requested_duration_months > 0
        else 1.0
    )

    values = df_use[column].astype(float).to_numpy()
    dates = df_use["date"].to_numpy()

    if len(values) < 2:
        return None

    start_value = float(values[0])
    end_value = float(values[-1])
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    std_value = float(np.std(values))
    absolute_change = end_value - start_value

    percentage_change = None
    percentage_change_reason = None
    if start_value == 0:
        percentage_change_reason = "start_value_zero"
    elif abs(start_value) < 1e-10:
        percentage_change_reason = "start_value_near_zero"
    else:
        percentage_change = (absolute_change / abs(start_value)) * 100
        percentage_change_reason = "valid"

    annualized_change = None
    annualized_change_reason = None
    duration_years = len(values) / 12.0
    if duration_years <= 0:
        annualized_change_reason = "insufficient_duration"
    elif start_value == 0:
        annualized_change_reason = "start_value_zero"
    elif start_value < 0:
        annualized_change_reason = "start_value_negative"
    elif end_value <= 0:
        annualized_change_reason = "end_value_non_positive"
    elif abs(start_value) < 1e-10:
        annualized_change_reason = "start_value_near_zero"
    else:
        annualized_change = (pow(end_value / start_value, 1 / duration_years) - 1) * 100
        annualized_change_reason = "valid"

    peak_idx = int(np.argmax(values))
    trough_idx = int(np.argmin(values))

    peak_is_start = peak_idx == 0
    peak_is_end = peak_idx == len(values) - 1
    trough_is_start = trough_idx == 0
    trough_is_end = trough_idx == len(values) - 1

    coef_var = (std_value / abs(mean_value)) if mean_value != 0 else None
    value_range = max_value - min_value

    return {
        "csv_source": csv_filename,
        "processing_type": processing_type,
        "data_used": column,
        "effective_points": effective_points,
        "fallback_to_value": fallback_to_value,
        "time_window": {
            "requested_start_date": requested_start.strftime("%Y-%m-%d"),
            "requested_end_date": requested_end.strftime("%Y-%m-%d"),
            "requested_duration_months": requested_duration_months,
            "actual_start_date": actual_start.strftime("%Y-%m-%d"),
            "actual_end_date": actual_end.strftime("%Y-%m-%d"),
            "actual_duration_months": actual_duration_months,
            "data_points_available": effective_points,
            "data_completeness_ratio": round(data_completeness, 4),
            "nan_months_dropped": nan_months_dropped,
            "data_aligned": nan_months_dropped == 0,
        },
        "values": {
            "start_value": round(start_value, 4),
            "end_value": round(end_value, 4),
            "min_value": round(min_value, 4),
            "max_value": round(max_value, 4),
            "mean_value": round(mean_value, 4),
            "median_value": round(median_value, 4),
            "std_value": round(std_value, 4),
        },
        "changes": {
            "absolute_change": round(absolute_change, 4),
            "percentage_change": (
                round(percentage_change, 4)
                if percentage_change is not None
                else None
            ),
            "percentage_change_reason": percentage_change_reason,
            "annualized_change": (
                round(annualized_change, 4)
                if annualized_change is not None
                else None
            ),
            "annualized_change_reason": annualized_change_reason,
        },
        "extremes": {
            "peak": {
                "date": pd.Timestamp(dates[peak_idx]).strftime("%Y-%m-%d"),
                "value": round(max_value, 4),
                "index_in_series": peak_idx,
                "is_start_point": peak_is_start,
                "is_end_point": peak_is_end,
            },
            "trough": {
                "date": pd.Timestamp(dates[trough_idx]).strftime("%Y-%m-%d"),
                "value": round(min_value, 4),
                "index_in_series": trough_idx,
                "is_start_point": trough_is_start,
                "is_end_point": trough_is_end,
            },
        },
        "volatility": {
            "coefficient_of_variation": (
                round(coef_var, 4) if coef_var is not None else None
            ),
            "range": round(value_range, 4),
        },
        "value_type": infer_value_type(csv_filename),
    }


def extract_anchors(
    images_dir: Path,
    csv_dir: Path,
    output_file: Path,
    show_progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Extract anchors for all PNG files in an image directory.

    Args:
        images_dir: Directory containing generated chart images.
        csv_dir: Directory containing cleaned CSV files.
        output_file: JSON path for the extracted anchor dictionary.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Dictionary keyed by image filename.
    """
    images_dir = Path(images_dir)
    csv_dir = Path(csv_dir)
    output_file = Path(output_file)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")

    image_files = sorted(images_dir.glob("*.png"))
    all_anchors: dict[str, dict[str, Any]] = {}
    failed_files: list[str] = []

    iterator = tqdm(
        image_files,
        desc="Extracting anchors",
        disable=not show_progress,
    )

    for image_path in iterator:
        parsed = parse_image_filename(image_path.name)
        if parsed is None:
            failed_files.append(image_path.name)
            continue

        csv_path = csv_dir / parsed["csv_filename"]
        if not csv_path.is_file():
            failed_files.append(image_path.name)
            continue

        start_date = parse_date_from_yyyymm(parsed["start_yyyymm"])
        end_date = parse_date_from_yyyymm(parsed["end_yyyymm"])
        if start_date is None or end_date is None:
            failed_files.append(image_path.name)
            continue

        anchors = extract_anchors_for_window(
            csv_path=csv_path,
            start_date=start_date,
            end_date=end_date,
            csv_filename=parsed["csv_filename"],
        )
        if anchors is None:
            failed_files.append(image_path.name)
            continue

        all_anchors[image_path.name] = anchors

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(all_anchors, handle, ensure_ascii=False, indent=2)

    total = len(image_files)
    success = len(all_anchors)
    failed = len(failed_files)

    LOGGER.info("Images found: %d", total)
    LOGGER.info("Anchors extracted: %d", success)
    LOGGER.info("Failures: %d", failed)

    if failed_files:
        LOGGER.warning("First failed files: %s", ", ".join(failed_files[:10]))
        if len(failed_files) > 10:
            LOGGER.warning("Additional failed files: %d", len(failed_files) - 10)

    if all_anchors:
        log_anchor_quality_summary(all_anchors)

    LOGGER.info("Saved anchors to %s", output_file)
    return all_anchors


def log_anchor_quality_summary(all_anchors: Mapping[str, dict[str, Any]]) -> None:
    """Log aggregate quality statistics for the extracted Stage 1 anchors."""
    success = len(all_anchors)
    if success == 0:
        return

    misaligned = sum(
        1 for anchor in all_anchors.values()
        if not anchor["time_window"]["data_aligned"]
    )
    peak_at_end = sum(
        1 for anchor in all_anchors.values()
        if anchor["extremes"]["peak"]["is_end_point"]
    )
    trough_at_start = sum(
        1 for anchor in all_anchors.values()
        if anchor["extremes"]["trough"]["is_start_point"]
    )
    completeness_values = [
        anchor["time_window"]["data_completeness_ratio"]
        for anchor in all_anchors.values()
    ]
    average_completeness = sum(completeness_values) / len(completeness_values)
    invalid_percentage = sum(
        1 for anchor in all_anchors.values()
        if anchor["changes"]["percentage_change_reason"] != "valid"
    )
    invalid_annualized = sum(
        1 for anchor in all_anchors.values()
        if anchor["changes"]["annualized_change_reason"] != "valid"
    )

    LOGGER.info(
        "Aligned anchors: %d/%d",
        success - misaligned,
        success,
    )
    LOGGER.info("Anchors with dropped trend NaNs: %d/%d", misaligned, success)
    LOGGER.info("Average data completeness: %.4f", average_completeness)
    LOGGER.info("Peaks at end point: %d/%d", peak_at_end, success)
    LOGGER.info("Troughs at start point: %d/%d", trough_at_start, success)
    LOGGER.info("Invalid percentage changes: %d/%d", invalid_percentage, success)
    LOGGER.info("Invalid annualized changes: %d/%d", invalid_annualized, success)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run Stage 1 anchor extraction and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Directory containing generated chart PNG files.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=CSV_DIR,
        help="Directory containing cleaned CSV files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help="Path for the Stage 1 anchor JSON file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        extract_anchors(
            images_dir=args.images_dir.expanduser().resolve(),
            csv_dir=args.csv_dir.expanduser().resolve(),
            output_file=args.output_file.expanduser().resolve(),
            show_progress=not args.no_progress,
        )
    except Exception:
        LOGGER.exception("Stage 1 anchor extraction failed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
