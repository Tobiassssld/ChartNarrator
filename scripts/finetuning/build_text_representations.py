#!/usr/bin/env python3
"""Build text representations for the text-only ablation dataset.

Inputs:
    data/finetune_dataset_5_0/images/
    data/cleaned_csv_data/

Outputs:
    data/text_only/text_representations.json

Run from the repository root:
    python scripts/finetuning/build_text_representations.py
    python scripts/finetuning/build_text_representations.py --dry_run --limit 10
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------------------------------------------
# Constants aligned with Stage 1 anchor extraction
# ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROLLING_SUM_FILES = {
    "cbs_Bankruptcies_Key_Figures_Companies.csv",
    "cbs_Bankruptcies_Key_Figures_Individuals.csv",
    "cbs_Bankruptcies_Key_Figures_Total.csv",
    "cbs_Existing_Homes_Prices_Sales_AvgPrice_Sold_Dwellings.csv",
    "cbs_Population_Dynamics_Monthly_Deaths.csv",
    "cbs_Population_Dynamics_Monthly_Live_Births.csv",
    "cbs_Population_Dynamics_Monthly_Net_Migration.csv",
}

_IMAGE_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<idx>\d{3})_(?P<start>\d{6})_(?P<end>\d{6})\.png$"
)

_MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_TREND_MIN_MONTHS = 24
_ROLLING_MIN_PERIODS = 12

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "finetune_dataset_5_0" / "images"
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "cleaned_csv_data"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "text_only" / "text_representations.json"


# ----------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------

def normalize_csv_stem(stem: str) -> str:
    """Remove duplicated cbs_ prefixes from legacy image stems."""
    while stem.startswith("cbs_cbs_"):
        stem = "cbs_" + stem[len("cbs_cbs_"):]
    return stem


def parse_image_filename(filename: str) -> Optional[Dict]:
    """Parse an image filename into CSV filename and date-window metadata."""
    m = _IMAGE_RE.match(os.path.basename(filename))
    if not m:
        return None
    prefix = normalize_csv_stem(m.group("prefix"))
    csv_filename = f"{prefix}.csv"

    start = _yyyymm_to_datetime(m.group("start"))
    end = _yyyymm_to_datetime(m.group("end"))
    if start is None or end is None:
        return None

    return {
        "csv_filename": csv_filename,
        "start_date": start,
        "end_date": end,
        "index": m.group("idx"),
    }


def _yyyymm_to_datetime(yyyymm: str) -> Optional[datetime]:
    """Convert a YYYYMM string to a first-of-month datetime."""
    try:
        return datetime(int(yyyymm[:4]), int(yyyymm[4:6]), 1)
    except Exception:
        return None


def fmt_date(dt) -> str:
    """Convert a datetime-like value to the token-safe 'Jan 1996' format."""
    if pd.isna(dt):
        return "N/A"
    try:
        dt = pd.Timestamp(dt)
        return f"{_MONTH_ABBR[dt.month - 1]} {dt.year}"
    except Exception:
        return str(dt)


def fmt_value(v) -> str:
    """Format numbers: integers without decimals, otherwise four significant digits."""
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e12:
            return str(int(f))
        return f"{f:.4g}"
    except Exception:
        return str(v)


# ----------------------------------------------------------------
# Data processing aligned with Stage 1 anchor extraction
# ----------------------------------------------------------------

def apply_processing(df: pd.DataFrame, csv_filename: str) -> Tuple[pd.DataFrame, str]:
    """Compute trend_value on the full series before slicing."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    if csv_filename in ROLLING_SUM_FILES:
        df["trend_value"] = df["value"].rolling(
            window=12, min_periods=_ROLLING_MIN_PERIODS
        ).sum()
        return df, "rolling_sum"
    df["trend_value"] = df["value"].rolling(
        window=12, min_periods=_ROLLING_MIN_PERIODS
    ).mean()
    return df, "moving_average"


def load_and_slice(
    csv_path: Path,
    csv_filename: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[Tuple[pd.DataFrame, str, str]]:
    """Load a cleaned CSV, compute trend_value, and slice the image time window."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    [ERROR] Failed to read CSV: {csv_path} - {e}")
        return None

    if "date" not in df.columns or "value" not in df.columns:
        print(f"    [ERROR] CSV missing date/value columns: {csv_path}")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df_full, proc_type = apply_processing(df, csv_filename)

    mask = (df_full["date"] >= pd.Timestamp(start_date)) & \
           (df_full["date"] <= pd.Timestamp(end_date))
    df_slice = df_full.loc[mask].copy().sort_values("date").reset_index(drop=True)

    if df_slice.empty:
        return None

    duration_months = (
        (df_slice["date"].iloc[-1].year - df_slice["date"].iloc[0].year) * 12 +
        (df_slice["date"].iloc[-1].month - df_slice["date"].iloc[0].month) + 1
    )
    plot_col = "value" if duration_months < _TREND_MIN_MONTHS else "trend_value"

    if plot_col == "trend_value" and df_slice["trend_value"].isna().all():
        plot_col = "value"

    df_slice = df_slice.dropna(subset=[plot_col]).reset_index(drop=True)
    if df_slice.empty:
        return None

    return df_slice, proc_type, plot_col


# ----------------------------------------------------------------
# Text formatting
# ----------------------------------------------------------------

def build_text_representation(
    img_name: str,
    df_slice: pd.DataFrame,
    proc_type: str,
    plot_col: str,
    csv_filename: str,
) -> str:
    """Build the model-readable time-series text representation."""
    vals = df_slice[plot_col].values
    dates = df_slice["date"].values

    n = len(vals)
    v_start = vals[0]
    v_end = vals[-1]
    v_min = vals.min()
    v_max = vals.max()
    v_mean = vals.mean()

    d_start = pd.Timestamp(dates[0])
    d_end = pd.Timestamp(dates[-1])

    idx_peak = vals.argmax()
    idx_trough = vals.argmin()
    d_peak = pd.Timestamp(dates[idx_peak])
    d_trough = pd.Timestamp(dates[idx_trough])

    abs_change = v_end - v_start
    pct_change = (abs_change / v_start * 100) if v_start != 0 else None

    value_type = "count" if csv_filename in ROLLING_SUM_FILES else "index"
    variable_name = csv_filename.replace("cbs_", "").replace(".csv", "").replace("_", " ")

    lines = []

    lines.append("[TIME SERIES DATA]")
    lines.append(f"Variable     : {variable_name}")
    lines.append(f"Value type   : {value_type}")
    lines.append(f"Processing   : {proc_type.replace('_', ' ')}")
    lines.append(f"Period       : {fmt_date(d_start)} to {fmt_date(d_end)}")
    lines.append(f"Data points  : {n}")
    lines.append("")

    lines.append("[KEY STATISTICS]")
    lines.append(f"Start value  : {fmt_value(v_start)} ({fmt_date(d_start)})")
    lines.append(f"End value    : {fmt_value(v_end)} ({fmt_date(d_end)})")
    lines.append(f"Peak         : {fmt_value(v_max)} on {fmt_date(d_peak)}")
    lines.append(f"Trough       : {fmt_value(v_min)} on {fmt_date(d_trough)}")
    lines.append(f"Mean         : {fmt_value(v_mean)}")
    lines.append(f"Abs change   : {fmt_value(abs_change)}")
    if pct_change is not None:
        lines.append(f"Pct change   : {pct_change:.2f}%")
    else:
        lines.append("Pct change   : N/A (start value is zero)")
    lines.append("")

    lines.append("[MONTHLY DATA]")
    lines.append(f"{'Date':<12} | Value")
    lines.append("-" * 28)
    for date_val, val in zip(dates, vals):
        lines.append(f"{fmt_date(pd.Timestamp(date_val)):<12} | {fmt_value(val)}")

    return "\n".join(lines)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def build_text_representations(
    images_dir: Path,
    csv_dir: Path,
    output_file: Path,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> int:
    """Build text representations for all PNG files under images_dir."""
    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        print(f"No image files found: {images_dir}")
        return 1

    if limit:
        png_files = png_files[:limit]

    print(f"{'=' * 60}")
    print("  Text Representation Builder")
    print(f"  Image dir : {images_dir}")
    print(f"  CSV dir   : {csv_dir}")
    print(f"  Images    : {len(png_files)}")
    print(f"{'=' * 60}\n")

    results: Dict[str, str] = {}
    failed: List[str] = []

    for i, png_path in enumerate(png_files, 1):
        img_name = png_path.name

        if i % 100 == 0 or i == 1:
            print(f"  [{i}/{len(png_files)}] {img_name}")

        parsed = parse_image_filename(img_name)
        if parsed is None:
            print(f"    [WARN] Failed to parse filename: {img_name}")
            failed.append(img_name)
            continue

        csv_filename = parsed["csv_filename"]
        csv_path = csv_dir / csv_filename

        if not csv_path.exists():
            print(f"    [WARN] CSV not found: {csv_filename}")
            failed.append(img_name)
            continue

        result = load_and_slice(
            csv_path,
            csv_filename,
            parsed["start_date"],
            parsed["end_date"],
        )
        if result is None:
            print(f"    [WARN] Data slicing failed or returned empty data: {img_name}")
            failed.append(img_name)
            continue

        df_slice, proc_type, plot_col = result

        text = build_text_representation(
            img_name, df_slice, proc_type, plot_col, csv_filename
        )
        results[img_name] = text

        if dry_run and i <= 2:
            print(f"\n{'-' * 60}")
            print(f"  SAMPLE: {img_name}")
            print(f"{'-' * 60}")
            print(text[:1200])
            print("  ...")
            print()

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(results)} succeeded, {len(failed)} failed")
    print(f"{'=' * 60}")

    if dry_run:
        print("\n[DRY RUN] No file written. Remove --dry_run to execute.")
        return 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWritten: {output_file.resolve()}")

    if failed:
        fail_log = output_file.parent / "text_repr_failed.txt"
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"Failed list: {fail_log}")

    return 0


def parse_args(argv: Optional[List[str]] = None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Build time-series text representations for the text-only ablation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--images_dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="Image directory to scan for .png files.",
    )
    p.add_argument(
        "--csv_dir",
        default=str(DEFAULT_CSV_DIR),
        help="Directory containing cleaned CSV files.",
    )
    p.add_argument(
        "--output_file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSON file path.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview the first two generated texts without writing files.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images for debugging.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run text-representation construction."""
    args = parse_args(argv)
    return build_text_representations(
        images_dir=Path(args.images_dir),
        csv_dir=Path(args.csv_dir),
        output_file=Path(args.output_file),
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
