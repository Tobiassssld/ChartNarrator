#!/usr/bin/env python3
"""Generate ChartNarrator charts from the cleaned CBS time-series dataset.

This batch runner uses chart_utils.py to sample random time windows, apply
the research pipeline's financial processing, and render PNG charts.

Inputs:
    data/cleaned_csv_data/*.csv
    Each file must contain date, value, dataset, and slice_id.
Outputs:
    data/images/*.png
    data/reports/chart_generation_stats.json

The default is 35 chart attempts per CSV. The original sampling probabilities,
financial processing, plotting styles, and image filename convention are
preserved. An optional seed makes a new run's sampling reproducible.

Run from the repository root:
    python scripts/data/generate_charts.py
    python scripts/data/generate_charts.py --seed 42
    python scripts/data/generate_charts.py --input-dir /path/to/cleaned_csv_data
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from tqdm import tqdm

from chart_utils import analyze_csv, generate_charts_from_csv

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "cleaned_csv_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "chart_generation_stats.json"
CHARTS_PER_CSV = 35


def generate_dataset_info(csv_file: Path) -> Optional[dict]:
    """Return the original report fields for a cleaned CSV, if valid."""
    stats = analyze_csv(csv_file)
    if stats is None:
        return None

    return {
        "filename": csv_file.name,
        "dataset": stats["dataset"],
        "slice_id": stats["slice_id"],
        "data_points": stats["points"],
        "time_range": f"{stats['start_date']} ~ {stats['end_date']}",
        "value_range": f"{stats['min_value']:.2f} ~ {stats['max_value']:.2f}",
        "has_negative": stats["has_negative"],
    }


def log_dataset_summary(dataset_entries: Sequence[dict]) -> None:
    """Log the original per-dataset and per-slice generation statistics."""
    dataset_groups = defaultdict(list)
    for item in dataset_entries:
        dataset_groups[item["dataset"]].append(item)

    for dataset_name, items in sorted(dataset_groups.items()):
        total_slices = len(items)
        total_charts = sum(item["charts_generated"] for item in items)
        LOGGER.info(
            "%s: %d slices, %d charts, %.1f charts per slice.",
            dataset_name,
            total_slices,
            total_charts,
            total_charts / total_slices,
        )
        for item in items:
            LOGGER.info(
                "  %s: %d points | %s | %d charts",
                item["slice_id"],
                item["data_points"],
                item["time_range"],
                item["charts_generated"],
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the chart-generation batch and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing cleaned CBS CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to save generated PNG charts.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the JSON generation report.",
    )
    parser.add_argument(
        "--charts-per-csv",
        type=int,
        default=CHARTS_PER_CSV,
        help="Number of chart attempts per CSV (default: 35).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed; omitted by default to preserve original behavior.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar, for example when running in CI.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.charts_per_csv < 1:
        parser.error("--charts-per-csv must be a positive integer.")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    if not input_dir.is_dir():
        LOGGER.error("Cleaned CSV directory does not exist: %s", input_dir)
        return 1

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        LOGGER.error("No cleaned CSV files found in %s", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed) if args.seed is not None else None

    LOGGER.info("Input directory: %s", input_dir)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Chart attempts per CSV: %d", args.charts_per_csv)
    LOGGER.info("Cleaned CSV files: %d", len(csv_files))

    stats_data = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "charts_per_csv": args.charts_per_csv,
        "total_csvs": len(csv_files),
        "datasets": [],
        "random_seed": args.seed,
    }
    total_charts = 0
    failed_csvs = []

    with tqdm(
        csv_files,
        desc="Generating charts",
        unit="CSV",
        disable=args.no_progress,
    ) as progress:
        for csv_file in progress:
            progress.set_description(f"Processing {csv_file.stem[:30]}")
            dataset_info = None

            try:
                dataset_info = generate_dataset_info(csv_file)
                success_count = generate_charts_from_csv(
                    csv_file,
                    num_charts=args.charts_per_csv,
                    output_dir=output_dir,
                    rng=rng,
                )
                total_charts += success_count

                if dataset_info is not None:
                    dataset_info["charts_generated"] = success_count
                    stats_data["datasets"].append(dataset_info)

                if success_count < args.charts_per_csv:
                    failed_csvs.append(csv_file.name)
                    LOGGER.warning(
                        "%s: generated %d/%d requested charts.",
                        csv_file.name,
                        success_count,
                        args.charts_per_csv,
                    )

                progress.set_postfix(
                    {"Charts": total_charts, "Current": success_count}
                )
            except Exception:
                LOGGER.exception("Failed to process %s", csv_file.name)
                failed_csvs.append(csv_file.name)

    # Keep the original report keys for compatibility with existing analyses.
    stats_data["total_charts_generated"] = total_charts
    stats_data["failed_csvs"] = failed_csvs

    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(stats_data, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    except OSError:
        LOGGER.exception("Failed to write generation report: %s", report_path)
        return 1

    LOGGER.info(
        "Batch summary: %d CSV files, %d charts, %.1f charts per CSV.",
        len(csv_files),
        total_charts,
        total_charts / len(csv_files),
    )
    if stats_data["datasets"]:
        log_dataset_summary(stats_data["datasets"])

    if failed_csvs:
        LOGGER.error(
            "%d CSV file(s) produced fewer charts than requested: %s",
            len(failed_csvs),
            ", ".join(failed_csvs),
        )
        return 1

    LOGGER.info("Generation report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

