"""Fetch raw CBS Open Data tables for the ChartNarrator dataset.

The script downloads the configured English-language CBS tables and saves each
table as a CSV file. It performs no cleaning, filtering, or chart generation.

Inputs:
    CBS table IDs listed in CBS_DATASETS.
Outputs:
    One cbs_<topic>.csv file per table in data/raw_cbs_data by default.
    The output directory can be overridden with --output-dir.

Run from the repository root:
    python scripts/data/fetch_cbs_data.py
    python scripts/data/fetch_cbs_data.py --output-dir /path/to/raw_data
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import cbsodata
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_cbs_data"

CBS_DATASETS = {
    # --- Strong financial narrative relevance ---
    "Consumer_Confidence": "83693ENG",
    "Inflation_CPI": "83131ENG",
    "Labour_Participation_Unemployment": "80590ENG",
    "Producer_Confidence": "81234ENG",

    # --- Medium financial narrative relevance ---
    "Energy_Prices_Consumers": "85592ENG",
    "Existing_Homes_Prices_Sales_AvgPrice": "85773ENG",
    "Government_Finance_Key_Figures": "85968ENG",
    "Industry_Production_Sales": "85806ENG",
    "Investment_Tangible_Assets_Volume": "85939ENG",
    "Manufacturing_Turnover": "81817ENG",
    "Retail_Turnover": "81810ENG",
    "Terms_of_Trade_Goods": "85962ENG",
    "Wages_CAO_Total": "82838ENG",

    # --- Additional medium-relevance indicators ---
    "Civil_Engineering_Input_Price_Index": "86049ENG",
    "Construction_Production_Index": "85809ENG",
    "Producer_Price_Index_Manufacturing": "85771ENG",
    "PPI_Output_2021": "85771ENG",

    # --- Weak financial narrative relevance ---
    "Household_Consumption_Expenditure": "85937ENG",
    "Bankruptcies_Key_Figures": "82242ENG",
    "Population_Dynamics_Monthly": "83474ENG",
}

def fetch_and_save_cbs(
    topic: str,
    table_id: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Download one CBS table and save it as a CSV file.

    Args:
        topic: Stable topic identifier used in the output filename.
        table_id: CBS Open Data table identifier.
        output_dir: Directory in which to save the CSV file.

    Returns:
        Path to the saved CSV file.

    Raises:
        Exception: Propagates download, conversion, and file-writing errors
            so the batch runner can report a failed table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cbs_{topic}.csv"

    LOGGER.info("Fetching %s (%s)", topic, table_id)
    data = cbsodata.get_data(table_id)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(output_path, index=False)
    LOGGER.info("Saved %s (%d rows)", output_path, len(dataframe))

    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Fetch all configured tables and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw CSV files (default: repository data/raw_cbs_data).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output_dir.expanduser().resolve()
    failures = []

    for topic, table_id in CBS_DATASETS.items():
        try:
            fetch_and_save_cbs(topic, table_id, output_dir)
        except Exception:
            LOGGER.exception("Failed to fetch %s (%s)", topic, table_id)
            failures.append(topic)

    if failures:
        LOGGER.error(
            "Batch finished with %d failed table(s): %s",
            len(failures),
            ", ".join(failures),
        )
        return 1

    LOGGER.info("Batch finished successfully: %d tables.", len(CBS_DATASETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
