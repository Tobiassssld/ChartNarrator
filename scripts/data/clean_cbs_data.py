#!/usr/bin/env python3
"""Clean raw CBS Open Data into monthly ChartNarrator time-series slices.

This script consumes the CSV files produced by fetch_cbs_data.py. It parses
CBS period labels, retains monthly observations, applies the configured
categorical filters, and converts selected measurement columns to numeric
values. Duplicate dates are aggregated by their arithmetic mean.

Inputs:
    data/raw_cbs_data/cbs_<topic>.csv
    DATASET_CONFIGS, which defines the selected series and category filters.

Outputs:
    data/cleaned_csv_data/cbs_<topic>_<slice_id>.csv
    Each CSV contains: date, value, dataset, slice_id.

The script preserves the original research selection rules and does not
perform the smoothing, window sampling, or chart rendering used downstream.

Run from the repository root:
    python scripts/data/clean_cbs_data.py
    python scripts/data/clean_cbs_data.py --input-dir /path/to/raw_data
    python scripts/data/clean_cbs_data.py --dataset cbs_Consumer_Confidence
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw_cbs_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned_csv_data"
OUTPUT_COLUMNS = ("date", "value", "dataset", "slice_id")

ENGLISH_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_cbs_date(period_str: Any) -> Optional[pd.Timestamp]:
    """Parse a CBS period label into the first day of its period.

    Supports English month labels, the original English/Dutch quarter
    labels, and yearly labels. Returns None for missing or invalid values.

    The parser retains the original date-recognition rules so that the
    historical dataset construction is not changed during cleanup.
    """
    if pd.isna(period_str):
        return None

    period_str = str(period_str).strip().lower()

    for month_name, month_num in ENGLISH_MONTHS.items():
        if month_name in period_str:
            try:
                year = int(period_str.split()[0])
                return pd.Timestamp(year=year, month=month_num, day=1)
            except (TypeError, ValueError, OverflowError):
                pass

    if "quarter" in period_str or "kwartaal" in period_str:
        try:
            year = int(period_str.split()[0])
            if "1st" in period_str or "1e" in period_str or "eerste" in period_str:
                return pd.Timestamp(year=year, month=1, day=1)
            elif "2nd" in period_str or "2e" in period_str or "tweede" in period_str:
                return pd.Timestamp(year=year, month=4, day=1)
            elif "3rd" in period_str or "3e" in period_str or "derde" in period_str:
                return pd.Timestamp(year=year, month=7, day=1)
            elif "4th" in period_str or "4e" in period_str or "vierde" in period_str:
                return pd.Timestamp(year=year, month=10, day=1)
        except (TypeError, ValueError, OverflowError):
            pass

    try:
        year = int(period_str.split()[0])
        if 1900 <= year <= 2100:
            return pd.Timestamp(year=year, month=1, day=1)
    except (TypeError, ValueError, OverflowError):
        pass

    return None


def detect_time_granularity(period_str: Any) -> Optional[str]:
    """Classify a CBS period as monthly, quarterly, yearly, or unknown."""
    if pd.isna(period_str):
        return None

    period_str = str(period_str).strip().lower()

    for month_name in ENGLISH_MONTHS:
        if month_name in period_str:
            return "monthly"

    if "quarter" in period_str or "kwartaal" in period_str:
        return "quarterly"

    try:
        parts = period_str.split()
        if len(parts) == 1:
            year = int(parts[0])
            if 1900 <= year <= 2100:
                return "yearly"
        elif len(parts) == 2 and "year" in parts[1]:
            return "yearly"
    except (TypeError, ValueError, OverflowError):
        pass

    return None


def is_numeric_column(col_name: str) -> bool:
    """Check whether a CBS column name ends with an underscore and digits.

    Retained as a helper from the original script. The configured value_col
    fields, rather than this naming heuristic, determine the selected series.
    """
    return bool(re.search(r"_\d+$", col_name))


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first column whose name contains 'period', if present."""
    for col in df.columns:
        if "period" in col.lower():
            return col
    return None


def clean_dataset(
    raw_path: Path,
    dataset_name: str,
    slice_configs: Sequence[Mapping[str, Any]],
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Clean one raw CBS table and write its configured monthly slices.

    Args:
        raw_path: Path to the CSV generated by fetch_cbs_data.py.
        dataset_name: Stable CBS dataset identifier, including the cbs_ prefix.
        slice_configs: Original series-selection and category-filter settings.
        output_dir: Optional output directory; defaults to OUTPUT_DIR.

    Returns:
        Paths of successfully written slice CSV files.

    Raises:
        OSError: If a raw file cannot be read or an output cannot be written.
        pandas.errors.ParserError: If pandas cannot parse the input CSV.
    """
    raw_path = Path(raw_path)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)

    LOGGER.info("Processing dataset: %s", dataset_name)
    df = pd.read_csv(raw_path, low_memory=False)
    LOGGER.info("Raw rows: %d", len(df))

    time_col = find_time_column(df)
    if time_col is None:
        LOGGER.warning("No period column found in %s; skipping.", raw_path)
        return []

    df["parsed_date"] = df[time_col].apply(parse_cbs_date)
    df["time_granularity"] = df[time_col].apply(detect_time_granularity)
    df = df.dropna(subset=["parsed_date"])

    if df.empty:
        LOGGER.warning("No valid dates remain in %s; skipping.", dataset_name)
        return []

    granularity_counts = df["time_granularity"].value_counts()
    for granularity, count in granularity_counts.items():
        LOGGER.info("Time granularity %s: %d rows", granularity, count)

    if "quarterly" in granularity_counts.index or "yearly" in granularity_counts.index:
        LOGGER.warning(
            "Quarterly or yearly observations detected in %s; "
            "only monthly observations will be retained.",
            dataset_name,
        )

    LOGGER.info(
        "Valid dates: %d; range: %s to %s",
        df["parsed_date"].nunique(),
        df["parsed_date"].min().date(),
        df["parsed_date"].max().date(),
    )

    written_paths = []
    for slice_config in slice_configs:
        output_path = process_slice(
            df,
            dataset_name,
            slice_config,
            time_col,
            output_dir=output_dir,
        )
        if output_path is not None:
            written_paths.append(output_path)

    LOGGER.info(
        "Completed %s: %d/%d slices written.",
        dataset_name,
        len(written_paths),
        len(slice_configs),
    )
    return written_paths


def process_slice(
    df: pd.DataFrame,
    dataset_name: str,
    slice_config: Mapping[str, Any],
    time_col: str,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Apply the original monthly and categorical filters to one series.

    Args:
        df: Dataframe with parsed_date and time_granularity columns.
        dataset_name: Stable dataset identifier used in the output filename.
        slice_config: Mapping containing slice_id, value_col, and filters.
        time_col: Original period column name, retained for compatibility.
        output_dir: Optional output directory; defaults to OUTPUT_DIR.

    Returns:
        Path to the cleaned CSV, or None when no valid output can be produced.

    Notes:
        Missing categorical filter columns are reported and skipped, matching
        the original behavior. They require review against the actual CBS
        schema before the final dataset is considered reproducible.
    """
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    slice_id = slice_config["slice_id"]
    filters = slice_config.get("filters", {})
    value_col = slice_config.get("value_col")

    LOGGER.debug(
        "Processing %s / %s using period column %s",
        dataset_name,
        slice_id,
        time_col,
    )

    df_filtered = df.copy()

    if "time_granularity" in df_filtered.columns:
        before_count = len(df_filtered)
        df_filtered = df_filtered[df_filtered["time_granularity"] == "monthly"].copy()
        filtered_count = before_count - len(df_filtered)
        if filtered_count:
            LOGGER.info(
                "%s: excluded %d non-monthly rows.",
                slice_id,
                filtered_count,
            )
    else:
        LOGGER.warning(
            "%s: time_granularity is missing; monthly filtering was not applied.",
            slice_id,
        )

    for filter_col, filter_values in filters.items():
        if filter_col in df_filtered.columns:
            mask = df_filtered[filter_col].isin(filter_values)
            df_filtered = df_filtered.loc[mask].copy()
        else:
            LOGGER.warning(
                "%s: configured filter column %s is missing; "
                "preserving the original skip-filter behavior.",
                slice_id,
                filter_col,
            )

    if df_filtered.empty:
        LOGGER.warning("%s: no rows remain after filtering.", slice_id)
        return None

    if not value_col or value_col not in df_filtered.columns:
        LOGGER.warning("%s: value column %s is missing.", slice_id, value_col)
        return None

    df_filtered["value"] = pd.to_numeric(df_filtered[value_col], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["value"])

    if df_filtered.empty:
        LOGGER.warning("%s: no valid numeric values remain.", slice_id)
        return None

    # Preserve the original mean aggregation for duplicate dates.
    df_clean = df_filtered.groupby("parsed_date", as_index=False)["value"].mean()

    date_counts = df_filtered.groupby("parsed_date").size()
    duplicate_dates = date_counts[date_counts > 1]
    if not duplicate_dates.empty:
        LOGGER.warning(
            "%s: %d dates contain duplicate observations; using their mean.",
            slice_id,
            len(duplicate_dates),
        )

    df_clean = df_clean.sort_values("parsed_date")
    df_clean["dataset"] = dataset_name
    df_clean["slice_id"] = slice_id
    df_clean = df_clean.rename(columns={"parsed_date": "date"})
    df_clean = df_clean[list(OUTPUT_COLUMNS)]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_{slice_id}.csv"
    df_clean.to_csv(output_path, index=False)

    LOGGER.info("%s: saved %d rows to %s", slice_id, len(df_clean), output_path)
    return output_path


DATASET_CONFIGS = {
    # Consumer confidence and component indicators.
    'cbs_Consumer_Confidence': {
        "slices": [
            {'slice_id': 'Confidence_Index', 'value_col': 'ConsumerConfidence_1'},
            {'slice_id': 'Economic_Climate', 'value_col': 'EconomicClimate_2'},
            {'slice_id': 'Willingness_Buy', 'value_col': 'WillingnessToBuy_3'},
            {'slice_id': 'Econ_Situation_Last12M', 'value_col': 'EconomicSituationLast12Months_4'},
            {'slice_id': 'Econ_Situation_Next12M', 'value_col': 'EconomicSituationNext12Months_5'},
            {'slice_id': 'Financial_Situation_Last12M', 'value_col': 'FinancialSituationLast12Months_6'},
            {'slice_id': 'Financial_Situation_Next12M', 'value_col': 'FinancialSituationNext12Months_7'},
            {'slice_id': 'Purchase_Intentions_12M', 'value_col': 'IntentionToMakeMajorPurchases_8'},
        ],
    },

    # Selected CPI expenditure categories.
    'cbs_Inflation_CPI': {
        "slices": [
            {'slice_id': 'All_Items',
             'filters': {'ExpenditureCategories': ['000000 All items']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Food',
             'filters': {'ExpenditureCategories': ['010000 Food and non-alcoholic beverages']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Housing',
             'filters': {'ExpenditureCategories': ['040000 Housing, water, electricity, gas, etc.']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Transport',
             'filters': {'ExpenditureCategories': ['070000 Transport']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Energy',
             'filters': {'ExpenditureCategories': ['045000 Electricity, gas and other fuels']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Clothing',
             'filters': {'ExpenditureCategories': ['030000 Clothing and footwear']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Health',
             'filters': {'ExpenditureCategories': ['060000 Health']},
             'value_col': 'CPI_1'},
            {'slice_id': 'Recreation',
             'filters': {'ExpenditureCategories': ['090000 Recreation and culture']},
             'value_col': 'CPI_1'},
        ],
    },

    # Selected sex and age groups; unemployment rate.
    'cbs_Labour_Participation_Unemployment': {
        "slices": [
            {'slice_id': 'Total_15_74',
             'filters': {'Sex': ['Total sex'], 'Age': ['15 to 74 years']},
             'value_col': 'NotSeasonallyAdjusted_7'},
            {'slice_id': 'Total_15_24_Youth',
             'filters': {'Sex': ['Total sex'], 'Age': ['15 to 24 years']},
             'value_col': 'NotSeasonallyAdjusted_7'},
            {'slice_id': 'Men_15_74',
             'filters': {'Sex': ['Men'], 'Age': ['15 to 74 years']},
             'value_col': 'NotSeasonallyAdjusted_7'},
            {'slice_id': 'Women_15_74',
             'filters': {'Sex': ['Women'], 'Age': ['15 to 74 years']},
             'value_col': 'NotSeasonallyAdjusted_7'},
        ],
    },

    # Value margin and seasonally adjusted observations.
    'cbs_Producer_Confidence': {
        "slices": [
            {'slice_id': 'Manufacturing_Total',
             'filters': {'SectorBranchesSIC2008': ['C Manufacturing'],
                         'Margins': ['Value'],
                         'SeasonalAdjustment': ['Seasonally adjusted data']},
             'value_col': 'ProducerConfidence_1'},
            {'slice_id': 'Food_Beverages',
             'filters': {'SectorBranchesSIC2008': ['10-12 Manufacture of food and beverages'],
                         'Margins': ['Value'],
                         'SeasonalAdjustment': ['Seasonally adjusted data']},
             'value_col': 'ProducerConfidence_1'},
            {'slice_id': 'Chemicals',
             'filters': {'SectorBranchesSIC2008': ['19-22 Refineries and chemistry'],
                         'Margins': ['Value'],
                         'SeasonalAdjustment': ['Seasonally adjusted data']},
             'value_col': 'ProducerConfidence_1'},
            {'slice_id': 'Machinery',
             'filters': {'SectorBranchesSIC2008': ['26-28 Manufact of electronics, machinery'],
                         'Margins': ['Value'],
                         'SeasonalAdjustment': ['Seasonally adjusted data']},
             'value_col': 'ProducerConfidence_1'},
        ],
    },

    # Consumer energy prices including VAT.
    'cbs_Energy_Prices_Consumers': {
        "slices": [
            {'slice_id': 'Electricity_Variable',
             'filters': {'VAT': ['Including VAT']},
             'value_col': 'VariableDeliveryRateContractPrices_3'},
            {'slice_id': 'Gas_Variable',
             'filters': {'VAT': ['Including VAT']},
             'value_col': 'VariableDeliveryRateContractPrices_8'},
        ],
    },

    # Housing price, sales, and average purchase price indicators.
    'cbs_Existing_Homes_Prices_Sales_AvgPrice': {
        "slices": [
            {'slice_id': 'Price_Index', 'value_col': 'PriceIndexSellingPrices_1'},
            {'slice_id': 'Sold_Dwellings', 'value_col': 'SoldHomes_4'},
            {'slice_id': 'Average_Price', 'value_col': 'AveragePurchasePrice_7'},
        ],
    },

    # Monthly observations only; quarterly and yearly rows are excluded.
    'cbs_Government_Finance_Key_Figures': {
        "slices": [
            {'slice_id': 'Revenue', 'value_col': 'GovernmentRevenue_1'},
            {'slice_id': 'Expenditure', 'value_col': 'GovernmentExpenditure_2'},
            {'slice_id': 'Balance', 'value_col': 'BalanceOfTheGeneralGovernmentSector_3'},
            {'slice_id': 'Debt', 'value_col': 'MaastrichtDebtEMU_4'},
        ],
    },

    # Selected industry and manufacturing sectors.
    'cbs_Industry_Production_Sales': {
        "slices": [
            {'slice_id': 'Total_Industry',
             'filters': {'SectorBranchesSIC2008': ['B-E Industry (no construction), energy']},
             'value_col': 'TotalTurnover_4'},
            {'slice_id': 'Manufacturing',
             'filters': {'SectorBranchesSIC2008': ['C Manufacturing']},
             'value_col': 'TotalTurnover_4'},
            {'slice_id': 'Food_Products',
             'filters': {'SectorBranchesSIC2008': ['10 Manufacture of food products']},
             'value_col': 'TotalTurnover_4'},
            {'slice_id': 'Chemicals',
             'filters': {'SectorBranchesSIC2008': ['20 Manufacture of chemicals and products']},
             'value_col': 'TotalTurnover_4'},
            {'slice_id': 'Machinery',
             'filters': {'SectorBranchesSIC2008': ['28 Manufacture of machinery and equipment n.e.c.']},
             'value_col': 'TotalTurnover_4'},
        ],
    },

    # Investment volume index.
    'cbs_Investment_Tangible_Assets_Volume': {
        "slices": [
            {'slice_id': 'Volume_Index', 'value_col': 'VolumeIndex_1'},
        ],
    },

    # Configured motor vehicle turnover series.
    'cbs_Manufacturing_Turnover': {
        "slices": [
            {'slice_id': 'Motor_Vehicles',
             'filters': {'SectorBranchesSIC2008': ['45 Sale and repair of motor vehicles']},
             'value_col': 'TurnoverIndices_1'},
        ],
    },

    # Configured sector-specific turnover series.
    'cbs_Retail_Turnover': {
        "slices": [
            {'slice_id': 'Total_Industry',
             'filters': {'SectorsBranchesSIC2008': ['B-E Industry (no construction), energy']},
             'value_col': 'TotalTurnover_4'},
            {'slice_id': 'Manufacturing',
             'filters': {'SectorsBranchesSIC2008': ['C Manufacturing']},
             'value_col': 'TotalTurnover_4'},
        ],
    },

    # Terms of trade and import/export price indices.
    'cbs_Terms_of_Trade_Goods': {
        "slices": [
            {'slice_id': 'Terms_of_Trade', 'value_col': 'TermsOfTrade_1'},
            {'slice_id': 'Import_Prices', 'value_col': 'ImportPrices_2'},
            {'slice_id': 'Export_Prices', 'value_col': 'ExportPrices_3'},
        ],
    },

    # Configured CAO wage series and current-figure filters.
    'cbs_Wages_CAO_Total': {
        "slices": [
            {'slice_id': 'Total_Monthly_Incl',
             'filters': {'CaoSectors': ['Total Collective labour agreement sector'],
                         'SectorBranchesSIC2008': ['A-U All economic activities'],
                         'Version': ['Current figures']},
             'value_col': 'MonthlyCaoWagesInclSpecialPayments_2'},
            {'slice_id': 'Private_Sector',
             'filters': {'CaoSectors': ['Sector private companies'],
                         'SectorBranchesSIC2008': ['A-U All economic activities'],
                         'Version': ['Current figures']},
             'value_col': 'MonthlyCaoWagesInclSpecialPayments_2'},
        ],
    },

    # Selected civil engineering categories.
    'cbs_Civil_Engineering_Input_Price_Index': {
        "slices": [
            {'slice_id': 'Total_Civil_Engineering',
             'filters': {'GWW': ['42/43 Civil engineering works']},
             'value_col': 'InputPriceIndices_1'},
            {'slice_id': 'Road_Construction_Brick',
             'filters': {'GWW': ['4211a Road construction; brick paving']},
             'value_col': 'InputPriceIndices_1'},
            {'slice_id': 'Road_Construction_Asphalt',
             'filters': {'GWW': ['4211b Road construction; asphalt paving']},
             'value_col': 'InputPriceIndices_1'},
            {'slice_id': 'Railways',
             'filters': {'GWW': ['4212 Above and underground railways']},
             'value_col': 'InputPriceIndices_1'},
        ],
    },

    # Total enterprise size and selected construction sectors.
    'cbs_Construction_Production_Index': {
        "slices": [
            {'slice_id': 'Total_Construction',
             'filters': {'EnterpriseSize': ['Total 1 or more employed persons'],
                         'SectorBranchesSIC2008': ['F Construction']},
             'value_col': 'TurnoverIndices_1'},
            {'slice_id': 'Construction_Buildings',
             'filters': {'EnterpriseSize': ['Total 1 or more employed persons'],
                         'SectorBranchesSIC2008': ['41 Construction buildings, development']},
             'value_col': 'TurnoverIndices_1'},
            {'slice_id': 'Civil_Engineering',
             'filters': {'EnterpriseSize': ['Total 1 or more employed persons'],
                         'SectorBranchesSIC2008': ['42 Civil engineering']},
             'value_col': 'TurnoverIndices_1'},
        ],
    },

    # Total sales and selected manufacturing sectors.
    'cbs_Producer_Price_Index_Manufacturing': {
        "slices": [
            {'slice_id': 'Total_Industry',
             'filters': {'Sales': ['Total sales'],
                         'SBI08SectionsAndDivisions': ['B-E Industry (no construction), energy']},
             'value_col': 'PriceIndexNumbersExcludingExcise_1'},
            {'slice_id': 'Manufacturing',
             'filters': {'Sales': ['Total sales'], 'SBI08SectionsAndDivisions': ['C Manufacturing']},
             'value_col': 'PriceIndexNumbersExcludingExcise_1'},
            {'slice_id': 'Food_Products',
             'filters': {'Sales': ['Total sales'],
                         'SBI08SectionsAndDivisions': ['10 Manufacture of food products']},
             'value_col': 'PriceIndexNumbersExcludingExcise_1'},
            {'slice_id': 'Chemicals',
             'filters': {'Sales': ['Total sales'],
                         'SBI08SectionsAndDivisions': ['20 Manufacture of chemicals and products']},
             'value_col': 'PriceIndexNumbersExcludingExcise_1'},
        ],
    },

    # Additional configured PPI series.
    'cbs_PPI_Output_2021': {
        "slices": [
            {'slice_id': 'Total_Industry',
             'filters': {'Sales': ['Total sales'],
                         'SBI08SectionsAndDivisions': ['B-E Industry (no construction), energy']},
             'value_col': 'PriceIndexNumbersExcludingExcise_1'},
            {'slice_id': 'Manufacturing',
             'filters': {'Sales': ['Total sales'], 'SBI08SectionsAndDivisions': ['C Manufacturing']},
             'value_col': 'PriceIndexNumbersExcludingExcise_1'},
        ],
    },

    # Total, goods, and services consumption.
    'cbs_Household_Consumption_Expenditure': {
        "slices": [
            {'slice_id': 'Total_Consumption',
             'filters': {'ConsumptionByHouseholds': ['Domestic consumption by households']},
             'value_col': 'Indices_1'},
            {'slice_id': 'Goods',
             'filters': {'ConsumptionByHouseholds': ['Consumption of goods by households']},
             'value_col': 'Indices_1'},
            {'slice_id': 'Services',
             'filters': {'ConsumptionByHouseholds': ['Consumption of services by households']},
             'value_col': 'Indices_1'},
        ],
    },

    # Selected legal forms and bankruptcy counts.
    'cbs_Bankruptcies_Key_Figures': {
        "slices": [
            {'slice_id': 'Total',
             'filters': {'TypeOfBankruptcy': ['Total legal forms all nationalities']},
             'value_col': 'PronouncedBankruptcies_1'},
            {'slice_id': 'Companies',
             'filters': {'TypeOfBankruptcy': ['Companies and institutions']},
             'value_col': 'PronouncedBankruptcies_1'},
            {'slice_id': 'Individuals',
             'filters': {'TypeOfBankruptcy': ['Nat. person with sole proprietorship']},
             'value_col': 'PronouncedBankruptcies_1'},
        ],
    },

    # Births, deaths, and configured population growth series.
    'cbs_Population_Dynamics_Monthly': {
        "slices": [
            {'slice_id': 'Live_Births', 'value_col': 'LiveBirths_2'},
            {'slice_id': 'Deaths', 'value_col': 'Deaths_3'},
            {'slice_id': 'Net_Migration', 'value_col': 'TotalPopulationGrowth_7'},
        ],
    },

}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the configured cleaning batch and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing raw CBS CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for cleaned monthly CSV files.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_CONFIGS),
        metavar="DATASET",
        help="Clean a selected dataset; repeat this option to select multiple datasets.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = (
        list(dict.fromkeys(args.dataset))
        if args.dataset
        else list(DATASET_CONFIGS)
    )

    LOGGER.info("Input directory: %s", input_dir)
    LOGGER.info("Output directory: %s", output_dir)

    datasets_processed = 0
    slices_written = 0
    expected_slices = 0
    incomplete_datasets = []

    for dataset_name in selected_names:
        config = DATASET_CONFIGS[dataset_name]
        slice_configs = config["slices"]
        expected_slices += len(slice_configs)
        raw_path = input_dir / f"{dataset_name}.csv"

        if not raw_path.is_file():
            LOGGER.warning("Raw CSV is missing: %s", raw_path)
            incomplete_datasets.append(dataset_name)
            continue

        try:
            written_paths = clean_dataset(
                raw_path,
                dataset_name,
                slice_configs,
                output_dir=output_dir,
            )
        except Exception:
            LOGGER.exception("Failed to clean dataset %s", dataset_name)
            incomplete_datasets.append(dataset_name)
            continue

        datasets_processed += 1
        slices_written += len(written_paths)
        if len(written_paths) != len(slice_configs):
            incomplete_datasets.append(dataset_name)

    LOGGER.info(
        "Batch summary: %d/%d datasets processed; %d/%d slices written.",
        datasets_processed,
        len(selected_names),
        slices_written,
        expected_slices,
    )

    if incomplete_datasets:
        LOGGER.error(
            "Incomplete datasets: %s",
            ", ".join(incomplete_datasets),
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
