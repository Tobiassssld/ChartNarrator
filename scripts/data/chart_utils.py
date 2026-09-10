#!/usr/bin/env python3
"""Utilities for generating ChartNarrator charts from cleaned CBS time series.

The functions implement the original research pipeline's financial processing,
random window sampling, chart-type selection, and adaptive rendering.

Inputs:
    Cleaned CSV files containing date, value, dataset, and slice_id.
Outputs:
    PNG charts in data/images by default, and optional CSV summary statistics.

Research behavior:
    The seven configured flow series use a 12-month rolling sum; other series
    use a 12-month moving average. Windows shorter than 24 observations use
    raw values, while longer windows use the precomputed trend. The original
    window probabilities, plotting styles, and filename convention are retained.

This module is used by generate_charts.py and is not a standalone batch runner.
"""

import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib

# Select a non-interactive backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
PathLike = Union[str, Path]

ROLLING_SUM_FILES = ['cbs_Bankruptcies_Key_Figures_Companies.csv',
 'cbs_Bankruptcies_Key_Figures_Individuals.csv',
 'cbs_Bankruptcies_Key_Figures_Total.csv',
 'cbs_Existing_Homes_Prices_Sales_AvgPrice_Sold_Dwellings.csv',
 'cbs_Population_Dynamics_Monthly_Deaths.csv',
 'cbs_Population_Dynamics_Monthly_Live_Births.csv',
 'cbs_Population_Dynamics_Monthly_Net_Migration.csv']

PLOT_STYLES = ['seaborn-v0_8-whitegrid',
 'seaborn-v0_8-darkgrid',
 'ggplot',
 'bmh',
 'fivethirtyeight',
 'seaborn-v0_8-ticks']

COLOR_POOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

MARKER_STYLES = ['o', 's', '^', 'v', 'D', 'p']

GRID_STYLES = [{'linestyle': '--', 'alpha': 0.3, 'linewidth': 0.8},
 {'linestyle': ':', 'alpha': 0.4, 'linewidth': 0.6},
 {'linestyle': '-', 'alpha': 0.2, 'linewidth': 0.5}]

WINDOW_STRATEGIES = {'micro': {'range': (12, 24), 'probability': 0.2},
 'cycle': {'range': (24, 96), 'probability': 0.6},
 'trend': {'range': (96, 240), 'probability': 0.2}}

def load_cleaned_csv(csv_path: PathLike):
    """Load and validate a cleaned CBS time series.

    Args:
        csv_path: Path to a CSV produced by clean_cbs_data.py.

    Returns:
        A tuple of (dataframe, dataset_name, slice_id). On a loading or
        validation failure, returns (None, None, None), preserving the
        original helper contract.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["date", "value"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid date/value observations remain.")

        dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else "Unknown"
        slice_id = df["slice_id"].iloc[0] if "slice_id" in df.columns else "Unknown"
        return df, dataset_name, slice_id
    except Exception:
        LOGGER.exception("Failed to load cleaned CSV: %s", csv_path)
        return None, None, None


def apply_financial_processing(df: pd.DataFrame, csv_filename: str):
    """Add the original 12-month financial processing series.

    Flow files in ROLLING_SUM_FILES use a rolling sum; all other files use
    a moving average. Both require 12 observations before producing a value.

    Returns:
        A copy of the dataframe with trend_value and the processing type.
    """
    df = df.copy()
    if csv_filename in ROLLING_SUM_FILES:
        df["trend_value"] = df["value"].rolling(window=12, min_periods=12).sum()
        return df, "rolling_sum"

    df["trend_value"] = df["value"].rolling(window=12, min_periods=12).mean()
    return df, "moving_average"


def generate_random_slice(
    df: pd.DataFrame,
    min_months: Optional[int] = None,
    max_months: Optional[int] = None,
    rng: Optional[random.Random] = None,
):
    """Sample a time window using the original 20/60/20 strategy.

    The micro, cycle, and trend ranges are 12-24, 24-96, and 96-240
    observations. Availability-dependent fallbacks are preserved.

    Args:
        df: Full time series to sample.
        min_months: Optional override of the selected minimum window size.
        max_months: Optional override of the selected maximum window size.
        rng: Optional random generator for reproducible sampling. When omitted,
            the original module-level random generator is used.

    Returns:
        A copy of the selected dataframe window and its strategy label.
    """
    rng = random if rng is None else rng
    total_points = len(df)

    if total_points <= 12:
        return df.copy(), "full"

    rand = rng.random()

    if rand < 0.20 and total_points > 24:
        strategy = "micro"
        min_w, max_w = WINDOW_STRATEGIES["micro"]["range"]
    elif rand < 0.80 and total_points > 48:
        strategy = "cycle"
        min_w, max_w = WINDOW_STRATEGIES["cycle"]["range"]
    else:
        strategy = "trend"
        min_w, max_w = WINDOW_STRATEGIES["trend"]["range"]

    if min_months is not None:
        min_w = min_months
    if max_months is not None:
        max_w = max_months

    min_w = min(min_w, total_points)
    max_w = min(max_w, total_points)

    if min_w >= max_w:
        return df.copy(), "full"

    window_size = rng.randint(min_w, max_w)
    max_start = total_points - window_size
    start_idx = rng.randint(0, max(0, max_start))
    slice_df = df.iloc[start_idx: start_idx + window_size].copy()
    return slice_df, strategy


def infer_chart_type(df: pd.DataFrame) -> str:
    """Select a diverging bar chart for negative data, otherwise a line chart."""
    has_negative = (df["value"] < 0).any()
    return "bar_diverging" if has_negative else "line"


def infer_value_semantics(dataset_name: str, slice_id: str, csv_filename: str) -> str:
    """Infer the original count, index, rate, level, or generic value label."""
    name = f"{dataset_name} {slice_id} {csv_filename}".lower()

    if csv_filename in ROLLING_SUM_FILES:
        return "count"
    if "index" in name:
        return "index"
    if "%" in name or "percent" in name or "rate" in name or "ratio" in name:
        return "rate"
    if (
        "price" in name
        or "wage" in name
        or "salary" in name
        or "amount" in name
        or "euro" in name
    ):
        return "level"
    return "value"


def format_ylabel(value_semantics: str, plot_column: str, processing_type: str) -> str:
    """Format the original axis label for the selected value representation."""
    base = {
        "count": "Count",
        "index": "Index (points)",
        "rate": "Rate (%)",
        "level": "Value",
        "value": "Value",
    }.get(value_semantics, "Value")

    if plot_column != "trend_value":
        return base
    if processing_type == "rolling_sum":
        return "12M rolling sum (count)"
    if processing_type == "moving_average":
        if value_semantics == "index":
            return "12M moving avg (index)"
        if value_semantics == "rate":
            return "12M moving avg (rate)"
        if value_semantics == "count":
            return "12M moving avg (count)"
        return "12M moving avg"
    return base


def select_plot_data(df: pd.DataFrame, time_horizon_months: int) -> str:
    """Select raw values below 24 observations and trend values otherwise."""
    if time_horizon_months < 24:
        return "value"
    return "trend_value"


def build_chart_stem(dataset_name: str, slice_id: str, chart_index: int) -> str:
    """Return the public chart filename stem without a duplicated cbs_ prefix."""
    dataset_stem = dataset_name.removeprefix("cbs_")
    return f"cbs_{dataset_stem}_{slice_id}_{chart_index:03d}"


def render_chart(
    df: pd.DataFrame,
    dataset_name: str,
    slice_id: str,
    csv_filename: str,
    preprocessed: bool = False,
    chart_type: Optional[str] = None,
    output_dir: PathLike = DEFAULT_IMAGE_DIR,
    save_name: Optional[str] = None,
    strategy: str = "unknown",
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Render one time-series chart using the original research rules.

    Args:
        df: Selected time-series window, optionally with trend_value.
        dataset_name: Dataset identifier from the cleaned CSV.
        slice_id: Series identifier from the cleaned CSV.
        csv_filename: Source filename used to select financial processing.
        preprocessed: Whether trend_value was computed on the full history.
        chart_type: Optional line or bar_diverging override.
        output_dir: Destination for PNG files.
        save_name: Optional filename stem before the date suffix.
        strategy: Window strategy label, retained for caller compatibility.
        rng: Optional generator for deterministic visual-style selection.

    Returns:
        String path to the saved PNG, or None when rendering fails.

    Notes:
        Filename prefixes, time-window boundaries, smoothing behavior, and
        visual styles are retained to match the original experimental pipeline.
    """
    rng = random if rng is None else rng
    fig = None
    try:
        if df is None or len(df) < 3:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if (not preprocessed) or ("trend_value" not in df.columns):
            df, processing_type = apply_financial_processing(df, csv_filename)
        else:
            processing_type = "preprocessed"

        time_horizon_months = len(df)
        plot_column = select_plot_data(df, time_horizon_months)

        if chart_type is None:
            chart_type = infer_chart_type(df)

        start_date = df["date"].min()
        end_date = df["date"].max()
        start_str = start_date.strftime("%Y%m")
        end_str = end_date.strftime("%Y%m")

        if save_name is None:
            filename = f"cbs_{dataset_name}_{slice_id}_{start_str}_{end_str}.png"
        else:
            filename = f"{save_name}_{start_str}_{end_str}.png"
        filepath = output_dir / filename

        style = rng.choice(PLOT_STYLES)
        color = rng.choice(COLOR_POOL)
        marker = rng.choice(MARKER_STYLES) if len(df) < 40 else None
        grid_style = rng.choice(GRID_STYLES)

        # Scope the style to this figure instead of changing global pyplot state.
        # Preserve the original cumulative pyplot style selection.
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=(10, 6))
        dates = df["date"]
        values = df[plot_column]

        # Drop the initial undefined trend values; retain the raw fallback.
        if plot_column == "trend_value":
            df_plot = df.dropna(subset=["trend_value"]).copy()
            if len(df_plot) >= 3:
                dates = df_plot["date"]
                values = df_plot["trend_value"]
            else:
                plot_column = "value"
                dates = df["date"]
                values = df["value"]

        if chart_type == "bar_diverging":
            bar_colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]
            ax.bar(dates, values, color=bar_colors, width=20, alpha=0.8)
            ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
        else:
            ax.plot(
                dates,
                values,
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=5 if marker else 0,
                linestyle="-",
                alpha=0.9,
            )

        title_start = start_date.strftime("%Y-%m")
        title_end = end_date.strftime("%Y-%m")
        title = (
            f"Netherlands {dataset_name.replace('_', ' ')}\n"
            f"{slice_id.replace('_', ' ')}\n"
            f"({title_start} to {title_end})"
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        n_points = len(df)
        if n_points <= 24:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        elif n_points <= 60:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif n_points <= 120:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())

        if n_points <= 60:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=45,
            ha="right",
            fontsize=9,
        )

        value_semantics = infer_value_semantics(dataset_name, slice_id, csv_filename)
        ax.set_ylabel(
            format_ylabel(value_semantics, plot_column, processing_type),
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=9)
        ax.grid(True, **grid_style)

        plt.tight_layout()
        fig.savefig(filepath, dpi=100, bbox_inches="tight")

        LOGGER.debug(
            "Rendered %s (strategy=%s, processing=%s, points=%d)",
            filepath,
            strategy,
            processing_type,
            len(df),
        )
        return str(filepath)
    except Exception:
        LOGGER.exception("Failed to render chart for %s / %s", dataset_name, slice_id)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def generate_charts_from_csv(
    csv_path: PathLike,
    num_charts: int = 35,
    output_dir: PathLike = DEFAULT_IMAGE_DIR,
    rng: Optional[random.Random] = None,
) -> int:
    """Generate multiple random-window charts from one cleaned CSV.

    The full-history trend is computed before sampling, preserving the
    original treatment of rolling windows at the start of each chart.

    Args:
        csv_path: Path to a cleaned time-series CSV.
        num_charts: Number of chart attempts.
        output_dir: Directory for generated PNG files.
        rng: Optional generator shared by sampling and visual-style selection.

    Returns:
        Number of successful chart renders.
    """
    if num_charts < 0:
        raise ValueError("num_charts must be non-negative.")

    df, dataset_name, slice_id = load_cleaned_csv(csv_path)
    if df is None or len(df) < 3:
        return 0

    csv_filename = Path(csv_path).name
    df_full, _processing_type = apply_financial_processing(df, csv_filename)
    success_count = 0

    for i in range(num_charts):
        slice_df, strategy = generate_random_slice(df_full, rng=rng)
        save_name = build_chart_stem(dataset_name, slice_id, i)

        result = render_chart(
            slice_df,
            dataset_name,
            slice_id,
            csv_filename,
            preprocessed=True,
            chart_type=None,
            output_dir=output_dir,
            save_name=save_name,
            strategy=strategy,
            rng=rng,
        )
        if result:
            success_count += 1

    return success_count


def analyze_csv(csv_path: PathLike) -> Optional[dict]:
    """Return summary statistics for one cleaned CSV, or None if invalid."""
    df, dataset_name, slice_id = load_cleaned_csv(csv_path)
    if df is None:
        return None

    return {
        "dataset": dataset_name,
        "slice_id": slice_id,
        "points": len(df),
        "start_date": df["date"].min().strftime("%Y-%m"),
        "end_date": df["date"].max().strftime("%Y-%m"),
        "min_value": float(df["value"].min()),
        "max_value": float(df["value"].max()),
        "mean_value": float(df["value"].mean()),
        "has_negative": bool((df["value"] < 0).any()),
    }
