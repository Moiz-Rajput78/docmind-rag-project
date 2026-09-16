from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


# ============================================================
# EVALUATION FILES
# ============================================================

EVALUATION_FILES = {
    "hybrid": "hybrid_retrieval_experiment.csv",
    "hybrid_weights": "hybrid_weight_experiment.csv",
    "reranker": "reranker_metrics_experiment.csv",
    "reranker_summary": "reranker_experiment.csv",
    "distance": "retrieval_distance_experiment.csv",
    "chunking": "chunking_experiment_results.csv",
}


# ============================================================
# LOAD SINGLE CSV
# ============================================================

def load_evaluation_csv(
    filename: str,
) -> pd.DataFrame:
    """
    Load an evaluation CSV from the project's docs directory.

    Returns an empty DataFrame if the file does not exist
    or cannot be read.
    """

    path = (
        PROJECT_ROOT
        / "docs"
        / filename
    )

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD ALL EVALUATION DATA
# ============================================================

def get_evaluation_data() -> dict[str, pd.DataFrame]:
    """
    Load all evaluation experiment datasets used
    by the DocMind Evaluation Dashboard.
    """

    return {
        key: load_evaluation_csv(filename)
        for key, filename in EVALUATION_FILES.items()
    }


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
) -> Optional[float]:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# PERCENTAGE
# ============================================================

def percent(
    value,
) -> str:
    """
    Convert a 0-1 metric to percentage text.
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.1f}%"

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


# ============================================================
# METRIC DELTA
# ============================================================

def render_metric_delta(
    current,
    baseline,
) -> Optional[str]:
    """
    Return a readable percentage-point delta.
    """

    current_value = safe_float(
        current
    )

    baseline_value = safe_float(
        baseline
    )

    if (
        current_value is None
        or baseline_value is None
    ):
        return None

    delta = (
        current_value
        - baseline_value
    )

    return f"{delta * 100:+.1f} pp"


# ============================================================
# COLUMN CHECK
# ============================================================

def has_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> bool:
    """
    Check whether a DataFrame contains all requested columns.
    """

    if dataframe.empty:
        return False

    return all(
        column in dataframe.columns
        for column in columns
    )


# ============================================================
# AVAILABLE COLUMNS
# ============================================================

def available_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    """
    Return only columns that actually exist in the DataFrame.
    """

    if dataframe.empty:
        return []

    return [
        column
        for column in columns
        if column in dataframe.columns
    ]


# ============================================================
# MISSING DATA MESSAGE TEXT
# ============================================================

def missing_evaluation_message(
    filename: str,
) -> str:
    """
    Return a standard message for missing evaluation data.
    """

    return (
        f"Evaluation data `{filename}` "
        "was not found in the `docs/` directory."
    )