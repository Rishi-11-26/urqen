"""
previa/feature_intelligence/analyzer.py
-----------------------------------------
Characterizes every feature in the dataset individually.

The feature analyzer answers:
  "What is each feature, and what are its key properties?"

It assigns a feature type to each column and computes type-specific statistics.
This output is used by the Preprocessing Recommender to make decisions.

Feature types assigned
-----------------------
  "numeric"      : numeric dtype, more than 2 unique values
  "boolean"      : exactly 2 unique values (including 0/1, True/False, yes/no)
  "categorical"  : non-numeric, more than 2 unique values
  "constant"     : only 1 unique value
  "datetime"     : pandas datetime dtype

For each feature the output contains:
  - feature_type
  - missing_count, missing_pct
  - unique_count, unique_pct
  - For numeric: mean, median, std, min, max, skewness
  - For categorical/boolean: top_value, top_freq, top_pct, n_categories
"""

import pandas as pd
import numpy as np


# Recognized boolean value pairs (lowercase)
BOOLEAN_VALUE_SETS = [
    {0, 1}, {"0", "1"}, {"true", "false"},
    {"yes", "no"}, {"y", "n"}, {"male", "female"}, {"m", "f"},
    {"positive", "negative"}, {"pass", "fail"},
]


def analyze_features(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """
    Analyze every column in the DataFrame and return a feature profile.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    target_column : str or None
        If provided, this column is marked as the target and still analyzed
        but labeled "target" in its feature_type.

    Returns
    -------
    dict[str -> dict]
        Keys are column names. Values are per-feature stat dicts.
    """
    results = {}

    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        info = _analyze_column(series, col, n_rows)

        # Mark target column explicitly (useful for the UI)
        if col == target_column:
            info["is_target"] = True
        else:
            info["is_target"] = False

        results[col] = info

    return results


# ---------------------------------------------------------------------------
# Internal: single-column analysis
# ---------------------------------------------------------------------------

def _analyze_column(series: pd.Series, col_name: str, n_rows: int) -> dict:
    """Analyze one column and return its stat dict."""

    missing_count = int(series.isnull().sum())
    missing_pct = round(missing_count / n_rows * 100, 2) if n_rows > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    unique_pct = round(unique_count / n_rows * 100, 2) if n_rows > 0 else 0.0

    # Base info present for every column
    info = {
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "unique_count": unique_count,
        "unique_pct": unique_pct,
        # Type-specific fields default to None; filled in below
        "feature_type": None,
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "skewness": None,
        "n_categories": None,
        "top_value": None,
        "top_freq": None,
        "top_pct": None,
    }

    # --- Constant column (only 1 unique non-null value, or entirely null) ---
    if unique_count <= 1:
        info["feature_type"] = "constant"
        return info

    # --- Datetime ---
    if pd.api.types.is_datetime64_any_dtype(series):
        info["feature_type"] = "datetime"
        return info

    # --- Numeric ---
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()

        # Could still be boolean-like (0/1 numeric)
        lower_vals = {str(int(v)) if float(v) == int(float(v)) else str(v)
                      for v in non_null.unique()}
        if unique_count == 2 and _is_boolean_set(lower_vals):
            info["feature_type"] = "boolean"
        else:
            info["feature_type"] = "numeric"

        # Numeric stats regardless of boolean/numeric subtype
        info["mean"] = round(float(non_null.mean()), 4) if len(non_null) > 0 else None
        info["median"] = round(float(non_null.median()), 4) if len(non_null) > 0 else None
        info["std"] = round(float(non_null.std()), 4) if len(non_null) > 1 else None
        info["min"] = round(float(non_null.min()), 4) if len(non_null) > 0 else None
        info["max"] = round(float(non_null.max()), 4) if len(non_null) > 0 else None
        info["skewness"] = round(float(non_null.skew()), 4) if len(non_null) > 2 else None
        return info

    # --- Categorical or boolean (non-numeric) ---
    non_null = series.dropna()
    lower_vals = {str(v).lower().strip() for v in non_null.unique()}

    if unique_count == 2 and _is_boolean_set(lower_vals):
        info["feature_type"] = "boolean"
    else:
        info["feature_type"] = "categorical"

    # Categorical stats
    info["n_categories"] = unique_count
    if len(non_null) > 0:
        top_value = non_null.value_counts().idxmax()
        top_freq = int(non_null.value_counts().max())
        info["top_value"] = str(top_value)
        info["top_freq"] = top_freq
        info["top_pct"] = round(top_freq / n_rows * 100, 2)

    return info


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_boolean_set(value_set: set) -> bool:
    """Return True if the value set matches a known boolean pair."""
    for boolean_pair in BOOLEAN_VALUE_SETS:
        # Cast both to str for comparison
        str_pair = {str(v) for v in boolean_pair}
        if value_set == str_pair or value_set.issubset(str_pair):
            return True
    return False
