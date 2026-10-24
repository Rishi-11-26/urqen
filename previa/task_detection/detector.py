"""
previa/task_detection/detector.py
----------------------------------
Automatically identifies the likely target column and ML task type.

The detector answers:
  "Which column is probably the target, and what kind of ML problem is this?"

Task types
----------
  "binary_classification"     : target has exactly 2 distinct values
  "multiclass_classification" : target has 3–N distinct values (categorical-like)
  "regression"                : target is continuous numerical

Confidence levels (heuristic, NOT a calibrated probability)
-------------------
  "high"   : multiple strong signals agree
  "medium" : some signals present but not conclusive
  "low"    : minimal signals, result is a best guess

Design principles
-----------------
- Evidence-based: every decision is supported by a list of reasons.
- Honest: confidence is heuristic. Never claim certainty.
- Transparent: the reasons list explains WHY each column was scored.
- Skippable: if nothing is found, return a clear "no_target" result rather
  than crashing or making a wild guess.

Scoring logic summary
---------------------
We score every column as a candidate target. Higher score = more likely target.
Positive evidence:
  +3  column name ends with common target keywords (target, label, outcome, etc.)
  +2  column is the LAST column (convention in many datasets)
  +2  binary values {0,1} or {"yes","no"} etc.
  +1  small number of unique values relative to rows (looks categorical)
  +1  not flagged as a likely identifier
Negative evidence:
  -3  looks like a row ID (near-100% unique non-numeric)
  -2  column name matches identifier patterns (id, index, key, etc.)
  -1  more than 50% missing values

The highest-scoring column is selected as the target.
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column name patterns that strongly suggest a target column
TARGET_NAME_KEYWORDS = re.compile(
    r"\b(target|label|class|outcome|output|y|result|response|"
    r"survived|purchased|churn|fraud|spam|default|approved|status|"
    r"price|salary|revenue|sales|score|grade|rating|value|amount|"
    r"cost|profit|loss|demand|count|quantity|tip|fare|charges)\b",
    re.IGNORECASE,
)

# Column name patterns that strongly suggest a row identifier
IDENTIFIER_NAME_KEYWORDS = re.compile(
    r"\b(id|index|idx|key|row|num|no|number|serial|code)\b",
    re.IGNORECASE,
)

# How many unique values can a classification target have at most?
MAX_CLASSIFICATION_UNIQUE = 20

# If unique values / n_rows > this, the column is likely continuous (regression)
REGRESSION_UNIQUE_RATIO = 0.05

# Common binary value sets (stored lowercase for comparison)
BINARY_VALUE_SETS = [
    {0, 1}, {"0", "1"},
    {"yes", "no"}, {"true", "false"},
    {"positive", "negative"}, {"pass", "fail"},
    {"male", "female"}, {"m", "f"},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_task_for_target(series: pd.Series, n_rows: int | None = None) -> tuple[str, list[str]]:
    """
    Public function to determine the task type ('regression', 'binary_classification',
    'multiclass_classification') for a specific target column.
    """
    rows = n_rows if n_rows is not None else len(series)
    return _classify_task(series, rows)


def detect_target_and_task(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """
    Identify the most likely target column and the ML task type.
    If target_column is specified and present in df, that column is evaluated directly.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (should not include preprocessing).
    target_column : str or None
        Optional override column to evaluate as target.

    Returns
    -------
    dict with keys:
        "target_column"  : str or None
        "task_type"      : "binary_classification" | "multiclass_classification"
                           | "regression" | "unknown"
        "confidence"     : "high" | "medium" | "low"
        "reasons"        : list[str]  — human-readable evidence
        "all_scores"     : dict[str -> float]  — score per column (for transparency)
    """
    n_rows, n_cols = df.shape

    if n_cols == 0:
        return _no_target(["Dataset has no columns."])

    # If target_column is explicitly requested and present
    if target_column and target_column in df.columns:
        task_type, task_reasons = _classify_task(df[target_column], n_rows)
        return {
            "target_column": target_column,
            "task_type": task_type,
            "confidence": "high",
            "reasons": [f"Target column '{target_column}' selected by user."] + task_reasons,
            "all_scores": {target_column: 10.0},
        }

    scores = {}
    column_reasons = {}

    for i, col in enumerate(df.columns):
        score, reasons = _score_column(col, df[col], i, n_cols, n_rows)
        scores[col] = score
        column_reasons[col] = reasons

    # Pick the column with the highest score
    best_col = max(scores, key=lambda c: scores[c])
    best_score = scores[best_col]

    # If the best score is very low, we default to the last column if it is numeric or categorical
    if best_score <= 0:
        last_col = df.columns[-1]
        task_type, task_reasons = _classify_task(df[last_col], n_rows)
        return {
            "target_column": last_col,
            "task_type": task_type,
            "confidence": "low",
            "reasons": ["Defaulted to last column as target."] + task_reasons,
            "all_scores": scores,
        }

    # Determine task type based on the winning column
    task_type, task_reasons = _classify_task(df[best_col], n_rows)

    # Determine confidence
    confidence = _confidence(best_score, scores)

    return {
        "target_column": best_col,
        "task_type": task_type,
        "confidence": confidence,
        "reasons": column_reasons[best_col] + task_reasons,
        "all_scores": scores,
    }


# ---------------------------------------------------------------------------
# Internal: column scoring
# ---------------------------------------------------------------------------

def _score_column(
    col: str,
    series: pd.Series,
    col_index: int,
    n_cols: int,
    n_rows: int,
) -> tuple[float, list[str]]:
    """
    Assign a score and list of evidence reasons to a column.

    Returns (score, reasons).
    """
    score = 0.0
    reasons = []

    # --- Positive: target-like name ---
    if TARGET_NAME_KEYWORDS.search(col):
        score += 3
        reasons.append(f"Column name '{col}' matches common target keywords.")

    # --- Positive: last column position (common convention) ---
    if col_index == n_cols - 1:
        score += 2
        reasons.append("Column is the last column (common convention for target).")

    # --- Negative: identifier-like name ---
    if IDENTIFIER_NAME_KEYWORDS.search(col):
        score -= 2
        reasons.append(f"Column name '{col}' matches identifier keywords (e.g. id, index).")

    # --- Missing values ---
    missing_pct = series.isnull().mean() * 100
    if missing_pct > 50:
        score -= 1
        reasons.append(f"Column has {missing_pct:.1f}% missing — less likely to be a target.")

    non_null = series.dropna()
    if len(non_null) == 0:
        score -= 2
        reasons.append("Column is entirely missing.")
        return score, reasons

    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / n_rows if n_rows > 0 else 0.0

    # --- Negative: looks like a row ID ---
    # For non-numeric columns: near-100% uniqueness strongly suggests an ID.
    # For numeric columns: we do NOT apply this penalty because a regression
    # target naturally has many unique values. We rely on name/position signals instead.
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if unique_ratio >= 0.9 and n_rows > 10 and not is_numeric:
        score -= 3
        reasons.append(
            f"Column has {unique_ratio*100:.0f}% unique values — likely a row identifier."
        )
        return score, reasons

    # --- Positive: binary values ---
    lower_vals = {str(v).lower().strip() for v in non_null.unique()}
    if _is_binary_set(lower_vals):
        score += 2
        reasons.append(f"Column has binary-like values: {lower_vals}.")

    # --- Positive: small unique count (classification-like) ---
    elif unique_count <= MAX_CLASSIFICATION_UNIQUE:
        score += 1
        reasons.append(
            f"Column has {unique_count} unique value(s) — could be a classification target."
        )

    return score, reasons


# ---------------------------------------------------------------------------
# Internal: task classification
# ---------------------------------------------------------------------------

def _classify_task(series: pd.Series, n_rows: int) -> tuple[str, list[str]]:
    """
    Decide whether the target column indicates classification or regression.

    Returns (task_type, reasons).
    """
    non_null = series.dropna()
    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / n_rows if n_rows > 0 else 0.0
    is_numeric = pd.api.types.is_numeric_dtype(series)

    reasons = []

    # Binary classification
    lower_vals = {str(v).lower().strip() for v in non_null.unique()}
    if _is_binary_set(lower_vals) or unique_count == 2:
        reasons.append("Target has 2 distinct values → binary classification.")
        return "binary_classification", reasons

    # Non-numeric with > 2 categories is multiclass classification
    if not is_numeric:
        reasons.append(
            f"Target is categorical with {unique_count} distinct classes → multiclass classification."
        )
        return "multiclass_classification", reasons

    # Numeric target: check if it behaves as discrete class labels
    try:
        has_fractions = any(float(v) % 1 != 0 for v in non_null)
        min_val = float(non_null.min())
        max_val = float(non_null.max())
    except (ValueError, TypeError):
        has_fractions = False
        min_val = 0
        max_val = 100

    is_discrete_small_classes = (
        not has_fractions
        and unique_count <= MAX_CLASSIFICATION_UNIQUE
        and min_val >= 0
        and max_val <= MAX_CLASSIFICATION_UNIQUE
        and unique_ratio < 0.7
    )

    if is_discrete_small_classes:
        reasons.append(
            f"Target has {unique_count} distinct small integer values ({int(min_val)} to {int(max_val)}) → multiclass classification."
        )
        return "multiclass_classification", reasons

    # Otherwise continuous numeric is regression
    reasons.append(
        f"Target is numeric with continuous values ({unique_count} unique values) → regression."
    )
    return "regression", reasons


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_binary_set(value_set: set) -> bool:
    """Return True if the value set matches a known binary pair."""
    for binary_pair in BINARY_VALUE_SETS:
        if value_set == binary_pair or value_set.issubset(binary_pair):
            return True
    return False


def _confidence(best_score: float, all_scores: dict) -> str:
    """
    Determine confidence from the winning score and the gap to second place.

    The larger the gap, the more confident we are.
    """
    sorted_scores = sorted(all_scores.values(), reverse=True)
    gap = best_score - (sorted_scores[1] if len(sorted_scores) > 1 else 0)

    if best_score >= 4 and gap >= 2:
        return "high"
    elif best_score >= 2 or gap >= 1:
        return "medium"
    else:
        return "low"


def _no_target(reasons: list[str]) -> dict:
    """Return a structured 'no target found' result."""
    return {
        "target_column": None,
        "task_type": "unknown",
        "confidence": "low",
        "reasons": reasons,
        "all_scores": {},
    }
