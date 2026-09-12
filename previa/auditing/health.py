"""
previa/auditing/health.py
--------------------------
Analyzes a dataset profile and identifies data-quality issues that
deserve human attention before model training.

The health analyzer answers: "What deserves attention?"

Design principles
-----------------
- Every issue is a warning, not a verdict. The language used is:
  "potential issue", "review recommended", "possible identifier" — never
  "this column is broken".
- Issues are generated from the profile dict (computed by profiler.py),
  so no raw DataFrame access is needed here.
- The output is a plain list of dicts, easy to display in Streamlit.

Issue severity levels
---------------------
  "high"   — strongly worth reviewing before modeling
  "medium" — worth noting, may or may not matter
  "low"    — informational

Detected issue types
---------------------
  - high_missing        : column has many missing values
  - duplicate_rows      : dataset contains duplicate rows
  - constant_column     : column has only one unique value (useless for ML)
  - high_cardinality    : categorical column has very many unique values
  - possible_identifier : column looks like a row ID (unique per row)
  - high_skewness       : numeric column is strongly skewed
  - potential_outliers  : numeric column may contain outliers (IQR method)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Thresholds — defined as named constants so they are easy to find/change
# ---------------------------------------------------------------------------

MISSING_HIGH_THRESHOLD = 20.0      # % missing → "high" severity
MISSING_MEDIUM_THRESHOLD = 5.0     # % missing → "medium" severity
DUPLICATE_PCT_THRESHOLD = 0.5      # % duplicates before flagging
CARDINALITY_HIGH_THRESHOLD = 50    # unique count for high-cardinality flag
CARDINALITY_HIGH_PCT = 50.0        # unique% for high-cardinality flag
IDENTIFIER_PCT_THRESHOLD = 90.0    # unique% for identifier flag
SKEWNESS_HIGH_THRESHOLD = 2.0      # |skewness| for high-skew flag
OUTLIER_IQR_MULTIPLIER = 1.5       # IQR fence multiplier


def analyze_health(profile: dict, df: pd.DataFrame) -> list[dict]:
    """
    Identify data-quality issues from the dataset profile.

    Parameters
    ----------
    profile : dict
        Output of previa.auditing.profiler.profile_dataset(df).
    df : pd.DataFrame
        The original DataFrame — needed for outlier detection (IQR).

    Returns
    -------
    list of dicts, each with keys:
        "column"      : str or None (None for dataset-level issues)
        "issue_type"  : str
        "severity"    : "high" | "medium" | "low"
        "message"     : str  (user-readable explanation)
        "evidence"    : str  (the specific statistic that triggered this)
    """
    issues = []

    summary = profile["summary"]
    columns = profile["columns"]

    # --- Dataset-level: duplicate rows ---
    if summary["n_duplicates"] > 0:
        dup_pct = summary["duplicate_pct"]
        severity = "high" if dup_pct > 5.0 else "medium"
        issues.append({
            "column": None,
            "issue_type": "duplicate_rows",
            "severity": severity,
            "message": (
                f"The dataset contains {summary['n_duplicates']} duplicate row(s) "
                f"({dup_pct}% of all rows). Duplicate rows can distort model training."
            ),
            "evidence": f"n_duplicates={summary['n_duplicates']}, duplicate_pct={dup_pct}%",
        })

    # --- Per-column issues ---
    n_rows = summary["n_rows"]

    for col, info in columns.items():
        dtype = info["dtype"]
        missing_pct = info["missing_pct"]
        unique_count = info["unique_count"]
        unique_pct = info["unique_pct"]

        # 1. Missing values
        if missing_pct >= MISSING_HIGH_THRESHOLD:
            issues.append({
                "column": col,
                "issue_type": "high_missing",
                "severity": "high",
                "message": (
                    f"Column '{col}' is missing {missing_pct}% of values. "
                    f"High missingness may reduce model reliability."
                ),
                "evidence": f"missing_pct={missing_pct}%",
            })
        elif missing_pct >= MISSING_MEDIUM_THRESHOLD:
            issues.append({
                "column": col,
                "issue_type": "high_missing",
                "severity": "medium",
                "message": (
                    f"Column '{col}' has {missing_pct}% missing values. "
                    f"Review recommended."
                ),
                "evidence": f"missing_pct={missing_pct}%",
            })

        # 2. Constant column (only one unique value — useless for ML)
        if unique_count <= 1:
            issues.append({
                "column": col,
                "issue_type": "constant_column",
                "severity": "high",
                "message": (
                    f"Column '{col}' has only {unique_count} unique value(s). "
                    f"A constant column provides no information to a model and "
                    f"should be removed."
                ),
                "evidence": f"unique_count={unique_count}",
            })
            continue  # No further checks needed for a constant column

        # 3. Possible identifier (unique per row, likely a row ID)
        #    Only flag non-numeric columns or numeric columns that look like IDs
        is_numeric = _is_numeric_dtype(dtype)
        if unique_pct >= IDENTIFIER_PCT_THRESHOLD and n_rows > 5:
            # Numeric columns with all-unique values could be IDs
            issues.append({
                "column": col,
                "issue_type": "possible_identifier",
                "severity": "medium",
                "message": (
                    f"Column '{col}' has {unique_pct}% unique values — this may be "
                    f"a row identifier (e.g. ID, index). Including identifiers in "
                    f"model training usually harms performance."
                ),
                "evidence": f"unique_pct={unique_pct}%",
            })

        # 4. High-cardinality categorical column
        #    Only relevant for non-numeric columns where OHE would explode
        elif not is_numeric and unique_count >= CARDINALITY_HIGH_THRESHOLD:
            issues.append({
                "column": col,
                "issue_type": "high_cardinality",
                "severity": "medium",
                "message": (
                    f"Column '{col}' has {unique_count} unique categories. "
                    f"One-hot encoding this many categories may cause dimensionality "
                    f"problems. Review recommended."
                ),
                "evidence": f"unique_count={unique_count}",
            })

        # 5. High skewness (numeric only)
        skewness = info.get("skewness")
        if skewness is not None and abs(skewness) >= SKEWNESS_HIGH_THRESHOLD:
            issues.append({
                "column": col,
                "issue_type": "high_skewness",
                "severity": "low",
                "message": (
                    f"Column '{col}' has a skewness of {skewness:.2f}. "
                    f"Strong skew can affect some models. A transformation "
                    f"(e.g. log) may be worth considering."
                ),
                "evidence": f"skewness={skewness:.4f}",
            })

        # 6. Potential outliers using IQR method (numeric only)
        if is_numeric and col in df.columns:
            series = df[col].dropna()
            if len(series) >= 4:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
                    upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
                    n_outliers = int(((series < lower) | (series > upper)).sum())
                    if n_outliers > 0:
                        outlier_pct = round(n_outliers / n_rows * 100, 2)
                        issues.append({
                            "column": col,
                            "issue_type": "potential_outliers",
                            "severity": "low",
                            "message": (
                                f"Column '{col}' has {n_outliers} potential outlier(s) "
                                f"({outlier_pct}%) based on the 1.5×IQR rule. "
                                f"Verify whether these are valid data points."
                            ),
                            "evidence": (
                                f"IQR fence: [{lower:.2f}, {upper:.2f}], "
                                f"n_outliers={n_outliers}"
                            ),
                        })

    return issues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_numeric_dtype(dtype_str: str) -> bool:
    """Return True if the dtype string corresponds to a numeric type."""
    return any(t in dtype_str for t in ("int", "float"))
