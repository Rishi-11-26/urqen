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
  along with df for outlier calculation.
- The output is an enriched 8-point finding structure for every issue:
  1. issue_type
  2. column
  3. evidence
  4. severity
  5. message
  6. why_it_matters
  7. recommendation
  8. trade_offs
  9. requires_approval
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
        "column"           : str or None (None for dataset-level issues)
        "issue_type"       : str
        "severity"         : "high" | "medium" | "low"
        "message"          : str  (user-readable explanation)
        "evidence"         : str  (the specific statistic that triggered this)
        "why_it_matters"   : str  (practical ML impact)
        "recommendation"   : str  (actionable advice)
        "trade_offs"       : str  (potential downsides/trade-offs)
        "requires_approval": bool (whether user should confirm)
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
            "why_it_matters": "Identical rows over-represent specific patterns and can cause evaluation metrics to appear artificially optimistic if duplicates land across train and test splits.",
            "recommendation": "Review dataset provenance. If duplicates represent repeated recording errors, deduplicate before modeling.",
            "trade_offs": "Deduplicating may discard genuine recurring business events if time or transaction context was omitted.",
            "requires_approval": True,
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
                "why_it_matters": "Estimating values for a heavily missing column may introduce synthetic noise or distort underlying feature relationships.",
                "recommendation": "Consider dropping this feature or creating a missingness indicator column alongside imputation.",
                "trade_offs": "Dropping loses partial signal; imputing heavily missing data risks artificial variance reduction.",
                "requires_approval": True,
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
                "why_it_matters": "Algorithms require complete data matrices; untreated missingness causes training errors.",
                "recommendation": "Apply median imputation for skewed numeric features, mean for symmetric, or mode for categorical features.",
                "trade_offs": "Standard imputation dampens natural feature variance slightly.",
                "requires_approval": False,
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
                "why_it_matters": "Zero variance features provide zero discriminative power across samples and waste computational resources.",
                "recommendation": "Exclude from feature set.",
                "trade_offs": "No downside — removing constant columns is always safe for tabular modeling.",
                "requires_approval": False,
            })
            continue  # No further checks needed for a constant column

        # 3. Possible identifier (unique per row, likely a row ID)
        is_numeric = _is_numeric_dtype(dtype)
        if unique_pct >= IDENTIFIER_PCT_THRESHOLD and n_rows > 5:
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
                "why_it_matters": "Unique identifiers allow complex models (like decision trees) to memorize training samples, causing severe overfitting and zero generalization.",
                "recommendation": "Exclude this column from model features unless it represents a continuous regression target.",
                "trade_offs": "Excluding an identifier prevents memorization; if the column actually has sequential temporal information, consider extracting trend features first.",
                "requires_approval": True,
            })

        # 4. High-cardinality categorical column
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
                "why_it_matters": "One-hot encoding high-cardinality features creates a massive, sparse matrix that increases memory overhead and risk of overfitting.",
                "recommendation": "Group rare categories into an 'Other' bucket, use target/frequency encoding, or evaluate feature exclusion.",
                "trade_offs": "Grouping categories simplifies dimensionality but collapses fine-grained subgroup distinctions.",
                "requires_approval": True,
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
                "why_it_matters": "Linear and distance-based models assume approximately symmetric distributions and can have coefficients distorted by long distribution tails.",
                "recommendation": "Evaluate log1p or power transformation if using linear or ridge models. Tree models are invariant to monotonic transformations.",
                "trade_offs": "Transformations make model coefficients harder to interpret directly in original units.",
                "requires_approval": False,
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
                            "why_it_matters": "Extreme values disproportionately influence squared-error loss functions and linear model slopes.",
                            "recommendation": "Use robust scaling (median/IQR) or tree-based models rather than blindly clipping observations.",
                            "trade_offs": "Clipping or trimming valid rare observations can blind models to true extreme events.",
                            "requires_approval": False,
                        })

    return issues


def _is_numeric_dtype(dtype_str: str) -> bool:
    """Return True if the dtype string corresponds to a numeric type."""
    return any(t in dtype_str for t in ("int", "float"))
