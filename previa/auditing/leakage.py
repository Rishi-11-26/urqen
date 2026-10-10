"""
previa/auditing/leakage.py
--------------------------
Detects potential data leakage and experimental reliability hazards.

Data leakage occurs when a model is trained using features that contain
information about the target that would NOT be accessible at real prediction time.

Design principles:
- Correlation alone does NOT prove leakage; we use careful wording:
  "Potential leakage warning: investigate whether this feature contains information
  that would not be available at prediction time."
- Do not automatically remove flagged features; allow the user to review and exclude explicitly.
- Flag statistical reliability hazards:
  - Extreme correlation with target (|r| >= 0.95)
  - Target-derived name hints (e.g. status, outcome, pred, result, target_...)
  - Severe class imbalance
  - Small dataset sample count (< 25)
  - Severe missingness (> 50%)
"""

import re
import pandas as pd
import numpy as np


SUSPICIOUS_LEAKAGE_KEYWORDS = {
    "target", "label", "outcome", "result", "status", "prediction",
    "pred", "ground_truth", "actual", "post_", "_post", "approved_at",
    "resolved_at", "closed_at", "decision"
}

HIGH_CORRELATION_THRESHOLD = 0.95
SEVERE_CLASS_IMBALANCE_RATIO = 0.10
SMALL_DATASET_THRESHOLD = 25
HIGH_FEATURE_MISSING_THRESHOLD = 50.0


def audit_leakage_and_reliability(
    df: pd.DataFrame,
    target_column: str | None = None,
    task_type: str | None = None,
) -> list[dict]:
    """
    Audit dataset for data leakage risks and evaluation reliability concerns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed dataset.
    target_column : str or None
        The designated target column name.
    task_type : str or None
        "binary_classification", "multiclass_classification", or "regression".

    Returns
    -------
    list of dicts, each with keys:
        "feature"       : str or None
        "warning_type"  : str
        "severity"      : "high" | "medium" | "low"
        "message"       : str
        "evidence"      : str
        "recommendation": str
    """
    warnings = []
    n_rows, n_cols = df.shape

    # 1. Dataset-level: small sample size warning
    if n_rows < SMALL_DATASET_THRESHOLD:
        warnings.append({
            "feature": None,
            "warning_type": "small_sample_size",
            "severity": "high",
            "message": (
                f"Dataset contains only {n_rows} rows (below the {SMALL_DATASET_THRESHOLD} sample threshold). "
                "Splits and evaluation metrics may exhibit high variance and instability."
            ),
            "evidence": f"n_samples={n_rows}",
            "recommendation": "Use adaptive cross-validation, avoid deep tree models, and treat reported metrics as directional indicators rather than definitive performance.",
        })

    # If no target specified, return dataset-level warnings only
    if not target_column or target_column not in df.columns:
        return warnings

    target_series = df[target_column]
    non_target_cols = [c for c in df.columns if c != target_column]

    # 2. Check for severe class imbalance (classification)
    if task_type in ("binary_classification", "multiclass_classification"):
        val_counts = target_series.dropna().value_counts(normalize=True)
        if len(val_counts) > 1:
            min_prop = float(val_counts.min())
            min_class = val_counts.idxmin()
            if min_prop < SEVERE_CLASS_IMBALANCE_RATIO:
                warnings.append({
                    "feature": target_column,
                    "warning_type": "severe_class_imbalance",
                    "severity": "medium",
                    "message": (
                        f"Target '{target_column}' exhibits severe class imbalance. "
                        f"Minority class '{min_class}' represents only {min_prop * 100:.1f}% of observations."
                    ),
                    "evidence": f"minority_class='{min_class}', proportion={min_prop * 100:.1f}%",
                    "recommendation": "Evaluate using PR-AUC, F1-score, or balanced accuracy rather than standard Accuracy. Use stratified train/test splitting.",
                })

    # 3. Check features for leakage signals
    target_clean = target_series.dropna()
    is_target_numeric = pd.api.types.is_numeric_dtype(target_series)

    for col in non_target_cols:
        series = df[col]
        col_lower = str(col).lower()

        # A. Suspicious column naming pattern
        name_tokens = set(re.split(r"[_\s\-]+", col_lower))
        matching_keywords = name_tokens & SUSPICIOUS_LEAKAGE_KEYWORDS
        target_name_tokens = set(re.split(r"[_\s\-]+", str(target_column).lower()))
        overlaps_target_name = bool((name_tokens & target_name_tokens) - {"the", "a", "of", "in", "is", "id"})

        if matching_keywords or overlaps_target_name:
            reason = f"tokens={matching_keywords}" if matching_keywords else f"shares root with target '{target_column}'"
            warnings.append({
                "feature": col,
                "warning_type": "suspicious_feature_name",
                "severity": "medium",
                "message": (
                    f"Potential leakage warning: Column '{col}' contains naming tokens that suggest post-event information ({reason}). "
                    "Investigate whether this feature would truly be accessible at prediction time."
                ),
                "evidence": f"column_name='{col}', match={reason}",
                "recommendation": "Verify feature availability at inference time. If this feature is recorded after or during the outcome event, exclude it from training.",
            })

        # B. Correlation with target (Numeric features)
        if is_target_numeric and pd.api.types.is_numeric_dtype(series):
            # Align non-null pairs
            valid_mask = series.notnull() & target_series.notnull()
            if valid_mask.sum() >= 5:
                corr = float(series[valid_mask].corr(target_series[valid_mask]))
                if abs(corr) >= HIGH_CORRELATION_THRESHOLD:
                    warnings.append({
                        "feature": col,
                        "warning_type": "near_perfect_target_correlation",
                        "severity": "high",
                        "message": (
                            f"Potential leakage warning: Feature '{col}' has near-perfect correlation with target '{target_column}' (|r| = {abs(corr):.4f}). "
                            "Investigate whether this feature directly encodes or approximates the target outcome."
                        ),
                        "evidence": f"pearson_r={corr:.4f}",
                        "recommendation": "Check if this feature is a duplicate or deterministic proxy of the target. Highly correlated features often cause data leakage.",
                    })

        # C. Categorical feature direct mapping
        elif not is_target_numeric and not pd.api.types.is_numeric_dtype(series):
            # If cross-tabulation shows near 1-to-1 mapping
            valid_mask = series.notnull() & target_series.notnull()
            if valid_mask.sum() >= 10:
                ct = pd.crosstab(series[valid_mask], target_series[valid_mask])
                # Check if each row of ct has a dominant class >= 98%
                row_max_pct = ct.max(axis=1) / ct.sum(axis=1)
                if (row_max_pct >= 0.98).all() and len(ct) == target_clean.nunique():
                    warnings.append({
                        "feature": col,
                        "warning_type": "direct_categorical_target_encoding",
                        "severity": "high",
                        "message": (
                            f"Potential leakage warning: Categorical feature '{col}' maps almost identically to the target '{target_column}'. "
                            "It may be a re-labeled version of the target."
                        ),
                        "evidence": f"exact_subgroup_purity={row_max_pct.mean():.2f}",
                        "recommendation": "Review column origin. If this feature is a post-event classification, exclude it.",
                    })

        # D. Excessive missingness hazard
        missing_pct = float(series.isnull().mean() * 100)
        if missing_pct >= HIGH_FEATURE_MISSING_THRESHOLD:
            warnings.append({
                "feature": col,
                "warning_type": "excessive_missingness",
                "severity": "medium",
                "message": (
                    f"Feature '{col}' is missing {missing_pct:.1f}% of observations. "
                    "Imputing more than half of a feature's values introduces high artificial bias."
                ),
                "evidence": f"missing_pct={missing_pct:.1f}%",
                "recommendation": "Consider dropping this feature unless missingness itself carries strong predictive signal.",
            })

    return warnings
