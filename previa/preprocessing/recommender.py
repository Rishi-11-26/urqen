"""
previa/preprocessing/recommender.py
-------------------------------------
Generates explainable preprocessing recommendations for every feature.

This module answers:
  "What preprocessing should be applied to each feature, and why?"

Design principles
-----------------
- Purely rule-based and deterministic. No LLM, no external API.
- Rules are driven by the feature intelligence output (feature type,
  missing %, cardinality, skewness, etc.).
- Every recommendation exposes the evidence that produced it.
- We NEVER claim recommendations are guaranteed to be correct.
- We prefer a clear "flag for review" over a silent wrong decision.
- The output list is ordered (constants and identifiers first) but that
  order does not affect the pipeline builder.

Recommendation actions (the "recommendation" field)
----------------------------------------------------
  "drop"                  : remove from model training (constant, identifier)
  "flag_for_review"       : needs human decision (high cardinality)
  "median_imputation"     : fill missing numeric values with median
  "most_frequent_imputation" : fill missing categorical values with mode
  "one_hot_encode"        : encode categorical to binary dummy columns
  "standard_scale"        : center and scale numeric features
  "log_transform"         : consider log transform for strongly skewed features
  "keep_as_is"            : no action needed

Each recommendation dict has:
  - feature         : column name
  - detected_type   : feature type from analyzer
  - issue           : short description of the detected condition
  - recommendation  : action string from the list above
  - reason          : human-readable justification
  - evidence        : the specific statistic that drove the recommendation
  - confidence      : "high" | "medium" | "low"
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds (named so they are easy to find and understand)
# ---------------------------------------------------------------------------

HIGH_MISSING_THRESHOLD = 20.0          # % missing that triggers imputation note
SKEWNESS_TRANSFORM_THRESHOLD = 2.0    # |skew| beyond which transform is suggested
HIGH_CARDINALITY_THRESHOLD = 50       # unique categories beyond which OHE is risky (raised from 15)


def generate_recommendations(
    feature_analysis: dict,
    target_column: str | None = None,
) -> list[dict]:
    """
    Generate a preprocessing recommendation for every feature.

    Parameters
    ----------
    feature_analysis : dict
        Output of previa.feature_intelligence.analyzer.analyze_features().
        Keys are column names; values are per-feature stat dicts.
    target_column : str or None
        The target column. It will receive a "keep_as_is" recommendation
        with an explanation that it is the prediction target.

    Returns
    -------
    list of dicts, one per feature. Each dict has the structure described
    in the module docstring.
    """
    recommendations = []

    for feature, info in feature_analysis.items():
        rec = _recommend_for_feature(feature, info, target_column)
        recommendations.append(rec)

    return recommendations


# ---------------------------------------------------------------------------
# Internal: single-feature recommendation logic
# ---------------------------------------------------------------------------

def _recommend_for_feature(
    feature: str,
    info: dict,
    target_column: str | None,
) -> dict:
    """
    Apply rules to a single feature and return its recommendation dict.

    Rules are applied in priority order:
      1. Target column → keep as-is (it is the prediction target)
      2. Constant column → drop
      3. Possible identifier → drop (flagged as possible_identifier)
      4. Feature type–specific rules
    """
    feature_type = info.get("feature_type", "unknown")
    missing_pct = info.get("missing_pct", 0.0)
    unique_pct = info.get("unique_pct", 0.0)
    skewness = info.get("skewness")
    n_categories = info.get("n_categories")
    is_target = info.get("is_target", False)

    # --- Rule 1: Target column ---
    if is_target:
        return _make_rec(
            feature=feature,
            detected_type=feature_type,
            issue="This is the prediction target.",
            recommendation="keep_as_is",
            reason="The target column is not preprocessed. It is what the model learns to predict.",
            evidence=f"is_target=True",
            confidence="high",
        )

    # --- Rule 2: Constant column ---
    if feature_type == "constant":
        return _make_rec(
            feature=feature,
            detected_type=feature_type,
            issue="Column has only one unique value.",
            recommendation="drop",
            reason=(
                "A constant column carries no information for a model — every row "
                "has the same value, so no pattern can be learned from it."
            ),
            evidence=f"unique_count=1",
            confidence="high",
        )

    # --- Rule 3: Possible identifier (near-unique non-numeric column) ---
    # We flag non-numeric columns with >90% unique values.
    # The pipeline builder will exclude these from training.
    if feature_type == "categorical" and unique_pct >= 90.0:
        return _make_rec(
            feature=feature,
            detected_type=feature_type,
            issue="Possible identifier column (nearly all values are unique).",
            recommendation="drop",
            reason=(
                "This column appears to be a row identifier (e.g., name, ID). "
                "Including identifiers in model training typically harms performance "
                "because models memorize them instead of learning real patterns."
            ),
            evidence=f"unique_pct={unique_pct}%",
            confidence="medium",
        )

    # --- Rule 4: Numeric feature ---
    if feature_type in ("numeric", "boolean"):
        return _recommend_numeric(feature, feature_type, missing_pct, skewness)

    # --- Rule 5: Categorical feature ---
    if feature_type == "categorical":
        return _recommend_categorical(feature, feature_type, missing_pct, n_categories)

    # --- Fallback ---
    return _make_rec(
        feature=feature,
        detected_type=feature_type,
        issue="Feature type could not be determined.",
        recommendation="flag_for_review",
        reason="URQEN could not classify this feature. Manual review is needed.",
        evidence=f"feature_type={feature_type}",
        confidence="low",
    )


def _recommend_numeric(
    feature: str,
    feature_type: str,
    missing_pct: float,
    skewness: float | None,
) -> dict:
    """Rules for numeric and boolean columns."""

    issues = []
    recommendations = []
    reasons = []
    evidence_parts = []

    # Missing values → imputation
    if missing_pct > 0:
        issues.append(f"has {missing_pct}% missing values")
        recommendations.append("median_imputation")
        reasons.append(
            f"Numeric feature has {missing_pct:.1f}% missing values. "
            "Median imputation is used because the median is not affected by extreme "
            "values (outliers), making it more robust than the mean."
        )
        evidence_parts.append(f"missing_pct={missing_pct}%")

    # Scaling — always recommended for numeric features
    recommendations.append("standard_scale")
    reasons.append(
        "Numeric features should be scaled so that features with large ranges "
        "do not dominate features with small ranges. StandardScaler centers the "
        "data (mean=0, std=1) which is compatible with most ML algorithms."
    )

    # Skewness → suggest transform
    if skewness is not None and abs(skewness) >= SKEWNESS_TRANSFORM_THRESHOLD:
        issues.append(f"strongly skewed (skewness={skewness:.2f})")
        recommendations.append("log_transform")
        reasons.append(
            f"This feature has a skewness of {skewness:.2f} (|skew| ≥ {SKEWNESS_TRANSFORM_THRESHOLD}). "
            "Strongly skewed distributions can affect distance-based and linear models. "
            "A log or power transform may help, but this is optional and depends on the context."
        )
        evidence_parts.append(f"skewness={skewness:.4f}")
        confidence = "medium"
    else:
        confidence = "high"

    if not issues:
        issues.append("no critical issues detected")

    return _make_rec(
        feature=feature,
        detected_type=feature_type,
        issue="; ".join(issues),
        recommendation=", ".join(recommendations),
        reason=" | ".join(reasons),
        evidence=", ".join(evidence_parts) if evidence_parts else "missing_pct=0%",
        confidence=confidence,
    )


def _recommend_categorical(
    feature: str,
    feature_type: str,
    missing_pct: float,
    n_categories: int | None,
) -> dict:
    """Rules for categorical columns."""

    issues = []
    recommendations = []
    reasons = []
    evidence_parts = []

    # Missing values → most-frequent imputation
    if missing_pct > 0:
        issues.append(f"has {missing_pct}% missing values")
        recommendations.append("most_frequent_imputation")
        reasons.append(
            f"Categorical feature has {missing_pct:.1f}% missing values. "
            "Most-frequent (mode) imputation replaces missing values with the most "
            "common category — a simple and often reasonable default."
        )
        evidence_parts.append(f"missing_pct={missing_pct}%")

    # High cardinality → flag for review, skip OHE
    if n_categories is not None and n_categories > HIGH_CARDINALITY_THRESHOLD:
        issues.append(f"high cardinality ({n_categories} categories)")
        recommendations.append("flag_for_review")
        reasons.append(
            f"This column has {n_categories} unique categories. "
            f"One-hot encoding {n_categories} categories would add {n_categories} new columns, "
            "which may cause dimensionality issues. Review recommended before encoding."
        )
        evidence_parts.append(f"n_categories={n_categories}")
        confidence = "low"
    else:
        # Normal cardinality → one-hot encode
        recommendations.append("one_hot_encode")
        reasons.append(
            "Categorical features must be converted to numbers before most ML algorithms "
            "can use them. One-hot encoding creates one binary column per category, "
            "which is the standard approach for nominal (unordered) categories."
        )
        evidence_parts.append(f"n_categories={n_categories}")
        confidence = "high"

    if not issues:
        issues.append("no critical issues detected")

    return _make_rec(
        feature=feature,
        detected_type=feature_type,
        issue="; ".join(issues),
        recommendation=", ".join(recommendations),
        reason=" | ".join(reasons),
        evidence=", ".join(evidence_parts),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Helper: build a consistent recommendation dict
# ---------------------------------------------------------------------------

def _make_rec(
    feature: str,
    detected_type: str,
    issue: str,
    recommendation: str,
    reason: str,
    evidence: str,
    confidence: str,
    why_it_matters: str = "",
    trade_offs: str = "",
) -> dict:
    """Return a recommendation dict with all required keys."""
    if not why_it_matters:
        why_it_matters = reason
    if not trade_offs:
        if "standard_scale" in recommendation:
            trade_offs = "Standard scaling centers mean to 0 and variance to 1; makes coefficients comparable across features but alters original measurement units."
        elif "one_hot_encode" in recommendation:
            trade_offs = "One-hot encoding expands categorical levels into binary columns; increases dimensionality proportional to category count."
        elif "drop" in recommendation:
            trade_offs = "Excluding this column avoids memorization or zero-variance waste; ensures model focuses on generalizable signals."
        elif "flag_for_review" in recommendation:
            trade_offs = "High-cardinality requires deliberate grouping or target encoding to prevent extreme column explosion."
        else:
            trade_offs = "Imputation replaces missing values with robust statistics (median/mode) to allow complete matrix operations."

    return {
        "feature": feature,
        "detected_type": detected_type,
        "issue": issue,
        "recommendation": recommendation,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "why_it_matters": why_it_matters,
        "trade_offs": trade_offs,
    }
