"""
tests/test_recommender.py
--------------------------
Tests for previa.preprocessing.recommender.generate_recommendations

We test:
- Output structure (list of dicts, one per feature)
- Required keys present in each recommendation
- Target column gets "keep_as_is"
- Constant columns get "drop"
- Numeric with missing gets "median_imputation"
- Numeric always gets "standard_scale"
- Strongly skewed numeric gets "log_transform"
- Categorical without missing gets "one_hot_encode"
- Categorical with missing gets "most_frequent_imputation"
- High-cardinality categorical gets "flag_for_review"
- Confidence values are always valid
"""

import pandas as pd
import pytest

from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations


# ---------------------------------------------------------------------------
# Helper: build feature analysis then generate recommendations
# ---------------------------------------------------------------------------

def _recommend(df: pd.DataFrame, target: str | None = None) -> list[dict]:
    analysis = analyze_features(df, target_column=target)
    return generate_recommendations(analysis, target_column=target)


def _rec_for(recs: list[dict], feature: str) -> dict:
    for r in recs:
        if r["feature"] == feature:
            return r
    raise KeyError(f"No recommendation found for feature '{feature}'")


REQUIRED_KEYS = [
    "feature", "detected_type", "issue",
    "recommendation", "reason", "evidence", "confidence",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_returns_list(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        recs = _recommend(df)
        assert isinstance(recs, list)

    def test_one_recommendation_per_feature(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        recs = _recommend(df)
        assert len(recs) == 2

    def test_each_rec_has_required_keys(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        recs = _recommend(df)
        for rec in recs:
            for key in REQUIRED_KEYS:
                assert key in rec, f"Missing key '{key}' in recommendation: {rec}"

    def test_confidence_values_are_valid(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        recs = _recommend(df)
        for rec in recs:
            assert rec["confidence"] in {"high", "medium", "low"}


class TestTargetColumn:

    def test_target_gets_keep_as_is(self):
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
        recs = _recommend(df, target="label")
        rec = _rec_for(recs, "label")
        assert "keep_as_is" in rec["recommendation"]

    def test_target_confidence_is_high(self):
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
        recs = _recommend(df, target="label")
        rec = _rec_for(recs, "label")
        assert rec["confidence"] == "high"


class TestConstantColumn:

    def test_constant_column_gets_drop(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "const": [5, 5, 5]})
        recs = _recommend(df)
        rec = _rec_for(recs, "const")
        assert "drop" in rec["recommendation"]

    def test_constant_column_high_confidence(self):
        df = pd.DataFrame({"const": [0, 0, 0, 0]})
        recs = _recommend(df)
        rec = _rec_for(recs, "const")
        assert rec["confidence"] == "high"


class TestNumericRecommendations:

    def test_numeric_always_gets_standard_scale(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        recs = _recommend(df)
        rec = _rec_for(recs, "val")
        assert "standard_scale" in rec["recommendation"]

    def test_numeric_with_missing_gets_median_imputation(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0, 4.0, 5.0]})
        recs = _recommend(df)
        rec = _rec_for(recs, "val")
        assert "median_imputation" in rec["recommendation"]

    def test_numeric_no_missing_no_imputation(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        recs = _recommend(df)
        rec = _rec_for(recs, "val")
        assert "median_imputation" not in rec["recommendation"]

    def test_strongly_skewed_numeric_gets_log_transform_suggestion(self):
        # Right-skewed distribution
        data = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1000]
        df = pd.DataFrame({"skewed": data})
        recs = _recommend(df)
        rec = _rec_for(recs, "skewed")
        assert "log_transform" in rec["recommendation"]

    def test_normal_numeric_no_log_transform(self):
        data = list(range(1, 11))  # 1..10, low skewness
        df = pd.DataFrame({"normal": data})
        recs = _recommend(df)
        rec = _rec_for(recs, "normal")
        assert "log_transform" not in rec["recommendation"]


class TestCategoricalRecommendations:

    def test_categorical_gets_one_hot_encode(self):
        df = pd.DataFrame({"edu": ["Bachelor", "Master", "PhD", "Bachelor", "Master"]})
        recs = _recommend(df)
        rec = _rec_for(recs, "edu")
        assert "one_hot_encode" in rec["recommendation"]

    def test_categorical_with_missing_gets_most_frequent_imputation(self):
        df = pd.DataFrame({"edu": ["Bachelor", None, "PhD", "Bachelor", "Master"]})
        recs = _recommend(df)
        rec = _rec_for(recs, "edu")
        assert "most_frequent_imputation" in rec["recommendation"]

    def test_high_cardinality_categorical_gets_flag_for_review(self):
        # 55 unique categories (above threshold of 50)
        categories = [f"cat_{i}" for i in range(55)]
        data = (categories * 2)[:100]  # 100 rows
        df = pd.DataFrame({"high_card": data})
        recs = _recommend(df)
        rec = _rec_for(recs, "high_card")
        assert "flag_for_review" in rec["recommendation"]
        assert rec["confidence"] == "low"
