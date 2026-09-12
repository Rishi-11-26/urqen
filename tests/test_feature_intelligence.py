"""
tests/test_feature_intelligence.py
------------------------------------
Tests for previa.feature_intelligence.analyzer.analyze_features

We test:
- Output structure (dict of dicts, one entry per column)
- Required keys present in each feature entry
- Numeric feature type assignment
- Categorical feature type assignment
- Boolean feature type assignment (0/1 and yes/no)
- Constant column detection
- Missing value stats
- Numeric stats (mean, median, std, min, max, skewness)
- Categorical stats (top_value, top_freq, top_pct, n_categories)
- Target column marked correctly
- Non-target columns not marked as target
"""

import pandas as pd
import numpy as np
import pytest

from previa.feature_intelligence.analyzer import analyze_features


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(df, target=None):
    return analyze_features(df, target_column=target)


REQUIRED_KEYS = [
    "feature_type", "missing_count", "missing_pct",
    "unique_count", "unique_pct", "is_target",
    "mean", "median", "std", "min", "max", "skewness",
    "n_categories", "top_value", "top_freq", "top_pct",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_returns_dict_keyed_by_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = _run(df)
        assert set(result.keys()) == {"a", "b"}

    def test_each_entry_has_required_keys(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _run(df)
        for key in REQUIRED_KEYS:
            assert key in result["a"], f"Missing key: {key}"


class TestNumericType:

    def test_numeric_column_gets_numeric_type(self):
        df = pd.DataFrame({"val": [1.0, 2.5, 3.0, 4.2, 5.0]})
        result = _run(df)
        assert result["val"]["feature_type"] == "numeric"

    def test_numeric_stats_computed(self):
        df = pd.DataFrame({"val": [2.0, 4.0, 6.0, 8.0, 10.0]})
        result = _run(df)
        col = result["val"]
        assert col["mean"] == 6.0
        assert col["min"] == 2.0
        assert col["max"] == 10.0
        assert col["std"] is not None
        assert col["skewness"] is not None

    def test_categorical_stats_none_for_numeric(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
        result = _run(df)
        assert result["val"]["top_value"] is None
        assert result["val"]["n_categories"] is None


class TestCategoricalType:

    def test_string_column_gets_categorical_type(self):
        df = pd.DataFrame({"edu": ["Bachelor", "Master", "PhD", "Bachelor"]})
        result = _run(df)
        assert result["edu"]["feature_type"] == "categorical"

    def test_categorical_top_value_is_most_frequent(self):
        df = pd.DataFrame({"edu": ["Bachelor", "Master", "Bachelor", "Bachelor"]})
        result = _run(df)
        assert result["edu"]["top_value"] == "Bachelor"
        assert result["edu"]["top_freq"] == 3

    def test_n_categories_correct(self):
        df = pd.DataFrame({"col": ["a", "b", "c", "a", "b"]})
        result = _run(df)
        assert result["col"]["n_categories"] == 3

    def test_numeric_stats_none_for_categorical(self):
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        result = _run(df)
        assert result["col"]["mean"] is None
        assert result["col"]["std"] is None


class TestBooleanType:

    def test_01_integer_column_is_boolean(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 1]})
        result = _run(df)
        assert result["flag"]["feature_type"] == "boolean"

    def test_yes_no_column_is_boolean(self):
        df = pd.DataFrame({"purchased": ["yes", "no", "yes", "yes"]})
        result = _run(df)
        assert result["purchased"]["feature_type"] == "boolean"

    def test_three_value_column_not_boolean(self):
        df = pd.DataFrame({"cat": ["a", "b", "c", "a"]})
        result = _run(df)
        assert result["cat"]["feature_type"] != "boolean"


class TestConstantColumn:

    def test_single_value_numeric_is_constant(self):
        df = pd.DataFrame({"c": [7, 7, 7, 7]})
        result = _run(df)
        assert result["c"]["feature_type"] == "constant"

    def test_single_value_categorical_is_constant(self):
        df = pd.DataFrame({"c": ["yes", "yes", "yes"]})
        result = _run(df)
        assert result["c"]["feature_type"] == "constant"


class TestMissingValues:

    def test_missing_count_correct(self):
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        result = _run(df)
        assert result["a"]["missing_count"] == 2

    def test_missing_pct_correct(self):
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        result = _run(df)
        assert result["a"]["missing_pct"] == 50.0

    def test_no_missing_is_zero(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _run(df)
        assert result["a"]["missing_count"] == 0
        assert result["a"]["missing_pct"] == 0.0


class TestTargetMarking:

    def test_target_column_marked(self):
        df = pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 0]})
        result = _run(df, target="label")
        assert result["label"]["is_target"] is True

    def test_non_target_not_marked(self):
        df = pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 0]})
        result = _run(df, target="label")
        assert result["a"]["is_target"] is False

    def test_no_target_all_false(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = _run(df, target=None)
        assert all(v["is_target"] is False for v in result.values())
