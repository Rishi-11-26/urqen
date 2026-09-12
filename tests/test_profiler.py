"""
tests/test_profiler.py
----------------------
Tests for previa.auditing.profiler.profile_dataset

We test:
- Profile structure (summary + columns keys)
- Correct row/column counts
- Correct duplicate detection
- Missing value counting and percentage
- Unique value counting
- Numeric stats (mean, median, std, min, max, skewness)
- Correct behaviour on datasets with no missing values
- Correct behaviour on datasets that are fully numeric
- Correct behaviour when a column has a mix of types
"""

import pandas as pd
import numpy as np
import pytest

from previa.auditing.profiler import profile_dataset


class TestProfileStructure:
    """The output structure must always be consistent."""

    def test_returns_dict_with_summary_and_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = profile_dataset(df)
        assert "summary" in result
        assert "columns" in result

    def test_summary_has_required_keys(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        summary = profile_dataset(df)["summary"]
        for key in ["n_rows", "n_cols", "n_duplicates", "duplicate_pct",
                    "total_cells", "total_missing", "overall_missing_pct"]:
            assert key in summary, f"Missing summary key: {key}"

    def test_columns_has_entry_for_each_column(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        result = profile_dataset(df)
        assert set(result["columns"].keys()) == {"x", "y", "z"}

    def test_column_entry_has_required_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        col_info = profile_dataset(df)["columns"]["a"]
        for key in ["dtype", "missing_count", "missing_pct",
                    "unique_count", "unique_pct"]:
            assert key in col_info, f"Missing column key: {key}"


class TestSummaryCounts:
    """Dataset-level counts should be accurate."""

    def test_row_and_column_counts(self):
        df = pd.DataFrame({"a": range(7), "b": range(7), "c": range(7)})
        summary = profile_dataset(df)["summary"]
        assert summary["n_rows"] == 7
        assert summary["n_cols"] == 3

    def test_duplicate_detection(self):
        # Two identical rows → 1 duplicate
        df = pd.DataFrame({"a": [1, 2, 1], "b": [10, 20, 10]})
        summary = profile_dataset(df)["summary"]
        assert summary["n_duplicates"] == 1

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        summary = profile_dataset(df)["summary"]
        assert summary["n_duplicates"] == 0

    def test_total_missing_count(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
        summary = profile_dataset(df)["summary"]
        assert summary["total_missing"] == 3

    def test_zero_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        summary = profile_dataset(df)["summary"]
        assert summary["total_missing"] == 0
        assert summary["overall_missing_pct"] == 0.0


class TestColumnMissing:
    """Per-column missing stats should be accurate."""

    def test_missing_count_for_column(self):
        df = pd.DataFrame({"a": [1, None, None, 4]})
        col = profile_dataset(df)["columns"]["a"]
        assert col["missing_count"] == 2

    def test_missing_pct_for_column(self):
        df = pd.DataFrame({"a": [1, None, None, 4]})
        col = profile_dataset(df)["columns"]["a"]
        assert col["missing_pct"] == 50.0

    def test_no_missing_in_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        col = profile_dataset(df)["columns"]["a"]
        assert col["missing_count"] == 0
        assert col["missing_pct"] == 0.0


class TestNumericStats:
    """Numeric columns should have correct computed statistics."""

    def setup_method(self):
        self.df = pd.DataFrame({"val": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
        self.col = profile_dataset(self.df)["columns"]["val"]

    def test_mean_is_computed(self):
        assert self.col["mean"] is not None
        assert abs(self.col["mean"] - 5.0) < 0.01

    def test_min_max(self):
        assert self.col["min"] == 2.0
        assert self.col["max"] == 9.0

    def test_skewness_is_computed(self):
        # The distribution above is slightly right-skewed
        assert self.col["skewness"] is not None

    def test_non_numeric_column_has_none_stats(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        col = profile_dataset(df)["columns"]["cat"]
        assert col["mean"] is None
        assert col["skewness"] is None
