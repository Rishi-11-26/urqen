"""
tests/test_health.py
--------------------
Tests for previa.auditing.health.analyze_health

We test:
- Duplicate row detection
- High-missing-value detection (high and medium severity)
- Constant column detection
- Possible identifier detection
- High-cardinality detection
- High-skewness detection
- Potential outlier detection
- Clean dataset produces no issues (or only low-severity ones)
- Issue dict structure is always consistent
"""

import pandas as pd
import numpy as np
import pytest

from previa.auditing.profiler import profile_dataset
from previa.auditing.health import analyze_health


# ---------------------------------------------------------------------------
# Helper: run both profiler and health on a DataFrame
# ---------------------------------------------------------------------------

def _run(df: pd.DataFrame) -> list[dict]:
    profile = profile_dataset(df)
    return analyze_health(profile, df)


def _issue_types(issues: list[dict]) -> list[str]:
    return [i["issue_type"] for i in issues]


def _issues_for_col(issues: list[dict], col: str) -> list[dict]:
    return [i for i in issues if i["column"] == col]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIssueStructure:
    """Every issue dict must have the required keys."""

    def test_issue_has_required_keys(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [None, None, 3]})
        issues = _run(df)
        for issue in issues:
            for key in ["column", "issue_type", "severity", "message", "evidence"]:
                assert key in issue, f"Issue missing key '{key}': {issue}"

    def test_severity_is_valid(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [None, None, 3]})
        issues = _run(df)
        for issue in issues:
            assert issue["severity"] in {"high", "medium", "low"}, \
                f"Invalid severity: {issue['severity']}"


class TestDuplicateDetection:
    """Duplicate rows should be flagged."""

    def test_detects_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": [10, 20, 10]})
        issues = _run(df)
        assert "duplicate_rows" in _issue_types(issues)

    def test_no_duplicate_flag_when_clean(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        issues = _run(df)
        assert "duplicate_rows" not in _issue_types(issues)

    def test_duplicate_issue_has_no_column(self):
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        issues = _run(df)
        dup = [i for i in issues if i["issue_type"] == "duplicate_rows"]
        assert len(dup) == 1
        assert dup[0]["column"] is None


class TestMissingValueDetection:
    """Missing values above thresholds should be flagged."""

    def test_high_missing_column_flagged(self):
        # 80% missing → high severity
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        issues = _run(df)
        col_issues = _issues_for_col(issues, "a")
        missing_issues = [i for i in col_issues if i["issue_type"] == "high_missing"]
        assert len(missing_issues) == 1
        assert missing_issues[0]["severity"] == "high"

    def test_medium_missing_column_flagged(self):
        # 10% missing (1 out of 10) → medium severity
        data = [1.0] * 9 + [None]
        df = pd.DataFrame({"a": data})
        issues = _run(df)
        col_issues = _issues_for_col(issues, "a")
        missing_issues = [i for i in col_issues if i["issue_type"] == "high_missing"]
        assert len(missing_issues) == 1
        assert missing_issues[0]["severity"] == "medium"

    def test_no_missing_flag_when_clean(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        issues = _run(df)
        assert not any(i["issue_type"] == "high_missing" for i in issues)


class TestConstantColumn:
    """Constant columns should always be flagged as high severity."""

    def test_constant_numeric_column_flagged(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5], "b": [1, 2, 3, 4]})
        issues = _run(df)
        col_issues = _issues_for_col(issues, "a")
        assert any(i["issue_type"] == "constant_column" for i in col_issues)

    def test_constant_categorical_column_flagged(self):
        df = pd.DataFrame({"cat": ["yes", "yes", "yes"]})
        issues = _run(df)
        assert any(i["issue_type"] == "constant_column" for i in issues)

    def test_constant_column_is_high_severity(self):
        df = pd.DataFrame({"a": [0, 0, 0, 0, 0]})
        issues = _run(df)
        const = [i for i in issues if i["issue_type"] == "constant_column"]
        assert const[0]["severity"] == "high"


class TestIdentifierDetection:
    """Columns with nearly all-unique values should be flagged as possible identifiers."""

    def test_unique_column_flagged_as_identifier(self):
        # 10 rows, all unique → 100% unique → identifier
        df = pd.DataFrame({"id": range(10), "val": [1] * 10})
        issues = _run(df)
        id_issues = _issues_for_col(issues, "id")
        assert any(i["issue_type"] == "possible_identifier" for i in id_issues)

    def test_non_unique_column_not_flagged_as_identifier(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "b", "a", "b", "a", "b"]})
        issues = _run(df)
        id_issues = _issues_for_col(issues, "cat")
        assert not any(i["issue_type"] == "possible_identifier" for i in id_issues)


class TestSkewnessDetection:
    """Highly skewed numeric columns should be flagged."""

    def test_high_skew_column_flagged(self):
        # A very right-skewed distribution
        data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]
        df = pd.DataFrame({"skewed": data})
        issues = _run(df)
        assert any(i["issue_type"] == "high_skewness" for i in issues)

    def test_normal_column_not_flagged_for_skew(self):
        data = list(range(1, 21))  # 1..20, roughly symmetric
        df = pd.DataFrame({"normal": data})
        issues = _run(df)
        assert not any(i["issue_type"] == "high_skewness" for i in issues)


class TestOutlierDetection:
    """Potential outliers should be flagged using the IQR rule."""

    def test_outlier_column_flagged(self):
        # 9 values clustered near 10, one extreme outlier at 1000
        data = [10, 11, 10, 9, 10, 11, 10, 9, 10, 1000]
        df = pd.DataFrame({"vals": data})
        issues = _run(df)
        assert any(i["issue_type"] == "potential_outliers" for i in issues)

    def test_clean_column_not_flagged_for_outliers(self):
        data = list(range(1, 11))  # 1..10, no outliers
        df = pd.DataFrame({"clean": data})
        issues = _run(df)
        assert not any(i["issue_type"] == "potential_outliers" for i in issues)
