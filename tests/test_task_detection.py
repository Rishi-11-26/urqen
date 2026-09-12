"""
tests/test_task_detection.py
-----------------------------
Tests for previa.task_detection.detector.detect_target_and_task

We test:
- Binary classification detection (0/1 labels)
- Binary classification with yes/no labels
- Multiclass classification detection
- Regression detection (continuous numeric target)
- Target keyword in column name boosts correct detection
- Last-column convention
- Identifier columns are not selected as target
- Output structure is always consistent
- 'unknown' result when no good target exists
"""

import pandas as pd
import numpy as np
import pytest

from previa.task_detection.detector import detect_target_and_task


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _detect(df: pd.DataFrame) -> dict:
    return detect_target_and_task(df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """Output must always contain the required keys."""

    def test_output_has_required_keys(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        result = _detect(df)
        for key in ["target_column", "task_type", "confidence", "reasons", "all_scores"]:
            assert key in result, f"Missing key: {key}"

    def test_reasons_is_a_list(self):
        df = pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 0]})
        result = _detect(df)
        assert isinstance(result["reasons"], list)

    def test_task_type_is_valid(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        result = _detect(df)
        valid = {"binary_classification", "multiclass_classification", "regression", "unknown"}
        assert result["task_type"] in valid

    def test_confidence_is_valid(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        result = _detect(df)
        assert result["confidence"] in {"high", "medium", "low"}


class TestBinaryClassification:
    """Binary {0,1} and {yes,no} targets should be correctly identified."""

    def test_binary_01_column_named_target(self):
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [40000, 55000, 70000, 85000, 100000],
            "target": [0, 1, 0, 1, 1],
        })
        result = _detect(df)
        assert result["target_column"] == "target"
        assert result["task_type"] == "binary_classification"

    def test_binary_01_column_named_label(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "label": [0, 1, 1, 0],
        })
        result = _detect(df)
        assert result["target_column"] == "label"
        assert result["task_type"] == "binary_classification"

    def test_binary_yes_no_target(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "purchased": ["yes", "no", "yes", "yes"],
        })
        result = _detect(df)
        assert result["task_type"] == "binary_classification"

    def test_binary_target_at_last_position(self):
        # No target keyword but it is the last column with binary values
        df = pd.DataFrame({
            "age": [20, 30, 40, 50],
            "income": [30000, 50000, 70000, 90000],
            "approved": [0, 1, 1, 0],
        })
        result = _detect(df)
        assert result["task_type"] == "binary_classification"


class TestMulticlassClassification:
    """Targets with 3–N distinct categorical values should be multiclass."""

    def test_three_class_target(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6],
            "class": ["A", "B", "C", "A", "B", "C"],
        })
        result = _detect(df)
        assert result["target_column"] == "class"
        assert result["task_type"] == "multiclass_classification"

    def test_five_class_numeric_target(self):
        df = pd.DataFrame({
            "x": range(15),
            "label": [0, 1, 2, 3, 4] * 3,
        })
        result = _detect(df)
        assert result["task_type"] == "multiclass_classification"


class TestRegression:
    """Continuous numeric targets should be detected as regression."""

    def test_continuous_numeric_target(self):
        # 50 unique float values → regression
        values = [float(i) * 1.5 + 0.3 for i in range(50)]
        df = pd.DataFrame({
            "feature_a": range(50),
            "feature_b": range(50, 100),
            "price": values,
        })
        result = _detect(df)
        assert result["target_column"] == "price"
        assert result["task_type"] == "regression"


class TestIdentifierExclusion:
    """Columns that look like row IDs should not be selected as targets."""

    def test_id_column_not_selected(self):
        df = pd.DataFrame({
            "id": range(20),          # all unique → identifier
            "feature": range(20),
            "target": [0, 1] * 10,    # binary target
        })
        result = _detect(df)
        assert result["target_column"] != "id"

    def test_id_named_column_penalized(self):
        df = pd.DataFrame({
            "customer_id": range(10),
            "age": range(10),
            "label": [0, 1] * 5,
        })
        result = _detect(df)
        assert result["target_column"] == "label"
