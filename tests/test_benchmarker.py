"""
tests/test_benchmarker.py
--------------------------
Tests for previa.benchmarking.benchmarker.run_benchmarks

We test:
- Output structure (required keys present)
- Correct task type recorded
- Binary classification produces accuracy, precision, recall, f1
- Regression produces mae, mse, rmse, r2
- Best model key is in the results
- CV mean is computed when dataset is large enough
- Small dataset: cv_warning is set and cv_folds is reduced
- Very small dataset: CV is skipped (cv_folds=0)
- Model count matches expected (3 for classification, 4 for regression)

We do NOT test for specific metric values — those depend on random_state,
sklearn version, and dataset content. We only test structure and sanity ranges.
"""

import numpy as np
import pandas as pd
import pytest

from previa.benchmarking.benchmarker import run_benchmarks


# ---------------------------------------------------------------------------
# Fixtures: small synthetic datasets
# ---------------------------------------------------------------------------

def _binary_data(n: int = 100):
    """n-sample binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _multiclass_data(n: int = 120):
    """n-sample 3-class classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 4))
    y = np.tile([0, 1, 2], n // 3 + 1)[:n]
    return X, y


def _regression_data(n: int = 100):
    """n-sample regression dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 3))
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.standard_normal(n) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_has_required_top_level_keys(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        for key in ["task_type", "n_samples", "n_features", "cv_folds",
                    "cv_warning", "test_size", "models", "best_model", "primary_metric"]:
            assert key in result, f"Missing top-level key: {key}"

    def test_models_is_a_list(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        assert isinstance(result["models"], list)

    def test_each_model_entry_has_model_key(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        for m in result["models"]:
            assert "model" in m


class TestBinaryClassification:

    def test_three_models_evaluated(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        assert len(result["models"]) == 3

    def test_classification_metrics_present(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        for m in result["models"]:
            for key in ["accuracy", "precision", "recall", "f1"]:
                assert key in m, f"Missing metric '{key}' in model: {m['model']}"

    def test_metrics_in_valid_range(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        for m in result["models"]:
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0

    def test_best_model_is_in_results(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        model_names = [m["model"] for m in result["models"]]
        assert result["best_model"] in model_names

    def test_primary_metric_is_f1(self):
        X, y = _binary_data()
        result = run_benchmarks(X, y, "binary_classification")
        assert result["primary_metric"] == "f1"

    def test_cv_mean_computed_for_large_dataset(self):
        X, y = _binary_data(n=100)
        result = run_benchmarks(X, y, "binary_classification")
        assert result["cv_folds"] == 5
        for m in result["models"]:
            assert m["cv_mean"] is not None


class TestMulticlassClassification:

    def test_three_models_evaluated(self):
        X, y = _multiclass_data()
        result = run_benchmarks(X, y, "multiclass_classification")
        assert len(result["models"]) == 3

    def test_f1_is_primary_metric(self):
        X, y = _multiclass_data()
        result = run_benchmarks(X, y, "multiclass_classification")
        assert result["primary_metric"] == "f1"


class TestRegression:

    def test_four_models_evaluated(self):
        X, y = _regression_data()
        result = run_benchmarks(X, y, "regression")
        assert len(result["models"]) == 4

    def test_regression_metrics_present(self):
        X, y = _regression_data()
        result = run_benchmarks(X, y, "regression")
        for m in result["models"]:
            for key in ["mae", "mse", "rmse", "r2"]:
                assert key in m, f"Missing metric '{key}' in model: {m['model']}"

    def test_primary_metric_is_r2(self):
        X, y = _regression_data()
        result = run_benchmarks(X, y, "regression")
        assert result["primary_metric"] == "r2"


class TestSmallDatasetHandling:

    def test_medium_dataset_uses_3fold_cv(self):
        # 30 samples → should use 3-fold
        X, y = _binary_data(n=30)
        result = run_benchmarks(X, y, "binary_classification")
        assert result["cv_folds"] == 3
        assert result["cv_warning"] is not None

    def test_small_dataset_uses_2fold_cv(self):
        # 15 samples → should use 2-fold
        X, y = _binary_data(n=15)
        result = run_benchmarks(X, y, "binary_classification")
        assert result["cv_folds"] == 2
        assert result["cv_warning"] is not None

    def test_very_small_dataset_skips_cv(self):
        # 8 samples → CV should be skipped
        X, y = _binary_data(n=8)
        result = run_benchmarks(X, y, "binary_classification")
        assert result["cv_folds"] == 0
        assert result["cv_warning"] is not None
        for m in result["models"]:
            assert m["cv_mean"] is None

    def test_n_samples_recorded_correctly(self):
        X, y = _binary_data(n=60)
        result = run_benchmarks(X, y, "binary_classification")
        assert result["n_samples"] == 60
