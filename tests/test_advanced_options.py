"""
tests/test_advanced_options.py
------------------------------
Tests for configurable options:
1. Scaler types: standard, minmax, robust, none
2. Custom CV folds override
3. Custom test_size split ratio
4. Model hyperparameters passing (XGBoost, RF, SVM, etc.)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from previa.preprocessing.pipeline_builder import build_pipeline
from previa.models.registry import create_model_instance, is_xgboost_available
from previa.benchmarking.benchmarker import run_benchmarks


@pytest.fixture
def sample_recommendations():
    return [
        {
            "feature": "feat_num1",
            "detected_type": "numeric",
            "issue": "no critical issues detected",
            "recommendation": "standard_scale",
            "reason": "Numeric feature",
            "evidence": "missing_pct=0%",
            "confidence": "high",
        },
        {
            "feature": "feat_cat",
            "detected_type": "categorical",
            "issue": "no critical issues detected",
            "recommendation": "one_hot_encode",
            "reason": "Categorical feature",
            "evidence": "n_categories=3",
            "confidence": "high",
        },
        {
            "feature": "target",
            "detected_type": "numeric",
            "issue": "Target column",
            "recommendation": "keep_as_is",
            "reason": "Target",
            "evidence": "is_target=True",
            "confidence": "high",
        },
    ]


def test_build_pipeline_scaler_types(sample_recommendations):
    # Helper to get numeric pipeline
    def get_numeric_pipe(ct):
        for name, trans, cols in ct.transformers:
            if name == "numeric":
                return trans
        return None

    # Standard
    pipe_std, num_f, cat_f, _ = build_pipeline(sample_recommendations, scaler_type="standard")
    scaler_step_std = dict(get_numeric_pipe(pipe_std).steps).get("scaler")
    assert isinstance(scaler_step_std, StandardScaler)

    # MinMax
    pipe_mm, _, _, _ = build_pipeline(sample_recommendations, scaler_type="minmax")
    scaler_step_mm = dict(get_numeric_pipe(pipe_mm).steps).get("scaler")
    assert isinstance(scaler_step_mm, MinMaxScaler)

    # Robust
    pipe_rob, _, _, _ = build_pipeline(sample_recommendations, scaler_type="robust")
    scaler_step_rob = dict(get_numeric_pipe(pipe_rob).steps).get("scaler")
    assert isinstance(scaler_step_rob, RobustScaler)

    # None
    pipe_none, _, _, _ = build_pipeline(sample_recommendations, scaler_type="none")
    assert "scaler" not in dict(get_numeric_pipe(pipe_none).steps)


def test_create_model_instance_with_hyperparameters():
    # Random Forest with custom n_estimators and max_depth
    rf = create_model_instance("Random Forest", "binary_classification", n_estimators=42, max_depth=5)
    assert rf.n_estimators == 42
    assert rf.max_depth == 5

    # Linear SVM with custom C
    svm = create_model_instance("Linear SVM (LVSM)", "binary_classification", C=0.25)
    assert svm.C == 0.25

    # XGBoost with custom eta and max_depth (if installed)
    if is_xgboost_available():
        xgb = create_model_instance("XGBoost", "binary_classification", n_estimators=60, max_depth=4, eta=0.05, min_child_weight=2.0)
        assert xgb.n_estimators == 60
        assert xgb.max_depth == 4
        assert xgb.learning_rate == 0.05
        assert xgb.min_child_weight == 2.0


def test_run_benchmarks_with_custom_cv_and_hyperparams():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = np.array([0, 1] * 30)

    res = run_benchmarks(
        X, y,
        task_type="binary_classification",
        models_to_evaluate=["Logistic Regression", "Random Forest"],
        cv_folds=4,
        test_size=0.25,
        model_hyperparams={
            "Random Forest": {"n_estimators": 25, "max_depth": 3},
            "Logistic Regression": {"C": 0.5},
        },
    )

    assert res["cv_folds"] == 4
    assert res["test_size"] == 0.25
    assert len(res["models"]) == 2
    assert res["models"][0]["status"] == "success"
    assert res["models"][1]["status"] == "success"
