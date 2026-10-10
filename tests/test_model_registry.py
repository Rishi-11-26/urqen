"""
tests/test_model_registry.py
----------------------------
Tests for previa.models.registry

We test:
- Listing available classification models
- Listing available regression models
- Model metadata has name, family, scale_sensitive, description
- Instantiating models by name
- Requesting invalid model name raises ValueError
- Optional XGBoost presence check
"""

import pytest
from previa.models.registry import (
    get_available_models,
    create_model_instance,
    is_xgboost_available,
)


class TestModelRegistry:

    def test_get_classification_models(self):
        models = get_available_models("binary_classification")
        names = [m["name"] for m in models]
        assert "Logistic Regression" in names
        assert "Decision Tree" in names
        assert "Random Forest" in names
        for m in models:
            assert "family" in m
            assert "scale_sensitive" in m
            assert "description" in m

    def test_get_regression_models(self):
        models = get_available_models("regression")
        names = [m["name"] for m in models]
        assert "Linear Regression" in names
        assert "Ridge" in names
        assert "Decision Tree" in names
        assert "Random Forest" in names

    def test_instantiate_logistic_regression(self):
        model = create_model_instance("Logistic Regression", "binary_classification")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_instantiate_random_forest(self):
        model = create_model_instance("Random Forest", "regression")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_instantiate_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_model_instance("NonexistentNeuralNetwork", "regression")

    def test_xgboost_flag_boolean(self):
        assert isinstance(is_xgboost_available(), bool)
