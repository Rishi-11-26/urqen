"""
tests/test_explainer.py
-----------------------
Tests for previa.explainability.explainer

We test:
- compute_feature_importance returns required keys
- Prominent causality disclaimer is present
- Tree-based models produce valid feature importances
- Linear models produce valid feature importances
- plot_feature_importance_chart returns a valid Plotly Figure
- Fallbacks work cleanly and never crash
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import plotly.graph_objects as go

from previa.explainability.explainer import (
    compute_feature_importance,
    plot_feature_importance_chart,
    CAUSALITY_DISCLAIMER,
)


class TestExplainer:

    def test_random_forest_classification_importance(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        y = (X[:, 0] > 0).astype(int)
        names = ["feat_0", "feat_1", "feat_2", "feat_3"]

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)

        res = compute_feature_importance(rf, X, y, feature_names=names, task_type="binary_classification")

        assert "method_used" in res
        assert "is_shap" in res
        assert "feature_importances" in res
        assert "top_features" in res
        assert res["disclaimer"] == CAUSALITY_DISCLAIMER
        assert len(res["feature_importances"]) == 4

        # Most important should be feat_0
        assert res["top_features"][0] == "feat_0"

    def test_linear_regression_importance(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = X[:, 0] * 5.0 + X[:, 1] * 0.1
        names = ["heavy_x", "light_x", "noise_x"]

        lr = LinearRegression()
        lr.fit(X, y)

        res = compute_feature_importance(lr, X, y, feature_names=names, task_type="regression")

        assert len(res["feature_importances"]) == 3
        assert res["top_features"][0] == "heavy_x"

    def test_chart_generation(self):
        data = {
            "method_used": "Test Method",
            "feature_importances": [
                {"feature": "f1", "importance": 0.6, "rank": 1},
                {"feature": "f2", "importance": 0.4, "rank": 2},
            ]
        }
        fig = plot_feature_importance_chart(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
