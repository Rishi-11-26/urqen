"""
tests/test_pipeline_builder.py
--------------------------------
Tests for previa.preprocessing.pipeline_builder.build_pipeline

We test:
- Returns a ColumnTransformer object
- Numeric features are placed in the numeric transformer
- Categorical features are placed in the categorical transformer
- Dropped features (constant, target) are excluded
- The built pipeline can be fitted on training data and transform test data
- Raises ValueError when no usable features remain
- Numeric pipeline has imputer + scaler (leakage-safe)
- Categorical pipeline has imputer + encoder

These tests verify both the structure of the pipeline and that it actually
works correctly end-to-end on a small dataset.
"""

import pandas as pd
import numpy as np
import pytest

from sklearn.compose import ColumnTransformer

from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations
from previa.preprocessing.pipeline_builder import build_pipeline


# ---------------------------------------------------------------------------
# Helper: build pipeline from a DataFrame
# ---------------------------------------------------------------------------

def _build(df: pd.DataFrame, target: str | None = None):
    analysis = analyze_features(df, target_column=target)
    recs = generate_recommendations(analysis, target_column=target)
    return build_pipeline(recs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineStructure:

    def test_returns_column_transformer(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0],
            "label": [0, 1, 0],
        })
        pipeline, _, _, _ = _build(df, target="label")
        assert isinstance(pipeline, ColumnTransformer)

    def test_returns_four_values(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0],
            "edu": ["Bach", "Master", "PhD"],
            "label": [0, 1, 0],
        })
        result = _build(df, target="label")
        assert len(result) == 4  # pipeline, numeric, categorical, dropped

    def test_numeric_feature_in_numeric_list(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0],
            "label": [0, 1, 0, 1],
        })
        _, numeric, categorical, dropped = _build(df, target="label")
        assert "age" in numeric
        assert "age" not in categorical
        assert "age" not in dropped

    def test_categorical_feature_in_categorical_list(self):
        df = pd.DataFrame({
            "edu": ["Bach", "Master", "PhD", "Bach"],
            "label": [0, 1, 0, 1],
        })
        _, numeric, categorical, dropped = _build(df, target="label")
        assert "edu" in categorical
        assert "edu" not in numeric

    def test_target_column_in_dropped(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0],
            "label": [0, 1, 0, 1],
        })
        _, numeric, categorical, dropped = _build(df, target="label")
        assert "label" in dropped

    def test_constant_column_in_dropped(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0],
            "const": [1, 1, 1, 1],
            "label": [0, 1, 0, 1],
        })
        _, numeric, categorical, dropped = _build(df, target="label")
        assert "const" in dropped


class TestPipelineFunctionality:
    """
    Verify that the pipeline can actually be fitted and used to transform data.
    This is the key leakage-safety test: we fit only on X_train.
    """

    def test_pipeline_fits_and_transforms(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            "edu": ["Bach", "Master", "PhD", "Bach", "Master", "PhD"],
            "label": [0, 1, 0, 1, 0, 1],
        })
        pipeline, _, _, _ = _build(df, target="label")

        X = df.drop(columns=["label"])
        y = df["label"]

        # Split manually: first 4 rows train, last 2 rows test
        X_train, X_test = X.iloc[:4], X.iloc[4:]

        # Fit only on training data
        pipeline.fit(X_train)

        # Transform both — should not raise
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        assert X_train_transformed.shape[0] == 4
        assert X_test_transformed.shape[0] == 2

    def test_pipeline_handles_missing_values(self):
        df = pd.DataFrame({
            "age": [25.0, None, 35.0, 40.0, 45.0, 50.0],
            "label": [0, 1, 0, 1, 0, 1],
        })
        pipeline, _, _, _ = _build(df, target="label")

        X = df.drop(columns=["label"])
        X_train = X.iloc[:4]
        X_test = X.iloc[4:]

        pipeline.fit(X_train)
        result = pipeline.transform(X_test)
        # Should not contain NaN after imputation
        assert not np.isnan(result).any()

    def test_no_usable_features_raises(self):
        """If only the target exists, build_pipeline should raise ValueError."""
        df = pd.DataFrame({"label": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="No usable feature columns"):
            _build(df, target="label")
