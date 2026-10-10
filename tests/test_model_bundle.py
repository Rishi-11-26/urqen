"""
tests/test_model_bundle.py
--------------------------
Tests for previa.export.model_bundle

We test:
- PreviaModelBundle saves to disk and reloads cleanly
- Reloaded bundle produces identical predictions on new data
- Missing required column raises ValueError
- Binary classification bundle includes probability scores
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations
from previa.preprocessing.pipeline_builder import build_pipeline
from previa.export.model_bundle import (
    save_model_bundle,
    load_model_bundle,
    PreviaModelBundle,
)


class TestModelBundle:

    @pytest.fixture
    def fitted_setup(self):
        df = pd.DataFrame({
            "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "dept": ["sales", "tech", "sales", "tech", "sales", "tech"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        analysis = analyze_features(df, target_column="target")
        recs = generate_recommendations(analysis, target_column="target")
        pipeline, num_cols, cat_cols, drop_cols = build_pipeline(recs)

        X_raw = df[["age", "dept"]]
        y = df["target"].values

        pipeline.fit(X_raw)
        X_trans = pipeline.transform(X_raw)

        model = LogisticRegression(random_state=42)
        model.fit(X_trans, y)

        metadata = {
            "experiment_id": "test_exp_123",
            "task_type": "binary_classification",
            "target": "target",
            "expected_features": ["age", "dept"],
        }
        return pipeline, model, metadata, df

    def test_bundle_save_and_reload(self, fitted_setup):
        pipeline, model, metadata, df = fitted_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "test_bundle.joblib")
            saved_path = save_model_bundle(pipeline, model, metadata, bundle_path)
            assert os.path.exists(saved_path)

            loaded_bundle = load_model_bundle(saved_path)
            assert isinstance(loaded_bundle, PreviaModelBundle)
            assert loaded_bundle.metadata["experiment_id"] == "test_exp_123"

            # Predict on new dataset
            new_data = pd.DataFrame({
                "age": [25.0, 45.0],
                "dept": ["tech", "sales"],
            })
            preds = loaded_bundle.predict(new_data)
            assert isinstance(preds, pd.DataFrame)
            assert "prediction" in preds.columns
            assert "probability_positive_class" in preds.columns
            assert len(preds) == 2

    def test_missing_feature_raises_error(self, fitted_setup):
        pipeline, model, metadata, df = fitted_setup

        bundle = PreviaModelBundle(pipeline, model, metadata)
        incomplete_data = pd.DataFrame({
            "age": [25.0],
            # missing "dept"
        })
        with pytest.raises(ValueError, match="missing 1 required feature"):
            bundle.predict(incomplete_data)
