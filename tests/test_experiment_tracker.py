"""
tests/test_experiment_tracker.py
--------------------------------
Tests for previa.experiments.tracker

We test:
- compute_dataset_fingerprint returns consistent hash
- create_experiment_manifest contains required metadata
- serialize_manifest_to_json produces valid JSON
"""

import json
import pandas as pd
import pytest

from previa.experiments.tracker import (
    compute_dataset_fingerprint,
    create_experiment_manifest,
    serialize_manifest_to_json,
)


class TestExperimentTracker:

    def test_dataset_fingerprint_deterministic(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        fp1 = compute_dataset_fingerprint(df1)
        fp2 = compute_dataset_fingerprint(df2)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_create_manifest_structure(self):
        df = pd.DataFrame({"age": [25, 30, 35], "target": [0, 1, 0]})
        prep_config = {"numeric_features": ["age"]}
        bench_results = {
            "primary_metric": "f1",
            "best_model": "Logistic Regression",
            "winner_explanation": "Won by highest F1",
            "cv_folds": 3,
            "models": [{"model": "Logistic Regression", "f1": 0.95}],
        }
        manifest = create_experiment_manifest(
            df=df,
            target_column="target",
            task_type="binary_classification",
            preprocessing_config=prep_config,
            benchmarking_results=bench_results,
        )

        assert "experiment_id" in manifest
        assert manifest["experiment_id"].startswith("exp_")
        assert "timestamp_utc" in manifest
        assert "software_versions" in manifest
        assert "dataset_metadata" in manifest
        assert manifest["dataset_metadata"]["n_rows"] == 3

    def test_serialize_manifest_valid_json(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        manifest = create_experiment_manifest(
            df=df,
            target_column="y",
            task_type="regression",
            preprocessing_config={},
            benchmarking_results={},
        )
        json_str = serialize_manifest_to_json(manifest)
        parsed = json.loads(json_str)
        assert parsed["experiment_id"] == manifest["experiment_id"]
