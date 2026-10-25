import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient

from backend.auth import UserIdentity, get_current_user
from backend.main import app
from backend.routers import artifacts as artifacts_router
from previa.benchmarking import benchmarker
from previa.benchmarking.evaluation import (
    build_split_consistency_report,
    verify_holdout_metrics,
)
from previa.export.report_generator import generate_html_report


class FitLengthTracker(BaseEstimator, TransformerMixin):
    fit_lengths = []

    def fit(self, X, y=None):
        type(self).fit_lengths.append(len(X))
        return self

    def transform(self, X):
        return np.asarray(X)


@pytest.fixture
def multiclass_frame():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(90, 3))
    return X


def _classification_data(n=90):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, 4))
    y = np.tile([0, 1, 2], int(np.ceil(n / 3)))[:n]
    return X, y


def test_classification_predictions_metrics_and_class_order_are_verifiable():
    X, y = _classification_data(90)
    labels = ["setosa", "versicolor", "virginica"]
    result = benchmarker.run_benchmarks(
        X,
        y,
        "multiclass_classification",
        models_to_evaluate=["Logistic Regression", "Decision Tree"],
        cv_folds=3,
        sample_ids=[f"row-{i}" for i in range(len(y))],
        class_labels=labels,
    )

    artifacts = result["evaluation_artifacts"]
    assert [item["holdout"]["sample_ids"] for item in artifacts] == [
        artifacts[0]["holdout"]["sample_ids"],
        artifacts[0]["holdout"]["sample_ids"],
    ]
    for item in artifacts:
        holdout = item["holdout"]
        assert holdout["class_labels"] == labels
        assert sum(sum(row) for row in holdout["confusion_matrix"]) == result["holdout_sample_count"]
        assert holdout["confusion_matrix"] == confusion_matrix(
            holdout["actual"], holdout["predicted"], labels=labels
        ).tolist()
        expected_macro = f1_score(
            holdout["actual"], holdout["predicted"], labels=labels,
            average="macro", zero_division=0,
        )
        assert item["cross_validation"]["evaluated_on"] == "training_partition_only"
        report = verify_holdout_metrics(
            holdout["actual"],
            holdout["predicted"],
            "multiclass_classification",
            item["holdout"]["metrics"],
            class_labels=labels,
        )
        assert report["f1"]["status"] == "verified"
        assert report["f1"]["recalculated"] == pytest.approx(expected_macro)


def test_zero_prediction_classes_keep_confusion_matrix_order(monkeypatch):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(90, 4))
    y = np.array([0] * 50 + [1] * 25 + [2] * 15)
    monkeypatch.setattr(
        benchmarker,
        "create_model_instance",
        lambda *args, **kwargs: DummyClassifier(strategy="constant", constant=0),
    )
    result = benchmarker.run_benchmarks(
        X,
        y,
        "multiclass_classification",
        models_to_evaluate=["Always first class"],
        cv_folds=3,
        class_labels=["class-a", "class-b", "class-c"],
    )
    holdout = result["evaluation_artifacts"][0]["holdout"]
    assert holdout["confusion_matrix"] == confusion_matrix(
        holdout["actual"],
        holdout["predicted"],
        labels=["class-a", "class-b", "class-c"],
    ).tolist()
    assert len(holdout["confusion_matrix"]) == 3
    assert holdout["metrics"]["f1_macro"] != holdout["metrics"]["f1_weighted"]


def test_regression_holdout_metrics_and_row_alignment_are_verifiable(monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.1, size=len(X))
    monkeypatch.setattr(
        benchmarker,
        "create_model_instance",
        lambda *args, **kwargs: DummyRegressor(strategy="mean"),
    )
    result = benchmarker.run_benchmarks(
        X,
        y,
        "regression",
        models_to_evaluate=["Mean baseline"],
        cv_folds=3,
        sample_ids=[f"row-{index}" for index in range(len(y))],
    )
    holdout = result["evaluation_artifacts"][0]["holdout"]
    expected = {
        "mae": mean_absolute_error(holdout["actual"], holdout["predicted"]),
        "mse": mean_squared_error(holdout["actual"], holdout["predicted"]),
        "rmse": np.sqrt(mean_squared_error(holdout["actual"], holdout["predicted"])),
        "r2": r2_score(holdout["actual"], holdout["predicted"]),
    }
    for key, value in expected.items():
        assert holdout["metrics"][key] == pytest.approx(value)
    assert len(holdout["sample_ids"]) == len(holdout["actual"]) == len(holdout["predicted"])
    verification = verify_holdout_metrics(
        holdout["actual"], holdout["predicted"], "regression", holdout["metrics"]
    )
    assert all(verification[key]["status"] == "verified" for key in expected)
    assert holdout["confusion_matrix"] is None


def test_cv_fits_preprocessing_separately_within_training_folds(monkeypatch):
    FitLengthTracker.fit_lengths = []
    X, y = _classification_data(90)
    result = benchmarker.run_benchmarks(
        X,
        y,
        "multiclass_classification",
        models_to_evaluate=["Logistic Regression"],
        cv_folds=3,
        test_size=0.2,
        preprocessor=FitLengthTracker(),
    )
    artifact = result["evaluation_artifacts"][0]
    training_count = result["training_sample_count"]
    assert artifact["cross_validation"]["fold_count"] == 3
    assert FitLengthTracker.fit_lengths[0] == training_count
    assert FitLengthTracker.fit_lengths[1:] == [training_count * 2 // 3] * 3


def test_same_holdout_count_with_different_row_ids_is_inconsistent():
    configuration = {"random_state": 42, "test_size": 0.2}
    report = build_split_consistency_report(
        [
            {
                "model": "A",
                "status": "success",
                "split_configuration": configuration,
                "holdout": {"sample_ids": ["row-1", "row-2"], "actual": [0, 1]},
            },
            {
                "model": "B",
                "status": "success",
                "split_configuration": configuration,
                "holdout": {"sample_ids": ["row-1", "row-3"], "actual": [0, 1]},
            },
        ],
        configuration,
    )
    assert report["holdout_sample_count_per_model"] == {"A": 2, "B": 2}
    assert report["holdout_row_identifiers_match"] is False
    assert report["status"] == "inconsistent_or_unavailable"


def test_missing_predictions_are_not_verified():
    result = verify_holdout_metrics(
        None, None, "binary_classification", {"f1": 0.8}
    )
    assert result["f1"]["status"] == "unavailable_for_verification"


def test_html_report_includes_audit_evidence_and_escapes_user_strings():
    manifest = {
        "experiment_id": "exp-test",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "software_versions": {"previa": "1.0"},
        "dataset_metadata": {
            "name": "<script>alert(1)</script>",
            "fingerprint_sha256": "abc123",
            "n_rows": 10,
            "n_columns": 2,
            "schema": {"<script>alert(1)</script>": "object", "target": "int64"},
        },
        "task_setup": {"target_column": "target", "task_type": "multiclass_classification", "random_seed": 42},
        "preprocessing_configuration": {
            "original_dataset_columns": [{"column": "Id", "dtype": "int64"}],
            "numeric_columns_processed": ["value"],
            "categorical_columns_processed": [],
            "excluded_columns": [{"column": "Id", "reason": "Possible identifier."}],
        },
        "model_benchmarking": {
            "primary_metric": "f1",
            "best_model": "model",
            "winner_explanation": "best",
            "models_evaluated": [{"model": "model", "status": "success", "primary_score": 1.0}],
        },
        "experiment_metadata": {
            "split_configuration": {"strategy": "train_test_split"},
            "primary_metric_definition": "Macro-F1",
        },
        "evaluation_summary": {
            "split_consistency": {
                "status": "consistent",
                "evaluated_model_count": 1,
                "holdout_sample_count_per_model": {"model": 2},
                "split_configuration_matches": True,
                "holdout_row_identifiers_match": True,
                "target_labels_aligned": True,
                "missing_or_inconsistent_artifacts": [],
            },
            "models": [
                {
                    "model": "model",
                    "status": "success",
                    "holdout": {
                        "sample_count": 2,
                        "metrics": {"f1": 1.0},
                        "class_labels": ["a", "b"],
                        "confusion_matrix": [[1, 0], [0, 1]],
                    },
                    "cross_validation": {
                        "fold_count": 2,
                        "fold_scores": [0.8, 1.0],
                        "mean": 0.9,
                        "standard_deviation": 0.1,
                        "metric": "f1_macro",
                    },
                    "metric_verification": {
                        "f1": {
                            "reported": 1.0,
                            "recalculated": 1.0,
                            "absolute_difference": 0.0,
                            "status": "verified",
                        },
                    },
                }
            ],
            "warnings": [],
        },
    }
    html_report = generate_html_report(manifest)
    assert "Independent Metric Verification" in html_report
    assert "Confusion matrix" in html_report
    assert "<script>" not in html_report
    assert "&lt;script&gt;" in html_report


def test_evaluation_download_and_samples_enforce_owner_and_pagination(monkeypatch, tmp_path):
    experiment_id = "214f498f-4f0e-40ed-b28f-8059db0c878f"
    sample_key = hashlib.sha256("Decision Tree".encode("utf-8")).hexdigest()[:16]
    experiment_dir = tmp_path / experiment_id
    sample_dir = experiment_dir / "evaluation_samples"
    sample_dir.mkdir(parents=True)
    evaluation_path = experiment_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    sample_path = sample_dir / f"{sample_key}.jsonl"
    sample_path.write_text(
        "\n".join(json.dumps({"sample_id": f"row-{i}", "actual": i, "predicted": i}) for i in range(3))
        + "\n",
        encoding="utf-8",
    )
    record = {
        "artifact_paths": {
            "evaluation": str(evaluation_path),
            f"evaluation_samples_{sample_key}": str(sample_path),
        },
        "results": {
            "holdout_sample_count": 3,
            "models": [{"model": "Decision Tree", "status": "success"}],
        },
    }
    monkeypatch.setattr(artifacts_router.settings, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(
        artifacts_router,
        "get_experiment_record",
        lambda current_id, user_id: record if user_id == "owner-id" else None,
    )
    app.dependency_overrides[get_current_user] = lambda: UserIdentity(
        user_id="owner-id", email="owner@example.test"
    )
    client = TestClient(app)
    try:
        export_response = client.get(f"/api/artifacts/{experiment_id}/evaluation")
        page_response = client.get(
            f"/api/artifacts/{experiment_id}/evaluation/samples",
            params={"model_name": "Decision Tree", "offset": 1, "limit": 1},
        )
        assert export_response.status_code == 200
        assert export_response.json()["format_version"] == 1
        assert page_response.status_code == 200
        assert page_response.json()["samples"] == [
            {"sample_id": "row-1", "actual": 1, "predicted": 1}
        ]
        app.dependency_overrides[get_current_user] = lambda: UserIdentity(
            user_id="intruder-id", email="intruder@example.test"
        )
        forbidden = client.get(f"/api/artifacts/{experiment_id}/evaluation")
        assert forbidden.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_user, None)
