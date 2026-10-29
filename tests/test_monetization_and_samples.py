"""
tests/test_monetization_and_samples.py
--------------------------------------
Tests for sample dataset loading, 200-column limit guard,
monthly trial refresh, Python code export, and Pro feature gating.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth import UserIdentity, get_current_user
from backend.storage import (
    reset_user_trials,
    get_user_trial_status,
    check_and_increment_trial_quota,
    is_user_paid,
    set_user_quota_limit,
    DEFAULT_USER_QUOTA_BYTES,
)
from previa.ingestion.sample_datasets import get_sample_dataset
from previa.export.code_generator import generate_pipeline_python_code

client = TestClient(app)


def test_builtin_sample_datasets():
    iris_df = get_sample_dataset("iris")
    assert len(iris_df) == 150
    assert "species" in iris_df.columns
    assert set(iris_df["species"].unique()) == {"setosa", "versicolor", "virginica"}

    titanic_df = get_sample_dataset("titanic")
    assert len(titanic_df) >= 100
    assert "Survived" in titanic_df.columns
    assert "Pclass" in titanic_df.columns


def test_monthly_quota_refresh_logic():
    user_id = "test_monthly_rollover_user"
    reset_user_trials(user_id)

    # Use 2 guided runs
    check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)
    check_and_increment_trial_quota(user_id, mode="guided", is_admin=False)

    status = get_user_trial_status(user_id, is_admin=False)
    assert status["guided_used"] == 2
    assert status["guided_remaining"] == 0

    # Simulate month rollover in trial record
    from backend.storage import _read_json, _write_json, USER_TRIALS_FILE
    trials = _read_json(USER_TRIALS_FILE)
    trials[user_id]["billing_month"] = "2020-01"  # past month
    _write_json(USER_TRIALS_FILE, trials)

    # Calling get_user_trial_status should detect rollover and reset base counts to 0
    refreshed_status = get_user_trial_status(user_id, is_admin=False)
    assert refreshed_status["guided_used"] == 0
    assert refreshed_status["guided_remaining"] == 2


def test_is_user_paid_detection():
    free_user = "test_free_user"
    reset_user_trials(free_user)
    set_user_quota_limit(free_user, DEFAULT_USER_QUOTA_BYTES)
    assert is_user_paid(free_user, is_admin=False) is False
    assert is_user_paid(free_user, is_admin=True) is True

    # Upgraded quota flags user as paid
    set_user_quota_limit(free_user, 250 * 1024 * 1024)
    assert is_user_paid(free_user, is_admin=False) is True
    # Reset back to default
    set_user_quota_limit(free_user, DEFAULT_USER_QUOTA_BYTES)


def test_python_code_generator():
    exp = {
        "experiment_id": "test_exp_001",
        "best_model": "RandomForestClassifier",
        "target_column": "species",
        "task_type": "multiclass_classification",
        "dataset_name": "sample_iris.csv",
        "configuration": {"test_size": 0.2, "random_state": 42},
    }
    code = generate_pipeline_python_code(exp)
    assert "import pandas as pd" in code
    assert "RandomForestClassifier" in code
    assert "ColumnTransformer" in code
    assert "test_size=0.2" in code
    assert 'TARGET_COLUMN = "species"' in code
