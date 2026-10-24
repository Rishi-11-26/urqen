"""
tests/test_api_endpoints.py
---------------------------
Integration tests for URQEN FastAPI REST endpoints.
Tests dataset upload, profiling, quality audit, experimentation, and cascade deletion.
"""

import io
import time
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth import UserIdentity, get_current_user, require_admin
from backend.routers import account as account_router, admin as admin_router

client = TestClient(app)


@pytest.fixture
def sample_csv_bytes():
    df = pd.DataFrame({
        "feature_num": [10.0, 20.0, 30.0, 40.0, 50.0] * 6,
        "feature_cat": ["A", "B", "A", "B", "C"] * 6,
        "target": [0, 1, 0, 1, 0] * 6,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def local_api_user():
    app.dependency_overrides[get_current_user] = lambda: UserIdentity(
        user_id="local_dev_user",
        email="dev@urqen.local",
    )
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_founder_can_remove_another_account_without_removing_self(monkeypatch):
    founder = UserIdentity(user_id="founder-id", email="founder@example.test")
    app.dependency_overrides[require_admin] = lambda: founder
    deleted_user_ids = []
    monkeypatch.setattr(
        admin_router,
        "delete_user_data",
        lambda user_id: deleted_user_ids.append(user_id) or {
            "success": True,
            "deleted_datasets": 0,
            "deleted_experiments": 0,
            "supabase_auth_deleted": True,
            "errors": [],
        },
    )
    try:
        response = client.delete("/api/admin/users/account-to-remove")
        self_response = client.delete("/api/admin/users/founder-id")
    finally:
        app.dependency_overrides.pop(require_admin, None)

    assert response.status_code == 200
    assert response.json()["details"]["success"] is True
    assert self_response.status_code == 400
    assert deleted_user_ids == ["account-to-remove"]


def test_user_removal_reports_unconfirmed_auth_account_deletion(monkeypatch):
    founder = UserIdentity(user_id="founder-id", email="founder@example.test")
    app.dependency_overrides[require_admin] = lambda: founder
    monkeypatch.setattr(
        admin_router,
        "delete_user_data",
        lambda user_id: {
            "success": False,
            "deleted_datasets": 1,
            "deleted_experiments": 0,
            "supabase_quota_deleted": False,
            "supabase_auth_deleted": False,
            "account_deletion_status": "not_confirmed",
            "errors": [
                "Could not remove the cloud storage quota record.",
                "Supabase Auth did not confirm account deletion.",
            ],
        },
    )
    try:
        response = client.delete("/api/admin/users/account-to-remove")
    finally:
        app.dependency_overrides.pop(require_admin, None)

    assert response.status_code == 200
    assert response.json()["details"]["supabase_auth_deleted"] is False
    assert "did not confirm login account deletion" in response.json()["message"]


def test_bulk_purge_requires_confirmation_and_retains_login_accounts(monkeypatch):
    founder = UserIdentity(user_id="founder-id", email="founder@example.test")
    app.dependency_overrides[require_admin] = lambda: founder
    monkeypatch.setattr(admin_router.settings, "ADMIN_EMAILS", "founder@example.test")
    called = []
    monkeypatch.setattr(
        admin_router,
        "purge_all_user_data",
        lambda initiated_by_user_id: called.append(initiated_by_user_id) or {
            "success": True,
            "processed_users": 1,
            "deleted_accounts": 0,
            "retained_auth_accounts": 1,
            "cleared_storage_entries": 0,
            "cleared_storage_bytes": 0,
            "errors": [],
            "results": [],
        },
    )
    try:
        rejected = client.post("/api/admin/purge-all", json={"confirm_text": "wrong"})
        accepted = client.post(
            "/api/admin/purge-all",
            json={"confirm_text": "PURGE ALL USER DATA"},
        )
    finally:
        app.dependency_overrides.pop(require_admin, None)

    assert rejected.status_code == 400
    assert accepted.status_code == 200
    assert accepted.json()["deleted_accounts"] == 0
    assert accepted.json()["retained_auth_accounts"] == 1
    assert called == [founder.user_id]


def test_bulk_purge_is_disabled_without_explicit_founder_allowlist(monkeypatch):
    founder = UserIdentity(user_id="founder-id", email="founder@example.test")
    app.dependency_overrides[require_admin] = lambda: founder
    monkeypatch.setattr(admin_router.settings, "ADMIN_EMAILS", "")
    called = []
    monkeypatch.setattr(
        admin_router,
        "purge_all_user_data",
        lambda: called.append(True),
    )
    try:
        response = client.post(
            "/api/admin/purge-all",
            json={"confirm_text": "PURGE ALL USER DATA"},
        )
    finally:
        app.dependency_overrides.pop(require_admin, None)

    assert response.status_code == 403
    assert called == []


def test_signed_in_user_can_read_system_notice(monkeypatch):
    user = UserIdentity(user_id="user-id", email="user@example.test")
    app.dependency_overrides[get_current_user] = lambda: user
    monkeypatch.setattr(
        account_router,
        "get_current_system_notice",
        lambda user_id: (
            {"notice_id": "notice-1", "message": "Your workspace history was cleared."}
            if user_id == user.user_id
            else None
        ),
    )
    try:
        response = client.get("/api/account/notice")
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    assert response.json()["notice_id"] == "notice-1"


def test_founder_can_download_history_pdf(monkeypatch):
    founder = UserIdentity(user_id="founder-id", email="founder@example.test")
    app.dependency_overrides[require_admin] = lambda: founder
    monkeypatch.setattr(admin_router.settings, "ADMIN_EMAILS", "founder@example.test")
    monkeypatch.setattr(admin_router.settings, "SUPABASE_URL", "")
    monkeypatch.setattr(admin_router.settings, "SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setattr(admin_router, "get_admin_system_overview", lambda: {
        "users": [{
            "user_id": "user-1",
            "email": "user@example.test",
            "dataset_count": 1,
            "experiment_count": 1,
            "total_bytes": 120,
            "quota_bytes": 1000,
        }]
    })
    monkeypatch.setattr(admin_router, "list_all_datasets_admin", lambda: [{
        "dataset_id": "ds-1",
        "user_id": "user-1",
        "name": "sample dataset",
        "row_count": 10,
        "column_count": 2,
    }])
    monkeypatch.setattr(admin_router, "list_all_experiments_admin", lambda: [{
        "experiment_id": "exp-1",
        "user_id": "user-1",
        "dataset_name": "sample dataset",
        "status": "completed",
    }])
    try:
        response = client.get("/api/admin/history.pdf")
    finally:
        app.dependency_overrides.pop(require_admin, None)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_dataset_upload_profile_and_audit(sample_csv_bytes, local_api_user):
    # 1. Upload
    files = {"file": ("test_dataset.csv", sample_csv_bytes, "text/csv")}
    upload_resp = client.post("/api/datasets/upload", files=files)
    assert upload_resp.status_code == 200
    up_data = upload_resp.json()
    assert "dataset_id" in up_data
    dataset_id = up_data["dataset_id"]
    assert up_data["row_count"] == 30
    assert up_data["column_count"] == 3

    # 2. Get Datasets List
    list_resp = client.get("/api/datasets")
    assert list_resp.status_code == 200
    assert any(d["dataset_id"] == dataset_id for d in list_resp.json())

    # 3. Preview
    prev_resp = client.get(f"/api/datasets/{dataset_id}/preview")
    assert prev_resp.status_code == 200
    assert len(prev_resp.json()["preview_rows"]) == 30

    # 4. Profile
    prof_resp = client.get(f"/api/datasets/{dataset_id}/profile")
    assert prof_resp.status_code == 200
    prof_data = prof_resp.json()
    assert prof_data["dimensions"]["rows"] == 30
    assert len(prof_data["columns"]) == 3

    # 5. Audit
    audit_resp = client.get(f"/api/datasets/{dataset_id}/audit?target_column=target")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["target_recommendation"]["suggested_target"] == "target"
    assert len(audit_data["recommendations"]) >= 1

    # Cleanup dataset
    client.delete(f"/api/datasets/{dataset_id}")


def test_experiment_creation_and_lifecycle(sample_csv_bytes, local_api_user):
    # 1. Upload dataset
    files = {"file": ("exp_dataset.csv", sample_csv_bytes, "text/csv")}
    up = client.post("/api/datasets/upload", files=files).json()
    dataset_id = up["dataset_id"]

    # 2. Trigger Experiment
    payload = {
        "dataset_id": dataset_id,
        "target_column": "target",
        "task_type": "binary_classification",
        "mode": "expert",
        "models": ["Logistic Regression", "Decision Tree"],
        "cv_folds": 2,
        "test_size": 0.25,
        "scaler_type": "standard",
    }
    exp_resp = client.post("/api/experiments/benchmark", json=payload)
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    experiment_id = exp_data["experiment_id"]
    assert exp_data["status"] == "running"

    # Wait briefly for background execution
    time.sleep(2)

    # 3. Get Experiment Status
    get_exp = client.get(f"/api/experiments/{experiment_id}")
    assert get_exp.status_code == 200
    data = get_exp.json()
    assert data["experiment_id"] == experiment_id
    assert data["status"] in ("running", "completed")

    # 4. Check Quota
    quota_resp = client.get("/api/experiments/quota/usage")
    assert quota_resp.status_code == 200
    assert quota_resp.json()["experiment_count"] >= 1

    # 5. Delete Experiment
    del_resp = client.delete(f"/api/experiments/{experiment_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Confirm deletion
    get_after = client.get(f"/api/experiments/{experiment_id}")
    assert get_after.status_code == 404

    # Cleanup dataset
    client.delete(f"/api/datasets/{dataset_id}")
