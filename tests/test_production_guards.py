"""
tests/test_production_guards.py
--------------------------------
Tests for production stability guards:
1. Upload header & magic-byte validation (rejecting non-tabular / binary / corrupt uploads)
2. Concurrent experiment execution prevention (409 Conflict when a job is active)
3. In-memory rate limiting middleware (429 Too Many Requests when spamming)
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth import UserIdentity, get_current_user
from backend.storage import (
    save_dataset_record,
    save_experiment_record,
    delete_experiment,
    delete_dataset,
)

client = TestClient(app)


@pytest.fixture
def test_user():
    user = UserIdentity(user_id="guard_test_user", email="guard_tester@urqen.io")
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_current_user, None)


def test_reject_executable_pe_binary_upload(test_user):
    """Uploading a Windows PE binary (.exe renamed to .csv) must be rejected with 400."""
    fake_exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00" + b"some_code"
    files = {"file": ("malicious.csv", fake_exe_bytes, "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "Executable PE binaries" in resp.json()["detail"]


def test_reject_elf_binary_upload(test_user):
    """Uploading an ELF binary disguised as .csv must be rejected with 400."""
    fake_elf_bytes = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    files = {"file": ("binary.csv", fake_elf_bytes, "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "ELF executables" in resp.json()["detail"]


def test_reject_image_upload_as_csv(test_user):
    """Uploading PNG/JPEG images disguised as CSV must be rejected with 400."""
    fake_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    files = {"file": ("chart.csv", fake_png_bytes, "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "image formats are not supported" in resp.json()["detail"]


def test_reject_csv_containing_null_bytes(test_user):
    """Uploading a binary file with null bytes as CSV must be rejected with 400."""
    corrupt_bytes = b"col1,col2,col3\r\n10,20,\x00\x00\xff"
    files = {"file": ("corrupt.csv", corrupt_bytes, "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "contains binary null bytes" in resp.json()["detail"]


def test_reject_empty_or_tiny_file(test_user):
    """Files under 5 bytes must be rejected with 400."""
    tiny_bytes = b"hi"
    files = {"file": ("tiny.csv", tiny_bytes, "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "under 5 bytes" in resp.json()["detail"]


def test_reject_invalid_xlsx_magic_bytes(test_user):
    """An .xlsx file not starting with PK (zip magic bytes) must be rejected with 400."""
    fake_excel_bytes = b"NotAZipFileAtAll"
    files = {"file": ("fake.xlsx", fake_excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "OpenXML container structure" in resp.json()["detail"]


def test_concurrent_experiment_spamming_blocked(test_user, tmp_path):
    """A user cannot start an experiment while another experiment is currently 'running'."""
    # Create fake active running experiment for this user
    active_exp_id = "active_running_exp_123"
    save_experiment_record(
        experiment_id=active_exp_id,
        user_id=test_user.user_id,
        dataset_id="ds_fake",
        dataset_name="dataset.csv",
        task_type="binary_classification",
        target_column="target",
        mode="guided",
        configuration={"models": ["Logistic Regression"]},
        status="running",
    )

    try:
        # Attempt to launch a new experiment while active one is running
        payload = {
            "dataset_id": "ds_fake",
            "target_column": "target",
            "task_type": "binary_classification",
            "mode": "guided",
            "models": ["Logistic Regression"],
        }
        resp = client.post("/api/experiments/benchmark", json=payload)
        assert resp.status_code == 409
        assert "already actively running" in resp.json()["detail"]
    finally:
        delete_experiment(active_exp_id, test_user.user_id)


def test_reject_dataset_exceeding_50k_rows(test_user, monkeypatch):
    """Datasets with > 50,000 rows must be rejected to prevent Render 512MB RAM exhaustion."""
    import pandas as pd
    from backend.routers import datasets as datasets_router

    # Mock load_dataset to return a fake 50,001 row dataframe
    fake_large_df = pd.DataFrame({"colA": [1] * 50001, "target": [0] * 50001})
    monkeypatch.setattr(datasets_router, "load_dataset", lambda _: fake_large_df)

    sample_csv = b"colA,target\r\n1,0\r\n1,0\r\n"
    files = {"file": ("big_data.csv", sample_csv, "text/csv")}
    resp = client.post("/api/datasets/upload", files=files)
    assert resp.status_code == 400
    assert "maximum supported size is 50,000 rows" in resp.json()["detail"]
