import json

import backend.storage as storage
import backend.supabase_client as supabase_client


def test_delete_user_data_removes_local_files_metadata_and_quota(tmp_path, monkeypatch):
    user_id = "user-to-remove"
    dataset_path = tmp_path / "datasets" / "dataset.csv"
    dataset_path.parent.mkdir()
    dataset_path.write_text("feature,target\n1,0\n", encoding="utf-8")

    artifact_root = tmp_path / "artifacts"
    experiment_dir = artifact_root / "experiment-1"
    experiment_dir.mkdir(parents=True)
    artifact_path = experiment_dir / "model.joblib"
    artifact_path.write_bytes(b"test model")

    datasets_meta = tmp_path / "datasets.json"
    experiments_meta = tmp_path / "experiments.json"
    quotas_meta = tmp_path / "user_quotas.json"
    datasets_meta.write_text(
        json.dumps({"dataset-1": {"user_id": user_id, "file_path": str(dataset_path)}}),
        encoding="utf-8",
    )
    experiments_meta.write_text(
        json.dumps({
            "experiment-1": {
                "user_id": user_id,
                "artifact_paths": {"model_bundle": str(artifact_path)},
            }
        }),
        encoding="utf-8",
    )
    quotas_meta.write_text(json.dumps({user_id: 1024}), encoding="utf-8")

    monkeypatch.setattr(storage, "DATASETS_META_FILE", datasets_meta)
    monkeypatch.setattr(storage, "EXPERIMENTS_META_FILE", experiments_meta)
    monkeypatch.setattr(storage, "USER_QUOTAS_FILE", quotas_meta)
    monkeypatch.setattr(storage.settings, "ARTIFACTS_DIR", artifact_root)
    monkeypatch.setattr(storage.settings, "SUPABASE_URL", "")
    monkeypatch.setattr(storage.settings, "SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: None)

    result = storage.delete_user_data(user_id)

    assert result["success"] is True
    assert result["deleted_datasets"] == 1
    assert result["deleted_experiments"] == 1
    assert result["supabase_auth_deleted"] is None
    assert result["account_deletion_status"] == "not_configured"
    assert not dataset_path.exists()
    assert not experiment_dir.exists()
    assert json.loads(datasets_meta.read_text(encoding="utf-8")) == {}
    assert json.loads(experiments_meta.read_text(encoding="utf-8")) == {}
    assert json.loads(quotas_meta.read_text(encoding="utf-8")) == {}


def test_bulk_purge_clears_all_app_storage_and_retains_every_auth_account(tmp_path, monkeypatch):
    user_id = "other-user"
    founder_id = "founder-user"
    storage_root = tmp_path / "storage"
    datasets_dir = storage_root / "datasets"
    artifacts_dir = storage_root / "artifacts"
    datasets_dir.mkdir(parents=True)
    artifacts_dir.mkdir()

    dataset_path = datasets_dir / "dataset.csv"
    dataset_path.write_text("x,y\n1,2\n", encoding="utf-8")
    artifact_path = artifacts_dir / "orphan.bin"
    artifact_path.write_bytes(b"orphan artifact")
    founder_file = datasets_dir / "founder.csv"
    founder_file.write_text("private data", encoding="utf-8")

    datasets_meta = tmp_path / "datasets.json"
    experiments_meta = tmp_path / "experiments.json"
    quotas_meta = tmp_path / "user_quotas.json"
    datasets_meta.write_text(json.dumps({
        "other-dataset": {"user_id": user_id, "file_path": str(dataset_path)},
        "founder-dataset": {"user_id": founder_id, "file_path": str(founder_file)},
    }), encoding="utf-8")
    experiments_meta.write_text("{}", encoding="utf-8")
    quotas_meta.write_text(json.dumps({user_id: 1024, founder_id: 2048}), encoding="utf-8")

    monkeypatch.setattr(storage, "DATASETS_META_FILE", datasets_meta)
    monkeypatch.setattr(storage, "EXPERIMENTS_META_FILE", experiments_meta)
    monkeypatch.setattr(storage, "USER_QUOTAS_FILE", quotas_meta)
    monkeypatch.setattr(storage.settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(storage.settings, "DATA_DIR", datasets_dir)
    monkeypatch.setattr(storage.settings, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(storage, "SYSTEM_NOTICE_FILE", tmp_path / "system_notice.json")
    monkeypatch.setattr(storage.settings, "SUPABASE_URL", "https://example.test")
    monkeypatch.setattr(storage.settings, "SUPABASE_SERVICE_ROLE_KEY", "configured")
    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: object())
    monkeypatch.setattr(
        supabase_client,
        "get_all_auth_users",
        lambda *, strict: {user_id: "user@example.test", founder_id: "founder@example.test"},
    )
    monkeypatch.setattr(supabase_client, "delete_all_app_records", lambda: {
        "deleted_rows": {"experiments": 1, "datasets": 2, "user_storage_quotas": 2},
        "errors": [],
    })
    monkeypatch.setattr(supabase_client, "clear_urqen_storage_objects", lambda: {
        "deleted_objects": 3,
        "deleted_bytes": 4096,
        "errors": [],
    })
    monkeypatch.setattr(
        supabase_client,
        "delete_auth_user",
        lambda _: (_ for _ in ()).throw(AssertionError("bulk purge must retain all accounts")),
    )

    result = storage.purge_all_user_data(initiated_by_user_id=founder_id)

    assert result["success"] is True
    assert result["processed_users"] == 2
    assert result["deleted_accounts"] == 0
    assert result["retained_auth_accounts"] == 2
    assert result["supabase_storage_objects_deleted"] == 3
    assert result["supabase_storage_bytes_deleted"] == 4096
    assert result["supabase_rows_deleted"] == {
        "experiments": 1,
        "datasets": 2,
        "user_storage_quotas": 2,
    }
    assert all(not item["account_deleted"] for item in result["results"])
    assert result["cleared_storage_bytes"] >= len(b"private data") + len(b"orphan artifact")
    assert list(datasets_dir.iterdir()) == []
    assert list(artifacts_dir.iterdir()) == []
    assert json.loads(datasets_meta.read_text(encoding="utf-8")) == {}
    assert json.loads(experiments_meta.read_text(encoding="utf-8")) == {}
    assert json.loads(quotas_meta.read_text(encoding="utf-8")) == {}
    notice = json.loads((tmp_path / "system_notice.json").read_text(encoding="utf-8"))
    assert "login account is safe" in notice["message"]
    assert set(notice["target_user_ids"]) == {user_id, founder_id}
    assert storage.get_current_system_notice(user_id) == notice
    assert storage.get_current_system_notice("created-after-purge") is None


def test_get_all_auth_users_paginates_and_strict_mode_surfaces_failure(monkeypatch):
    class FakeUser:
        def __init__(self, user_id, email):
            self.id = user_id
            self.email = email

    class FakeAuthAdmin:
        def __init__(self):
            self.calls = []

        def list_users(self, *, page, per_page):
            self.calls.append((page, per_page))
            if page == 1:
                return [FakeUser(f"user-{index}", f"user{index}@example.test") for index in range(per_page)]
            if page == 2:
                return [FakeUser("user-last", "last@example.test")]
            return []

    class FakeClient:
        def __init__(self):
            self.auth = type("Auth", (), {"admin": FakeAuthAdmin()})()

    fake_client = FakeClient()
    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: fake_client)

    users = supabase_client.get_all_auth_users(strict=True)
    assert users["user-0"]["email"] == "user0@example.test"
    assert users["user-last"]["email"] == "last@example.test"
    assert len(users) == 1001
    assert fake_client.auth.admin.calls == [(1, 1000), (2, 1000)]

    class BrokenAdmin:
        def list_users(self, **kwargs):
            raise OSError("provider unavailable")

    fake_client.auth.admin = BrokenAdmin()
    try:
        supabase_client.get_all_auth_users(strict=True)
    except RuntimeError as exc:
        assert "purge was aborted" in str(exc)
    else:
        raise AssertionError("strict listing must surface provider failures")


def test_supabase_app_record_cleanup_targets_only_urqen_tables(monkeypatch):
    rows = {
        "experiments": [{"experiment_id": "exp-1"}],
        "datasets": [{"dataset_id": f"dataset-{index}"} for index in range(501)],
        "user_storage_quotas": [{"user_id": "user-1"}],
    }
    deleted_tables = []

    class Query:
        def __init__(self, table):
            self.table = table
            self.operation = None
            self.key = None
            self.start = 0
            self.end = 499

        def select(self, key):
            self.operation = "select"
            self.key = key
            return self

        def range(self, start, end):
            self.start = start
            self.end = end
            return self

        def delete(self):
            self.operation = "delete"
            return self

        def in_(self, key, ids):
            assert all(isinstance(value, str) for value in ids)
            self.key = key
            self.ids = ids
            return self

        def execute(self):
            if self.operation == "select":
                return type("Response", (), {
                    "data": rows[self.table][self.start : self.end + 1],
                })()
            deleted_tables.append(self.table)
            rows[self.table] = [
                row for row in rows[self.table] if row[self.key] not in self.ids
            ]
            return type("Response", (), {"data": []})()

    class FakeClient:
        def table(self, table):
            return Query(table)

    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: FakeClient())

    result = supabase_client.delete_all_app_records()
    assert result == {
        "deleted_rows": {"experiments": 1, "datasets": 501, "user_storage_quotas": 1},
        "errors": [],
    }
    assert deleted_tables == ["experiments", "datasets", "datasets", "user_storage_quotas"]
    assert not any(rows.values())


def test_supabase_app_record_cleanup_reports_a_failed_table(monkeypatch):
    class Query:
        def __init__(self, table):
            self.table = table

        def select(self, key):
            return self

        def range(self, start, end):
            return self

        def execute(self):
            if self.table == "datasets":
                raise RuntimeError("database unavailable")
            return type("Response", (), {"data": []})()

    class FakeClient:
        def table(self, table):
            return Query(table)

    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: FakeClient())

    result = supabase_client.delete_all_app_records()

    assert result["deleted_rows"] == {
        "experiments": 0,
        "datasets": 0,
        "user_storage_quotas": 0,
    }
    assert result["errors"] == ["Could not clear Supabase table datasets: database unavailable"]


def test_supabase_storage_cleanup_recursively_removes_urqen_objects(monkeypatch):
    removed = {}

    class FakeBucket:
        def __init__(self, bucket_name):
            self.bucket_name = bucket_name

        def list(self, path=None, options=None):
            if self.bucket_name == "previa-artifacts":
                return []
            if not path:
                return [
                    {"name": "root.csv", "id": "root-id", "metadata": {"size": 5}},
                    {"name": "user-1", "id": None, "metadata": None},
                ]
            if path == "user-1":
                return [
                    {"name": "dataset.csv", "id": "dataset-id", "metadata": {"size": "7"}},
                    {"name": "experiment", "id": None, "metadata": None},
                ]
            if path == "user-1/experiment":
                return [
                    {"name": "model.bin", "id": "model-id", "metadata": {"size": 11}},
                ]
            raise AssertionError(f"Unexpected folder {path}")

        def remove(self, paths):
            removed[self.bucket_name] = paths
            return [{"name": path} for path in paths]

    class FakeStorage:
        def list_buckets(self):
            return [
                type("Bucket", (), {"name": "previa-datasets"})(),
                type("Bucket", (), {"name": "previa-artifacts"})(),
                type("Bucket", (), {"name": "unrelated"})(),
            ]

        def from_(self, bucket_name):
            return FakeBucket(bucket_name)

    class FakeClient:
        storage = FakeStorage()

    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: FakeClient())

    result = supabase_client.clear_urqen_storage_objects()

    assert result == {"deleted_objects": 3, "deleted_bytes": 23, "errors": []}
    assert removed == {
        "previa-datasets": ["root.csv", "user-1/dataset.csv", "user-1/experiment/model.bin"],
    }


def test_auth_account_delete_requests_hard_deletion(monkeypatch):
    calls = []

    class FakeAuthAdmin:
        def delete_user(self, user_id, *, should_soft_delete):
            calls.append((user_id, should_soft_delete))

    class FakeClient:
        auth = type("Auth", (), {"admin": FakeAuthAdmin()})()

    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: FakeClient())

    assert supabase_client.delete_auth_user("account-id") is True
    assert calls == [("account-id", False)]


def test_auth_account_delete_exposes_failure_reason(monkeypatch):
    class FakeAuthAdmin:
        def delete_user(self, user_id, *, should_soft_delete):
            raise RuntimeError("Auth policy denied the delete")

    class FakeClient:
        auth = type("Auth", (), {"admin": FakeAuthAdmin()})()

    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: FakeClient())

    deleted, error = supabase_client.delete_auth_user_with_error("account-id")

    assert deleted is False
    assert error == "Auth policy denied the delete"


def test_bulk_purge_aborts_before_local_deletion_if_auth_listing_fails(tmp_path, monkeypatch):
    storage_root = tmp_path / "storage"
    datasets_dir = storage_root / "datasets"
    artifacts_dir = storage_root / "artifacts"
    datasets_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    retained_file = datasets_dir / "must-remain.csv"
    retained_file.write_text("still here", encoding="utf-8")

    monkeypatch.setattr(storage.settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(storage.settings, "DATA_DIR", datasets_dir)
    monkeypatch.setattr(storage.settings, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(storage.settings, "SUPABASE_URL", "https://example.test")
    monkeypatch.setattr(storage.settings, "SUPABASE_SERVICE_ROLE_KEY", "configured")
    monkeypatch.setattr(supabase_client, "get_admin_client", lambda: object())
    monkeypatch.setattr(
        supabase_client,
        "get_all_auth_users",
        lambda *, strict: (_ for _ in ()).throw(RuntimeError("Auth listing failed")),
    )

    try:
        storage.purge_all_user_data()
    except RuntimeError as exc:
        assert "Auth listing failed" in str(exc)
    else:
        raise AssertionError("purge must abort when Auth accounts cannot be enumerated")

    assert retained_file.read_text(encoding="utf-8") == "still here"


def test_bulk_purge_aborts_when_supabase_configuration_is_partial(tmp_path, monkeypatch):
    storage_root = tmp_path / "storage"
    datasets_dir = storage_root / "datasets"
    artifacts_dir = storage_root / "artifacts"
    datasets_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    retained_file = datasets_dir / "must-remain.csv"
    retained_file.write_text("still here", encoding="utf-8")

    monkeypatch.setattr(storage.settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(storage.settings, "DATA_DIR", datasets_dir)
    monkeypatch.setattr(storage.settings, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(storage.settings, "SUPABASE_URL", "https://example.test")
    monkeypatch.setattr(storage.settings, "SUPABASE_SERVICE_ROLE_KEY", "")

    try:
        storage.purge_all_user_data()
    except RuntimeError as exc:
        assert "partially configured" in str(exc)
    else:
        raise AssertionError("partial Supabase configuration must block the purge")

    assert retained_file.read_text(encoding="utf-8") == "still here"


def test_bulk_purge_refuses_dataset_paths_outside_configured_storage(tmp_path, monkeypatch):
    storage_root = tmp_path / "storage"
    datasets_dir = storage_root / "datasets"
    artifacts_dir = storage_root / "artifacts"
    datasets_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    external_file = tmp_path / "do-not-delete.txt"
    external_file.write_text("keep this file", encoding="utf-8")
    datasets_meta = tmp_path / "datasets.json"
    datasets_meta.write_text(json.dumps({
        "unexpected-dataset": {
            "user_id": "user-1",
            "file_path": str(external_file),
        }
    }), encoding="utf-8")

    monkeypatch.setattr(storage, "DATASETS_META_FILE", datasets_meta)
    monkeypatch.setattr(storage, "EXPERIMENTS_META_FILE", tmp_path / "experiments.json")
    monkeypatch.setattr(storage, "USER_QUOTAS_FILE", tmp_path / "user_quotas.json")
    monkeypatch.setattr(storage.settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(storage.settings, "DATA_DIR", datasets_dir)
    monkeypatch.setattr(storage.settings, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(storage.settings, "SUPABASE_URL", "")
    monkeypatch.setattr(storage.settings, "SUPABASE_SERVICE_ROLE_KEY", "")

    try:
        storage.purge_all_user_data()
    except RuntimeError as exc:
        assert "outside configured storage" in str(exc)
    else:
        raise AssertionError("bulk purge must refuse paths outside URQEN storage")

    assert external_file.read_text(encoding="utf-8") == "keep this file"
    assert json.loads(datasets_meta.read_text(encoding="utf-8"))["unexpected-dataset"]["user_id"] == "user-1"
