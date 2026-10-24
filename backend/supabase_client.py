"""
backend/supabase_client.py
--------------------------
Admin-level Supabase client (uses service role key) for server-side operations:
  - Writing datasets / experiments to Postgres tables
  - Uploading files to Supabase Storage
  - Storage quota tracking in user_storage_quotas table

This module is OPTIONAL. If SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY are not
configured the client will be None and all sync calls become no-ops, so the app
runs perfectly in local-only mode.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_admin_client = None


def get_admin_client():
    """Returns the admin Supabase client, creating it on first call."""
    global _admin_client
    if _admin_client is not None:
        return _admin_client

    try:
        from backend.config import settings
        from supabase import create_client

        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            logger.info("Supabase admin client disabled — credentials not set.")
            return None

        _admin_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY,
        )
        logger.info("Supabase admin client initialized.")
        return _admin_client
    except Exception as e:
        logger.warning(f"Could not initialize Supabase admin client: {e}")
        return None


def sync_dataset(user_id: str, dataset_record: dict) -> bool:
    """
    Upsert a dataset record to Supabase 'datasets' table.
    Returns True on success, False if sync is skipped or fails.
    """
    client = get_admin_client()
    if not client:
        return False
    try:
        payload = {
            "dataset_id": dataset_record.get("dataset_id"),
            "user_id": user_id,
            "name": dataset_record.get("name") or dataset_record.get("filename", ""),
            "filename": dataset_record.get("filename", ""),
            "row_count": dataset_record.get("row_count") or dataset_record.get("n_rows", 0),
            "column_count": dataset_record.get("column_count") or dataset_record.get("n_columns", 0),
            "file_size_bytes": dataset_record.get("file_size_bytes", 0),
            "storage_path": f"previa-datasets/{user_id}/{dataset_record.get('dataset_id')}",
            "status": dataset_record.get("status", "ready"),
            "created_at": dataset_record.get("created_at"),
        }
        client.table("datasets").upsert(payload, on_conflict="dataset_id").execute()
        return True
    except Exception as e:
        logger.warning(f"Dataset sync failed: {e}")
        return False


def sync_experiment(user_id: str, experiment_record: dict) -> bool:
    """
    Upsert an experiment record to Supabase 'experiments' table.
    """
    client = get_admin_client()
    if not client:
        return False
    try:
        models = experiment_record.get("models", [])
        best = models[0] if models else {}
        payload = {
            "experiment_id": experiment_record.get("experiment_id"),
            "user_id": user_id,
            "dataset_id": experiment_record.get("dataset_id"),
            "task_type": experiment_record.get("task_type", "classification"),
            "target_column": experiment_record.get("target_column", ""),
            "status": experiment_record.get("status", "pending"),
            "best_model_name": best.get("model_name", ""),
            "best_metric_value": best.get("holdout_score"),
            "primary_metric": experiment_record.get("primary_metric", ""),
            "models_evaluated": len(models),
            "created_at": experiment_record.get("created_at"),
            "completed_at": experiment_record.get("completed_at"),
        }
        client.table("experiments").upsert(payload, on_conflict="experiment_id").execute()
        return True
    except Exception as e:
        logger.warning(f"Experiment sync failed: {e}")
        return False


def update_storage_quota(
    user_id: str,
    dataset_bytes: int = 0,
    artifact_bytes: int = 0,
    quota_limit_bytes: int = 52_428_800,
) -> bool:
    """
    Update the user_storage_quotas table with current usage.
    Default quota: 50 MB (52_428_800 bytes).
    """
    client = get_admin_client()
    if not client:
        return False
    try:
        # Check if user_id is a valid UUID (Supabase requires UUID for user_id)
        import uuid
        try:
            uuid.UUID(user_id)
        except ValueError:
            # Skip local dev user IDs that are not UUIDs
            return False

        client.table("user_storage_quotas").upsert(
            {
                "user_id": user_id,
                "dataset_bytes": dataset_bytes,
                "artifact_bytes": artifact_bytes,
                "quota_limit_bytes": quota_limit_bytes,
            },
            on_conflict="user_id",
        ).execute()
        return True
    except Exception as e:
        logger.warning(f"Storage quota update failed: {e}")
        return False


def delete_dataset_record(dataset_id: str) -> bool:
    """Remove a dataset row from the Supabase datasets table."""
    client = get_admin_client()
    if not client:
        return False
    try:
        client.table("datasets").delete().eq("dataset_id", dataset_id).execute()
        return True
    except Exception as e:
        logger.warning(f"Dataset delete sync failed: {e}")
        return False


def delete_experiment_record(experiment_id: str) -> bool:
    """Remove an experiment row from the Supabase experiments table."""
    client = get_admin_client()
    if not client:
        return False
    try:
        client.table("experiments").delete().eq("experiment_id", experiment_id).execute()
        return True
    except Exception as e:
        logger.warning(f"Experiment delete sync failed: {e}")
        return False


def delete_user_storage_quota(user_id: str) -> bool:
    """Remove a user's storage quota row from Supabase."""
    deleted, _ = delete_user_storage_quota_with_error(user_id)
    return deleted


def delete_user_storage_quota_with_error(user_id: str) -> tuple[bool, Optional[str]]:
    """Deletes a quota row and returns a safe provider error for founder diagnostics."""
    client = get_admin_client()
    if not client:
        return False, "Supabase admin client is unavailable."
    try:
        client.table("user_storage_quotas").delete().eq("user_id", user_id).execute()
        return True, None
    except Exception as e:
        logger.warning(f"Storage quota delete failed for user {user_id}: {e}")
        message = getattr(e, "message", None) or str(e) or type(e).__name__
        return False, " ".join(message.split())[:300]


def delete_all_app_records() -> dict:
    """Deletes URQEN metadata rows in dependency order and reports row counts."""
    client = get_admin_client()
    if not client:
        return {
            "deleted_rows": {},
            "errors": ["Supabase admin client is unavailable."],
        }

    errors = []
    deleted_rows = {}
    page_size = 500
    for table, key in (
        ("experiments", "experiment_id"),
        ("datasets", "dataset_id"),
        ("user_storage_quotas", "user_id"),
    ):
        deleted_count = 0
        try:
            offset = 0
            ids = []
            while True:
                response = (
                    client.table(table)
                    .select(key)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                rows = response.data or []
                ids.extend(row[key] for row in rows if row.get(key) is not None)
                if len(rows) < page_size:
                    break
                offset += page_size
            for start in range(0, len(ids), page_size):
                batch = ids[start : start + page_size]
                client.table(table).delete().in_(key, batch).execute()
                deleted_count += len(batch)
            deleted_rows[table] = deleted_count
        except Exception as e:
            deleted_rows[table] = deleted_count
            logger.warning(f"Could not clear URQEN table {table}: {e}")
            message = getattr(e, "message", None) or str(e) or type(e).__name__
            errors.append(f"Could not clear Supabase table {table}: {message}")
    return {"deleted_rows": deleted_rows, "errors": errors}


def clear_urqen_storage_objects() -> dict:
    """Removes objects only from URQEN's known private dataset and artifact buckets."""
    client = get_admin_client()
    if not client:
        return {
            "deleted_objects": 0,
            "deleted_bytes": 0,
            "errors": ["Supabase admin client is unavailable."],
        }

    bucket_names = {"previa-datasets", "previa-artifacts"}
    try:
        existing_buckets = {
            bucket.name
            for bucket in client.storage.list_buckets()
            if bucket.name in bucket_names
        }
    except Exception as e:
        logger.warning("Could not enumerate Supabase Storage buckets: %s", e)
        message = getattr(e, "message", None) or str(e) or type(e).__name__
        return {
            "deleted_objects": 0,
            "deleted_bytes": 0,
            "errors": [f"Could not enumerate URQEN Storage buckets: {message}"],
        }

    deleted_objects = 0
    deleted_bytes = 0
    errors = []
    page_size = 1000
    delete_batch_size = 100

    for bucket_name in sorted(existing_buckets):
        bucket = client.storage.from_(bucket_name)
        objects = []
        visited_folders = set()
        bucket_errors = []

        def list_folder(prefix: str) -> None:
            if prefix in visited_folders:
                return
            visited_folders.add(prefix)
            offset = 0
            while True:
                try:
                    entries = bucket.list(
                        prefix or None,
                        options={"limit": page_size, "offset": offset},
                    )
                except Exception as exc:
                    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
                    bucket_errors.append(f"Could not list objects in {bucket_name}/{prefix}: {message}")
                    return

                for entry in entries:
                    name = entry.get("name")
                    if not name:
                        bucket_errors.append(f"Supabase returned an unnamed object in {bucket_name}/{prefix}.")
                        continue
                    object_path = f"{prefix}/{name}" if prefix else name
                    if entry.get("id") is None and entry.get("metadata") is None:
                        list_folder(object_path)
                    else:
                        metadata = entry.get("metadata") or {}
                        try:
                            size = int(metadata.get("size") or metadata.get("contentLength") or 0)
                        except (TypeError, ValueError):
                            size = 0
                        objects.append((object_path, size))

                if len(entries) < page_size:
                    break
                offset += page_size

        list_folder("")
        if bucket_errors:
            errors.extend(bucket_errors)
            continue

        paths = [path for path, _ in objects]
        for start in range(0, len(paths), delete_batch_size):
            batch = paths[start : start + delete_batch_size]
            try:
                bucket.remove(batch)
                deleted_objects += len(batch)
                deleted_bytes += sum(size for _, size in objects[start : start + delete_batch_size])
            except Exception as e:
                logger.warning("Could not remove Supabase Storage objects from %s: %s", bucket_name, e)
                message = getattr(e, "message", None) or str(e) or type(e).__name__
                errors.append(f"Could not remove objects from Supabase bucket {bucket_name}: {message}")

    return {
        "deleted_objects": deleted_objects,
        "deleted_bytes": deleted_bytes,
        "errors": errors,
    }


def get_all_auth_users(*, strict: bool = False) -> dict:
    """
    Fetches all registered users from Supabase Auth admin API.
    Returns mapping of {user_id: email}.
    """
    client = get_admin_client()
    if not client:
        if strict:
            raise RuntimeError("Supabase admin client is unavailable.")
        return {}
    try:
        user_map = {}
        page = 1
        page_size = 1000
        while True:
            resp = client.auth.admin.list_users(page=page, per_page=page_size)
            users_list = getattr(resp, "users", resp) if not isinstance(resp, list) else resp
            for user in users_list:
                user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
                email = getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else None)
                if user_id:
                    user_map[str(user_id)] = email or ""
            if len(users_list) < page_size:
                break
            page += 1
        return user_map
    except Exception as e:
        logger.warning(f"Could not list Supabase auth users: {e}")
        if strict:
            raise RuntimeError("Could not enumerate all Supabase Auth users; purge was aborted.") from e
        return {}


def delete_auth_user(user_id: str) -> bool:
    """Permanently deletes a user account from Supabase Auth."""
    deleted, _ = delete_auth_user_with_error(user_id)
    return deleted


def delete_auth_user_with_error(user_id: str) -> tuple[bool, Optional[str]]:
    """Hard-deletes a Supabase Auth account and returns a safe provider error."""
    client = get_admin_client()
    if not client:
        return False, "Supabase admin client is unavailable."
    try:
        client.auth.admin.delete_user(user_id, should_soft_delete=False)
        logger.info("Supabase Auth confirmed deletion for user %s.", user_id)
        return True, None
    except Exception as e:
        logger.warning(f"Could not delete Supabase auth user {user_id}: {e}")
        message = getattr(e, "message", None) or str(e) or type(e).__name__
        return False, " ".join(message.split())[:300]
