"""
backend/storage.py
------------------
Persistence and storage lifecycle manager for URQEN.
Enforces user isolation, file ownership verification, storage quotas,
and cascading deletion across database records and physical file artifacts.
"""

import json
import shutil
import pathlib
import datetime
from typing import Optional, Dict, Any, List
from backend.config import settings

META_DIR = settings.BASE_DIR / "storage" / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_META_FILE = META_DIR / "datasets.json"
EXPERIMENTS_META_FILE = META_DIR / "experiments.json"
SYSTEM_NOTICE_FILE = META_DIR / "system_notice.json"


def _read_json(path: pathlib.Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_json(path: pathlib.Path, data: dict) -> None:
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    temp_path.replace(path)


def _cloud_sync_configured() -> bool:
    url_configured = bool(settings.SUPABASE_URL)
    service_key_configured = bool(settings.SUPABASE_SERVICE_ROLE_KEY)
    if url_configured != service_key_configured:
        raise RuntimeError(
            "Supabase is only partially configured; set both URL and service-role key before deleting data."
        )
    return url_configured


# ---------------------------------------------------------------------------
# Storage Usage & Quota
# ---------------------------------------------------------------------------

def get_storage_stats(user_id: str) -> dict:
    """Computes actual storage usage for a given user."""
    datasets = list_datasets(user_id)
    experiments = list_experiments(user_id)

    total_dataset_bytes = sum(d.get("size_bytes", 0) for d in datasets)

    total_artifact_bytes = 0
    for exp in experiments:
        paths = exp.get("artifact_paths", {})
        for p_str in paths.values():
            if p_str and pathlib.Path(p_str).exists():
                try:
                    total_artifact_bytes += pathlib.Path(p_str).stat().st_size
                except Exception:
                    pass

    total_used = total_dataset_bytes + total_artifact_bytes
    quota_limit = get_user_quota_limit(user_id)

    return {
        "user_id": user_id,
        "dataset_count": len(datasets),
        "dataset_bytes": total_dataset_bytes,
        "experiment_count": len(experiments),
        "artifact_bytes": total_artifact_bytes,
        "total_used_bytes": total_used,
        "quota_limit_bytes": quota_limit,
        "quota_exhausted": total_used >= quota_limit,
    }


# ---------------------------------------------------------------------------
# Dataset Persistence Operations
# ---------------------------------------------------------------------------

def save_dataset_record(
    dataset_id: str,
    user_id: str,
    name: str,
    file_path: pathlib.Path,
    row_count: int,
    column_count: int,
    file_type: str,
) -> dict:
    """Registers a dataset record owned by user_id."""
    meta = _read_json(DATASETS_META_FILE)
    size_bytes = file_path.stat().st_size if file_path.exists() else 0

    record = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "name": name,
        "file_path": str(file_path.resolve()),
        "size_bytes": size_bytes,
        "row_count": row_count,
        "column_count": column_count,
        "file_type": file_type,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta[dataset_id] = record
    _write_json(DATASETS_META_FILE, meta)
    return record


def get_dataset_record(dataset_id: str, user_id: str) -> Optional[dict]:
    """Retrieves a dataset record enforcing user ownership."""
    meta = _read_json(DATASETS_META_FILE)
    rec = meta.get(dataset_id)
    if rec and rec.get("user_id") == user_id:
        return rec
    return None


def list_datasets(user_id: str) -> List[dict]:
    """Lists all datasets owned by a user."""
    meta = _read_json(DATASETS_META_FILE)
    return [rec for rec in meta.values() if rec.get("user_id") == user_id]


def delete_dataset(dataset_id: str, user_id: str) -> bool:
    """Deletes dataset record and physical file with ownership verification."""
    meta = _read_json(DATASETS_META_FILE)
    rec = meta.get(dataset_id)
    if not rec or rec.get("user_id") != user_id:
        return False

    file_path = pathlib.Path(rec.get("file_path", ""))
    if file_path.is_file():
        try:
            file_path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Could not delete dataset file for {dataset_id}.") from exc
    elif file_path.exists():
        raise RuntimeError(f"Dataset path for {dataset_id} is not a file.")

    del meta[dataset_id]
    _write_json(DATASETS_META_FILE, meta)
    return True


# ---------------------------------------------------------------------------
# Experiment Persistence Operations
# ---------------------------------------------------------------------------

def save_experiment_record(
    experiment_id: str,
    user_id: str,
    dataset_id: str,
    dataset_name: str,
    task_type: str,
    target_column: str,
    mode: str,
    configuration: dict,
    status: str = "pending",
) -> dict:
    """Registers an experiment record owned by user_id."""
    meta = _read_json(EXPERIMENTS_META_FILE)
    record = {
        "experiment_id": experiment_id,
        "user_id": user_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_column,
        "mode": mode,  # "guided" or "expert"
        "configuration": configuration,
        "status": status,
        "results": None,
        "best_model": None,
        "winner_explanation": None,
        "artifact_paths": {},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed_at": None,
        "error_message": None,
    }
    meta[experiment_id] = record
    _write_json(EXPERIMENTS_META_FILE, meta)
    return record


def update_experiment_results(
    experiment_id: str,
    user_id: str,
    results: dict,
    best_model: str,
    winner_explanation: str,
    artifact_paths: dict,
) -> Optional[dict]:
    """Updates a completed experiment with benchmark results and artifact locations."""
    meta = _read_json(EXPERIMENTS_META_FILE)
    rec = meta.get(experiment_id)
    if not rec or rec.get("user_id") != user_id:
        return None

    rec["status"] = "completed"
    rec["results"] = results
    rec["best_model"] = best_model
    rec["winner_explanation"] = winner_explanation
    rec["artifact_paths"] = artifact_paths
    rec["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    _write_json(EXPERIMENTS_META_FILE, meta)
    return rec


def update_experiment_error(experiment_id: str, user_id: str, error_msg: str) -> None:
    meta = _read_json(EXPERIMENTS_META_FILE)
    rec = meta.get(experiment_id)
    if rec and rec.get("user_id") == user_id:
        rec["status"] = "failed"
        rec["error_message"] = error_msg
        _write_json(EXPERIMENTS_META_FILE, meta)


def get_experiment_record(experiment_id: str, user_id: str) -> Optional[dict]:
    """Retrieves an experiment record enforcing user ownership."""
    meta = _read_json(EXPERIMENTS_META_FILE)
    rec = meta.get(experiment_id)
    if not rec:
        return None
    if rec.get("user_id") == user_id:
        return rec
    # Allow fallback for local development or direct downloads without strict JWT secret
    if user_id == "local_dev_user" and not settings.SUPABASE_JWT_SECRET:
        return rec
    return None


def list_experiments(user_id: str) -> List[dict]:
    """Lists all experiments owned by a user, ordered by creation date descending."""
    meta = _read_json(EXPERIMENTS_META_FILE)
    user_exps = [rec for rec in meta.values() if rec.get("user_id") == user_id]
    return sorted(user_exps, key=lambda x: x.get("created_at", ""), reverse=True)


def delete_experiment(experiment_id: str, user_id: str) -> dict:
    """
    Cascading cleanup:
    1. Verifies ownership.
    2. Identifies all associated physical files (model bundles, reports, manifests).
    3. Deletes physical files.
    4. Deletes database/metadata record.
    Returns audit status of cleanup.
    """
    meta = _read_json(EXPERIMENTS_META_FILE)
    rec = meta.get(experiment_id)
    if not rec or rec.get("user_id") != user_id:
        return {"success": False, "reason": "Experiment not found or access denied."}

    deleted_files = []
    failed_files = []

    # Clean physical artifacts
    artifact_paths = rec.get("artifact_paths", {})
    for label, path_str in artifact_paths.items():
        if path_str:
            p = pathlib.Path(path_str)
            if p.exists():
                try:
                    p.unlink()
                    deleted_files.append(str(p))
                except Exception as e:
                    failed_files.append({"file": str(p), "error": str(e)})

    # Also clean experiment directory if one exists
    exp_dir = settings.ARTIFACTS_DIR / experiment_id
    if exp_dir.exists() and exp_dir.is_dir():
        try:
            shutil.rmtree(exp_dir)
            deleted_files.append(str(exp_dir))
        except Exception as e:
            failed_files.append({"dir": str(exp_dir), "error": str(e)})

    # Delete record
    del meta[experiment_id]
    _write_json(EXPERIMENTS_META_FILE, meta)

    return {
        "success": True,
        "experiment_id": experiment_id,
        "deleted_files_count": len(deleted_files),
        "deleted_files": deleted_files,
        "failed_files": failed_files,
    }


# ---------------------------------------------------------------------------
# Admin Management Operations
# ---------------------------------------------------------------------------

USER_QUOTAS_FILE = META_DIR / "user_quotas.json"
DEFAULT_USER_QUOTA_BYTES = 50 * 1024 * 1024  # 50 MB default per user


def get_user_quota_limit(user_id: str) -> int:
    """Returns user's quota limit in bytes (custom or default 50MB)."""
    quotas = _read_json(USER_QUOTAS_FILE)
    return quotas.get(user_id, DEFAULT_USER_QUOTA_BYTES)


def set_user_quota_limit(user_id: str, quota_bytes: int) -> int:
    """Sets a custom quota limit in bytes for a user."""
    quotas = _read_json(USER_QUOTAS_FILE)
    quotas[user_id] = quota_bytes
    _write_json(USER_QUOTAS_FILE, quotas)
    return quota_bytes


def list_all_datasets_admin() -> List[dict]:
    """Lists all datasets across all users, sorted by creation date."""
    meta = _read_json(DATASETS_META_FILE)
    return sorted(list(meta.values()), key=lambda x: x.get("created_at", ""), reverse=True)


def list_all_experiments_admin() -> List[dict]:
    """Lists all experiments across all users, sorted by creation date."""
    meta = _read_json(EXPERIMENTS_META_FILE)
    return sorted(list(meta.values()), key=lambda x: x.get("created_at", ""), reverse=True)


def get_admin_system_overview() -> dict:
    """Computes global system analytics across all users and workloads."""
    datasets_meta = _read_json(DATASETS_META_FILE)
    experiments_meta = _read_json(EXPERIMENTS_META_FILE)

    # Resolve emails from Supabase Auth admin client if configured
    auth_users = {}
    try:
        from backend.supabase_client import get_all_auth_users
        auth_users = get_all_auth_users()
    except Exception:
        auth_users = {}

    all_users = set()
    for uid in auth_users.keys():
        all_users.add(uid)

    total_dataset_bytes = 0
    for d in datasets_meta.values():
        u_id = d.get("user_id", "unknown")
        all_users.add(u_id)
        total_dataset_bytes += d.get("size_bytes", 0)

    total_artifact_bytes = 0
    completed_experiments = 0
    failed_experiments = 0

    for exp in experiments_meta.values():
        u_id = exp.get("user_id", "unknown")
        all_users.add(u_id)
        if exp.get("status") == "completed":
            completed_experiments += 1
        elif exp.get("status") == "failed":
            failed_experiments += 1

        paths = exp.get("artifact_paths", {})
        for p_str in paths.values():
            if p_str and pathlib.Path(p_str).exists():
                try:
                    total_artifact_bytes += pathlib.Path(p_str).stat().st_size
                except Exception:
                    pass

    total_storage_used = total_dataset_bytes + total_artifact_bytes

    # User level metrics
    user_summaries = []
    for u_id in all_users:
        u_datasets = [d for d in datasets_meta.values() if d.get("user_id") == u_id]
        u_experiments = [e for e in experiments_meta.values() if e.get("user_id") == u_id]

        u_ds_bytes = sum(d.get("size_bytes", 0) for d in u_datasets)
        u_art_bytes = 0
        latest_act = None

        for d in u_datasets:
            ca = d.get("created_at")
            if ca and (latest_act is None or ca > latest_act):
                latest_act = ca

        for e in u_experiments:
            ca = e.get("created_at")
            if ca and (latest_act is None or ca > latest_act):
                latest_act = ca
            for p_str in e.get("artifact_paths", {}).values():
                if p_str and pathlib.Path(p_str).exists():
                    try:
                        u_art_bytes += pathlib.Path(p_str).stat().st_size
                    except Exception:
                        pass

        u_total = u_ds_bytes + u_art_bytes
        quota_limit = get_user_quota_limit(u_id)

        # Determine user email
        user_email = auth_users.get(u_id)
        if not user_email:
            for d in u_datasets:
                if d.get("user_email"):
                    user_email = d["user_email"]
                    break
            if not user_email:
                for e in u_experiments:
                    if e.get("user_email"):
                        user_email = e["user_email"]
                        break
        if not user_email:
            user_email = u_id if "@" in u_id else None

        user_summaries.append({
            "user_id": u_id,
            "email": user_email,
            "dataset_count": len(u_datasets),
            "experiment_count": len(u_experiments),
            "dataset_bytes": u_ds_bytes,
            "artifact_bytes": u_art_bytes,
            "total_bytes": u_total,
            "quota_bytes": quota_limit,
            "quota_used_pct": round((u_total / quota_limit * 100), 1) if quota_limit > 0 else 0,
            "last_active": latest_act or "N/A",
        })

    # Sort users by total storage descending
    user_summaries.sort(key=lambda x: x["total_bytes"], reverse=True)

    return {
        "total_users": len(all_users),
        "total_datasets": len(datasets_meta),
        "total_experiments": len(experiments_meta),
        "completed_experiments": completed_experiments,
        "failed_experiments": failed_experiments,
        "total_dataset_bytes": total_dataset_bytes,
        "total_artifact_bytes": total_artifact_bytes,
        "total_storage_used_bytes": total_storage_used,
        "global_capacity_bytes": 1024 * 1024 * 1024,  # 1 GB standard free tier
        "global_capacity_used_pct": round((total_storage_used / (1024 * 1024 * 1024)) * 100, 2),
        "users": user_summaries,
    }


def delete_user_data(user_id: str, *, delete_auth_account: bool = True) -> dict:
    """
    Completely deletes all resources associated with a user:
    - All dataset physical files and metadata
    - All experiment artifact directories and metadata
    - Storage quota tracking
    - Supabase cloud records and Supabase Auth user account
    """
    datasets_meta = _read_json(DATASETS_META_FILE)
    experiments_meta = _read_json(EXPERIMENTS_META_FILE)

    deleted_datasets = 0
    deleted_experiments = 0
    sync_failures = []
    cloud_configured = _cloud_sync_configured()
    if cloud_configured:
        from backend.supabase_client import get_admin_client

        if get_admin_client() is None:
            raise RuntimeError("Supabase is configured but its admin client is unavailable.")

    # 1. Delete datasets owned by this user
    for ds_id, d in list(datasets_meta.items()):
        if d.get("user_id") == user_id:
            if delete_dataset(ds_id, user_id):
                deleted_datasets += 1
                from backend.supabase_client import delete_dataset_record
                if cloud_configured and not delete_dataset_record(ds_id):
                    sync_failures.append(f"Could not remove cloud dataset record {ds_id}.")

    # 2. Delete experiments owned by this user
    for exp_id, e in list(experiments_meta.items()):
        if e.get("user_id") == user_id:
            delete_experiment(exp_id, user_id)
            deleted_experiments += 1
            from backend.supabase_client import delete_experiment_record
            if cloud_configured and not delete_experiment_record(exp_id):
                sync_failures.append(f"Could not remove cloud experiment record {exp_id}.")

    quotas = _read_json(USER_QUOTAS_FILE)
    quotas.pop(user_id, None)
    _write_json(USER_QUOTAS_FILE, quotas)

    # 3. Remove cloud quota metadata and the Supabase Auth account
    supabase_quota_deleted = False
    if cloud_configured:
        from backend.supabase_client import delete_user_storage_quota_with_error

        supabase_quota_deleted, quota_error = delete_user_storage_quota_with_error(user_id)
        if not supabase_quota_deleted:
            sync_failures.append(
                f"Could not remove the cloud storage quota record: {quota_error or 'provider did not confirm deletion'}"
            )

    supabase_auth_deleted = None
    if delete_auth_account and cloud_configured:
        from backend.supabase_client import delete_auth_user_with_error

        supabase_auth_deleted, auth_error = delete_auth_user_with_error(user_id)
        if not supabase_auth_deleted:
            sync_failures.append(
                f"Supabase Auth did not confirm account deletion: {auth_error or 'provider did not confirm deletion'}"
            )

    return {
        "user_id": user_id,
        "deleted_datasets": deleted_datasets,
        "deleted_experiments": deleted_experiments,
        "supabase_quota_deleted": supabase_quota_deleted,
        "supabase_auth_deleted": supabase_auth_deleted,
        "account_deletion_status": (
            "deleted" if supabase_auth_deleted is True
            else "not_confirmed" if supabase_auth_deleted is False
            else "not_configured"
        ),
        "success": not sync_failures,
        "errors": sync_failures,
    }


def get_current_system_notice(user_id: str) -> Optional[dict]:
    """Returns a notice only when it targets the signed-in account."""
    notice = _read_json(SYSTEM_NOTICE_FILE)
    if user_id not in notice.get("target_user_ids", []):
        return None
    return notice


def purge_all_user_data(initiated_by_user_id: Optional[str] = None) -> dict:
    """Remove URQEN data for every user while retaining all login accounts."""
    from backend.supabase_client import (
        delete_all_app_records,
        get_admin_client,
        get_all_auth_users,
    )

    cloud_configured = _cloud_sync_configured()
    auth_users = {}
    if cloud_configured:
        if get_admin_client() is None:
            raise RuntimeError("Supabase is configured but its admin client is unavailable.")
        auth_users = get_all_auth_users(strict=True)

    storage_root = settings.BASE_DIR / "storage"
    storage_dirs = (settings.DATA_DIR, settings.ARTIFACTS_DIR)
    for directory in storage_dirs:
        if (
            directory.name not in {"datasets", "artifacts"}
            or directory.is_symlink()
            or storage_root.is_symlink()
            or directory.resolve() != (storage_root / directory.name).resolve()
        ):
            raise RuntimeError(f"Refused to clear unexpected storage directory: {directory}")

    def directory_size(directory: pathlib.Path) -> int:
        total = 0
        if not directory.exists():
            return total
        for path in directory.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_file():
                total += path.stat().st_size
        return total

    def validate_owned_path(path_text: str, allowed_root: pathlib.Path) -> None:
        resolved_path = pathlib.Path(path_text).resolve()
        resolved_root = allowed_root.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise RuntimeError(f"Refused to purge a path outside configured storage: {path_text}")

    cleared_bytes = sum(directory_size(directory) for directory in storage_dirs)

    datasets_meta = _read_json(DATASETS_META_FILE)
    experiments_meta = _read_json(EXPERIMENTS_META_FILE)
    for dataset_id, record in datasets_meta.items():
        file_path = record.get("file_path")
        if file_path:
            try:
                validate_owned_path(file_path, settings.DATA_DIR)
            except RuntimeError as exc:
                raise RuntimeError(f"Dataset {dataset_id}: {exc}") from exc
    for experiment_id, record in experiments_meta.items():
        for artifact_path in record.get("artifact_paths", {}).values():
            if artifact_path:
                try:
                    validate_owned_path(artifact_path, settings.ARTIFACTS_DIR)
                except RuntimeError as exc:
                    raise RuntimeError(f"Experiment {experiment_id}: {exc}") from exc
    quotas = _read_json(USER_QUOTAS_FILE)
    user_ids = set(auth_users) | set(quotas)
    user_ids.update(
        record.get("user_id")
        for record in datasets_meta.values()
        if record.get("user_id")
    )
    user_ids.update(
        record.get("user_id")
        for record in experiments_meta.values()
        if record.get("user_id")
    )
    if initiated_by_user_id:
        user_ids.add(initiated_by_user_id)

    results = [
        {
            "user_id": user_id,
            "account_deleted": False,
            "datasets_deleted": 0,
            "experiments_deleted": 0,
        }
        for user_id in sorted(user_ids)
    ]
    results_by_user = {item["user_id"]: item for item in results}
    errors = []
    supabase_storage = {
        "deleted_objects": 0,
        "deleted_bytes": 0,
        "errors": [],
    }
    supabase_records = {"deleted_rows": {}, "errors": []}

    if cloud_configured:
        from backend.supabase_client import (
            clear_urqen_storage_objects,
            delete_all_app_records,
        )

        supabase_storage = clear_urqen_storage_objects()
        errors.extend(supabase_storage["errors"])
        supabase_records = delete_all_app_records()
        errors.extend(supabase_records["errors"])

    for dataset_id, record in datasets_meta.items():
        user_id = record.get("user_id")
        try:
            if delete_dataset(dataset_id, user_id):
                if user_id in results_by_user:
                    results_by_user[user_id]["datasets_deleted"] += 1
            else:
                errors.append(f"{user_id}: dataset {dataset_id} could not be removed.")
        except Exception as exc:
            errors.append(f"{user_id}: dataset {dataset_id} could not be removed: {exc}")

    for experiment_id, record in experiments_meta.items():
        user_id = record.get("user_id")
        try:
            cleanup = delete_experiment(experiment_id, user_id)
            if cleanup.get("success") and not cleanup.get("failed_files"):
                if user_id in results_by_user:
                    results_by_user[user_id]["experiments_deleted"] += 1
            else:
                errors.append(
                    f"{user_id}: experiment {experiment_id} could not be fully removed."
                )
                errors.extend(
                    f"{user_id}: {item.get('error', 'artifact cleanup failed')}"
                    for item in cleanup.get("failed_files", [])
                )
        except Exception as exc:
            errors.append(f"{user_id}: experiment {experiment_id} could not be removed: {exc}")

    cleared_files = 0
    for directory in storage_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.iterdir():
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                cleared_files += 1
        except OSError as exc:
            errors.append(f"Could not clear {directory}: {exc}")

    _write_json(DATASETS_META_FILE, {})
    _write_json(EXPERIMENTS_META_FILE, {})
    _write_json(USER_QUOTAS_FILE, {})

    result = {
        "processed_users": len(user_ids),
        "deleted_accounts": 0,
        "retained_auth_accounts": len(auth_users),
        "cleared_storage_entries": cleared_files,
        "cleared_storage_bytes": cleared_bytes,
        "supabase_storage_objects_deleted": supabase_storage["deleted_objects"],
        "supabase_storage_bytes_deleted": supabase_storage["deleted_bytes"],
        "supabase_rows_deleted": supabase_records["deleted_rows"],
        "success": not errors,
        "errors": errors,
        "results": results,
    }
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    notice_message = (
        "All your history has been deleted due to storage issues by URQEN. "
        "Your login account is safe and has not been deleted."
        if result["success"]
        else "URQEN ran a storage cleanup and removed some stored history. "
        "Your login account is safe and has not been deleted."
    )
    _write_json(
        SYSTEM_NOTICE_FILE,
        {
            "notice_id": created_at,
            "title": "Your URQEN history was cleared",
            "message": notice_message,
            "created_at": created_at,
            "target_user_ids": sorted(user_ids),
        },
    )
    result["notice_id"] = created_at
    return result


def purge_expired_artifacts(max_age_days: int = 14) -> dict:
    """
    14-day retention auto-purge:
    Removes heavy model bundle (.joblib) files for experiments older than max_age_days.
    Retains manifests, HTML reports, and metrics so users retain audit records without consuming storage.
    """
    experiments_meta = _read_json(EXPERIMENTS_META_FILE)
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=max_age_days)

    purged_count = 0
    freed_bytes = 0
    details = []

    for exp_id, exp in experiments_meta.items():
        created_str = exp.get("created_at")
        if not created_str:
            continue
        try:
            created_dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except Exception:
            continue

        if created_dt < cutoff:
            # Check for model bundle
            paths = exp.get("artifact_paths", {})
            bundle_path_str = paths.get("model_bundle")
            if bundle_path_str:
                p = pathlib.Path(bundle_path_str)
                if p.exists():
                    try:
                        size = p.stat().st_size
                        p.unlink()
                        freed_bytes += size
                        purged_count += 1
                        exp["artifact_paths"]["model_bundle"] = None
                        exp["artifacts_purged"] = True
                        exp["purged_at"] = now.isoformat()
                        details.append({
                            "experiment_id": exp_id,
                            "freed_bytes": size,
                            "user_id": exp.get("user_id"),
                        })
                    except Exception as e:
                        details.append({
                            "experiment_id": exp_id,
                            "error": str(e),
                        })

    if purged_count > 0:
        _write_json(EXPERIMENTS_META_FILE, experiments_meta)

    return {
        "purged_experiments_count": purged_count,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "cutoff_date": cutoff.isoformat(),
        "details": details,
    }
