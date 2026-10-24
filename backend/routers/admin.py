"""
backend/routers/admin.py
------------------------
Administrative control panel endpoints:
  - Global infrastructure metrics and Supabase sync health
  - User quota management and per-user storage allocation
  - 14-day retention artifact purge executor
  - Global experiment tracking & dataset oversight
"""

from io import BytesIO
from html import escape
import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

from backend.config import settings
from backend.auth import get_current_user, require_admin, UserIdentity
from backend.storage import (
    get_admin_system_overview,
    list_all_experiments_admin,
    list_all_datasets_admin,
    purge_expired_artifacts,
    purge_all_user_data,
    set_user_quota_limit,
    get_user_quota_limit,
    delete_experiment,
    delete_dataset,
    delete_user_data,
    get_storage_stats,
    _read_json,
    EXPERIMENTS_META_FILE,
    DATASETS_META_FILE,
)
from backend.supabase_client import get_admin_client, update_storage_quota, delete_dataset_record, delete_experiment_record
from backend.presence import get_live_users

router = APIRouter(prefix="/api/admin", tags=["Admin"])


class QuotaUpdateRequest(BaseModel):
    quota_mb: int = Field(..., ge=1, le=10000, description="Quota in Megabytes")


class RetentionPurgeRequest(BaseModel):
    max_age_days: int = Field(default=14, ge=1, le=365)


class PurgeAllRequest(BaseModel):
    confirm_text: str


@router.get("/check")
async def check_admin_status(user: UserIdentity = Depends(get_current_user)):
    """Checks if the currently authenticated user has admin access."""
    is_adm = settings.is_admin(user.email)
    return {
        "is_admin": is_adm,
        "email": user.email,
        "user_id": user.user_id,
        "admins_configured": bool(settings.ADMIN_EMAILS.strip()),
    }


@router.get("/overview")
async def get_overview(admin: UserIdentity = Depends(require_admin)):
    """Provides high-level system analytics, global storage breakdown, and user stats."""
    overview = get_admin_system_overview()

    # Check Supabase status
    supabase_client = get_admin_client()
    supabase_connected = False
    supabase_users_count = None

    if supabase_client:
        try:
            # Quick ping to Supabase
            res = supabase_client.table("user_storage_quotas").select("user_id", count="exact").limit(1).execute()
            supabase_connected = True
            supabase_users_count = res.count
        except Exception:
            supabase_connected = False

    overview["supabase"] = {
        "configured": bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY),
        "connected": supabase_connected,
        "tracked_users": supabase_users_count,
        "url": settings.SUPABASE_URL,
    }

    return overview


@router.get("/users")
async def get_users_list(admin: UserIdentity = Depends(require_admin)):
    """Lists all users registered in the system with their storage footprint."""
    overview = get_admin_system_overview()
    return overview.get("users", [])


@router.get("/live-users")
async def get_live_users_list(admin: UserIdentity = Depends(require_admin)):
    """Lists authenticated users whose foreground heartbeat is still current."""
    return get_live_users()


@router.get("/experiments")
async def get_all_experiments(admin: UserIdentity = Depends(require_admin)):
    """Returns global history of experiments across all users."""
    return list_all_experiments_admin()


@router.get("/datasets")
async def get_all_datasets(admin: UserIdentity = Depends(require_admin)):
    """Returns all datasets hosted across all users."""
    return list_all_datasets_admin()


@router.post("/purge-retention")
async def trigger_retention_purge(
    req: RetentionPurgeRequest,
    admin: UserIdentity = Depends(require_admin),
):
    """
    Executes the 14-day automated retention policy:
    Deletes heavy model bundle (.joblib) binaries for experiments older than max_age_days,
    preserving evaluation metrics, HTML reports, and audit manifests.
    """
    res = purge_expired_artifacts(max_age_days=req.max_age_days)
    return {
        "message": f"Purged model bundles older than {req.max_age_days} days.",
        "results": res,
    }


@router.post("/purge-all")
async def purge_all_storage(
    req: PurgeAllRequest,
    admin: UserIdentity = Depends(require_admin),
):
    """Deletes all URQEN user data while retaining every Auth account."""
    if not settings.ADMIN_EMAILS.strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bulk purge requires an explicit ADMIN_EMAILS founder allowlist.",
        )
    if req.confirm_text != "PURGE ALL USER DATA":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Confirmation text must exactly match "PURGE ALL USER DATA".',
        )
    try:
        return purge_all_user_data(initiated_by_user_id=admin.user_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.put("/users/{user_id}/quota")
async def update_user_quota(
    user_id: str,
    req: QuotaUpdateRequest,
    admin: UserIdentity = Depends(require_admin),
):
    """Sets a custom storage quota (in MB) for a specific user."""
    quota_bytes = req.quota_mb * 1024 * 1024
    set_user_quota_limit(user_id, quota_bytes)

    # Sync to Supabase table if available
    current_stats = get_storage_stats(user_id)
    update_storage_quota(
        user_id,
        dataset_bytes=current_stats.get("dataset_bytes", 0),
        artifact_bytes=current_stats.get("artifact_bytes", 0),
        quota_limit_bytes=quota_bytes,
    )

    return {
        "user_id": user_id,
        "quota_mb": req.quota_mb,
        "quota_bytes": quota_bytes,
        "message": f"Storage quota updated to {req.quota_mb} MB.",
    }


@router.get("/history.pdf")
async def export_admin_history_pdf(admin: UserIdentity = Depends(require_admin)):
    """Downloads a founder-only snapshot of account, dataset, and experiment history."""
    if not settings.ADMIN_EMAILS.strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="History export requires an explicit ADMIN_EMAILS founder allowlist.",
        )
    try:
        overview = get_admin_system_overview()
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
            from backend.supabase_client import get_admin_client, get_all_auth_users

            if get_admin_client() is None:
                raise RuntimeError("Supabase admin client is unavailable.")
            auth_users = get_all_auth_users(strict=True)
            overview_users = {user.get("user_id"): user for user in overview.get("users", [])}
            for user_id, email in auth_users.items():
                if user_id not in overview_users:
                    overview_users[user_id] = {
                        "user_id": user_id,
                        "email": email,
                        "dataset_count": 0,
                        "experiment_count": 0,
                        "total_bytes": 0,
                        "quota_bytes": 0,
                    }
                elif not overview_users[user_id].get("email"):
                    overview_users[user_id]["email"] = email
            overview["users"] = list(overview_users.values())
        datasets = list_all_datasets_admin()
        experiments = list_all_experiments_admin()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        alignment=TA_LEFT,
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0f766e"),
    ))

    def cell(value):
        return Paragraph(
            escape(str(value if value is not None else "—")),
            styles["ReportCell"],
        )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.4 * inch,
        leftMargin=0.4 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title="URQEN Founder History Export",
        author="URQEN",
    )
    story = [
        Paragraph("URQEN Founder History Export", styles["ReportTitle"]),
        Paragraph(
            f"Generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
            "Confidential account and workspace metadata",
            styles["BodyText"],
        ),
        Spacer(1, 14),
        Paragraph("User Accounts & Storage", styles["Heading2"]),
    ]

    user_rows = [[cell(value) for value in ("Email", "User ID", "Datasets", "Experiments", "Storage", "Quota")]]
    for user in overview.get("users", []):
        user_rows.append([
            cell(user.get("email") or "Email unavailable"),
            cell(user.get("user_id")),
            cell(user.get("dataset_count", 0)),
            cell(user.get("experiment_count", 0)),
            cell(f'{user.get("total_bytes", 0)} bytes'),
            cell(f'{user.get("quota_bytes", 0)} bytes'),
        ])
    user_table = LongTable(
        user_rows,
        colWidths=[2.15 * inch, 2.0 * inch, 0.75 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch],
        repeatRows=1,
    )
    story.extend([user_table, Spacer(1, 14), Paragraph("Dataset History", styles["Heading2"])])
    dataset_rows = [[cell(value) for value in ("Dataset", "Owner ID", "Rows", "Columns", "Format", "Size", "Created")]]
    for dataset in datasets:
        dataset_rows.append([
            cell(dataset.get("name") or dataset.get("filename")),
            cell(dataset.get("user_id")),
            cell(dataset.get("row_count", 0)),
            cell(dataset.get("column_count", 0)),
            cell(dataset.get("file_type")),
            cell(f'{dataset.get("size_bytes", 0)} bytes'),
            cell(dataset.get("created_at")),
        ])
    dataset_table = LongTable(
        dataset_rows,
        colWidths=[2.2 * inch, 1.9 * inch, 0.65 * inch, 0.65 * inch, 0.8 * inch, 1.0 * inch, 1.5 * inch],
        repeatRows=1,
    )
    story.extend([dataset_table, Spacer(1, 14), Paragraph("Experiment History", styles["Heading2"])])
    experiment_rows = [[cell(value) for value in (
        "Experiment ID",
        "Owner ID",
        "Dataset",
        "Task",
        "Status",
        "Best Model",
        "Best Score",
        "Created",
    )]]
    for experiment in experiments:
        models = (experiment.get("results") or {}).get("models") or []
        best_score = models[0].get("holdout_score") if models else None
        experiment_rows.append([
            cell(experiment.get("experiment_id")),
            cell(experiment.get("user_id")),
            cell(experiment.get("dataset_name")),
            cell(experiment.get("task_type")),
            cell(experiment.get("status")),
            cell(experiment.get("best_model")),
            cell(best_score),
            cell(experiment.get("created_at")),
        ])
    experiment_table = LongTable(
        experiment_rows,
        colWidths=[1.45 * inch, 1.5 * inch, 1.5 * inch, 0.8 * inch, 0.7 * inch, 1.1 * inch, 0.9 * inch, 1.4 * inch],
        repeatRows=1,
    )
    story.append(experiment_table)
    for table in (user_table, dataset_table, experiment_table):
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
    document.build(story)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="urqen-founder-history.pdf"'},
    )


@router.delete("/users/{user_id}")
async def admin_delete_user(
    user_id: str,
    admin: UserIdentity = Depends(require_admin),
):
    """Permanently deletes a user, their datasets, experiments, artifacts, and auth account."""
    if user_id == admin.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete the account used for the current founder session.",
        )
    try:
        res = delete_user_data(user_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    if res["supabase_auth_deleted"] is True:
        message = "The Supabase login account was deleted."
    elif res["supabase_auth_deleted"] is False:
        message = "URQEN data was removed, but Supabase Auth did not confirm login account deletion."
    else:
        message = "URQEN data was removed. No Supabase Auth account was configured for deletion."
    if res["errors"]:
        message += " One or more URQEN cloud data cleanups failed."
    return {"message": message, "details": res}


@router.delete("/experiments/{experiment_id}")
async def admin_delete_experiment(
    experiment_id: str,
    admin: UserIdentity = Depends(require_admin),
):
    """Force deletes an experiment across any user."""
    meta = _read_json(EXPERIMENTS_META_FILE)
    rec = meta.get(experiment_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    res = delete_experiment(experiment_id, rec.get("user_id"))
    delete_experiment_record(experiment_id)
    return {"message": "Experiment deleted by admin.", "details": res}


@router.delete("/datasets/{dataset_id}")
async def admin_delete_dataset(
    dataset_id: str,
    admin: UserIdentity = Depends(require_admin),
):
    """Force deletes a dataset across any user."""
    meta = _read_json(DATASETS_META_FILE)
    rec = meta.get(dataset_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    u_id = rec.get("user_id")
    res = delete_dataset(dataset_id, u_id)
    if not res:
        raise HTTPException(status_code=404, detail="Dataset could not be deleted.")
    delete_dataset_record(dataset_id)
    if u_id:
        stats_after = get_storage_stats(u_id)
        update_storage_quota(
            u_id,
            dataset_bytes=stats_after.get("dataset_bytes", 0),
            artifact_bytes=stats_after.get("artifact_bytes", 0),
            quota_limit_bytes=stats_after.get("quota_limit_bytes", 52428800),
        )
    return {"message": "Dataset deleted by admin.", "success": res}
