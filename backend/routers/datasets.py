"""
backend/routers/datasets.py
---------------------------
Dataset endpoints: upload, validation, profiling, and data-quality audit.
"""

import uuid
import pathlib
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status

from backend.config import settings
from backend.auth import get_current_user, UserIdentity
from backend.storage import (
    save_dataset_record,
    get_dataset_record,
    list_datasets,
    delete_dataset,
    get_storage_stats,
)
from backend.schemas.dataset import (
    DatasetUploadResponse,
    DatasetProfileResponse,
    DatasetAuditResponse,
    ColumnSummary,
    TargetRecommendation,
    QualityFinding,
    PreprocessingRecommendation,
)

from previa.ingestion.loader import load_dataset, validate_dataset
from previa.auditing.profiler import profile_dataset
from previa.auditing.health import analyze_health
from previa.auditing.leakage import audit_leakage_and_reliability
from previa.task_detection.detector import detect_target_and_task, classify_task_for_target
from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations
from backend.supabase_client import sync_dataset, delete_dataset_record, update_storage_quota

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    user: UserIdentity = Depends(get_current_user),
):
    """
    Ingests and validates an uploaded tabular CSV or Excel file.
    Enforces file size quotas and ownership boundaries.
    """
    # 1. Quota verification
    stats = get_storage_stats(user.user_id)
    if stats["quota_exhausted"]:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Storage quota exceeded. Please delete older experiments or datasets before uploading.",
        )

    # 2. File extension validation
    filename = file.filename or "uploaded_data.csv"
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only CSV (.csv) and Excel (.xlsx, .xls) files are supported.",
        )

    # 3. Read content with size check
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    # 4. Save to disk under unique ID
    dataset_id = str(uuid.uuid4())
    save_filename = f"{dataset_id}_{filename}"
    file_path = settings.DATA_DIR / save_filename

    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist file: {str(e)}",
        )

    # 5. Load and validate using previa engine
    try:
        df = load_dataset(str(file_path))
        val_report = validate_dataset(df)
        if not val_report["is_valid"]:
            file_path.unlink(missing_ok=True)
            issues = "; ".join(val_report.get("critical_issues", ["Dataset validation failed."]))
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=issues)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse dataset: {str(e)}")

    record = save_dataset_record(
        dataset_id=dataset_id,
        user_id=user.user_id,
        name=filename,
        file_path=file_path,
        row_count=len(df),
        column_count=len(df.columns),
        file_type=ext[1:],
    )

    # Sync to Supabase (best-effort — never blocks the response)
    sync_dataset(user.user_id, record)
    stats_after = get_storage_stats(user.user_id)
    update_storage_quota(
        user.user_id,
        dataset_bytes=stats_after.get("dataset_bytes", 0),
        artifact_bytes=stats_after.get("artifact_bytes", 0),
        quota_limit_bytes=stats_after.get("quota_limit_bytes", 52428800),
    )

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        name=filename,
        row_count=len(df),
        column_count=len(df.columns),
        file_type=ext[1:],
        size_bytes=len(content),
        created_at=record["created_at"],
        message="Dataset uploaded and validated successfully.",
    )


@router.get("/storage/quota")
async def get_storage_quota(user: UserIdentity = Depends(get_current_user)):
    """Provides user storage consumption and quota boundaries."""
    return get_storage_stats(user.user_id)


@router.get("", response_model=List[dict])
@router.get("/", response_model=List[dict], include_in_schema=False)
async def get_user_datasets(user: UserIdentity = Depends(get_current_user)):
    """Lists all datasets owned by the authenticated user."""
    return list_datasets(user.user_id)


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, user: UserIdentity = Depends(get_current_user)):
    """Retrieves dataset metadata enforcing ownership."""
    record = get_dataset_record(dataset_id, user.user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")
    return record


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    limit: int = 50,
    user: UserIdentity = Depends(get_current_user)
):
    """Returns the top rows and column schema for interactive table exploration."""
    record = get_dataset_record(dataset_id, user.user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    df = load_dataset(record["file_path"])
    preview_df = df.head(min(limit, 200)).fillna("").to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "name": record["name"],
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "total_rows": len(df),
        "preview_rows": preview_df,
    }


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def get_dataset_profile(
    dataset_id: str,
    user: UserIdentity = Depends(get_current_user)
):
    """Generates comprehensive descriptive statistics and distribution metrics."""
    record = get_dataset_record(dataset_id, user.user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    df = load_dataset(record["file_path"])
    profile = profile_dataset(df)
    summary = profile["summary"]
    cols_meta = profile["columns"]

    cols_list = []
    for col_name, c_data in cols_meta.items():
        stats = c_data.get("numeric_stats") or {}
        cat_stats = c_data.get("categorical_stats") or {}
        top_cats = []
        if cat_stats and "top_categories" in cat_stats:
            top_cats = [{"category": k, "count": v} for k, v in cat_stats["top_categories"].items()]

        cols_list.append(
            ColumnSummary(
                name=col_name,
                dtype=c_data.get("dtype", "unknown"),
                missing_count=c_data.get("missing_count", 0),
                missing_pct=c_data.get("missing_pct", 0.0),
                unique_count=c_data.get("unique_count", 0),
                unique_pct=c_data.get("unique_pct", 0.0),
                is_numeric=c_data.get("is_numeric", False),
                is_categorical=c_data.get("is_categorical", False),
                is_constant=c_data.get("is_constant", False),
                is_identifier_candidate=c_data.get("is_identifier_candidate", False),
                mean=stats.get("mean"),
                median=stats.get("median"),
                std=stats.get("std"),
                min=stats.get("min"),
                max=stats.get("max"),
                skewness=stats.get("skewness"),
                top_categories=top_cats if top_cats else None,
            )
        )

    preview_rows = df.head(15).fillna("").to_dict(orient="records")

    return DatasetProfileResponse(
        dataset_id=dataset_id,
        name=record["name"],
        dimensions={"rows": summary.get("row_count", len(df)), "columns": summary.get("column_count", len(df.columns))},
        missing_summary={
            "total_missing": summary.get("total_missing_values", 0),
            "missing_pct": summary.get("missing_cell_percentage", 0.0),
        },
        duplicate_rows=summary.get("duplicate_row_count", 0),
        numeric_column_count=summary.get("numeric_column_count", 0),
        categorical_column_count=summary.get("categorical_column_count", 0),
        columns=cols_list,
        preview_rows=preview_rows,
    )


@router.get("/{dataset_id}/audit", response_model=DatasetAuditResponse)
async def get_dataset_audit(
    dataset_id: str,
    target_column: Optional[str] = None,
    user: UserIdentity = Depends(get_current_user),
):
    """
    Executes URQEN's evidence-backed data quality audit, target detection,
    leakage inspection, and preprocessing recommendations.
    """
    record = get_dataset_record(dataset_id, user.user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")

    df = load_dataset(record["file_path"])

    # 1. Target & Task Detection
    target_det = detect_target_and_task(df, target_column=target_column)
    suggested = target_det.get("target_column") or (df.columns[-1] if len(df.columns) > 0 else "")
    target = target_column or suggested
    if target not in df.columns:
        target = suggested

    if target and target in df.columns:
        detected_task, task_reasons = classify_task_for_target(df[target], len(df))
    else:
        detected_task = target_det.get("task_type", "binary_classification")
        task_reasons = target_det.get("reasons", [])

    target_rec = TargetRecommendation(
        suggested_target=target or (df.columns[-1] if len(df.columns) > 0 else "target"),
        detected_task=detected_task,
        confidence=target_det.get("confidence", "high" if target_column else "medium"),
        reasons=task_reasons or target_det.get("reasons", []),
        candidate_targets=list(df.columns),
    )

    # 2. Profile for Health Audit
    profile = profile_dataset(df)
    health_issues = analyze_health(profile, df)
    findings = []
    for issue in health_issues:
        findings.append(
            QualityFinding(
                issue=issue.get("issue_type", "issue").replace("_", " ").title(),
                affected_column=issue.get("column") or "Dataset-Level",
                evidence=issue.get("evidence", ""),
                severity=issue.get("severity", "info"),
                why_it_matters=issue.get("why_it_matters", "Affects model generalization."),
                recommended_action=issue.get("recommendation", "Review before modeling."),
                trade_offs=issue.get("trade_offs", "None documented."),
                user_review_required=bool(issue.get("requires_approval", False)),
            )
        )

    # 3. Leakage & Reliability Warnings
    leakage_warnings = audit_leakage_and_reliability(df, target_column=target, task_type=target_rec.detected_task)

    # 4. Feature Intelligence & Preprocessing Recommendations
    f_analysis = analyze_features(df, target_column=target)
    recs_list = generate_recommendations(f_analysis, target_column=target)
    recs = [
        PreprocessingRecommendation(
            feature=r["feature"],
            detected_type=r["detected_type"],
            issue=r["issue"],
            recommendation=r["recommendation"],
            reason=r["reason"],
            evidence=r["evidence"],
            confidence=r["confidence"],
            why_it_matters=r["why_it_matters"],
            trade_offs=r["trade_offs"],
        )
        for r in recs_list
    ]

    # 5. Compute Overall Health Score
    penalty = 0.0
    for f in findings:
        if f.severity == "high":
            penalty += 15.0
        elif f.severity == "medium":
            penalty += 7.0
        elif f.severity == "low":
            penalty += 3.0
    for lw in leakage_warnings:
        if lw.get("severity") == "high":
            penalty += 15.0
        elif lw.get("severity") == "medium":
            penalty += 7.0

    health_score = max(0.0, round(100.0 - penalty, 1))
    if health_score >= 80.0:
        health_status = "healthy"
    elif health_score >= 50.0:
        health_status = "caution"
    else:
        health_status = "critical"

    return DatasetAuditResponse(
        dataset_id=dataset_id,
        target_recommendation=target_rec,
        quality_findings=findings,
        leakage_warnings=leakage_warnings,
        recommendations=recs,
        health_score=health_score,
        health_status=health_status,
    )


@router.delete("/{dataset_id}")
async def remove_dataset(dataset_id: str, user: UserIdentity = Depends(get_current_user)):
    """Deletes dataset record and physical file with ownership verification."""
    success = delete_dataset(dataset_id, user.user_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied.")
    # Sync deletion to Supabase and update quota
    delete_dataset_record(dataset_id)
    stats_after = get_storage_stats(user.user_id)
    update_storage_quota(
        user.user_id,
        dataset_bytes=stats_after.get("dataset_bytes", 0),
        artifact_bytes=stats_after.get("artifact_bytes", 0),
        quota_limit_bytes=stats_after.get("quota_limit_bytes", 52428800),
    )
    return {"message": "Dataset deleted successfully.", "dataset_id": dataset_id}
