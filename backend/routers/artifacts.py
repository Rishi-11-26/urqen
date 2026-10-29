"""
backend/routers/artifacts.py
----------------------------
Artifact download and live model inference endpoints.
"""

import hashlib
import io
import json
import pathlib
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import FileResponse, Response

from backend.config import settings
from backend.auth import get_current_user, UserIdentity
from backend.storage import get_experiment_record
from backend.schemas.prediction import PredictionResponse, PredictionRecord
from previa.export.model_bundle import load_model_bundle

router = APIRouter(prefix="/api/artifacts", tags=["Artifacts & Predictions"])
MAX_EVALUATION_EXPORT_BYTES = 25 * 1024 * 1024


def _evaluation_path(experiment_id: str, user: UserIdentity) -> pathlib.Path:
    try:
        uuid.UUID(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid experiment identifier.") from exc
    experiment = get_experiment_record(experiment_id, user.user_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or access denied.",
        )
    path_string = experiment.get("artifact_paths", {}).get("evaluation")
    if not path_string:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored holdout evaluation artifacts are unavailable for this experiment.",
        )
    path = pathlib.Path(path_string)
    expected_directory = (settings.ARTIFACTS_DIR / experiment_id).resolve()
    if path.resolve().parent != expected_directory or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation artifact is missing or outside the experiment artifact directory.",
        )
    return path


def _evaluation_record(
    experiment_id: str,
    user: UserIdentity,
) -> tuple[dict, pathlib.Path]:
    path = _evaluation_path(experiment_id, user)
    experiment = get_experiment_record(experiment_id, user.user_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found or access denied.")
    return experiment, path


@router.get("/{experiment_id}/report")
async def download_report(
    experiment_id: str,
    format: str = "html",
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads the generated decision and benchmarking report (HTML or Markdown)."""
    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    paths = exp.get("artifact_paths", {})
    key = "html_report" if format.lower() == "html" else "markdown_report"
    path_str = paths.get(key)

    if not path_str or not pathlib.Path(path_str).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report artifact not generated or missing.")

    media_type = "text/html" if format.lower() == "html" else "text/markdown"
    filename = f"previa_report_{experiment_id[:8]}.{'html' if format.lower() == 'html' else 'md'}"
    return FileResponse(path=path_str, media_type=media_type, filename=filename)


@router.get("/{experiment_id}/manifest")
async def download_manifest(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads the self-contained experiment JSON manifest."""
    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    path_str = exp.get("artifact_paths", {}).get("manifest")
    if not path_str or not pathlib.Path(path_str).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manifest artifact not found.")

    return FileResponse(path=path_str, media_type="application/json", filename=f"previa_manifest_{experiment_id[:8]}.json")


@router.get("/{experiment_id}/evaluation")
async def download_evaluation_export(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads stored holdout predictions and evaluation verification artifacts."""
    evaluation_path = _evaluation_path(experiment_id, user)
    if evaluation_path.stat().st_size > MAX_EVALUATION_EXPORT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Evaluation export exceeds the 25 MiB download limit. "
                "Use the paginated evaluation samples endpoint to retrieve predictions."
            ),
        )
    return FileResponse(
        path=evaluation_path,
        media_type="application/json",
        filename=f"previa_evaluation_{experiment_id[:8]}.json",
    )


@router.get("/{experiment_id}/evaluation/samples")
async def get_evaluation_samples(
    experiment_id: str,
    model_name: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=500),
    user: UserIdentity = Depends(get_current_user),
):
    """Returns a bounded page of one model's stored holdout predictions."""
    experiment, _ = _evaluation_record(experiment_id, user)
    model = next(
        (
            item for item in (experiment.get("results") or {}).get("models", [])
            if item.get("model") == model_name
        ),
        None,
    )
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model evaluation artifact not found.",
        )
    if model.get("status") != "success":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holdout predictions are unavailable for this model.",
        )
    sample_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
    sample_path_string = (experiment.get("artifact_paths") or {}).get(
        f"evaluation_samples_{sample_key}"
    )
    if not sample_path_string:
        raise HTTPException(status_code=404, detail="Paginated holdout samples are unavailable.")
    sample_path = pathlib.Path(sample_path_string)
    expected_directory = (settings.ARTIFACTS_DIR / experiment_id / "evaluation_samples").resolve()
    if sample_path.resolve().parent != expected_directory or not sample_path.is_file():
        raise HTTPException(status_code=404, detail="Paginated holdout sample artifact is missing.")
    total = int((experiment.get("results") or {}).get("holdout_sample_count") or 0)
    end = min(offset + limit, total)
    samples = []
    with sample_path.open("r", encoding="utf-8") as sample_file:
        for index, line in enumerate(sample_file):
            if index < offset:
                continue
            if index >= end:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Stored holdout sample artifact is malformed.",
                ) from exc
    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": end < total,
        "samples": samples,
    }


@router.get("/{experiment_id}/model-bundle")
async def download_model_bundle(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads the deployable .joblib model bundle (Pro feature)."""
    from backend.storage import is_user_paid
    if not is_user_paid(user.user_id, settings.is_admin(user.email)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Model Bundle (.joblib) download is a Pro feature. Upgrade to Student Plan (₹249/mo) "
                "or get the ₹99 Top-Up Pack to download deployable model pipelines. Contact urqen0@gmail.com for instant activation."
            ),
        )

    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    path_str = exp.get("artifact_paths", {}).get("model_bundle")
    if not path_str or not pathlib.Path(path_str).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model bundle not found.")

    return FileResponse(path=path_str, media_type="application/octet-stream", filename=f"previa_model_{experiment_id[:8]}.joblib")


@router.get("/{experiment_id}/code-export")
async def export_python_code(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """
    Exports a standalone scikit-learn Python script reproducing the winning model pipeline.
    Pro users get full code; free users receive an illustrative preview with unlock details.
    """
    from previa.export.code_generator import generate_pipeline_python_code
    from backend.storage import is_user_paid

    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    full_code = generate_pipeline_python_code(exp)
    paid = is_user_paid(user.user_id, settings.is_admin(user.email))

    if not paid:
        preview_lines = "\n".join(full_code.splitlines()[:16])
        return {
            "experiment_id": experiment_id,
            "is_locked": True,
            "best_model": exp.get("best_model", "Model"),
            "preview_code": preview_lines,
            "message": (
                "Python code export is a Pro feature. Unlock the full scikit-learn reproduction "
                "script with the ₹99 Top-Up Pack or Student Plan (₹249/mo). Contact urqen0@gmail.com for instant activation."
            ),
        }

    return {
        "experiment_id": experiment_id,
        "is_locked": False,
        "best_model": exp.get("best_model", "Model"),
        "filename": f"pipeline_{exp.get('best_model', 'model').lower()}_{experiment_id[:8]}.py",
        "code": full_code,
    }


@router.get("/{experiment_id}/code-export/download")
async def download_python_code_file(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads the standalone Python reproduction script as a .py file (Pro feature)."""
    from previa.export.code_generator import generate_pipeline_python_code
    from backend.storage import is_user_paid
    from fastapi.responses import Response

    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    if not is_user_paid(user.user_id, settings.is_admin(user.email)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Python code export download is a Pro feature. Upgrade with ₹99 Top-Up or Student Plan (₹249/mo). Contact urqen0@gmail.com.",
        )

    full_code = generate_pipeline_python_code(exp)
    filename = f"urqen_pipeline_{exp.get('best_model', 'model').lower()}_{experiment_id[:8]}.py"
    return Response(
        content=full_code,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{experiment_id}/preprocessed-dataset")
async def download_preprocessed_dataset(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads the preprocessed/transformed dataset CSV."""
    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    path_str = exp.get("artifact_paths", {}).get("preprocessed_data")
    if not path_str or not pathlib.Path(path_str).exists():
        # Fallback: check if preprocessed_dataset.csv exists in artifact directory
        exp_dir = settings.ARTIFACTS_DIR / experiment_id
        fallback_path = exp_dir / "preprocessed_dataset.csv"
        if fallback_path.exists():
            path_str = str(fallback_path)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preprocessed dataset artifact not found for this experiment.")

    ds_name = (exp.get("dataset_name") or "dataset").replace(".csv", "").replace(".tsv", "")
    filename = f"{ds_name}_preprocessed_{experiment_id[:8]}.csv"
    return FileResponse(path=path_str, media_type="text/csv", filename=filename)


@router.post("/{experiment_id}/predict", response_model=PredictionResponse)
async def predict_with_model(
    experiment_id: str,
    file: UploadFile = File(...),
    user: UserIdentity = Depends(get_current_user),
):
    """
    Live inference endpoint: Loads the trained model bundle and generates
    leakage-safe predictions on an uploaded unlabelled CSV dataset (Pro feature).
    """
    from backend.storage import is_user_paid
    if not is_user_paid(user.user_id, settings.is_admin(user.email)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Model inference on new test datasets is a Pro feature. Upgrade to Student Plan (₹249/mo) "
                "or get the ₹99 Top-Up Pack to run predictions on unlabelled datasets. Contact urqen0@gmail.com for instant activation."
            ),
        )

    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    bundle_path = exp.get("artifact_paths", {}).get("model_bundle")
    if not bundle_path or not pathlib.Path(bundle_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trained model bundle not found.")

    bundle = load_model_bundle(bundle_path)

    # Read uploaded inference data
    try:
        content = await file.read()
        df_new = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse CSV: {str(e)}")

    # Remove target column if user passed it
    target = bundle.target_name
    if target in df_new.columns:
        df_new = df_new.drop(columns=[target])

    try:
        preds = bundle.predict(df_new)
        probs = None
        if hasattr(bundle, "predict_proba") and "classification" in bundle.task_type:
            try:
                probs_raw = bundle.predict_proba(df_new)
                classes = [str(c) for c in bundle.classes] if bundle.classes is not None else [f"class_{i}" for i in range(probs_raw.shape[1])]
                probs = [dict(zip(classes, [round(float(p), 4) for p in row])) for row in probs_raw]
            except Exception:
                probs = None

        pred_records = []
        for idx, val in enumerate(preds):
            prob_dict = probs[idx] if probs else None
            pred_records.append(PredictionRecord(prediction=val, probability=prob_dict))

        return PredictionResponse(
            experiment_id=experiment_id,
            model_name=exp.get("best_model", "Model"),
            task_type=bundle.task_type,
            predictions_count=len(pred_records),
            predictions=pred_records,
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Prediction error: {str(e)}")
