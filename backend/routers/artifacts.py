"""
backend/routers/artifacts.py
----------------------------
Artifact download and live model inference endpoints.
"""

import io
import pathlib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, Response

from backend.config import settings
from backend.auth import get_current_user, UserIdentity
from backend.storage import get_experiment_record
from backend.schemas.prediction import PredictionResponse, PredictionRecord
from previa.export.model_bundle import load_model_bundle

router = APIRouter(prefix="/api/artifacts", tags=["Artifacts & Predictions"])


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


@router.get("/{experiment_id}/model-bundle")
async def download_model_bundle(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Downloads the deployable .joblib model bundle."""
    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    path_str = exp.get("artifact_paths", {}).get("model_bundle")
    if not path_str or not pathlib.Path(path_str).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model bundle not found.")

    return FileResponse(path=path_str, media_type="application/octet-stream", filename=f"previa_model_{experiment_id[:8]}.joblib")


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
    leakage-safe predictions on an uploaded unlabelled CSV dataset.
    """
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
