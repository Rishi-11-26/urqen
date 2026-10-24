"""
backend/routers/experiments.py
------------------------------
Experiment execution, benchmarking, historical tracking, and quota monitoring.
"""

import uuid
import pathlib
import traceback
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status

from backend.config import settings
from backend.auth import get_current_user, UserIdentity
from backend.storage import (
    save_experiment_record,
    update_experiment_results,
    update_experiment_error,
    get_experiment_record,
    list_experiments,
    delete_experiment,
    get_dataset_record,
    get_storage_stats,
)
from backend.schemas.experiment import (
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentSummary,
    ModelBenchmarkResult,
    StorageQuotaResponse,
)

from previa.ingestion.loader import load_dataset
from previa.task_detection.detector import detect_target_and_task
from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations
from previa.preprocessing.pipeline_builder import build_pipeline
from previa.models.registry import get_available_models, create_model_instance, is_xgboost_available
from previa.benchmarking.benchmarker import run_benchmarks
from previa.export.model_bundle import save_model_bundle, PreviaModelBundle
from previa.export.report_generator import generate_html_report, generate_markdown_report
from previa.experiments.tracker import create_experiment_manifest, serialize_manifest_to_json
from previa.explainability.explainer import compute_feature_importance
from backend.supabase_client import sync_experiment, delete_experiment_record

router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


def _run_experiment_background(
    experiment_id: str,
    user_id: str,
    req_data: dict,
    dataset_path: str,
    dataset_name: str,
) -> None:
    """Background worker that trains models, computes metrics, and persists artifacts."""
    try:
        df = load_dataset(dataset_path)
        target = req_data["target_column"]

        # 1. Determine task type
        task = req_data.get("task_type")
        if not task:
            target_det = detect_target_and_task(df)
            task = target_det.get("detected_task", "binary_classification")

        # 2. Extract X and clean target
        if not target or (target not in df.columns and (target == "target" or target.lower() == "target")):
            det = detect_target_and_task(df)
            suggested = det.get("target_column")
            if suggested and suggested in df.columns:
                target = suggested
            elif len(df.columns) > 0:
                target = df.columns[-1]

        if not target or target not in df.columns:
            available = ", ".join(df.columns.tolist()[:15])
            raise ValueError(
                f"Target column '{target}' was not found in the dataset. "
                f"Available columns: {available}. "
                f"Please select a valid target column to predict."
            )

        user_features = req_data.get("feature_columns")
        if user_features and isinstance(user_features, list) and len(user_features) > 0:
            valid_user_feats = [c for c in user_features if c in df.columns and c != target]
            if not valid_user_feats:
                raise ValueError("None of the selected input feature columns exist in the dataset.")
            cols_to_use = valid_user_feats + [target]
            df_filtered = df[cols_to_use]
            X_raw = df_filtered.drop(columns=[target])
        else:
            df_filtered = df
            X_raw = df.drop(columns=[target])

        y_raw = df_filtered[target]

        valid_idx = y_raw.dropna().index
        X_clean = X_raw.loc[valid_idx]
        y_clean = y_raw.loc[valid_idx]

        # Target encoding
        if "classification" in task:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_clean.astype(str))
            class_labels = [str(c) for c in le.classes_]
        else:
            # For regression, ensure numeric float values with fallback for messy formats
            try:
                y_encoded = pd.to_numeric(y_clean, errors="coerce").fillna(0.0).values.astype(float)
            except Exception:
                y_encoded = y_clean.values.astype(float)
            class_labels = None

        # 3. Preprocessing Recommendations & Pipeline Construction
        f_analysis = analyze_features(df_filtered, target_column=target)
        recs = generate_recommendations(f_analysis, target_column=target)

        scaler_type = req_data.get("scaler_type", "standard")
        overrides = req_data.get("user_overrides") or {}

        pipeline, num_feats, cat_feats, dropped_feats = build_pipeline(
            recs,
            user_overrides=overrides,
            scaler_type=scaler_type,
        )

        # Train / Test split on RAW data to eliminate preprocessing data leakage
        _req_test_size = req_data.get("test_size", 0.20)
        split_ratio = _req_test_size if len(y_encoded) >= 20 else 0.3
        from sklearn.model_selection import train_test_split
        stratify = (
            y_encoded
            if ("classification" in task and len(np.unique(y_encoded)) > 1 and np.min(np.bincount(y_encoded)) >= 2)
            else None
        )

        if len(y_encoded) >= 10:
            try:
                X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                    X_clean, y_encoded, test_size=split_ratio, random_state=42, stratify=stratify
                )
            except Exception:
                X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                    X_clean, y_encoded, test_size=split_ratio, random_state=42
                )
        else:
            X_tr_raw, X_te_raw, y_tr, y_te = X_clean, X_clean, y_encoded, y_encoded

        # Leakage-safe fit strictly on training partition
        pipeline.fit(X_tr_raw, y_tr)
        X_tr = pipeline.transform(X_tr_raw)
        X_te = pipeline.transform(X_te_raw)
        X_transformed = pipeline.transform(X_clean)

        # 4. Determine models and hyper-parameters
        cv_folds = req_data.get("cv_folds", 10)
        test_size = req_data.get("test_size", 0.20)
        model_hyperparams = req_data.get("hyperparameters") or {}

        chosen_models = req_data.get("models")
        if not chosen_models:
            avail = get_available_models(task, include_optional=False)
            chosen_models = [m["name"] for m in avail]
            if is_xgboost_available() and "XGBoost" not in chosen_models:
                chosen_models.append("XGBoost")

        # 5. Execute Benchmarks
        bench_res = run_benchmarks(
            X_transformed,
            y_encoded,
            task_type=task,
            models_to_evaluate=chosen_models,
            cv_folds=cv_folds,
            test_size=test_size,
            model_hyperparams=model_hyperparams,
        )

        # 6. Fit Winning Model on Holdout Split for Artifact Serialization
        best_name = bench_res["best_model"]
        best_kwargs = model_hyperparams.get(best_name, {})
        fitted_model = create_model_instance(best_name, task, random_state=42, **best_kwargs)
        fitted_model.fit(X_tr, y_tr)

        # 7. Generate Feature Importance
        try:
            feat_names = [n.split("__")[-1] for n in pipeline.get_feature_names_out()]
        except Exception:
            feat_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        feat_importance = compute_feature_importance(
            model=fitted_model,
            X=X_te,
            y=y_te,
            feature_names=feat_names,
            task_type=task,
        )

        # 8. Save Artifacts (Bundle, Manifest, Reports)
        exp_dir = settings.ARTIFACTS_DIR / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = exp_dir / "model_bundle.joblib"
        bundle_metadata = {
            "experiment_id": experiment_id,
            "target_column": target,
            "task_type": task,
            "expected_features": [c for c in df.columns if c != target],
            "feature_names_out": feat_names,
            "classes": class_labels if "classification" in task else None,
            "metrics": bench_res,
        }
        save_model_bundle(pipeline, fitted_model, bundle_metadata, str(bundle_path))

        # Manifest
        preprocessing_config = {
            "scaler": scaler_type,
            "cv_folds": cv_folds,
            "test_size": test_size,
        }
        manifest = create_experiment_manifest(
            df=df,
            target_column=target,
            task_type=task,
            preprocessing_config=preprocessing_config,
            benchmarking_results=bench_res,
            explainability_summary=feat_importance,
            dataset_name=dataset_name,
            random_seed=42,
        )
        manifest["experiment_id"] = experiment_id
        manifest_path = exp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            mf.write(serialize_manifest_to_json(manifest))

        # Reports
        html_report = generate_html_report(manifest)
        html_path = exp_dir / "report.html"
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write(html_report)

        md_report = generate_markdown_report(manifest)
        md_path = exp_dir / "report.md"
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(md_report)

        # Preprocessed Clean Dataset Export (Preserves exact tabular schema and human-readable format)
        prep_path = exp_dir / "preprocessed_dataset.csv"
        try:
            from previa.export.exporter import create_clean_preprocessed_dataframe
            clean_prep_df = create_clean_preprocessed_dataframe(
                df=df_filtered,
                target_column=target,
                numeric_features=num_feats,
                categorical_features=cat_feats,
                dropped_features=dropped_feats,
                train_indices=X_tr_raw.index,
            )
            clean_prep_df.to_csv(prep_path, index=False)
        except Exception:
            try:
                df_filtered.to_csv(prep_path, index=False)
            except Exception:
                pass

        artifact_paths = {
            "model_bundle": str(bundle_path),
            "manifest": str(manifest_path),
            "html_report": str(html_path),
            "markdown_report": str(md_path),
            "preprocessed_data": str(prep_path),
        }

        # 9. Update Database Record
        bench_res["feature_importance"] = feat_importance
        bench_res["expected_features"] = [c for c in df.columns if c != target]
        bench_res["feature_names"] = feat_names

        # Compute confusion matrix for the winner (classification only)
        if "classification" in task:
            try:
                from sklearn.metrics import confusion_matrix as sk_cm
                y_winner_pred = fitted_model.predict(X_te)
                cm = sk_cm(y_te, y_winner_pred)
                bench_res["confusion_matrix"] = cm.tolist()
            except Exception:
                bench_res["confusion_matrix"] = None
        else:
            bench_res["confusion_matrix"] = None

        update_experiment_results(
            experiment_id=experiment_id,
            user_id=user_id,
            results=bench_res,
            best_model=best_name,
            winner_explanation=bench_res["winner_explanation"],
            artifact_paths=artifact_paths,
        )

        # Sync completed experiment to Supabase (best-effort)
        try:
            from backend.storage import get_experiment_record
            final_record = get_experiment_record(experiment_id, user_id)
            if final_record:
                sync_experiment(user_id, final_record)
        except Exception:
            pass

    except Exception as e:
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        update_experiment_error(experiment_id, user_id, err_msg)


@router.post("/benchmark", response_model=ExperimentResponse)
async def create_and_run_experiment(
    req: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
    user: UserIdentity = Depends(get_current_user),
):
    """
    Submits a machine learning experimentation job in Guided or Expert Mode.
    Spawns background training and returns the tracked experiment object.
    """
    record = get_dataset_record(req.dataset_id, user.user_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Referenced dataset not found or access denied.",
        )

    experiment_id = str(uuid.uuid4())
    task_type = req.task_type or "binary_classification"

    # Validate or auto-detect target column from dataset
    target_col = req.target_column
    try:
        sample_df = load_dataset(record["file_path"], sample_rows=50)
        if not target_col or (target_col not in sample_df.columns and (target_col == "target" or target_col.lower() == "target")):
            det = detect_target_and_task(sample_df)
            suggested = det.get("target_column")
            if suggested and suggested in sample_df.columns:
                target_col = suggested
            elif len(sample_df.columns) > 0:
                target_col = sample_df.columns[-1]
            if not req.task_type and det.get("task_type"):
                task_type = det["task_type"]
    except Exception:
        pass

    req_payload = req.model_dump()
    req_payload["target_column"] = target_col
    req_payload["task_type"] = task_type

    exp_record = save_experiment_record(
        experiment_id=experiment_id,
        user_id=user.user_id,
        dataset_id=req.dataset_id,
        dataset_name=record["name"],
        task_type=task_type,
        target_column=target_col,
        mode=req.mode,
        configuration=req_payload,
        status="running",
    )

    background_tasks.add_task(
        _run_experiment_background,
        experiment_id=experiment_id,
        user_id=user.user_id,
        req_data=req_payload,
        dataset_path=record["file_path"],
        dataset_name=record["name"],
    )

    return ExperimentResponse(
        experiment_id=experiment_id,
        user_id=user.user_id,
        dataset_id=req.dataset_id,
        dataset_name=record["name"],
        task_type=task_type,
        target_column=target_col,
        mode=req.mode,
        status="running",
        created_at=exp_record["created_at"],
        configuration=req_payload,
    )


@router.get("", response_model=List[ExperimentSummary])
@router.get("/", response_model=List[ExperimentSummary], include_in_schema=False)
async def get_user_experiments(user: UserIdentity = Depends(get_current_user)):
    """Lists all historical experiments for the authenticated user."""
    exps = list_experiments(user.user_id)
    return [
        ExperimentSummary(
            experiment_id=e["experiment_id"],
            dataset_name=e.get("dataset_name", "Dataset"),
            task_type=e.get("task_type", "classification"),
            target_column=e.get("target_column", "target"),
            mode=e.get("mode", "guided"),
            status=e.get("status", "completed"),
            best_model=e.get("best_model"),
            created_at=e.get("created_at", ""),
        )
        for e in exps
    ]


@router.get("/quota/usage", response_model=StorageQuotaResponse)
async def get_quota_usage(user: UserIdentity = Depends(get_current_user)):
    """Provides user storage consumption and quota boundaries."""
    return get_storage_stats(user.user_id)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Retrieves full experiment details, benchmarking results, and winner rationale."""
    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found or access denied.")

    models_res = None
    if exp.get("results") and "models" in exp["results"]:
        models_res = [
            ModelBenchmarkResult(
                model=m.get("model", "Unknown"),
                family=m.get("family", "general"),
                status=m.get("status", "success"),
                training_time_seconds=m.get("training_time_seconds", 0.0),
                cv_mean=m.get("cv_mean"),
                cv_std=m.get("cv_std"),
                accuracy=m.get("accuracy"),
                f1=m.get("f1"),
                precision=m.get("precision"),
                recall=m.get("recall"),
                sensitivity=m.get("sensitivity"),
                specificity=m.get("specificity"),
                roc_auc=m.get("roc_auc"),
                r2=m.get("r2"),
                mae=m.get("mae"),
                rmse=m.get("rmse"),
                error_message=m.get("error_message"),
            )
            for m in exp["results"]["models"]
        ]

    # Build results dict with feature_importance, confusion_matrix and primary_metric
    # so the frontend ResultsView can read experiment.results.models, primary_metric, etc.
    results_dict = None
    if exp.get("results"):
        raw_results = exp["results"]
        results_dict = {
            "models": [
                {
                    "model": m.get("model", "Unknown"),
                    "family": m.get("family", "general"),
                    "status": m.get("status", "success"),
                    "training_time_seconds": m.get("training_time_seconds", 0.0),
                    "cv_mean": m.get("cv_mean"),
                    "cv_std": m.get("cv_std"),
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "sensitivity": m.get("sensitivity"),
                    "specificity": m.get("specificity"),
                    "roc_auc": m.get("roc_auc"),
                    "r2": m.get("r2"),
                    "mae": m.get("mae"),
                    "rmse": m.get("rmse"),
                    "error_message": m.get("error_message"),
                }
                for m in raw_results.get("models", [])
            ],
            "primary_metric": raw_results.get("primary_metric", "f1"),
            "feature_importance": raw_results.get("feature_importance"),
            "confusion_matrix": raw_results.get("confusion_matrix"),
            "best_model": raw_results.get("best_model"),
            "winner_explanation": raw_results.get("winner_explanation"),
            "expected_features": raw_results.get("expected_features"),
            "feature_names": raw_results.get("feature_names"),
            "n_samples": raw_results.get("n_samples"),
            "n_features": raw_results.get("n_features"),
            "cv_folds": raw_results.get("cv_folds"),
            "test_size": raw_results.get("test_size"),
        }

    return ExperimentResponse(
        experiment_id=exp["experiment_id"],
        user_id=exp["user_id"],
        dataset_id=exp["dataset_id"],
        dataset_name=exp.get("dataset_name", "Dataset"),
        task_type=exp.get("task_type", "classification"),
        target_column=exp.get("target_column", "target"),
        mode=exp.get("mode", "guided"),
        status=exp.get("status", "pending"),
        created_at=exp.get("created_at", ""),
        completed_at=exp.get("completed_at"),
        best_model=exp.get("best_model"),
        winner_explanation=exp.get("winner_explanation"),
        models=models_res,
        results=results_dict,
        cv_folds=exp.get("results", {}).get("cv_folds") if exp.get("results") else None,
        test_size=exp.get("results", {}).get("test_size") if exp.get("results") else None,
        configuration=exp.get("configuration"),
        error_message=exp.get("error_message"),
    )


@router.delete("/{experiment_id}")
async def remove_experiment(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Cascading cleanup of experiment record and all physical artifacts."""
    res = delete_experiment(experiment_id, user.user_id)
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=res.get("reason", "Experiment not found or access denied."),
        )
    # Cascading deletion to Supabase and storage quota sync
    try:
        delete_experiment_record(experiment_id)
        stats_after = get_storage_stats(user.user_id)
        from backend.supabase_client import update_storage_quota
        update_storage_quota(
            user.user_id,
            dataset_bytes=stats_after.get("dataset_bytes", 0),
            artifact_bytes=stats_after.get("artifact_bytes", 0),
            quota_limit_bytes=stats_after.get("quota_limit_bytes", 52428800),
        )
    except Exception:
        pass
    return res


@router.post("/{experiment_id}/cancel")
async def cancel_experiment(
    experiment_id: str,
    user: UserIdentity = Depends(get_current_user),
):
    """Cancels an in-progress experiment and marks it as stopped."""
    exp = get_experiment_record(experiment_id, user.user_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found or access denied.",
        )

    update_experiment_error(experiment_id, user.user_id, "Benchmark cancelled by user.")
    return {"success": True, "message": "Experiment benchmarking was cancelled by user."}
