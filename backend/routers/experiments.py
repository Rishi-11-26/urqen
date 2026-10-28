"""
backend/routers/experiments.py
------------------------------
Experiment execution, benchmarking, historical tracking, and quota monitoring.
"""

import uuid
import hashlib
import pathlib
import json
import datetime
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
from previa.models.registry import get_available_models, is_xgboost_available
from previa.benchmarking.benchmarker import run_benchmarks
from previa.benchmarking.evaluation import (
    build_split_consistency_report,
    verify_holdout_metrics,
)
from previa.export.model_bundle import save_model_bundle
from previa.export.report_generator import generate_html_report, generate_markdown_report
from previa.experiments.tracker import (
    compute_dataset_fingerprint,
    create_experiment_manifest,
    serialize_manifest_to_json,
)
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
        if "classification" in task:
            valid_idx = y_raw.dropna().index
            y_clean = y_raw.loc[valid_idx].astype(str)
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_clean)
            class_labels = [str(label) for label in label_encoder.classes_]
        else:
            numeric_target = pd.to_numeric(y_raw, errors="coerce")
            valid_idx = numeric_target.dropna().index
            y_encoded = numeric_target.loc[valid_idx].to_numpy(dtype=float)
            class_labels = None
        X_clean = X_raw.loc[valid_idx]
        if len(X_clean) == 0:
            raise ValueError("No rows with valid target values are available for evaluation.")
        y_encoded = np.asarray(y_encoded)

        # Build the same preprocessing blueprint for holdout and fold-local clones.
        f_analysis = analyze_features(df_filtered, target_column=target)
        recs = generate_recommendations(f_analysis, target_column=target)
        rec_by_feature = {rec["feature"]: rec for rec in recs}
        scaler_type = req_data.get("scaler_type", "standard")
        overrides = req_data.get("user_overrides") or {}
        pipeline, num_feats, cat_feats, dropped_feats = build_pipeline(
            recs, user_overrides=overrides, scaler_type=scaler_type
        )

        candidate_features = list(X_raw.columns)
        excluded_columns = [{"column": target, "reason": "Target column; predicted as the response."}]
        user_selected = req_data.get("feature_columns")
        user_selected = set(user_selected) if user_selected else set(df.columns) - {target}
        for column in df.columns:
            if column == target:
                continue
            if column not in user_selected:
                excluded_columns.append({
                    "column": column,
                    "reason": "Excluded because it was not selected in the experiment feature list.",
                })
            elif column in dropped_feats:
                recommendation = rec_by_feature.get(column, {})
                excluded_columns.append({
                    "column": column,
                    "reason": recommendation.get("reason") or "Excluded by the preprocessing recommendation.",
                })
            elif column not in num_feats and column not in cat_feats:
                excluded_columns.append({
                    "column": column,
                    "reason": "Excluded because no supported preprocessing transformer handles this feature type.",
                })
        processed_features = [*num_feats, *cat_feats]

        cv_folds = req_data.get("cv_folds", 10)
        requested_test_size = req_data.get("test_size", 0.20)
        effective_test_size = requested_test_size if len(y_encoded) >= 20 else 0.30
        model_hyperparams = req_data.get("hyperparameters") or {}
        chosen_models = req_data.get("models")
        if not chosen_models:
            chosen_models = [m["name"] for m in get_available_models(task, include_optional=False)]
            if is_xgboost_available() and "XGBoost" not in chosen_models:
                chosen_models.append("XGBoost")
        row_positions = {index: position for position, index in enumerate(df.index)}
        sample_ids = [f"row-{row_positions[index]}" for index in X_clean.index]

        bench_res = run_benchmarks(
            X_clean,
            y_encoded,
            task_type=task,
            models_to_evaluate=chosen_models,
            cv_folds=cv_folds,
            test_size=effective_test_size,
            model_hyperparams=model_hyperparams,
            preprocessor=pipeline,
            sample_ids=sample_ids,
            class_labels=class_labels,
            return_fitted_estimators=True,
        )
        split_indices = bench_res.pop("_split_indices")
        fitted_estimators = bench_res.pop("_fitted_estimators")
        evaluation_artifacts = bench_res.pop("evaluation_artifacts")
        models_by_name = {model["model"]: model for model in bench_res["models"]}
        for evaluation in evaluation_artifacts:
            model_result = models_by_name.get(evaluation["model"], {})
            holdout = evaluation.get("holdout") or {}
            if evaluation["status"] == "success":
                evaluation["metric_verification"] = verify_holdout_metrics(
                    actual=holdout.get("actual"),
                    predicted=holdout.get("predicted"),
                    task_type=task,
                    reported_metrics=model_result,
                    class_labels=holdout.get("class_labels"),
                )
                holdout["metrics"]["class_labels"] = holdout.get("class_labels", [])
            else:
                evaluation["metric_verification"] = {}
        split_configuration = bench_res["split_configuration"]
        split_consistency = build_split_consistency_report(evaluation_artifacts, split_configuration)

        warnings = []
        if split_configuration["strategy"] == "in_sample_fallback_too_few_rows":
            warnings.append("Fewer than 10 valid rows: training and holdout rows overlap; metrics are not independent.")
        for evaluation in evaluation_artifacts:
            if evaluation["status"] != "success":
                warnings.append(f"{evaluation['model']} evaluation failed: {evaluation.get('error_message')}")
            elif evaluation.get("cross_validation", {}).get("error_message"):
                warnings.append(f"{evaluation['model']} cross-validation is unavailable.")
            elif "classification" in task and evaluation["holdout"].get("probabilities") is None:
                warnings.append(f"{evaluation['model']} does not provide holdout class probabilities.")
            for metric, evidence in evaluation.get("metric_verification", {}).items():
                if isinstance(evidence, dict) and evidence.get("status") == "mismatch":
                    warnings.append(f"{evaluation['model']} reported {metric} differs from independent recalculation.")
        if split_consistency["status"] != "consistent":
            warnings.extend(split_consistency["missing_or_inconsistent_artifacts"])

        best_name = bench_res.get("best_model")
        best_estimator = fitted_estimators.get(best_name)
        if best_estimator is None:
            best_name = None
            bench_res["best_model"] = None
            feat_importance = {}
            feat_names = []
            fitted_pipeline = None
            fitted_model = None
            X_holdout_transformed = None
            y_holdout = None
        else:
            fitted_pipeline = best_estimator.named_steps["preprocessor"]
            fitted_model = best_estimator.named_steps["model"]
            train_indices = np.asarray(split_indices["train"], dtype=int)
            holdout_indices = np.asarray(split_indices["holdout"], dtype=int)
            X_holdout_raw = X_clean.iloc[holdout_indices]
            y_holdout = y_encoded[holdout_indices]
            X_holdout_transformed = fitted_pipeline.transform(X_holdout_raw)
            try:
                feat_names = [name.split("__")[-1] for name in fitted_pipeline.get_feature_names_out()]
            except (AttributeError, ValueError):
                feat_names = [f"feature_{index}" for index in range(X_holdout_transformed.shape[1])]
            feat_importance = compute_feature_importance(
                model=fitted_model,
                X=X_holdout_transformed,
                y=y_holdout,
                feature_names=feat_names,
                task_type=task,
            )

        exp_dir = settings.ARTIFACTS_DIR / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = {}
        expected_features = processed_features
        if best_estimator is not None:
            bundle_path = exp_dir / "model_bundle.joblib"
            bundle_metadata = {
                "experiment_id": experiment_id,
                "target_column": target,
                "task_type": task,
                "expected_features": expected_features,
                "feature_names_out": feat_names,
                "classes": class_labels if "classification" in task else None,
                "metrics": bench_res,
            }
            save_model_bundle(
                fitted_pipeline, fitted_model, bundle_metadata, str(bundle_path)
            )
            artifact_paths["model_bundle"] = str(bundle_path)

        preprocessing_steps = []
        if num_feats:
            preprocessing_steps.extend([
                {"columns": num_feats, "transformers": ["median imputation"]},
            ])
            if scaler_type.lower() != "none":
                preprocessing_steps[-1]["transformers"].append(f"{scaler_type} scaling")
        if cat_feats:
            preprocessing_steps.append({
                "columns": cat_feats,
                "transformers": ["most-frequent imputation", "one-hot encoding"],
            })
        if not preprocessing_steps:
            preprocessing_steps.append({
                "columns": [],
                "transformers": [],
                "status": "No columns passed through a preprocessing transformer.",
            })
        final_feature_count = (
            int(X_holdout_transformed.shape[1]) if X_holdout_transformed is not None else None
        )
        preprocessing_audit = {
            "original_dataset_columns": [
                {"column": str(column), "dtype": str(dtype)}
                for column, dtype in df.dtypes.items()
            ],
            "target_column": target,
            "candidate_features": candidate_features,
            "numeric_columns_processed": num_feats,
            "categorical_columns_processed": cat_feats,
            "excluded_columns": excluded_columns,
            "preprocessing_steps": preprocessing_steps,
            "feature_selection_or_transformation": (
                "One-hot encoding expands categorical columns; numeric columns are imputed and scaled."
                if num_feats or cat_feats else "No preprocessing transformations were applied."
            ),
            "final_transformed_feature_count": final_feature_count,
            "fit_scope": "Training partition for holdout; separately cloned and fitted within each CV training fold.",
        }
        split_configuration["training_sample_count"] = int(len(split_indices["train"]))
        split_configuration["holdout_sample_count"] = int(len(split_indices["holdout"]))
        dataset_fingerprint = compute_dataset_fingerprint(df)

        metric_verification = {
            item["model"]: item.get("metric_verification", {})
            for item in evaluation_artifacts
        }
        bench_res["evaluation_summary"] = {
            "split_consistency": split_consistency,
            "metric_verification": metric_verification,
            "warnings": warnings,
        }
        bench_res["feature_importance"] = feat_importance
        bench_res["expected_features"] = expected_features
        bench_res["feature_names"] = feat_names
        winner_evaluation = next(
            (item for item in evaluation_artifacts if item.get("model") == best_name),
            None,
        )
        bench_res["confusion_matrix"] = (
            winner_evaluation.get("holdout", {}).get("confusion_matrix")
            if winner_evaluation and "classification" in task
            else None
        )

        evaluation_export = {
            "format_version": 1,
            "experiment": {
                "experiment_id": experiment_id,
                "dataset_id": req_data.get("dataset_id"),
                "dataset_name": dataset_name,
                "dataset_fingerprint_sha256": dataset_fingerprint,
                "task_type": task,
                "target_column": target,
                "primary_metric": bench_res["primary_metric"],
                "primary_metric_definition": (
                    "Macro-F1" if bench_res["primary_metric"] == "f1"
                    else "R-squared" if bench_res["primary_metric"] == "r2"
                    else bench_res["primary_metric"]
                ),
                "model_parameters": model_hyperparams,
                "preprocessing_configuration": {
                    "scaler": scaler_type,
                    "user_overrides": overrides,
                    "cv_folds_requested": cv_folds,
                },
                "random_seed": 42,
                "split_configuration": split_configuration,
                "training_sample_count": int(len(split_indices["train"])),
                "holdout_sample_count": int(len(split_indices["holdout"])),
                "training_status": "completed",
                "evaluation_status": "completed" if best_name else "completed_without_successful_models",
            },
            "dataset_schema": preprocessing_audit["original_dataset_columns"],
            "preprocessing_audit": preprocessing_audit,
            "data_artifact_types": {
                "original_input_data": "Source dataset; not included in this export.",
                "preprocessed_dataset_csv": "Human-readable imputed table; not a holdout-only dataset or transformed model matrix.",
                "holdout_predictions": "Actual stored predictions with safe row IDs; does not include raw feature values.",
                "cross_validation": "Fold scores computed on the training partition with fold-local preprocessing.",
            },
            "model_comparison": bench_res["models"],
            "models": evaluation_artifacts,
            "metric_verification": metric_verification,
            "split_consistency": split_consistency,
            "warnings": warnings,
        }
        evaluation_path = exp_dir / "evaluation.json"
        evaluation_path.write_text(
            serialize_manifest_to_json(evaluation_export), encoding="utf-8"
        )
        artifact_paths["evaluation"] = str(evaluation_path)
        sample_directory = exp_dir / "evaluation_samples"
        sample_directory.mkdir(exist_ok=True)
        for evaluation in evaluation_artifacts:
            holdout = evaluation.get("holdout")
            if not holdout:
                continue
            sample_key = hashlib.sha256(
                evaluation["model"].encode("utf-8")
            ).hexdigest()[:16]
            sample_path = sample_directory / f"{sample_key}.jsonl"
            with sample_path.open("w", encoding="utf-8") as sample_file:
                for index, sample_id in enumerate(holdout["sample_ids"]):
                    sample = {
                        "sample_id": sample_id,
                        "actual": holdout["actual"][index],
                        "predicted": holdout["predicted"][index],
                        "probabilities": (
                            holdout["probabilities"][index]
                            if holdout.get("probabilities") is not None else None
                        ),
                    }
                    sample_file.write(json.dumps(sample, separators=(",", ":")) + "\n")
            artifact_paths[f"evaluation_samples_{sample_key}"] = str(sample_path)

        manifest = create_experiment_manifest(
            df=df,
            target_column=target,
            task_type=task,
            preprocessing_config={
                "scaler": scaler_type,
                "cv_folds": cv_folds,
                "test_size": effective_test_size,
                **preprocessing_audit,
            },
            benchmarking_results=bench_res,
            explainability_summary=feat_importance,
            dataset_name=dataset_name,
            random_seed=42,
        )
        report_evaluations = []
        for evaluation in evaluation_artifacts:
            holdout = evaluation.get("holdout") or {}
            report_holdout = None
            if holdout:
                report_holdout = {
                    "sample_count": holdout.get("sample_count"),
                    "metrics": holdout.get("metrics"),
                    "class_labels": holdout.get("class_labels"),
                    "confusion_matrix": holdout.get("confusion_matrix"),
                }
                if holdout.get("confusion_matrix") is None:
                    report_holdout["example_pairs"] = [
                        {
                            "sample_id": holdout["sample_ids"][index],
                            "actual": holdout["actual"][index],
                            "predicted": holdout["predicted"][index],
                        }
                        for index in range(min(10, holdout.get("sample_count", 0)))
                    ]
            report_evaluations.append({
                "model": evaluation["model"],
                "status": evaluation["status"],
                "error_message": evaluation.get("error_message"),
                "holdout": report_holdout,
                "cross_validation": evaluation.get("cross_validation"),
                "metric_verification": evaluation.get("metric_verification"),
            })
        manifest.update({
            "experiment_id": experiment_id,
            "experiment_metadata": evaluation_export["experiment"],
            "evaluation_summary": {
                "split_consistency": split_consistency,
                "models": report_evaluations,
                "warnings": warnings,
            },
        })
        manifest_path = exp_dir / "manifest.json"
        manifest_path.write_text(serialize_manifest_to_json(manifest), encoding="utf-8")
        html_path = exp_dir / "report.html"
        html_path.write_text(generate_html_report(manifest), encoding="utf-8")
        md_path = exp_dir / "report.md"
        md_path.write_text(generate_markdown_report(manifest), encoding="utf-8")
        artifact_paths.update({
            "manifest": str(manifest_path),
            "html_report": str(html_path),
            "markdown_report": str(md_path),
        })

        prep_path = exp_dir / "preprocessed_dataset.csv"
        from previa.export.exporter import create_clean_preprocessed_dataframe
        clean_prep_df = create_clean_preprocessed_dataframe(
            df=df_filtered.loc[valid_idx],
            target_column=target,
            numeric_features=num_feats,
            categorical_features=cat_feats,
            dropped_features=dropped_feats,
            train_indices=X_clean.iloc[split_indices["train"]].index,
        )
        clean_prep_df.to_csv(prep_path, index=False)
        artifact_paths["preprocessed_data"] = str(prep_path)

        update_experiment_results(
            experiment_id=experiment_id,
            user_id=user_id,
            results=bench_res,
            best_model=best_name or "",
            winner_explanation=bench_res["winner_explanation"],
            artifact_paths=artifact_paths,
        )

        try:
            from backend.storage import get_experiment_record
            final_record = get_experiment_record(experiment_id, user_id)
            if final_record:
                sync_experiment(user_id, final_record)
        except Exception:
            pass

    except Exception as e:
        update_experiment_error(
            experiment_id,
            user_id,
            f"Experiment failed ({type(e).__name__}); evaluation artifacts may be unavailable.",
        )


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
    from backend.storage import get_maintenance_status, check_and_increment_trial_quota
    m_status = get_maintenance_status()
    if m_status.get("active") and not settings.is_admin(user.email):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=m_status.get("message", "Platform is currently in maintenance mode. Training is paused."),
        )

    is_admin_user = settings.is_admin(user.email)

    # Prevent concurrent experiment execution for the same user (stops job spamming & quota racing)
    if not is_admin_user:
        active_exps = [e for e in list_experiments(user.user_id) if e.get("status") == "running"]
        now = datetime.datetime.now(datetime.timezone.utc)
        for active in active_exps:
            created_str = active.get("created_at")
            is_stale = False
            if created_str:
                try:
                    created_dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if (now - created_dt).total_seconds() > 900:  # 15 min timeout for stale jobs
                        is_stale = True
                except Exception:
                    is_stale = True
            else:
                is_stale = True

            if is_stale:
                update_experiment_error(
                    active.get("experiment_id"),
                    user.user_id,
                    "Experiment execution timed out.",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="An experiment is already actively running for your account. Please wait for it to complete.",
                )

    # Enforce trial quota (2 free Guided runs, 1 free Expert run; unlimited for Developer)
    try:
        check_and_increment_trial_quota(user.user_id, mode=req.mode or "guided", is_admin=is_admin_user)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

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
                cv_folds=m.get("cv_folds"),
                cv_fold_scores=m.get("cv_fold_scores"),
                holdout_sample_count=exp.get("results", {}).get("holdout_sample_count"),
                accuracy=m.get("accuracy"),
                f1=m.get("f1"),
                f1_macro=m.get("f1_macro"),
                f1_weighted=m.get("f1_weighted"),
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
                    "cv_folds": m.get("cv_folds"),
                    "cv_fold_scores": m.get("cv_fold_scores"),
                    "holdout_sample_count": raw_results.get("holdout_sample_count"),
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "f1_macro": m.get("f1_macro"),
                    "f1_weighted": m.get("f1_weighted"),
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
            "evaluation_summary": raw_results.get("evaluation_summary"),
            "evaluation_available": bool(exp.get("artifact_paths", {}).get("evaluation")),
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
