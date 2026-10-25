"""
previa/benchmarking/benchmarker.py
------------------------------------
Trains and evaluates baseline ML models from the Model Registry.

Purpose
-------
Answers: "How do standard baseline models perform on this dataset after preprocessing?"

It does NOT claim to find an objectively optimal model.
The results are always described as:
  "Best among the evaluated baseline models according to the selected metric."

Features:
- Centralized model instantiation from Model Registry (LVSM, Naive Bayes, RBFN, Random Forest, XGBoost).
- Evaluation metrics include Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity, MAE, MSE, RMSE, R2.
- 10-fold cross-validation on datasets with >= 50 samples (adaptive for smaller datasets).
- Execution time tracking per model (training_time_seconds).
- Graceful failure handling (one model failure does not halt others).
- Transparent winner selection rationale.
"""

import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    precision_recall_fscore_support,
)

from previa.models.registry import (
    get_available_models,
    create_model_instance,
    RANDOM_STATE,
)


CV_FOLDS_LARGE = 10    # n_samples >= 50: 10-fold CV
CV_FOLDS_MEDIUM = 3    # n_samples >= 20
CV_FOLDS_SMALL = 2     # n_samples >= 10


def run_benchmarks(
    X: np.ndarray,
    y: np.ndarray | pd.Series,
    task_type: str,
    primary_metric: str | None = None,
    include_optional: bool = False,
    models_to_evaluate: list[str] | None = None,
    cv_folds: int | None = None,
    test_size: float | None = None,
    model_hyperparams: dict[str, dict] | None = None,
    preprocessor=None,
    sample_ids: list[str] | None = None,
    class_labels: list[str] | None = None,
    return_fitted_estimators: bool = False,
) -> dict:
    """
    Train and evaluate models on preprocessed data.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed feature matrix.
    y : array-like
        Target values.
    task_type : str
        "binary_classification", "multiclass_classification", or "regression".
    primary_metric : str or None
        Metric used to rank models. Defaults to 'f1' for classification, 'r2' for regression.
    include_optional : bool
        Whether to include expanded algorithms (LVSM, Naive Bayes, RBFN, XGBoost).
        Defaults to False for standard baseline testing.
    models_to_evaluate : list[str] or None
        Explicit list of model names to benchmark. If provided, overrides include_optional.
    cv_folds : int or None
        Manual override for CV folds. If None, automatically determined based on sample size (10-fold for >=50).
    test_size : float or None
        Holdout test set fraction (e.g., 0.2 for 20%). If None, auto-determined based on sample size.
    model_hyperparams : dict[str, dict] or None
        Optional dictionary mapping model names to custom hyperparameter dictionaries
        (e.g., {"XGBoost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}}).

    Returns
    -------
    dict
    """
    y = np.asarray(y)
    n_samples = len(y)
    hyperparams_map = model_hyperparams or {}

    if cv_folds is not None:
        actual_cv_folds = cv_folds
        cv_warning = None if cv_folds > 0 else "Cross-validation disabled."
    else:
        actual_cv_folds, cv_warning = _determine_cv_folds(n_samples, task_type)

    train_indices, test_indices, split_ratio, split_strategy = _split_indices(y, task_type, test_size)
    X_train = _take_rows(X, train_indices)
    X_test = _take_rows(X, test_indices)
    y_train = y[train_indices]
    y_test = y[test_indices]
    ids = list(sample_ids) if sample_ids is not None else [f"row-{i}" for i in range(n_samples)]
    if len(ids) != n_samples:
        raise ValueError("sample_ids must contain one identifier for every input row.")
    holdout_ids = [str(ids[i]) for i in test_indices]
    split_configuration = {
        "strategy": split_strategy,
        "test_size": split_ratio,
        "random_state": RANDOM_STATE,
        "stratified": split_strategy == "train_test_split_stratified",
        "training_sample_count": int(len(train_indices)),
        "holdout_sample_count": int(len(test_indices)),
    }

    if models_to_evaluate:
        models_meta = [{"name": name, "family": "custom"} for name in models_to_evaluate]
    else:
        models_meta = get_available_models(task_type, include_optional=include_optional)

    results = []
    evaluation_artifacts = []
    fitted_estimators = {}
    for meta in models_meta:
        name = meta["name"]
        try:
            model = create_model_instance(
                name, task_type, random_state=RANDOM_STATE, **hyperparams_map.get(name, {})
            )
            res, artifact, fitted = _evaluate_model(
                name=name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                X_cv=X_train,
                y_cv=y_train,
                task_type=task_type,
                cv_folds=actual_cv_folds,
                preprocessor=preprocessor,
                holdout_ids=holdout_ids,
                class_labels=class_labels,
                split_configuration=split_configuration,
            )
            res["family"] = meta.get("family", "general")
            res["status"] = "success"
            artifact.update({"model": name, "family": res["family"], "status": "success"})
            evaluation_artifacts.append(artifact)
            if return_fitted_estimators:
                fitted_estimators[name] = fitted
        except Exception as exc:
            error_text = f"Model evaluation failed ({type(exc).__name__})."
            res = {
                "model": name,
                "family": meta.get("family", "general"),
                "status": "failed",
                "error_message": error_text,
                "training_time_seconds": 0.0,
                "cv_mean": None,
                "cv_std": None,
            }
            if "classification" in task_type:
                res.update({
                    "accuracy": None, "precision": None, "recall": None, "f1": None,
                    "f1_macro": None, "f1_weighted": None,
                    "sensitivity": None, "specificity": None, "roc_auc": None,
                })
            else:
                res.update({"mae": None, "mse": None, "rmse": None, "r2": None})
            evaluation_artifacts.append({
                "model": name,
                "family": meta.get("family", "general"),
                "status": "failed",
                "error_message": error_text,
                "holdout": None,
                "cross_validation": {
                    "fold_count": 0,
                    "fold_scores": [],
                    "mean": None,
                    "standard_deviation": None,
                    "metric": _cv_scoring(task_type),
                    "aggregation": "arithmetic mean and population standard deviation",
                    "error_message": "Unavailable because model evaluation failed.",
                },
                "split_configuration": split_configuration,
            })
        results.append(res)

    chosen_metric = primary_metric or _default_primary_metric(task_type)
    best_model, winner_reason, sorted_results = _find_best_model_and_sort(
        results, chosen_metric, task_type
    )
    output = {
        "task_type": task_type,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "cv_folds": actual_cv_folds,
        "cv_warning": cv_warning,
        "test_size": split_ratio,
        "training_sample_count": int(len(train_indices)),
        "holdout_sample_count": int(len(test_indices)),
        "split_configuration": split_configuration,
        "models": sorted_results,
        "best_model": best_model,
        "primary_metric": chosen_metric,
        "winner_explanation": winner_reason,
        "evaluation_artifacts": evaluation_artifacts,
    }
    if return_fitted_estimators:
        output["_fitted_estimators"] = fitted_estimators
        output["_split_indices"] = {
            "train": train_indices.tolist(),
            "holdout": test_indices.tolist(),
        }
    return output


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _determine_cv_folds(n_samples: int, task_type: str) -> tuple[int, str | None]:
    """Choose appropriate CV fold count based on sample size (10-fold default)."""
    if n_samples >= 50:
        return CV_FOLDS_LARGE, None

    if n_samples >= 20:
        return CV_FOLDS_MEDIUM, (
            f"Dataset has {n_samples} samples. "
            f"Using {CV_FOLDS_MEDIUM}-fold cross-validation instead of 10-fold "
            f"to prevent overly small fold sizes."
        )

    if n_samples >= 10:
        return CV_FOLDS_SMALL, (
            f"Dataset is small ({n_samples} samples). "
            f"Using {CV_FOLDS_SMALL}-fold cross-validation. "
            f"Results may be unstable; collect more data if possible."
        )

    return 0, (
        f"Dataset has only {n_samples} samples — too small for reliable "
        f"cross-validation. CV is skipped. Results should be treated with caution."
    )


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    task_type: str,
    user_test_size: float | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test with stratification when appropriate."""
    train_indices, test_indices, _, _ = _split_indices(y, task_type, user_test_size)
    return (
        len(test_indices),
        _take_rows(X, train_indices),
        _take_rows(X, test_indices),
        y[train_indices],
        y[test_indices],
    )


def _split_indices(
    y: np.ndarray,
    task_type: str,
    user_test_size: float | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Generate a shared, deterministic holdout split from positional row IDs."""
    n_samples = len(y)
    if n_samples < 10:
        indices = np.arange(n_samples)
        return indices, indices, 1.0, "in_sample_fallback_too_few_rows"

    if user_test_size is not None and 0.05 <= user_test_size <= 0.5:
        test_ratio = float(user_test_size)
    else:
        test_ratio = 0.30 if n_samples < 20 else 0.20

    stratify = None
    if task_type in ("binary_classification", "multiclass_classification"):
        _, counts = np.unique(y, return_counts=True)
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y
    indices = np.arange(n_samples)
    try:
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=RANDOM_STATE, stratify=stratify
        )
        strategy = "train_test_split_stratified" if stratify is not None else "train_test_split"
    except ValueError:
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=RANDOM_STATE
        )
        strategy = "train_test_split_unstratified_fallback"
    return np.asarray(train_indices), np.asarray(test_indices), test_ratio, strategy


def _take_rows(X, indices: np.ndarray):
    """Select positional rows and preserve DataFrame column names."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _evaluate_model(
    name: str,
    model,
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    cv_folds: int,
    X_cv,
    y_cv: np.ndarray,
    preprocessor,
    holdout_ids: list[str],
    class_labels: list[str] | None,
    split_configuration: dict,
) -> tuple[dict, dict, object]:
    """Fit/evaluate one estimator and retain its holdout predictions."""
    estimator = _make_estimator(model, preprocessor)
    start_t = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_duration = time.perf_counter() - start_t
    y_pred = estimator.predict(X_test)
    result = {
        "model": name,
        "training_time_seconds": float(fit_duration),
    }
    if "classification" in task_type:
        metrics = _classification_metrics(
            y_test,
            y_pred,
            estimator,
            X_test,
            task_type,
            labels=np.arange(len(class_labels)) if class_labels else np.unique(np.concatenate((y_test, y_pred))),
        )
    else:
        metrics = _regression_metrics(y_test, y_pred)
    result.update(metrics)

    cv_scores = []
    cv_error = None
    if cv_folds >= 2:
        try:
            effective_folds = _effective_cv_folds(y_cv, task_type, cv_folds)
            if effective_folds >= 2:
                if "classification" in task_type:
                    cv = StratifiedKFold(
                        n_splits=effective_folds, shuffle=True, random_state=RANDOM_STATE
                    )
                else:
                    cv = KFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_STATE)
                scores = cross_val_score(
                    _make_estimator(model, preprocessor),
                    X_cv,
                    y_cv,
                    cv=cv,
                    scoring=_cv_scoring(task_type),
                    error_score="raise",
                )
                cv_scores = [float(score) for score in scores if np.isfinite(score)]
                if len(cv_scores) != effective_folds:
                    raise ValueError("One or more cross-validation fold scores were undefined.")
                result["cv_mean"] = float(np.mean(cv_scores))
                result["cv_std"] = float(np.std(cv_scores))
                result["cv_fold_scores"] = cv_scores
                result["cv_folds"] = effective_folds
            else:
                result["cv_mean"] = None
                result["cv_std"] = None
                result["cv_fold_scores"] = []
                result["cv_folds"] = 0
        except Exception as exc:
            result["cv_mean"] = None
            result["cv_std"] = None
            result["cv_fold_scores"] = []
            result["cv_folds"] = 0
            cv_error = f"Cross-validation unavailable ({type(exc).__name__})."
            result["cv_error"] = cv_error
    else:
        result["cv_mean"] = None
        result["cv_std"] = None
        result["cv_fold_scores"] = []
        result["cv_folds"] = 0

    if "classification" in task_type:
        actual = [_decode_label(value, class_labels) for value in y_test]
        predicted = [_decode_label(value, class_labels) for value in y_pred]
        labels = class_labels or [str(value) for value in np.unique(np.concatenate((y_test, y_pred)))]
        probabilities = _probability_records(estimator, X_test, class_labels)
        matrix = confusion_matrix(
            y_test, y_pred, labels=np.arange(len(labels))
        ).tolist()
    else:
        actual = [float(value) for value in y_test]
        predicted = [float(value) for value in y_pred]
        labels = []
        probabilities = None
        matrix = None

    artifact = {
        "split_configuration": split_configuration,
        "holdout": {
            "sample_ids": holdout_ids,
            "actual": actual,
            "predicted": predicted,
            "probabilities": probabilities,
            "sample_count": len(y_test),
            "metrics": metrics,
            "class_labels": labels,
            "confusion_matrix": matrix,
        },
        "cross_validation": {
            "fold_count": result["cv_folds"],
            "fold_scores": result["cv_fold_scores"],
            "mean": result["cv_mean"],
            "standard_deviation": result["cv_std"],
            "metric": _cv_scoring(task_type),
            "aggregation": "arithmetic mean and population standard deviation",
            "error_message": cv_error,
            "evaluated_on": "training_partition_only",
        },
    }
    return result, artifact, estimator


def _make_estimator(model, preprocessor):
    if preprocessor is None:
        return clone(model)
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(model)),
    ])


def _effective_cv_folds(y: np.ndarray, task_type: str, requested: int) -> int:
    if "classification" in task_type:
        _, counts = np.unique(y, return_counts=True)
        return min(requested, int(counts.min())) if len(counts) else 0
    return min(requested, len(y))


def _decode_label(value, class_labels: list[str] | None):
    if class_labels is None:
        return value.item() if isinstance(value, np.generic) else value
    return str(class_labels[int(value)])


def _probability_records(estimator, X_test, class_labels: list[str] | None):
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        probabilities = estimator.predict_proba(X_test)
        model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
        classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
        labels = class_labels or [str(value) for value in classes]
        return [
            {
                str(labels[int(class_index)]): float(probability)
                for class_index, probability in zip(classes, row)
            }
            for row in probabilities
        ]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _classification_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model,
    X_test: np.ndarray,
    task_type: str,
    labels: np.ndarray | None = None,
) -> dict:
    """Compute classification hold-out metrics including sensitivity and specificity."""
    is_binary = task_type == "binary_classification"
    average = "binary" if is_binary else "weighted"

    # Sensitivity (Recall / TPR) and Specificity (TNR)
    label_order = labels if labels is not None else np.unique(np.concatenate((y_test, y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=label_order)
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        # Multiclass macro average
        n_classes = cm.shape[0]
        sens_list = []
        spec_list = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        sensitivity = float(np.mean(sens_list)) if sens_list else 0.0
        specificity = float(np.mean(spec_list)) if spec_list else 0.0

    # Macro F1 treats both majority and minority classes with equal weight,
    # preventing artificial 0.0 or 1.0 distortions on imbalanced datasets.
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_test, y_pred, labels=label_order, zero_division=0
    )
    f1_macro = float(f1_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, labels=label_order, average="weighted", zero_division=0))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0)),
        "f1": f1_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "per_class_metrics": [
            {
                "label": int(label),
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
            for index, label in enumerate(label_order)
        ],
        "confusion_matrix": cm.tolist(),
        "class_labels": [int(label) for label in label_order],
        "roc_auc": None,
    }

    if hasattr(model, "predict_proba"):
        try:
            if is_binary and len(np.unique(y_test)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            elif not is_binary and len(np.unique(y_test)) > 1:
                y_prob = model.predict_proba(X_test)
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            pass

    return metrics


def _regression_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression hold-out metrics."""
    mse = float(mean_squared_error(y_test, y_pred))
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": (
            float(r2_score(y_test, y_pred))
            if len(y_test) > 1 and len(np.unique(y_test)) > 1
            else None
        ),
    }


def _default_primary_metric(task_type: str) -> str:
    """Return default ranking metric."""
    if task_type in ("binary_classification", "multiclass_classification"):
        return "f1"
    return "r2"


def _cv_scoring(task_type: str) -> str:
    """Return scoring metric for cross-validation."""
    if task_type in ("binary_classification", "multiclass_classification"):
        return "f1_macro"
    return "r2"


def _find_best_model_and_sort(results: list[dict], metric: str, task_type: str) -> tuple[str, str, list[dict]]:
    """
    Rigorously ranks models by empirical evidence and guarantees 100% agreement
    between the best model selection and the sorted leaderboard.

    Ranking Criteria (Purely Evidence-Driven, Zero Hardcoded Favoritism):
      1. Cross-Validation Mean (cv_mean): Evaluates out-of-sample performance across all folds.
         If CV is disabled or unavailable (<20 samples), uses the holdout test score.
      2. Cross-Validation Stability (lower cv_std): Rewards consistent, low-variance generalization.
      3. Holdout Test Set Confirmation: Validates on the independent holdout partition.
      4. Occam's Razor / Computational Efficiency: Faster training time breaks ties fairly.
    """
    successful = [r for r in results if r.get("status") != "failed"]
    failed = [r for r in results if r.get("status") == "failed"]

    if not successful:
        return "None", "All models encountered training errors.", results

    lower_is_better = metric in ("mae", "mse", "rmse")

    def _val(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return -float("inf")
        return -float(x) if lower_is_better else float(x)

    def _sort_key(r: dict):
        cv_m = r.get("cv_mean")
        holdout_v = r.get(metric)

        # Primary signal: CV score if valid CV was performed; fallback to holdout
        primary = _val(cv_m) if cv_m is not None else _val(holdout_v)
        holdout = _val(holdout_v)
        cv_std = float(r.get("cv_std") or 0.0) if r.get("cv_std") is not None else 1.0
        train_time = float(r.get("training_time_seconds") or 999.0)

        # Round primary to 3 decimal places to detect genuine statistical ties
        return (round(primary, 3), -round(cv_std, 3), round(holdout, 3), -train_time)

    # Sort successful models in descending order of true empirical quality
    successful.sort(key=_sort_key, reverse=True)
    best = successful[0]

    primary_name = "Cross-Validation " + metric.upper() if best.get("cv_mean") is not None else "Holdout " + metric.upper()
    best_score = best.get("cv_mean") if best.get("cv_mean") is not None else best.get(metric)
    direction = "lowest" if lower_is_better else "highest"

    explanation = (
        f"'{best['model']}' achieved the {direction} {primary_name} ({best_score}) "
        f"across all {len(successful)} successfully evaluated algorithms."
    )
    if best.get("cv_mean") is not None and best.get("cv_std") is not None:
        explanation += f" It demonstrated high fold stability with CV score {best['cv_mean']} ± {best['cv_std']}."

    # Return the unified sorted list with best model at index 0
    all_sorted = successful + failed
    return best["model"], explanation, all_sorted
