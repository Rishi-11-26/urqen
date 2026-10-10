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
- Centralized model instantiation from Model Registry.
- Execution time tracking per model (training_time_seconds).
- Graceful failure handling (one model failure does not halt others).
- Small-dataset adaptive cross-validation.
- Support for primary metric selection and transparent winner rationale.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from previa.models.registry import (
    get_available_models,
    create_model_instance,
    RANDOM_STATE,
)


CV_FOLDS_LARGE = 5     # n_samples >= 50
CV_FOLDS_MEDIUM = 3    # n_samples >= 20
CV_FOLDS_SMALL = 2     # n_samples >= 10


def run_benchmarks(
    X: np.ndarray,
    y: np.ndarray | pd.Series,
    task_type: str,
    primary_metric: str | None = None,
    include_optional: bool = False,
) -> dict:
    """
    Train and evaluate baseline models on preprocessed data.

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
        Whether to include optional algorithms (e.g. XGBoost) in addition to core baselines.
        Defaults to False to guarantee identical evaluation baselines.

    Returns
    -------
    dict with keys:
        "task_type"          : str
        "n_samples"          : int
        "n_features"         : int
        "cv_folds"           : int
        "cv_warning"         : str or None
        "test_size"          : int
        "models"             : list[dict]
        "best_model"         : str
        "primary_metric"     : str
        "winner_explanation" : str
    """
    y = np.array(y)
    n_samples = len(y)

    # --- 1. Determine CV fold count ---
    cv_folds, cv_warning = _determine_cv_folds(n_samples, task_type)

    # --- 2. Train / test split ---
    test_size, X_train, X_test, y_train, y_test = _split_data(X, y, n_samples, task_type)

    # --- 3. Model registry catalog ---
    available_model_meta = get_available_models(task_type, include_optional=include_optional)

    # --- 4. Evaluate each model ---
    results = []
    for meta in available_model_meta:
        name = meta["name"]
        try:
            model = create_model_instance(name, task_type, random_state=RANDOM_STATE)
            res = _evaluate_model(
                name, model, X_train, X_test, y_train, y_test,
                task_type, cv_folds, X, y
            )
            res["family"] = meta["family"]
            res["status"] = "success"
        except Exception as exc:
            res = {
                "model": name,
                "family": meta["family"],
                "status": "failed",
                "error_message": str(exc),
                "training_time_seconds": 0.0,
                "cv_mean": None,
                "cv_std": None,
            }
            if task_type in ("binary_classification", "multiclass_classification"):
                res.update({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": None})
            else:
                res.update({"mae": float("inf"), "mse": float("inf"), "rmse": float("inf"), "r2": -float("inf")})

        results.append(res)

    # --- 5. Determine ranking metric & winner ---
    chosen_metric = primary_metric or _default_primary_metric(task_type)
    best_model, winner_reason = _find_best_model(results, chosen_metric, task_type)

    return {
        "task_type": task_type,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "cv_folds": cv_folds,
        "cv_warning": cv_warning,
        "test_size": test_size,
        "models": results,
        "best_model": best_model,
        "primary_metric": chosen_metric,
        "winner_explanation": winner_reason,
    }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _determine_cv_folds(n_samples: int, task_type: str) -> tuple[int, str | None]:
    """Choose appropriate CV fold count based on sample size."""
    if n_samples >= 50:
        return CV_FOLDS_LARGE, None

    if n_samples >= 20:
        return CV_FOLDS_MEDIUM, (
            f"Dataset has {n_samples} samples. "
            f"Using {CV_FOLDS_MEDIUM}-fold cross-validation instead of 5-fold "
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
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test with stratification when appropriate."""
    if n_samples < 10:
        return n_samples, X, X, y, y

    test_ratio = 0.30 if n_samples < 20 else 0.20

    stratify = None
    if task_type in ("binary_classification", "multiclass_classification"):
        # Check minimum class count for stratification
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=RANDOM_STATE, stratify=stratify
        )
    except Exception:
        # Fallback to unstratified if class counts fail constraint
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=RANDOM_STATE
        )

    return len(y_test), X_train, X_test, y_train, y_test


def _evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    cv_folds: int,
    X_full: np.ndarray,
    y_full: np.ndarray,
) -> dict:
    """Fit one model, measure execution time, compute hold-out metrics and CV."""
    start_t = time.perf_counter()
    model.fit(X_train, y_train)
    fit_duration = time.perf_counter() - start_t

    y_pred = model.predict(X_test)
    result = {
        "model": name,
        "training_time_seconds": round(float(fit_duration), 4),
    }

    # Hold-out metrics
    if task_type in ("binary_classification", "multiclass_classification"):
        result.update(_classification_metrics(y_test, y_pred, model, X_test, task_type))
    else:
        result.update(_regression_metrics(y_test, y_pred))

    # Cross-validation
    if cv_folds >= 2:
        cv_scoring = _cv_scoring(task_type)
        try:
            cv_scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring=cv_scoring)
            result["cv_mean"] = round(float(cv_scores.mean()), 4)
            result["cv_std"] = round(float(cv_scores.std()), 4)
        except Exception as exc:
            result["cv_mean"] = None
            result["cv_std"] = None
            result["cv_error"] = str(exc)
    else:
        result["cv_mean"] = None
        result["cv_std"] = None

    return result


def _classification_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model,
    X_test: np.ndarray,
    task_type: str,
) -> dict:
    """Compute classification hold-out metrics."""
    is_binary = task_type == "binary_classification"
    average = "binary" if is_binary else "weighted"

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average=average, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average=average, zero_division=0)), 4),
        "roc_auc": None,
    }

    if is_binary and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            pass

    return metrics


def _regression_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression hold-out metrics."""
    mse = float(mean_squared_error(y_test, y_pred))
    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def _default_primary_metric(task_type: str) -> str:
    """Return default ranking metric."""
    if task_type in ("binary_classification", "multiclass_classification"):
        return "f1"
    return "r2"


def _cv_scoring(task_type: str) -> str:
    """Return scoring metric for cross-validation."""
    if task_type == "binary_classification":
        return "f1"
    elif task_type == "multiclass_classification":
        return "f1_weighted"
    return "r2"


def _find_best_model(results: list[dict], metric: str, task_type: str) -> tuple[str, str]:
    """Find best model by primary metric and generate explanation."""
    successful = [r for r in results if r.get("status") != "failed"]
    if not successful:
        return "None", "All models encountered training errors."

    # Lower is better for error metrics (mae, mse, rmse)
    lower_is_better = metric in ("mae", "mse", "rmse")

    def _sort_key(r):
        val = r.get(metric)
        if val is None:
            return float("inf") if lower_is_better else float("-inf")
        return -val if lower_is_better else val

    best = max(successful, key=_sort_key)
    score = best.get(metric)

    direction = "lowest" if lower_is_better else "highest"
    explanation = (
        f"'{best['model']}' achieved the {direction} {metric.upper()} ({score}) "
        f"on the hold-out evaluation set among all {len(successful)} successfully trained baselines."
    )
    return best["model"], explanation
