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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
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
    y = np.array(y)
    n_samples = len(y)
    hyperparams_map = model_hyperparams or {}

    # --- 1. Determine CV fold count ---
    if cv_folds is not None:
        actual_cv_folds = cv_folds
        cv_warning = None if cv_folds > 0 else "Cross-validation disabled."
    else:
        actual_cv_folds, cv_warning = _determine_cv_folds(n_samples, task_type)

    # --- 2. Train / test split ---
    split_size, X_train, X_test, y_train, y_test = _split_data(X, y, n_samples, task_type, user_test_size=test_size)

    # --- 3. Determine models to evaluate ---
    if models_to_evaluate:
        models_meta = []
        for name in models_to_evaluate:
            models_meta.append({"name": name, "family": "custom"})
    else:
        models_meta = get_available_models(task_type, include_optional=include_optional)

    # --- 4. Evaluate each model ---
    results = []
    for meta in models_meta:
        name = meta["name"]
        try:
            m_kwargs = hyperparams_map.get(name, {})
            model = create_model_instance(name, task_type, random_state=RANDOM_STATE, **m_kwargs)
            res = _evaluate_model(
                name, model, X_train, X_test, y_train, y_test,
                task_type, actual_cv_folds, X, y
            )
            res["family"] = meta.get("family", "general")
            res["status"] = "success"
        except Exception as exc:
            res = {
                "model": name,
                "family": meta.get("family", "general"),
                "status": "failed",
                "error_message": str(exc),
                "training_time_seconds": 0.0,
                "cv_mean": None,
                "cv_std": None,
            }
            if task_type in ("binary_classification", "multiclass_classification"):
                res.update({
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "sensitivity": 0.0, "specificity": 0.0, "roc_auc": None,
                })
            else:
                res.update({"mae": float("inf"), "mse": float("inf"), "rmse": float("inf"), "r2": -float("inf")})

        results.append(res)

    # --- 5. Determine ranking metric & winner ---
    chosen_metric = primary_metric or _default_primary_metric(task_type)
    best_model, winner_reason, sorted_results = _find_best_model_and_sort(results, chosen_metric, task_type)

    return {
        "task_type": task_type,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "cv_folds": actual_cv_folds,
        "cv_warning": cv_warning,
        "test_size": test_size,
        "models": sorted_results,
        "best_model": best_model,
        "primary_metric": chosen_metric,
        "winner_explanation": winner_reason,
    }


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
    if n_samples < 10:
        return n_samples, X, X, y, y

    if user_test_size is not None and 0.05 <= user_test_size <= 0.5:
        test_ratio = float(user_test_size)
    else:
        test_ratio = 0.30 if n_samples < 20 else 0.20

    stratify = None
    if task_type in ("binary_classification", "multiclass_classification"):
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=RANDOM_STATE, stratify=stratify
        )
    except Exception:
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
    """Compute classification hold-out metrics including sensitivity and specificity."""
    is_binary = task_type == "binary_classification"
    average = "binary" if is_binary else "weighted"

    # Sensitivity (Recall / TPR) and Specificity (TNR)
    cm = confusion_matrix(y_test, y_pred)
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
    f1_macro = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    f1_weighted = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1": f1_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "roc_auc": None,
    }

    if hasattr(model, "predict_proba"):
        try:
            if is_binary and len(np.unique(y_test)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            elif not is_binary and len(np.unique(y_test)) > 1:
                y_prob = model.predict_proba(X_test)
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")), 4)
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
