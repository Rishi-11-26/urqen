"""
previa/benchmarking/benchmarker.py
------------------------------------
Trains and evaluates a set of baseline ML models.

Purpose
-------
This module answers:
  "How do standard baseline models perform on this dataset after preprocessing?"

It does NOT claim to find the best model in the world.
The results are always described as:
  "Best among the evaluated baseline models according to the selected metric."

How it works
------------
1. Receives preprocessed X (numpy array) and y (numpy array/series).
2. Determines the appropriate CV fold count based on dataset size.
3. For each baseline model:
   a. Fits the model on the full training set for the hold-out metrics.
   b. Also runs cross-validation for a more reliable CV mean score.
4. Returns a structured results dict with per-model metrics.

Small dataset handling
-----------------------
The function automatically adjusts the CV fold count:
  - n_samples >= 50  : 5-fold CV
  - n_samples >= 20  : 3-fold CV
  - n_samples < 20   : 2-fold CV with a warning
  - n_samples < 10   : CV is skipped entirely with a clear message

This prevents misleading results on very small datasets.

Train/test split
-----------------
We do an 80/20 split with a fixed random_state for reproducibility.
For very small datasets (<20 samples), we may fall back to 70/30 or
skip the hold-out test set and use only CV.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


# Fixed random state for reproducibility across all models and splits
RANDOM_STATE = 42

# CV fold thresholds (based on dataset size)
CV_FOLDS_LARGE = 5     # n_samples >= 50
CV_FOLDS_MEDIUM = 3    # n_samples >= 20
CV_FOLDS_SMALL = 2     # n_samples >= 10


def run_benchmarks(
    X: np.ndarray,
    y: np.ndarray | pd.Series,
    task_type: str,
) -> dict:
    """
    Train and evaluate baseline models on preprocessed data.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed feature matrix (output of the fitted pipeline transformer).
    y : array-like
        Target values.
    task_type : str
        "binary_classification", "multiclass_classification", or "regression".

    Returns
    -------
    dict with keys:
        "task_type"      : str
        "n_samples"      : int
        "n_features"     : int
        "cv_folds"       : int (the number of folds actually used)
        "cv_warning"     : str or None (message if dataset is very small)
        "test_size"      : int (number of samples in the hold-out test set)
        "models"         : list[dict]  — one dict per model with metrics
        "best_model"     : str  — name of the best model by primary metric
        "primary_metric" : str  — the metric used to rank models
    """
    y = np.array(y)
    n_samples = len(y)

    # --- Determine CV configuration based on dataset size ---
    cv_folds, cv_warning = _determine_cv_folds(n_samples, task_type)

    # --- Train/test split ---
    test_size, X_train, X_test, y_train, y_test = _split_data(X, y, n_samples)

    # --- Select models for this task ---
    models = _get_models(task_type)

    # --- Train and evaluate each model ---
    results = []
    for name, model in models:
        result = _evaluate_model(
            name, model, X_train, X_test, y_train, y_test,
            task_type, cv_folds, X, y
        )
        results.append(result)

    # --- Determine primary metric and best model ---
    primary_metric = _primary_metric(task_type)
    best_model = _find_best(results, primary_metric)

    return {
        "task_type": task_type,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "cv_folds": cv_folds,
        "cv_warning": cv_warning,
        "test_size": test_size,
        "models": results,
        "best_model": best_model,
        "primary_metric": primary_metric,
    }


# ---------------------------------------------------------------------------
# Internal: dataset size / CV configuration
# ---------------------------------------------------------------------------

def _determine_cv_folds(n_samples: int, task_type: str) -> tuple[int, str | None]:
    """
    Choose an appropriate CV fold count for the dataset size.

    Returns (cv_folds, warning_message_or_None).
    """
    if n_samples >= 50:
        return CV_FOLDS_LARGE, None

    if n_samples >= 20:
        return CV_FOLDS_MEDIUM, (
            f"Dataset has {n_samples} samples. "
            f"Using {CV_FOLDS_MEDIUM}-fold cross-validation instead of 5-fold "
            f"to avoid overly small fold sizes."
        )

    if n_samples >= 10:
        return CV_FOLDS_SMALL, (
            f"Dataset is small ({n_samples} samples). "
            f"Using {CV_FOLDS_SMALL}-fold cross-validation. "
            f"Results may be unstable. Collect more data if possible."
        )

    # Too small for meaningful CV
    return 0, (
        f"Dataset has only {n_samples} samples — too small for reliable "
        f"cross-validation. CV is skipped. Results should be treated with caution."
    )


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split X, y into train/test. Adjusts the split ratio for small datasets.

    Returns (test_size_count, X_train, X_test, y_train, y_test).
    """
    if n_samples < 10:
        # Too small: use everything as "training"; test set = train set
        # (results are meaningless but we don't crash)
        return n_samples, X, X, y, y

    if n_samples < 20:
        test_ratio = 0.30
    else:
        test_ratio = 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=RANDOM_STATE
    )
    return len(y_test), X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Internal: model selection
# ---------------------------------------------------------------------------

def _get_models(task_type: str) -> list[tuple[str, object]]:
    """Return the list of (name, model) pairs for the given task."""
    if task_type in ("binary_classification", "multiclass_classification"):
        return [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]
    else:  # regression
        return [
            ("Linear Regression", LinearRegression()),
            ("Ridge", Ridge(random_state=RANDOM_STATE)),
            ("Decision Tree", DecisionTreeRegressor(random_state=RANDOM_STATE)),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        ]


# ---------------------------------------------------------------------------
# Internal: single-model evaluation
# ---------------------------------------------------------------------------

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
    """Fit one model, compute hold-out metrics, and run CV."""

    # Fit on training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {"model": name}

    # --- Hold-out metrics ---
    if task_type in ("binary_classification", "multiclass_classification"):
        result.update(_classification_metrics(y_test, y_pred, model, X_test, task_type))
    else:
        result.update(_regression_metrics(y_test, y_pred))

    # --- Cross-validation ---
    if cv_folds >= 2:
        cv_scoring = _cv_scoring(task_type)
        try:
            cv_scores = cross_val_score(
                model, X_full, y_full,
                cv=cv_folds,
                scoring=cv_scoring,
            )
            result["cv_mean"] = round(float(cv_scores.mean()), 4)
            result["cv_std"] = round(float(cv_scores.std()), 4)
        except Exception as e:
            result["cv_mean"] = None
            result["cv_std"] = None
            result["cv_error"] = str(e)
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

    # ROC-AUC: only for binary classification with probability support
    if is_binary and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            pass  # Not all binary datasets produce a valid ROC-AUC

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


# ---------------------------------------------------------------------------
# Internal: helpers
# ---------------------------------------------------------------------------

def _primary_metric(task_type: str) -> str:
    """Return the metric used to rank models."""
    if task_type == "binary_classification":
        return "f1"
    elif task_type == "multiclass_classification":
        return "f1"
    else:
        return "r2"


def _cv_scoring(task_type: str) -> str:
    """Return the sklearn scoring string for cross_val_score."""
    if task_type == "binary_classification":
        return "f1"
    elif task_type == "multiclass_classification":
        return "f1_weighted"
    else:
        return "r2"


def _find_best(results: list[dict], primary_metric: str) -> str:
    """Return the model name with the highest primary metric value."""
    best = max(
        results,
        key=lambda r: r.get(primary_metric) if r.get(primary_metric) is not None else float("-inf"),
    )
    return best["model"]
