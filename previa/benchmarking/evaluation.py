"""Independent verification helpers for persisted holdout evaluation artifacts."""

import hashlib
import json
import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)


CLASSIFICATION_METRICS = {
    "accuracy": "accuracy",
    "precision": "macro_precision",
    "recall": "macro_recall",
    "f1": "macro_f1",
    "f1_macro": "macro_f1",
    "f1_weighted": "weighted_f1",
}
REGRESSION_METRICS = ("mae", "mse", "rmse", "r2")


def verify_holdout_metrics(
    actual: list[Any] | None,
    predicted: list[Any] | None,
    task_type: str,
    reported_metrics: dict[str, Any],
    class_labels: list[Any] | None = None,
    tolerance: float = 1e-8,
) -> dict:
    """Recalculate reported holdout metrics from stored labels and predictions."""
    if actual is None or predicted is None or len(actual) != len(predicted) or not actual:
        return {
            metric: {
                "reported": reported_metrics.get(metric),
                "recalculated": None,
                "absolute_difference": None,
                "status": "unavailable_for_verification",
                "explanation": "Stored holdout labels and predictions are unavailable or misaligned.",
            }
            for metric in _reported_metric_names(task_type, reported_metrics)
        }

    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)
    if "classification" in task_type:
        labels = class_labels or sorted(set(actual) | set(predicted), key=str)
        actual_values = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        }
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_class = [
            {
                "label": _native(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ]
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        verification = _compare_reported(reported_metrics, actual_values, tolerance)
        verification["confusion_matrix"] = matrix
        verification["class_labels"] = [_native(label) for label in labels]
        verification["per_class"] = per_class
        verification["zero_division"] = 0
        return verification

    actual_values = {}
    if len(y_true) >= 1:
        actual_values.update(
            mae=float(mean_absolute_error(y_true, y_pred)),
            mse=float(mean_squared_error(y_true, y_pred)),
            rmse=float(math.sqrt(mean_squared_error(y_true, y_pred))),
        )
    if len(y_true) >= 2 and len(np.unique(y_true)) >= 2:
        actual_values["r2"] = float(r2_score(y_true, y_pred))
    verification = _compare_reported(reported_metrics, actual_values, tolerance)
    if "r2" in reported_metrics and "r2" not in actual_values:
        verification["r2"] = {
            "reported": reported_metrics.get("r2"),
            "recalculated": None,
            "absolute_difference": None,
            "status": "unavailable_for_verification",
            "explanation": "R² is undefined for a single sample or a constant actual target.",
        }
    return verification


def build_split_consistency_report(
    model_artifacts: list[dict],
    split_configuration: dict,
) -> dict:
    """Compare model holdout identity, configuration, and target alignment."""
    evaluated = [item for item in model_artifacts if item.get("status") == "success"]
    missing = []
    ids_by_model = {}
    targets_by_model = {}
    configurations = []

    for item in evaluated:
        model = item.get("model", "Unknown")
        holdout = item.get("holdout") or {}
        sample_ids = holdout.get("sample_ids")
        actual = holdout.get("actual")
        if sample_ids is None or actual is None or len(sample_ids) != len(actual):
            missing.append(f"{model}: holdout identifiers or actual targets are missing/misaligned.")
            continue
        ids_by_model[model] = list(sample_ids)
        targets_by_model[model] = dict(zip(sample_ids, actual))
        configurations.append(item.get("split_configuration"))

    first_ids = next(iter(ids_by_model.values()), None)
    identifiers_match = (
        first_ids is not None
        and all(ids == first_ids for ids in ids_by_model.values())
        and len(ids_by_model) == len(evaluated)
    )
    config_key = json.dumps(split_configuration, sort_keys=True, default=str)
    configurations_match = all(
        json.dumps(config, sort_keys=True, default=str) == config_key
        for config in configurations
    ) and len(configurations) == len(evaluated)

    targets_aligned = bool(identifiers_match)
    if targets_aligned and targets_by_model:
        first_targets = next(iter(targets_by_model.values()))
        targets_aligned = all(targets == first_targets for targets in targets_by_model.values())

    all_ids = sorted({sample_id for ids in ids_by_model.values() for sample_id in ids})
    fingerprint = hashlib.sha256(
        json.dumps(all_ids, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "evaluated_model_count": len(evaluated),
        "holdout_sample_count_per_model": {
            item.get("model", "Unknown"): len((item.get("holdout") or {}).get("sample_ids") or [])
            for item in model_artifacts
        },
        "split_configuration_matches": configurations_match,
        "holdout_row_identifiers_match": identifiers_match,
        "target_labels_aligned": targets_aligned,
        "holdout_identifiers_fingerprint_sha256": fingerprint if all_ids else None,
        "missing_or_inconsistent_artifacts": missing,
        "status": (
            "consistent"
            if evaluated and identifiers_match and configurations_match and targets_aligned and not missing
            else "inconsistent_or_unavailable"
        ),
        "limitations": [
            "Shared holdout identity is established by stable row identifiers, not sample count alone."
        ],
    }


def _reported_metric_names(task_type: str, reported_metrics: dict) -> list[str]:
    candidates = CLASSIFICATION_METRICS if "classification" in task_type else {
        key: key for key in REGRESSION_METRICS
    }
    return [key for key in candidates if key in reported_metrics]


def _compare_reported(
    reported_metrics: dict[str, Any],
    calculated_metrics: dict[str, float],
    tolerance: float,
) -> dict:
    results = {}
    metric_pairs = (
        [("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"),
         ("f1", "f1"), ("f1_macro", "f1_macro"), ("f1_weighted", "f1_weighted")]
    )
    for reported_name, calculated_name in metric_pairs:
        if reported_name not in reported_metrics:
            continue
        results[reported_name] = _metric_comparison(
            reported_metrics[reported_name],
            calculated_metrics.get(calculated_name),
            tolerance,
        )
    for metric in REGRESSION_METRICS:
        if metric in reported_metrics:
            results[metric] = _metric_comparison(
                reported_metrics[metric], calculated_metrics.get(metric), tolerance
            )
    return results


def _metric_comparison(reported: Any, calculated: float | None, tolerance: float) -> dict:
    if reported is None or calculated is None or not math.isfinite(float(calculated)):
        return {
            "reported": _native(reported),
            "recalculated": None,
            "absolute_difference": None,
            "status": "unavailable_for_verification",
            "explanation": "The reported value or independently calculated metric is unavailable.",
        }
    difference = abs(float(reported) - calculated)
    return {
        "reported": float(reported),
        "recalculated": calculated,
        "absolute_difference": difference,
        "status": "verified" if difference <= tolerance else "mismatch",
        "explanation": None if difference <= tolerance else "Difference exceeds tolerance.",
    }


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
