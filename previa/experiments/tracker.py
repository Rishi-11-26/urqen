"""
previa/experiments/tracker.py
------------------------------
Lightweight reproducible experiment manifest tracker.

Records complete provenance for each tabular ML experiment:
- Unique Experiment ID
- ISO Timestamp
- Dataset SHA-256 fingerprint (hash of shape + column names + head data)
- Dataset dimensions and schema
- Target and task
- Feature configuration & preprocessing decisions
- Model configurations and winner selection
- Software version metadata (Python, scikit-learn, URQEN)
- Downloadable JSON manifest string
"""

import sys
import json
import hashlib
import datetime
import uuid
import pandas as pd
import sklearn

import previa


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 fingerprint of a DataFrame without
    persisting or exposing raw sensitive data rows.
    """
    hasher = hashlib.sha256()
    # Hash shape, columns, and data dtypes
    hasher.update(str(df.shape).encode("utf-8"))
    hasher.update(",".join(df.columns.astype(str)).encode("utf-8"))
    hasher.update(",".join(str(t) for t in df.dtypes).encode("utf-8"))
    # Hash the complete frame without persisting any cell values.
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return hasher.hexdigest()


def create_experiment_manifest(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    preprocessing_config: dict,
    benchmarking_results: dict,
    explainability_summary: dict | None = None,
    dataset_name: str = "dataset.csv",
    random_seed: int = 42,
) -> dict:
    """
    Assemble a comprehensive, reproducible experiment manifest dictionary.

    Returns
    -------
    dict
        Structured manifest ready for JSON serialization.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    short_uid = uuid.uuid4().hex[:6]
    experiment_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{short_uid}"

    dataset_fingerprint = compute_dataset_fingerprint(df)

    schema = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    manifest = {
        "experiment_id": experiment_id,
        "timestamp_utc": now.isoformat(),
        "software_versions": {
            "previa": previa.__version__,
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
        },
        "dataset_metadata": {
            "name": dataset_name,
            "fingerprint_sha256": dataset_fingerprint,
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "schema": schema,
        },
        "task_setup": {
            "target_column": target_column,
            "task_type": task_type,
            "random_seed": random_seed,
        },
        "preprocessing_configuration": preprocessing_config,
        "model_benchmarking": {
            "primary_metric": benchmarking_results.get("primary_metric"),
            "best_model": benchmarking_results.get("best_model"),
            "winner_explanation": benchmarking_results.get("winner_explanation"),
            "cv_folds": benchmarking_results.get("cv_folds"),
            "cv_warning": benchmarking_results.get("cv_warning"),
            "test_sample_count": benchmarking_results.get("test_size"),
            "models_evaluated": [
                {
                    "model": m.get("model"),
                    "status": m.get("status", "success"),
                    "family": m.get("family"),
                    "training_time_seconds": m.get("training_time_seconds"),
                    "evaluation_type": "holdout",
                    "holdout_sample_count": benchmarking_results.get("holdout_sample_count"),
                    "primary_score": m.get(benchmarking_results.get("primary_metric")),
                    "cv_mean": m.get("cv_mean"),
                    "cv_std": m.get("cv_std"),
                    "cv_fold_scores": m.get("cv_fold_scores", []),
                    "cv_metric": (
                        "macro_f1" if "classification" in benchmarking_results.get("task_type", "")
                        else "r2"
                    ),
                }
                for m in benchmarking_results.get("models", [])
            ],
        },
        "explainability": explainability_summary or {},
    }

    return manifest


def serialize_manifest_to_json(manifest: dict, indent: int = 2) -> str:
    """Serialize the manifest dictionary to pretty-printed JSON."""
    return json.dumps(manifest, indent=indent, default=str)
