"""
previa/export/model_bundle.py
-----------------------------
Joblib-based export, loading, and inference prediction engine for Previa.

Bundles the fitted ColumnTransformer preprocessing pipeline and trained
estimator into a single self-contained artifact alongside essential metadata:
- Experiment ID
- Target column name
- Task type
- Expected input feature columns and types
- Preprocessing steps
- Evaluation metrics
"""

import os
import pathlib
import joblib
import pandas as pd
import numpy as np


class PreviaModelBundle:
    """
    Self-contained deployment artifact holding a fitted preprocessing transformer,
    a trained estimator, and dataset schema metadata.
    """

    def __init__(
        self,
        pipeline,
        model,
        metadata: dict,
    ):
        self.pipeline = pipeline
        self.model = model
        self.metadata = metadata

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for a new incoming raw DataFrame.

        Validates that required features are present, runs them through
        the fitted preprocessing pipeline, and outputs predictions with
        associated class probabilities where supported.

        Parameters
        ----------
        df : pd.DataFrame
            New raw dataset to score.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and probability scores.
        """
        expected_features = self.metadata.get("expected_features", [])

        # Schema validation
        missing_cols = [c for c in expected_features if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Prediction data is missing {len(missing_cols)} required feature(s): {missing_cols}"
            )

        # Select only expected columns in expected order
        X_raw = df[expected_features].copy()

        # Transform using fitted preprocessing pipeline
        X_trans = self.pipeline.transform(X_raw)

        # Generate predictions
        preds = self.model.predict(X_trans)

        result_df = pd.DataFrame({"prediction": preds})

        # Add probability scores if classification model supports predict_proba
        task_type = self.metadata.get("task_type", "")
        if task_type in ("binary_classification", "multiclass_classification"):
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X_trans)
                    if probs.shape[1] == 2:
                        result_df["probability_positive_class"] = np.round(probs[:, 1], 4)
                    else:
                        for class_idx in range(probs.shape[1]):
                            result_df[f"prob_class_{class_idx}"] = np.round(probs[:, class_idx], 4)
                except Exception:
                    pass

        return result_df


def save_model_bundle(
    pipeline,
    model,
    metadata: dict,
    output_path: str,
) -> str:
    """
    Serialize a PreviaModelBundle to a .joblib file on disk.

    Parameters
    ----------
    pipeline : ColumnTransformer or Pipeline
        Fitted preprocessing pipeline.
    model : estimator
        Fitted scikit-learn or XGBoost model.
    metadata : dict
        Context metadata (experiment_id, expected_features, target, task_type).
    output_path : str
        Target destination file path (.joblib).

    Returns
    -------
    str
        Resolved absolute path.
    """
    path = pathlib.Path(output_path)
    if not path.parent.exists():
        os.makedirs(path.parent, exist_ok=True)

    bundle = PreviaModelBundle(
        pipeline=pipeline,
        model=model,
        metadata=metadata,
    )

    joblib.dump(bundle, str(path))
    return str(path.resolve())


def load_model_bundle(filepath: str) -> PreviaModelBundle:
    """
    Load a PreviaModelBundle from a .joblib file on disk.

    Parameters
    ----------
    filepath : str
        Path to the saved .joblib file.

    Returns
    -------
    PreviaModelBundle
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle file not found: '{filepath}'")

    bundle = joblib.load(str(path))
    if not isinstance(bundle, PreviaModelBundle):
        raise TypeError(f"Loaded object is not a valid PreviaModelBundle (type: {type(bundle)}).")

    return bundle
