"""
previa/explainability/explainer.py
----------------------------------
Model explainability engine supporting SHAP values with deterministic
model-derived fallbacks.

Mandatory Engineering Principles:
1. Predictive feature importance does NOT establish real-world causality.
   A prominent disclaimer is attached to every explanation result.
2. If SHAP is unavailable or incompatible, use a clearly labeled fallback
   (e.g., Tree Gini Importance, Linear Magnitude, Permutation Importance).
   Never label a fallback as SHAP.
3. Explainability failures must never crash or invalidate successful model training.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance


CAUSALITY_DISCLAIMER = (
    "Predictive feature importance measures model dependency and statistical association "
    "on this dataset. It reflects what the algorithm used to minimize its loss function; "
    "it does NOT establish real-world cause-and-effect relationships."
)


def compute_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    task_type: str = "binary_classification",
    max_shap_samples: int = 100,
) -> dict:
    """
    Compute global feature importances for a trained model.

    Tries SHAP if installed and compatible, with graceful fallbacks
    to native model feature importances, linear coefficient magnitudes,
    or permutation importance.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn or XGBoost model.
    X : np.ndarray
        Evaluation or test feature matrix.
    y : np.ndarray
        True target labels or values.
    feature_names : list[str] or None
        Names of the features corresponding to X's columns.
    task_type : str
        "binary_classification", "multiclass_classification", or "regression".
    max_shap_samples : int
        Maximum rows from X to evaluate with SHAP for speed.

    Returns
    -------
    dict with:
        "method_used"         : str
        "is_shap"             : bool
        "feature_importances" : list[dict] sorted descending
        "top_features"        : list[str]
        "disclaimer"          : str
        "summary"             : str
    """
    n_features = X.shape[1]
    if feature_names is None or len(feature_names) != n_features:
        names = [f"feature_{i}" for i in range(n_features)]
    else:
        names = [str(f) for f in feature_names]

    # Sample X for SHAP if large
    if len(X) > max_shap_samples:
        sample_indices = np.linspace(0, len(X) - 1, max_shap_samples, dtype=int)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    # 1. Try SHAP first
    shap_result = _try_shap_explanation(model, X_sample, names)
    if shap_result is not None:
        return shap_result

    # 2. Try native model feature_importances_ (Trees & Ensembles)
    if hasattr(model, "feature_importances_"):
        raw_vals = np.array(model.feature_importances_, dtype=float)
        return _format_importance_dict(
            raw_vals, names,
            method_used="Model-Derived Feature Importance (Gini / Split Gain)",
            is_shap=False
        )

    # 3. Try linear model coefficients (coef_)
    if hasattr(model, "coef_"):
        raw_coef = np.array(model.coef_, dtype=float)
        if raw_coef.ndim > 1:
            raw_vals = np.mean(np.abs(raw_coef), axis=0)
        else:
            raw_vals = np.abs(raw_coef)
        return _format_importance_dict(
            raw_vals, names,
            method_used="Linear Coefficient Magnitudes (|coef_|)",
            is_shap=False
        )

    # 4. Fallback to Permutation Importance
    try:
        perm_res = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        raw_vals = np.maximum(0, perm_res.importances_mean)
        return _format_importance_dict(
            raw_vals, names,
            method_used="Permutation Feature Importance (Hold-out Scoring)",
            is_shap=False
        )
    except Exception as exc:
        # Ultimate safe fallback: uniform dummy importances
        raw_vals = np.zeros(n_features)
        return _format_importance_dict(
            raw_vals, names,
            method_used=f"Fallback Importance (Error: {exc})",
            is_shap=False
        )


def plot_feature_importance_chart(importance_data: dict, top_k: int = 15) -> go.Figure:
    """
    Generate a Plotly horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_data : dict
        Output from compute_feature_importance().
    top_k : int
        Maximum number of top features to display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    items = importance_data.get("feature_importances", [])[:top_k]
    # Reverse so top feature is at the top of horizontal bar chart
    items = items[::-1]

    features = [item["feature"] for item in items]
    importances = [item["importance"] for item in items]

    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation="h",
        marker=dict(color="#1f77b4", opacity=0.85),
    ))

    method = importance_data.get("method_used", "Feature Importance")
    fig.update_layout(
        title=f"Global Feature Importance — {method}",
        xaxis_title="Relative Importance Score",
        yaxis_title="Feature",
        height=max(380, 25 * len(features)),
        margin=dict(l=150, r=30, t=50, b=40),
    )

    return fig


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _try_shap_explanation(model, X_sample: np.ndarray, names: list[str]) -> dict | None:
    """Attempt SHAP calculation, returning None on any incompatibility."""
    try:
        import shap

        # For tree models
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                # Multiclass: average absolute values across classes
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                mean_abs = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            return _format_importance_dict(
                mean_abs, names,
                method_used="SHAP (TreeExplainer mean absolute values)",
                is_shap=True
            )

        # For linear models
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            return _format_importance_dict(
                mean_abs, names,
                method_used="SHAP (LinearExplainer mean absolute values)",
                is_shap=True
            )

    except Exception:
        # Graceful fallback on any SHAP error
        return None

    return None


def _format_importance_dict(
    raw_scores: np.ndarray,
    feature_names: list[str],
    method_used: str,
    is_shap: bool,
) -> dict:
    """Format raw score array into clean sorted feature importance list."""
    total = np.sum(raw_scores)
    if total > 0:
        normalized = raw_scores / total
    else:
        normalized = raw_scores

    records = []
    for name, norm_val in zip(feature_names, normalized):
        records.append({
            "feature": name,
            "importance": round(float(norm_val), 4),
        })

    # Sort descending
    records.sort(key=lambda r: r["importance"], reverse=True)
    for rank, r in enumerate(records, start=1):
        r["rank"] = rank

    top_names = [r["feature"] for r in records[:5]]
    summary = (
        f"Top influential features identified by {method_used}: "
        + ", ".join([f"'{name}'" for name in top_names]) + "."
    )

    return {
        "method_used": method_used,
        "is_shap": is_shap,
        "feature_importances": records,
        "top_features": top_names,
        "disclaimer": CAUSALITY_DISCLAIMER,
        "summary": summary,
    }
