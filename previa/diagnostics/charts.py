"""
previa/diagnostics/charts.py
------------------------------
Plotly-based diagnostic charts for model evaluation.

For classification:
  - Confusion matrix heatmap
  - ROC curve (binary classification only, if model has predict_proba)

For regression:
  - Actual vs. predicted scatter plot
  - Residual plot (residuals = actual - predicted, plotted against predicted)

Design principles
-----------------
- All functions return a Plotly Figure object (not displayed here).
  The Streamlit app calls st.plotly_chart(fig) to render them.
- Charts are informative and labeled clearly.
- No matplotlib — only Plotly.
- No unnecessary decoration.

Why these charts?
-----------------
Confusion matrix: shows WHERE the model makes mistakes, not just how many.
ROC curve: shows the trade-off between true positive and false positive rates.
Actual vs predicted: for regression, shows how far predictions deviate from truth.
Residual plot: reveals patterns in errors (e.g., heteroscedasticity).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> go.Figure:
    """
    Create a Plotly confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list or None
        Class label names for axis ticks. If None, uses the unique values from y_true.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = sorted(set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    str_labels = [str(l) for l in labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=str_labels,
            y=str_labels,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14},
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(autorange="reversed"),  # top-left is True Positive by convention
        width=500,
        height=450,
    )

    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
    """
    Create a Plotly ROC curve for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {roc_auc:.3f})",
        line=dict(color="#1f77b4", width=2),
    ))

    # Random baseline (diagonal)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Baseline",
        line=dict(color="gray", dash="dash", width=1),
    ))

    fig.update_layout(
        title=f"ROC Curve — AUC = {roc_auc:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        width=550,
        height=450,
    )

    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create an actual vs. predicted scatter plot for regression.

    A perfect model would produce all points on the diagonal line (y = x).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    fig = go.Figure()

    # Scatter: actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=y_true,
        mode="markers",
        marker=dict(color="#1f77b4", opacity=0.6, size=7),
        name="Predictions",
    ))

    # Perfect prediction line
    all_vals = np.concatenate([y_true, y_pred])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="red", dash="dash", width=1.5),
    ))

    fig.update_layout(
        title="Actual vs. Predicted",
        xaxis_title="Predicted Value",
        yaxis_title="Actual Value",
        width=550,
        height=450,
    )

    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Create a residual plot for regression diagnostics.

    Residuals = Actual − Predicted.
    A good model has residuals scattered randomly around zero.
    Patterns in the residuals (e.g., a funnel shape) suggest the model
    has systematic errors.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig = go.Figure()

    # Residual scatter
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        marker=dict(color="#ff7f0e", opacity=0.6, size=7),
        name="Residuals",
    ))

    # Zero line (ideal: all residuals = 0)
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1.5)

    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual − Predicted)",
        width=550,
        height=450,
    )

    return fig
