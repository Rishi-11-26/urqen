"""
tests/test_diagnostics.py
-------------------------
Tests for previa.diagnostics.charts

We test:
- plot_confusion_matrix returns a plotly Figure
- plot_roc_curve returns a plotly Figure
- plot_actual_vs_predicted returns a plotly Figure
- plot_residuals returns a plotly Figure
- Valid structure and data in returned figures
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from previa.diagnostics.charts import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_actual_vs_predicted,
    plot_residuals,
)


class TestDiagnosticsCharts:

    def test_confusion_matrix_returns_figure(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "heatmap"

    def test_confusion_matrix_with_custom_labels(self):
        y_true = np.array(["cat", "dog", "cat"])
        y_pred = np.array(["cat", "cat", "dog"])
        fig = plot_confusion_matrix(y_true, y_pred, labels=["cat", "dog"])
        assert isinstance(fig, go.Figure)
        assert list(fig.data[0].x) == ["cat", "dog"]

    def test_roc_curve_returns_figure(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        fig = plot_roc_curve(y_true, y_prob)
        assert isinstance(fig, go.Figure)
        # Should have 2 traces: ROC curve + random baseline
        assert len(fig.data) == 2
        assert fig.data[0].type == "scatter"

    def test_actual_vs_predicted_returns_figure(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.5, 29.0])
        fig = plot_actual_vs_predicted(y_true, y_pred)
        assert isinstance(fig, go.Figure)
        # Should have scatter trace + diagonal reference line trace
        assert len(fig.data) == 2
        assert fig.data[0].type == "scatter"

    def test_residuals_returns_figure(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.5, 29.0])
        fig = plot_residuals(y_true, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].type == "scatter"
