"""
tests/test_report_generator.py
------------------------------
Tests for previa.export.report_generator

We test:
- generate_markdown_report produces non-empty string with required sections
- generate_html_report produces valid HTML document
"""

import pandas as pd
import pytest

from previa.experiments.tracker import create_experiment_manifest
from previa.export.report_generator import (
    generate_markdown_report,
    generate_html_report,
)


class TestReportGenerator:

    @pytest.fixture
    def sample_manifest(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})
        return create_experiment_manifest(
            df=df,
            target_column="target",
            task_type="binary_classification",
            preprocessing_config={"numeric_features": ["feat"]},
            benchmarking_results={
                "primary_metric": "f1",
                "best_model": "Logistic Regression",
                "winner_explanation": "Highest hold-out F1 score",
                "models": [{"model": "Logistic Regression", "status": "success", "primary_score": 0.95}],
            },
        )

    def test_markdown_report_sections(self, sample_manifest):
        md = generate_markdown_report(sample_manifest)
        assert "# URQEN — Model Decision Report" in md
        assert "## 1. Experiment Overview" in md
        assert "## 2. Dataset Summary & Schema" in md
        assert "## 6. Evaluated Models & Winner Selection" in md
        assert "Logistic Regression" in md

    def test_html_report_structure(self, sample_manifest):
        html_out = generate_html_report(sample_manifest)
        assert "<!DOCTYPE html>" in html_out
        assert "<title>URQEN Model Decision Report</title>" in html_out
        assert "<h1>URQEN — Model Decision Report</h1>" in html_out
        assert "</html>" in html_out
