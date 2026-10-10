"""
tests/test_leakage.py
---------------------
Tests for previa.auditing.leakage.audit_leakage_and_reliability

We test:
- Small dataset size warning
- Severe class imbalance detection
- Highly correlated numeric feature (|r| >= 0.95)
- Suspicious feature naming pattern
- Clean dataset with no leakage produces no false high-severity warnings
- Returns well-structured warning dicts
"""

import pandas as pd
import numpy as np
import pytest

from previa.auditing.leakage import audit_leakage_and_reliability


class TestLeakageAuditor:

    def test_small_dataset_flagged(self):
        df = pd.DataFrame({
            "feat": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        })
        warnings = audit_leakage_and_reliability(df, target_column="target", task_type="binary_classification")
        types = [w["warning_type"] for w in warnings]
        assert "small_sample_size" in types

    def test_high_correlation_numeric_leakage(self):
        rng = np.random.default_rng(42)
        target = rng.standard_normal(50)
        # Perfectly correlated feature
        leaky_feat = target * 1.0001
        clean_feat = rng.standard_normal(50)

        df = pd.DataFrame({
            "clean_x": clean_feat,
            "leak_x": leaky_feat,
            "target_y": target,
        })
        warnings = audit_leakage_and_reliability(df, target_column="target_y", task_type="regression")
        leak_warnings = [w for w in warnings if w["feature"] == "leak_x"]
        assert len(leak_warnings) == 1
        assert leak_warnings[0]["warning_type"] == "near_perfect_target_correlation"
        assert leak_warnings[0]["severity"] == "high"

    def test_suspicious_name_pattern(self):
        df = pd.DataFrame({
            "age": [20, 30, 40] * 10,
            "post_outcome_status": ["a", "b", "c"] * 10,
            "approved": [1, 0, 1] * 10,
        })
        warnings = audit_leakage_and_reliability(df, target_column="approved", task_type="binary_classification")
        suspicious = [w for w in warnings if w["feature"] == "post_outcome_status"]
        assert len(suspicious) >= 1
        assert suspicious[0]["warning_type"] == "suspicious_feature_name"

    def test_severe_class_imbalance(self):
        # 95% class 0, 5% class 1
        y = [0] * 95 + [1] * 5
        x = list(range(100))
        df = pd.DataFrame({"x": x, "target": y})
        warnings = audit_leakage_and_reliability(df, target_column="target", task_type="binary_classification")
        imbalance = [w for w in warnings if w["warning_type"] == "severe_class_imbalance"]
        assert len(imbalance) == 1
        assert "target" in imbalance[0]["feature"]

    def test_clean_data_no_severe_leakage(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feat_a": rng.standard_normal(60),
            "feat_b": rng.standard_normal(60),
            "target": rng.integers(0, 2, 60),
        })
        warnings = audit_leakage_and_reliability(df, target_column="target", task_type="binary_classification")
        high_sev = [w for w in warnings if w["severity"] == "high"]
        assert len(high_sev) == 0
