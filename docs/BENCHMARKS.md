# URQEN Competitive Benchmark Report

## 1. Executive Summary

This document presents benchmark results comparing URQEN's pipeline and baseline models against a Naive Baseline (dummy classifier/regressor).

The evaluation protocol measures:
1. **Predictive Performance**: Hold-out test evaluation score.
2. **Evaluation Correctness**: Strict prevention of data leakage via unfitted transformers.
3. **Data Quality Awareness**: Number of data anomalies (missingness, skew, outliers) detected.
4. **Execution Runtime**: End-to-end audit, preprocessing, and training duration.
5. **Explainability**: Presence of feature importance and causality disclaimers.

---

## 2. Experimental Setup

The benchmark is executed locally using the reproducible benchmark suite (`previa.benchmarking.competitive_benchmarks`):
- **Hardware/Environment**: Python 3.13 on x86_64 CPU.
- **Random Seed**: `42` across all splits and stochastic algorithms.
- **Data Splits**: 80% train / 20% test split.
- **Evaluation Metrics**:
  - Binary Classification: $F_1$-score (macro/binary).
  - Multiclass Classification: Weighted $F_1$-score.
  - Continuous Regression: $R^2$ coefficient of determination.

---

## 3. Benchmark Datasets

| Dataset | Samples | Features | Characteristics | Target Column |
| :--- | :---: | :---: | :--- | :--- |
| **Binary Classification** | 200 | 3 | Exponential skew, missing values in Age, Categorical levels | `purchased` (0/1) |
| **Multiclass Classification** | 240 | 3 | Multi-region categories, class imbalance (50/35/15%) | `category` (0/1/2) |
| **Continuous Regression** | 200 | 3 | Non-linear tier multipliers, missing bedroom counts | `price` ($) |

---

## 4. Empirical Results

| Task | Metric | Naive Dummy Baseline | URQEN Winner Baseline | URQEN Winning Model | URQEN Runtime | Quality Findings Flagged |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Binary Classification** | $F_1$ Score | 0.0000 | **0.8718** | Logistic Regression | ~0.42s | 3 (Missingness, Skew, Outliers) |
| **Multiclass Classification** | $F_1$ Score | 0.3871 | **0.5833** | Random Forest | ~0.48s | 2 (Class Imbalance, Regions) |
| **Continuous Regression** | $R^2$ Score | -0.0051 | **0.9576** | Random Forest Regressor | ~0.39s | 2 (Missingness, Outliers) |

---

## 5. Comparative Evaluation

| Feature / Dimension | Naive Baseline | URQEN Workbench |
| :--- | :---: | :---: |
| **Data Leakage Safety** | ❌ Fails (Requires manual prep) | ✅ Guaranteed (`ColumnTransformer` on train split) |
| **Missing Value Handling** | ❌ None (Errors on raw data) | ✅ Robust Median / Mode Imputation |
| **Categorical Conversion** | ❌ None | ✅ One-Hot Encoding (`ignore_unknown=True`) |
| **Adaptive Cross-Validation** | ❌ None | ✅ Dynamic folds based on sample size |
| **Model Comparison** | ❌ Single guess | ✅ Centralized Registry (Linear, Tree, Ensemble, XGBoost) |
| **Explainability** | ❌ Black-box | ✅ SHAP / Model Feature Importance with Disclaimer |
| **Reproducibility** | ❌ Ad-hoc | ✅ SHA-256 Fingerprint + JSON Manifest |

---

## 6. Honest Limitations & Boundary

1. URQEN does NOT perform exhaustive hyperparameter Bayesian optimization; it focuses on pre-model auditing and baseline evaluation.
2. On massive datasets (>1M rows), tree ensembles and cross-validation will take longer; runtime caching in Streamlit mitigates redundant re-evaluations.
3. URQEN is not a replacement for domain expertise: flagged outliers and high-cardinality features require user confirmation before final deployment.
