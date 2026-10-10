# Previa

**Trustworthy Pre-Model Data Intelligence Workbench**

> *"Understand the data before trusting the model."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-159%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()
[![Status](https://img.shields.io/badge/Version-v1.0.0%20Release-blue.svg)]()

---

## 1. Overview & Problem Statement

In practical machine learning, models fail not because algorithms are weak, but because datasets are misunderstood, corrupted, or contaminated with **data leakage**. Beginners and experienced practitioners alike often rush to train complex models while missing critical issues:
- **Data Leakage**: Preprocessing (imputation, scaling, encoding) fitted on the entire dataset rather than strictly on training splits.
- **Silent Quality Issues**: High missingness, constant features, or row identifiers silently distorting weights.
- **Post-Outcome Leakage**: Features recorded after the target outcome event leaking ground truth into training features.
- **Blind AutoML**: Tools that treat data preparation as an opaque black box, leaving users unable to explain *why* transformations were chosen.
- **Overconfident Target Inferences**: Hardcoding assumptions about target labels without human verification.

**Previa** is an explainable, leakage-safe data intelligence workbench designed to audit, prepare, benchmark, explain, and validate tabular datasets before machine learning models are deployed. Previa automates repetitive pre-modeling tasks while exposing evidence, trade-offs, confidence levels, and clear reasoning for every recommendation.

---

## 2. Core Differentiators

1. **Evidence-Backed Data Intelligence**: Identifies data problems, presents measured statistical evidence, recommends actions, explains trade-offs, and prompts for human review.
2. **Trustworthy ML Experiments**: Enforces leakage prevention via `scikit-learn` `ColumnTransformer` fitted strictly on training data, adaptive cross-validation for small datasets, and execution timing.
3. **Transparent Model Decisions & Explainability**: Explains why a model was selected, how it performed, which features influenced its predictions (via SHAP or robust model fallbacks), and attaches scientific causality disclaimers.
4. **Reproducible Provenance**: Generates unique experiment IDs, SHA-256 dataset fingerprints, downloadable JSON manifests, and self-contained `.joblib` model bundles with real-time inference prediction.

---

## 3. End-to-End Workflow Architecture

```
Raw CSV / XLSX
      ↓
[1. Ingestion & Validation] ────→ Format, empty-column & duplicate header checks
      ↓
[2. Automated Profiler]     ────→ Summary counts, memory, quartiles & category frequencies
      ↓
[3. Evidence-Backed Health] ────→ 8-point findings: Evidence, severity, why it matters, trade-offs
      ↓
[4. Leakage & Reliability]  ────→ Near-perfect correlation (|r| >= 0.95), suspicious naming, imbalance
      ↓
[5. Target & Task Detection]────→ Inferred target & task (binary, multiclass, regression) with override
      ↓
[6. Feature Intelligence]   ────→ Type classification (numeric, boolean, categorical, constant)
      ↓
[7. Preprocessing Engine]   ────→ Rule-based recommendations with user acceptance & customization
      ↓
[8. Leakage-Safe Pipeline]  ────→ ColumnTransformer blueprint (fitted strictly on training splits)
      ↓
[9. Centralized Benchmark]  ────→ Model Registry (LR, Ridge, DT, RF, XGBoost) with adaptive CV & timing
      ↓
[10. Interactive Diagnostics]───→ Plotly Confusion Matrix, ROC curve, Actual vs. Predicted, Residuals
      ↓
[11. Explainability Engine] ────→ SHAP values & model feature importance with causality disclaimer
      ↓
[12. Tracking & Export]     ────→ SHA-256 manifest, HTML/Markdown report, and Joblib prediction bundle
```

---

## 4. Technical Documentation

For in-depth architectural and mathematical details, consult the dedicated guides in `docs/`:

- [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md): Complete technical blueprint, module boundaries, and design principles.
- [**docs/PREVIA_DEEP_DIVE.md**](docs/PREVIA_DEEP_DIVE.md): Detailed function-by-function reference, inputs, outputs, failure modes, and ML concepts.
- [**docs/BENCHMARKS.md**](docs/BENCHMARKS.md): Reproducible benchmarks comparing Previa vs naive baselines across standard tabular tasks.
- [**docs/CHANGELOG.md**](docs/CHANGELOG.md): Record of changes and enhancements in v1.0.0.

---

## 5. Tech Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Optional Gradient Boosting**: `xgboost`
- **Explainability**: `shap` (with deterministic model feature importance fallbacks)
- **Interactive Visualizations**: `plotly`
- **Web UI**: `streamlit`
- **Model Serialization & Excel**: `joblib`, `openpyxl`
- **Testing**: `pytest`

---

## 6. Installation & Quickstart

### Prerequisites
Ensure Python 3.10+ is installed on your machine.

### 1. Clone & Navigate
```bash
git clone https://github.com/Rishi-11-26/previa.git
cd previa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

### 4. Launch the Streamlit Workbench
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Model Registry & Evaluation

### Supported Models

| Task | Core Baseline Models | Optional Accelerated Model |
| :--- | :--- | :--- |
| **Classification** | Logistic Regression, Decision Tree, Random Forest | XGBoost Classifier |
| **Regression** | Linear Regression, Ridge Regression, Decision Tree, Random Forest | XGBoost Regressor |

### Metrics Computed
- **Classification**: Accuracy, Precision, Recall, $F_1$-score (macro/weighted), ROC-AUC (binary).
- **Regression**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), $R^2$ Score.
- **Adaptive Cross-Validation**:
  - $n \ge 50$: 5-fold CV
  - $20 \le n < 50$: 3-fold CV
  - $10 \le n < 20$: 2-fold CV
  - $n < 10$: CV skipped with stability advisory.

---

## 8. Automated Testing

Previa maintains a comprehensive test suite across 13 test files:

```text
tests/test_benchmarker.py .............. (18 passed)
tests/test_competitive_benchmarks.py .. (2 passed)
tests/test_diagnostics.py ..... (5 passed)
tests/test_experiment_tracker.py ... (3 passed)
tests/test_explainer.py ... (3 passed)
tests/test_exporter.py .......... (10 passed)
tests/test_feature_intelligence.py .................... (20 passed)
tests/test_health.py ................. (17 passed)
tests/test_leakage.py ..... (5 passed)
tests/test_loader.py .......... (10 passed)
tests/test_model_bundle.py .. (2 passed)
tests/test_model_registry.py ...... (6 passed)
tests/test_pipeline_builder.py ......... (9 passed)
tests/test_profiler.py ................ (16 passed)
tests/test_recommender.py ................ (16 passed)
tests/test_report_generator.py .. (2 passed)
tests/test_task_detection.py ............. (13 passed)
tests/test_workflow.py .. (2 passed)
============================ 159 passed in 27.26s =============================
```

---

## 9. Known Limitations & Scope Boundary (v1.0.0)

- **Tabular Scope**: Previa is built for structured tabular data (CSV/XLSX). Unstructured text, audio, and image arrays are explicitly out of scope.
- **Local-First**: No mandatory cloud accounts, billing keys, or paid LLM APIs are required. Everything runs locally.
- **Explainability**: Predictive feature importance quantifies model dependency and loss minimization; it does NOT establish real-world causality.
- **AutoML Boundary**: Previa prioritizes transparent baseline comparison, data health auditing, and leakage-safe preprocessing over black-box deep search.
