# URQEN Architecture & Technical Blueprint

## 1. System Philosophy

URQEN is a tabular machine-learning workbench for data inspection, experiment setup, and result review. Its core engineering tenet is:
> **"Understand the data before trusting the model."**

The platform operates on a local-first, zero-external-API architecture designed to prevent data leakage, expose evidence for every recommendation, transparently evaluate baseline models, and export reproducible inference artifacts.

---

## 2. Complete Workflow Architecture

```
Raw CSV / XLSX Upload
         │
         ▼
[1. Ingestion & Pre-flight Validation] (loader.py)
         │  ├── Format, size & schema checks
         │  └── Empty column, duplicate header & memory estimation
         ▼
[2. Automated Profiling] (profiler.py)
         │  ├── Dimensions, null counts, duplicate rows
         │  └── Quartiles (Q25, Q75), skewness, top category frequencies
         ▼
[3. Evidence-Backed Quality Audit] (health.py)
         │  ├── High missingness, constant columns, identifiers
         │  └── 8-point findings: Evidence, Severity, Why it matters, Trade-offs
         ▼
[4. Data Leakage & Reliability Audit] (leakage.py)
         │  ├── Near-perfect correlation (|r| >= 0.95), suspicious naming
         │  └── Severe class imbalance & micro-sample alerts
         ▼
[5. Target & Task Inference] (detector.py)
         │  ├── Heuristic evidence-based target scoring
         │  ├── Binary, Multiclass, or Regression classification
         │  └── Human-in-the-loop manual override
         ▼
[6. Feature Intelligence] (analyzer.py)
         │  └── Type classification: numeric, boolean, categorical, constant
         ▼
[7. Preprocessing Recommender] (recommender.py)
         │  ├── Deterministic rules: imputation, OHE, scaling, dropping
         │  └── Interactive user acceptance & customization
         ▼
[8. Leakage-Safe Pipeline Construction] (pipeline_builder.py)
         │  └── sklearn ColumnTransformer (fitted strictly on training splits)
         ▼
[9. Centralized Model Benchmarking] (benchmarker.py & models/registry.py)
         │  ├── Catalog: Logistic/Ridge/Linear Regression, Decision Tree, Random Forest, XGBoost
         │  ├── Adaptive cross-validation (5-fold, 3-fold, 2-fold, or warning)
         │  └── Execution timing & failure resilience
         ▼
[10. Interactive Diagnostics] (diagnostics/charts.py)
         │  ├── Classification: Plotly Confusion Matrix & ROC curves
         │  └── Regression: Actual vs. Predicted & Residual plots
         ▼
[11. Explainability Engine] (explainability/explainer.py)
         │  ├── SHAP values (TreeExplainer/LinearExplainer)
         │  ├── Fallbacks: Gini Importance, Permutation Importance
         │  └── Prominent causality disclaimer
         ▼
[12. Tracking, Reporting & Deployment] (tracker.py, report_generator.py, model_bundle.py)
         ├── SHA-256 fingerprinting & reproducible JSON experiment manifest
         ├── Comprehensive HTML & Markdown decision reports
         └── Joblib model bundle export & live inference prediction engine
```

---

## 3. Module Boundaries & Data Flow

| Subpackage | Main Responsibilities | Primary Input | Primary Output |
| :--- | :--- | :--- | :--- |
| `previa.ingestion` | File validation, CSV/XLSX loading, size limits | File path / buffer | `pd.DataFrame`, validation dict |
| `previa.auditing` | Statistical profiling, quality audits, leakage detection | `pd.DataFrame` | Profiler dict, 8-point health issues, leakage warnings |
| `previa.task_detection` | Target heuristic scoring & task inference | `pd.DataFrame` | Target column, task type, confidence, reasons |
| `previa.feature_intelligence` | Feature type classification & distributions | `pd.DataFrame`, target | Feature dictionary |
| `previa.preprocessing` | Rule recommendations, ColumnTransformer builder | Feature analysis, user choices | `ColumnTransformer` blueprint |
| `previa.models` | Algorithmic registry, reproducible instantiations | Task type, model name | Unfitted estimator |
| `previa.benchmarking` | Model evaluation, adaptive CV, timing, winner logic | `X_trans`, `y`, task type | Benchmark dictionary |
| `previa.diagnostics` | Plotly diagnostic figures | `y_true`, `y_pred`, `y_prob` | `plotly.graph_objects.Figure` |
| `previa.explainability` | SHAP & model-derived feature importance | Fitted model, `X`, `y` | Importance dict, Plotly bar chart |
| `previa.experiments` | SHA-256 fingerprinting, manifest assembly | Data & run metadata | JSON experiment manifest |
| `previa.export` | Processed data export, decision reports, model bundles | Trained objects, metadata | CSV, XLSX, Markdown, HTML, `.joblib` |

---

## 4. Key Design Decisions

1. **Strict Data Leakage Prevention**:
   Transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are never fitted on full dataset matrices. In `run_benchmarks`, train/test splitting occurs first; transformers are fitted strictly on `X_train` and evaluated on `X_test`.
2. **Deterministic Over Black-Box**:
   All preprocessing recommendations are derived from statistical rules rather than opaque LLM prompts.
3. **Graceful Degradation**:
   If a model (such as XGBoost) or explainability method (such as SHAP) fails or is uninstalled, URQEN logs the condition and uses available fallbacks where implemented.
4. **Reproducibility by Design**:
   Fixed random seeds (`42`), dataset SHA-256 fingerprinting, and standardized metadata bundles ensure experiments can be audited and reproduced.
