# URQEN Changelog

## [1.0.0] — Official Final Release

### Added
- **Pre-flight Dataset Validation**:
  - File size safety limits (`max_size_mb`).
  - Detection of duplicate headers, unnamed columns, and completely null features.
  - Deep memory calculation and formatting in bytes/KB/MB.
- **Enhanced Profiling**:
  - Numeric quartiles ($Q_1$ and $Q_3$).
  - Top category distribution breakdown for categorical features.
  - Datetime column recognition.
- **Evidence-Backed 8-Point Data Health Audit**:
  - Structured fields: Issue Name, Affected Column, Evidence, Severity, Why It Matters, Recommended Action, Trade-offs, and User Approval requirements.
- **Data Leakage & Reliability Warnings**:
  - Near-perfect target correlation detection ($|r| \ge 0.95$).
  - Target-derived and post-event feature name pattern recognition.
  - Severe class imbalance alerts.
  - Micro-sample size warnings ($< 25$ rows).
- **Centralized Model Registry**:
  - Catalog for classification: Logistic Regression, Decision Tree, Random Forest, XGBoost (optional).
  - Catalog for regression: Linear Regression, Ridge, Decision Tree, Random Forest, XGBoost (optional).
  - Algorithmic metadata: family, scale-sensitivity, training duration tracking.
  - Failure resilience: single model exception does not halt remaining models.
- **Explainability Engine**:
  - SHAP value explanations for Tree and Linear models.
  - Transparent fallback to Model Feature Importances (Gini/weights) and Permutation Importance.
  - Prominent scientific causality disclaimer.
- **Reproducible Experiment Manifest**:
  - Dataset SHA-256 fingerprinting without leaking sensitive row data.
  - Full tracking of schema, target, task, preprocessing configurations, and software versions.
  - Downloadable JSON experiment manifest.
- **Decision Report Generator**:
  - Self-contained downloadable HTML and Markdown reports covering all 11 experiment sections.
- **Model Bundle Export & Inference Prediction**:
  - Joblib-based serialization of combined fitted `ColumnTransformer` + estimator.
  - Interactive prediction interface supporting CSV upload, schema verification, and probability estimation.
- **Competitive Benchmark Suite**:
  - Reproducible benchmark across Binary, Multiclass, and Regression datasets documented in `docs/BENCHMARKS.md`.
- **Comprehensive Documentation Suite**:
  - `docs/ARCHITECTURE.md`
  - `docs/URQEN_DEEP_DIVE.md`
  - `docs/BENCHMARKS.md`
  - `docs/CHANGELOG.md`

### Enhanced
- Expanded automated test suite from 136 to 159 tests (100% passing).
- Leakage-safe pipeline builder now accepts interactive user overrides and model-specific scaling toggles.
- Upgraded Streamlit UI with complete 12-page guided workflow, state management, and real-time inference prediction.
