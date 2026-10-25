# URQEN Deep Dive — Module Technical Reference

This document provides a module-by-module breakdown of URQEN's internal architecture, mathematical formulations, and engineering design rationale.

---

## 1. Ingestion (`previa/ingestion/loader.py`)
- **Purpose:** Securely ingest tabular files, validate schemas, and calculate memory footprints.
- **Inputs:** File path string (`.csv` or `.xlsx`), optional `max_size_mb`.
- **Outputs:** `pd.DataFrame` and validation dictionary.
- **Main Functions:** `load_dataset()`, `validate_dataset()`.
- **Key Libraries:** `pandas`, `pathlib`, `openpyxl`.
- **Concepts:** Untrusted file handling, memory estimation (`deep=True`), duplicate column detection.
- **Design Decisions:** Explicitly disallows arbitrary file types to prevent security risks. Never auto-drops columns silently.
- **Failure Cases:** Non-existent paths (`FileNotFoundError`), unsupported extensions (`ValueError`), empty files (`ValueError`), parse failures (`RuntimeError`).

---

## 2. Profiling (`previa/auditing/profiler.py`)
- **Purpose:** Compute summary statistics once for downstream auditing and UI rendering.
- **Inputs:** `pd.DataFrame`.
- **Outputs:** Dictionary with `"summary"` and `"columns"` keys.
- **Main Functions:** `profile_dataset()`.
- **Key Libraries:** `pandas`, `numpy`.
- **Concepts:** Skewness calculation (Fisher-Pearson third moment), quartiles ($Q_1, Q_2, Q_3$), cardinality ratios.
- **Design Decisions:** Returns plain Python primitives (dicts, lists, floats) rather than custom objects for trivial JSON/Streamlit serialization.

---

## 3. Data Quality Audit (`previa/auditing/health.py`)
- **Purpose:** Identify actionable data quality hazards before model training.
- **Inputs:** Profiler dictionary, raw `pd.DataFrame`.
- **Outputs:** List of 8-point finding dicts (Issue, Column, Evidence, Severity, Why it matters, Recommendation, Trade-offs, Requires approval).
- **Main Functions:** `analyze_health()`.
- **Key Libraries:** `pandas`, `numpy`.
- **Concepts:** Tukey's 1.5×IQR outlier fences:
  $$\text{IQR} = Q_3 - Q_1, \quad \text{Lower} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$
- **Design Decisions:** Frames findings as evidence-backed advisories rather than definitive verdicts.

---

## 4. Leakage & Reliability Warnings (`previa/auditing/leakage.py`)
- **Purpose:** Detect features that directly encode target outcomes or signal high experimental variance.
- **Inputs:** `pd.DataFrame`, `target_column`, `task_type`.
- **Outputs:** List of leakage warning dicts.
- **Main Functions:** `audit_leakage_and_reliability()`.
- **Key Libraries:** `pandas`, `numpy`, `re`.
- **Concepts:** Pearson correlation coefficient ($|r| \ge 0.95$), post-event naming tokens, class imbalance ratios.
- **Design Decisions:** Never automatically drops suspected leaky columns; issues prominent warnings requiring user confirmation.

---

## 5. Target & Task Detection (`previa/task_detection/detector.py`)
- **Purpose:** Infer the most likely target column and classify problem type (Binary, Multiclass, Regression).
- **Inputs:** `pd.DataFrame`.
- **Outputs:** Dict with `target_column`, `task_type`, `confidence`, `reasons`, `all_scores`.
- **Main Functions:** `detect_target_and_task()`.
- **Key Libraries:** `pandas`, `numpy`.
- **Concepts:** Heuristic scoring, discrete integer class analysis vs continuous regression floating points.
- **Design Decisions:** Protects against false identifier penalties on continuous regression targets. Always provides manual override in the UI.

---

## 6. Feature Intelligence (`previa/feature_intelligence/analyzer.py`)
- **Purpose:** Characterize individual feature roles, datatypes, and distribution properties.
- **Inputs:** `pd.DataFrame`, optional `target_column`.
- **Outputs:** Dictionary mapping each column name to its detailed profile.
- **Main Functions:** `analyze_features()`.
- **Concepts:** Boolean value set matching ({0,1}, {yes,no}, {true,false}), categorical mode extraction.

---

## 7. Preprocessing Recommender (`previa/preprocessing/recommender.py`)
- **Purpose:** Provide deterministic, explainable recommendations for every feature.
- **Inputs:** Feature analysis dictionary, target column name.
- **Outputs:** List of recommendation dicts including why it matters and trade-offs.
- **Main Functions:** `generate_recommendations()`.
- **Concepts:** Robust median imputation (insensitive to outliers) vs mode imputation; one-hot encoding vs high-cardinality review.
- **Design Decisions:** Rule-based and deterministic — no external LLMs or black-box heuristics.

---

## 8. Leakage-Safe Pipeline Builder (`previa/preprocessing/pipeline_builder.py`)
- **Purpose:** Assemble a scikit-learn `ColumnTransformer` that is fitted strictly on training data.
- **Inputs:** Recommendations list, optional `user_overrides`, optional `scale_numeric`.
- **Outputs:** `(ColumnTransformer, numeric_features, categorical_features, dropped_features)`.
- **Main Functions:** `build_pipeline()`.
- **Key Libraries:** `sklearn.pipeline.Pipeline`, `sklearn.compose.ColumnTransformer`, `sklearn.impute.SimpleImputer`, `sklearn.preprocessing.StandardScaler`, `sklearn.preprocessing.OneHotEncoder`.
- **Concepts:** Data leakage prevention, unknown category handling (`handle_unknown='ignore'`).

---

## 9. Model Registry & Benchmarking (`previa/models/registry.py` & `benchmarker.py`)
- **Purpose:** Catalog and benchmark standard baseline models across hold-out splits and cross-validation.
- **Inputs:** `X_trans` (numpy array), `y` (target array), `task_type`.
- **Outputs:** Benchmark results dictionary with per-model scores, timing, and winner explanation.
- **Main Functions:** `get_available_models()`, `create_model_instance()`, `run_benchmarks()`.
- **Key Libraries:** `scikit-learn`, optional `xgboost`.
- **Concepts:**
  - Adaptive cross-validation:
    - $n \ge 50$: 5-fold CV
    - $20 \le n < 50$: 3-fold CV
    - $10 \le n < 20$: 2-fold CV
    - $n < 10$: CV skipped with caution warning
  - Primary metric ranking (F1 for classification, $R^2$ for regression).

---

## 10. Explainability (`previa/explainability/explainer.py`)
- **Purpose:** Compute global feature importance via SHAP values with seamless fallback to native model weights.
- **Inputs:** Fitted model, evaluation matrix `X`, target `y`, feature names.
- **Outputs:** Importance dictionary and Plotly bar chart figure.
- **Main Functions:** `compute_feature_importance()`, `plot_feature_importance_chart()`.
- **Key Libraries:** `shap`, `sklearn.inspection.permutation_importance`, `plotly`.
- **Concepts:** Cooperative game theory Shapley values vs Gini impurity reduction. Causality vs predictive correlation.

---

## 11. Experiment Tracking & Model Bundle (`previa/experiments/tracker.py` & `model_bundle.py`)
- **Purpose:** Maintain reproducible experiment manifests and export deployable Joblib inference artifacts.
- **Inputs:** Fitted pipelines, trained models, schema metadata, new inference data.
- **Outputs:** JSON manifest string, `.joblib` model bundle, prediction `pd.DataFrame`.
- **Main Functions:** `create_experiment_manifest()`, `save_model_bundle()`, `load_model_bundle()`.
- **Concepts:** SHA-256 dataset fingerprinting, model serialization, inference schema validation.
