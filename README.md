# Previa

**Trustworthy Pre-Model Data Intelligence Workbench**

> *"Understand the data before trusting the model."*

---

## 1. Overview & Problem Statement

In practical machine learning, models fail not because algorithms are weak, but because datasets are misunderstood, corrupted, or contaminated with data leakage. Beginners and experienced practitioners alike often rush to train complex models while missing critical issues:
- **Data Leakage**: Preprocessing (imputation, scaling, encoding) fitted on the entire dataset rather than strictly on training splits.
- **Silent Failures**: High missingness, constant columns, or high-cardinality features silently distorting weights.
- **Blind AutoML**: Tools that treat data preparation as an opaque black box, leaving users unable to explain *why* transformations were chosen.
- **Overconfident Target Inferences**: Hardcoding assumptions about target labels without human verification.

**Previa** is an explainable, leakage-safe data intelligence workbench designed to audit, prepare, benchmark, and validate tabular datasets before machine learning models are deployed. Previa automates repetitive pre-modeling tasks while exposing evidence, confidence levels, and clear reasoning for every recommendation.

---

## 2. Core Principles

1. **Explainable Over Clever**: Recommendations are generated using deterministic rules and statistical evidence, not opaque black-box APIs.
2. **Leakage-Safe by Design**: All preprocessing pipelines (`scikit-learn` `Pipeline` + `ColumnTransformer`) are fitted strictly on training data splits.
3. **Evidence-Based Warnings**: Data quality issues are flagged with explicit statistical evidence and severity badges (High, Medium, Low), not alarmist verdicts.
4. **Human-in-the-Loop**: Automated target and task inferences provide heuristic confidence and can be manually confirmed or overridden.
5. **Honest Benchmarking**: Evaluates standard baselines with adaptive cross-validation for small datasets. Models are reported as *"best among evaluated baselines according to the selected metric"*, never "objectively optimal."

---

## 3. Architecture & 10-Step Workflow

```
Raw CSV / XLSX
      ↓
[1. Ingestion]  ──────→ Validates format, shape, and integrity
      ↓
[2. Profiler]   ──────→ Computes dataset & per-column statistics
      ↓
[3. Health]     ──────→ Audits missingness, duplicates, outliers, IDs, constants
      ↓
[4. Detection]  ──────→ Infers target column & ML task type (with manual override)
      ↓
[5. Features]   ──────→ Classifies types (numeric, boolean, categorical, constant)
      ↓
[6. Recommender] ─────→ Generates rule-based, evidence-backed transformations
      ↓
[7. Pipeline]   ──────→ Assembles sklearn ColumnTransformer (fitted strictly on train folds)
      ↓
[8. Benchmark]  ──────→ Trains baselines with small-dataset adaptive cross-validation
      ↓
[9. Diagnostics] ─────→ Generates Plotly charts (Confusion Matrix, ROC, Residuals)
      ↓
[10. Export]    ──────→ Exports verified Processed / Transformed Dataset (CSV / XLSX)
```

---

## 4. Tech Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Interactive Visualizations**: `plotly`
- **Web UI**: `streamlit`
- **Spreadsheet Support**: `openpyxl`
- **Testing**: `pytest`

---

## 5. Repository Structure

```
PREVIA/
├── previa/                         # Core Python package
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py               # Robust CSV & XLSX loading
│   ├── auditing/
│   │   ├── __init__.py
│   │   ├── profiler.py             # Dataset & column profiling
│   │   └── health.py               # Data quality audits
│   ├── task_detection/
│   │   ├── __init__.py
│   │   └── detector.py             # Target & task inference
│   ├── feature_intelligence/
│   │   ├── __init__.py
│   │   └── analyzer.py             # Individual feature characterization
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── recommender.py          # Explainable recommendations engine
│   │   └── pipeline_builder.py     # Leakage-safe ColumnTransformer
│   ├── benchmarking/
│   │   ├── __init__.py
│   │   └── benchmarker.py          # Adaptive baseline model evaluation
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   └── charts.py               # Plotly diagnostic figures
│   └── export/
│       ├── __init__.py
│       └── exporter.py             # Processed dataset export
│
├── app/
│   └── app.py                      # Guided 10-step Streamlit application
│
├── tests/
│   ├── test_loader.py              # Ingestion tests
│   ├── test_profiler.py            # Profiling tests
│   ├── test_health.py              # Health audit tests
│   ├── test_task_detection.py      # Target detection tests
│   ├── test_feature_intelligence.py# Feature intelligence tests
│   ├── test_recommender.py         # Recommendation engine tests
│   ├── test_pipeline_builder.py    # Pipeline construction tests
│   ├── test_exporter.py            # Export tests
│   ├── test_benchmarker.py         # Model benchmarking tests
│   ├── test_diagnostics.py         # Diagnostic charts tests
│   └── test_workflow.py            # End-to-end integration tests
│
├── examples/
│   └── sample.csv                  # Multi-type sample dataset
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## 6. Installation & Quickstart

### Prerequisites
Make sure Python 3.10+ is installed on your machine.

### 1. Clone & Navigate
```bash
git clone <your-repo-url>
cd PREVIA
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

### 4. Launch the Streamlit Interface
```bash
streamlit run app/app.py
```

---

## 7. Baseline Models & Evaluation Metrics

### Classification
- **Models**: Logistic Regression, Decision Tree, Random Forest
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC (binary classification)
- **Cross-Validation**: 5-fold (stratified where applicable), automatically adjusted to 3-fold or 2-fold for smaller datasets.

### Regression
- **Models**: Linear Regression, Ridge Regression, Decision Tree Regressor, Random Forest Regressor
- **Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), $R^2$ Score
- **Cross-Validation**: 5-fold, automatically adjusted to 3-fold or 2-fold for smaller datasets.

---

## 8. Small Dataset Robustness

When auditing small datasets (such as academic or experimental datasets):
- Datasets with $\ge 50$ samples utilize standard 5-fold cross-validation.
- Datasets with $20 \le n < 50$ samples adaptively switch to 3-fold CV to prevent micro-fold variance.
- Datasets with $10 \le n < 20$ samples switch to 2-fold CV and display a prominent stability advisory.
- Datasets with $< 10$ samples skip cross-validation and display an explicit caution message rather than failing or reporting misleading numbers.

---

## 9. Limitations & Scope Boundary (V1)

- **V1 Scope**: Previa focuses strictly on tabular data (CSV/XLSX). It does not handle audio, images, or unstructured text.
- **AutoML**: Previa is intentionally not a black-box AutoML system; it builds transparent baselines rather than complex hyperparameter optimization.
- **Model Explainability**: Detailed SHAP/LIME feature attributions and inference deployment are reserved for future major versions.

---

## 10. Key Engineering & Data Science Concepts

- **Data Leakage Prevention**: Never fitting scalers or encoders before splitting data.
- **Heuristic vs. Calibrated Confidence**: Explicitly separating deterministic heuristics from empirical statistical certainty.
- **ColumnTransformer Remainder**: Explicitly dropping unused, identifier, or constant columns to prevent dimensionality explosion.
- **Robust Imputation**: Choosing median over mean for skewed numeric distributions to minimize outlier distortion.
