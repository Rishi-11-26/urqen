# URQEN

**Trustworthy Tabular Machine Learning Experimentation & Decision Support Platform**

> *"Understand the data before trusting the model."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20REST-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-168%20Passed%20(100%25)-brightgreen.svg)]()
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

**URQEN** is a tabular machine-learning workbench for inspecting data, configuring experiments, comparing model results, and reviewing available diagnostics.

---

## 2. Core Differentiators & Product Principles

1. **Both Guided & Expert Modes are 100% Free**: No paywalls, no fake payments, and no artificial restriction of Expert Mode.
2. **Evidence-Backed Data Intelligence**: Identifies data problems, presents measured statistical evidence, recommends actions, explains trade-offs, and prompts for human review across an 8-point finding schema.
3. **25 Algorithmic Baselines Across 5 Families**: Benchmarking spans Linear Models, Tree Ensembles, Distance/Kernel Models, Probabilistic Networks (Naive Bayes & RBFN), and modern Gradient Boosting (XGBoost).
4. **Interactive Validation & Hyperparameter Control**:
   - Customizable Cross-Validation folds (2 to 20, default 10 with adaptive small-dataset fallbacks).
   - Adjustable Holdout Test Split (10% to 40%, default 20%).
   - Numerical Scaler selection (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`).
   - Hyperparameter fine-tuning for XGBoost (`learning_rate`, `max_depth`, `n_estimators`, `min_child_weight`), Random Forest, Decision Tree, SVM (`C`), and Logistic Regression.
5. **Transparent Model Decisions & Explainability**: Explains why a model was selected, how it performed, which features influenced its predictions (via SHAP or robust model fallbacks), and attaches scientific causality disclaimers.
6. **Deployable Provenance & Real-Time Inference**: Generates unique experiment IDs, SHA-256 dataset fingerprints, downloadable JSON manifests, self-contained `.joblib` model bundles, and provides a built-in browser Inference Sandbox.

---

## 3. Technology Stack

- **Frontend**: React 19, Vite, Lucide Icons, custom CSS design system.
- **Backend API**: Python FastAPI, Pydantic v2 schemas, REST endpoints, Background Tasks.
- **ML Core Engine**: Python (`previa/`), Pandas, NumPy, scikit-learn, XGBoost, SHAP, Joblib.
- **Authentication and cloud administration**: Supabase Auth and optional Supabase database/storage administration.
- **Testing**: `pytest` unit and integration tests.

---

## 4. End-to-End Architecture

```
Raw CSV / XLSX
      ↓
[1. Ingestion & Validation] ────→ Format, empty-column & duplicate header checks
      ↓
[2. Automated Profiler]     ────→ Summary counts, memory, quartiles & category frequencies
      ↓
[3. Evidence-Backed Health] ────→ 8-point findings: Evidence, severity, why it matters, trade-offs
      ↓
[4. Leakage & Reliability]  ────→ Correlation (|r| >= 0.95), suspicious naming, class imbalance
      ↓
[5. Target & Task Detection]────→ Inferred target & task (binary, multiclass, regression) with candidate override
      ↓
[6. Feature Intelligence]   ────→ Type classification (numeric, boolean, categorical, constant)
      ↓
[7. Preprocessing Engine]   ────→ Rule-based recommendations with user acceptance & customization
      ↓
[8. Leakage-Safe Pipeline]  ────→ ColumnTransformer blueprint (fitted strictly on training splits)
      ↓
[9. Centralized Benchmark]  ────→ 25 Model Registry with CV folds, split slider, scalers & hyperparams
      ↓
[10. Interactive Diagnostics]───→ Confusion Matrix heatmap, Holdout evaluation & leaderboards
      ↓
[11. Explainability Engine] ────→ SHAP values & model feature importance with causality disclaimer
      ↓
[12. Tracking & Live Sandbx]────→ SHA-256 manifest, HTML/Markdown reports, .joblib bundle & browser inference
```

---

## 5. Quickstart & Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ (tested on Node 24)

### 1. Configure the environment
Copy `.env.example` to `.env` and set the Supabase values used by the backend. For the frontend, create `frontend/.env` with:

```dotenv
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-public-key
```

Use the Supabase **Project URL** as `VITE_SUPABASE_URL` (do not append `/rest/v1` or `/auth/v1`).

Never put the Supabase service-role key in a frontend environment variable or commit either `.env` file.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

```bash
cd frontend
npm ci
cd ..
```

### 3. Run automated tests
```bash
python -m pytest -v
```

### 4. Launch URQEN Full-Stack Application

#### Option A: FastAPI Server with Built-in Frontend (Recommended)
Build the frontend once and serve everything from FastAPI:
```bash
cd frontend
npm ci
npm run build
cd ..
uvicorn backend.main:app --reload --port 8000
```
Open **`http://localhost:8000`**. FastAPI serves the API, OpenAPI docs at `/docs`, and the built frontend.

#### Option B: Independent Vite Hot-Reloading Dev Server
For active frontend development with Instant HMR:
```bash
# Terminal 1: Backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```
Open **`http://localhost:5173`** (Vite proxies all `/api` calls directly to FastAPI on port 8000).

---

## 6. Deploy on Render

The repository includes a root-level `render.yaml` Blueprint and a multi-stage `Dockerfile`. In Render, create a **New → Blueprint** deployment, connect this GitHub repository, and apply the Blueprint. Supply these environment variables when prompted:

- `SUPABASE_URL` and `SUPABASE_KEY` (the Supabase project URL and anon/public key)
- `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` (same URL and anon/public key; used to build the frontend)
- `SUPABASE_SERVICE_ROLE_KEY` (backend only; keep secret)
- `SUPABASE_JWT_SECRET`
- `ADMIN_EMAILS` (comma-separated founder email allowlist)

If the URQEN tables have not been created, run `supabase_schema.sql` in the Supabase SQL Editor. Set the Supabase Auth site URL to the deployed Render URL and add the relevant redirect URLs in Supabase Auth settings.

**Storage note:** Render's filesystem is ephemeral by default. URQEN's local dataset and artifact files can be lost when the service restarts or redeploys. Use durable cloud storage or configure a persistent disk before storing production user data; persistent disks require a paid Render service.

---

## 7. Model Catalog & Evaluation

### Supported Algorithms (25 Models)

| Algorithmic Family | Classification Algorithms | Regression Algorithms |
| :--- | :--- | :--- |
| **Linear Models** | Logistic Regression, Ridge Classifier, SGD Classifier | Linear Regression, Ridge, Lasso, ElasticNet |
| **Tree & Ensembles** | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost | Decision Tree Regressor, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor, AdaBoost Regressor |
| **Distance & Kernels** | K-Nearest Neighbors, Linear SVM, Kernel SVM (RBF) | K-Nearest Neighbors Regressor, Support Vector Regressor |
| **Probabilistic / Networks**| Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Radial Basis Function Network (RBFN) | Radial Basis Function Regressor |
| **Modern Boosting** | XGBoost Classifier | XGBoost Regressor |

---

## 8. Automated Test Suite

The automated tests cover ML correctness, leakage prevention, edge-case safety, API contracts, administrative cleanup, and serializability. Run the current suite with `python -m pytest -v`.


---
