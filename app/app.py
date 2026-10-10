"""
app/app.py
----------
Streamlit Application for Previa: Trustworthy Pre-Model Data Intelligence Workbench.

Complete 12-Step Guided Workflow:
  STEP 1: Ingestion & Pre-flight Validation
  STEP 2: Automated Profiler
  STEP 3: Evidence-Backed Health Audit
  STEP 4: Leakage & Reliability Warnings
  STEP 5: Target & Task Detection
  STEP 6: Feature Intelligence
  STEP 7: Preprocessing & Recommendations (with User Overrides)
  STEP 8: Leakage-Safe Pipeline Blueprint
  STEP 9: Model Training & Centralized Benchmarking
  STEP 10: Interactive Diagnostics & Evaluation
  STEP 11: Explainability & SHAP Feature Importance
  STEP 12: Manifest, Decision Report & Live Inference Prediction
"""

import os
import sys
import tempfile
import pathlib
import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from previa.ingestion.loader import load_dataset, validate_dataset
from previa.auditing.profiler import profile_dataset
from previa.auditing.health import analyze_health
from previa.auditing.leakage import audit_leakage_and_reliability
from previa.task_detection.detector import detect_target_and_task
from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations
from previa.preprocessing.pipeline_builder import build_pipeline
from previa.models.registry import get_available_models, is_xgboost_available
from previa.benchmarking.benchmarker import run_benchmarks
from previa.diagnostics.charts import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_actual_vs_predicted,
    plot_residuals,
)
from previa.explainability.explainer import (
    compute_feature_importance,
    plot_feature_importance_chart,
)
from previa.experiments.tracker import (
    create_experiment_manifest,
    serialize_manifest_to_json,
)
from previa.export.report_generator import (
    generate_markdown_report,
    generate_html_report,
)
from previa.export.model_bundle import (
    save_model_bundle,
    load_model_bundle,
    PreviaModelBundle,
)
from previa.export.exporter import export_dataframe


# --- Page Configuration ---
st.set_page_config(
    page_title="Previa — Data Intelligence Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Previa")
st.caption("Trustworthy Pre-Model Data Intelligence Workbench — *Understand the data before trusting the model.*")


def get_sample_path():
    base = pathlib.Path(__file__).resolve().parent.parent
    sample = base / "examples" / "sample.csv"
    return str(sample) if sample.exists() else None


# --- Sidebar Navigation ---
st.sidebar.header("Workflow Steps")
steps = [
    "1. Ingestion & Validation",
    "2. Automated Profiler",
    "3. Evidence-Backed Health Audit",
    "4. Leakage & Reliability Warnings",
    "5. Target & Task Detection",
    "6. Feature Intelligence",
    "7. Preprocessing & Recommendations",
    "8. Leakage-Safe Pipeline",
    "9. Model Training & Comparison",
    "10. Diagnostics & Evaluation",
    "11. Explainability & SHAP",
    "12. Manifest, Report & Prediction",
]
selected_step = st.sidebar.radio("Navigate", steps)

# --- Session State Initialization ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "validation_report" not in st.session_state:
    st.session_state.validation_report = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "health_issues" not in st.session_state:
    st.session_state.health_issues = None
if "leakage_warnings" not in st.session_state:
    st.session_state.leakage_warnings = None
if "target_detection" not in st.session_state:
    st.session_state.target_detection = None
if "selected_target" not in st.session_state:
    st.session_state.selected_target = None
if "selected_task" not in st.session_state:
    st.session_state.selected_task = None
if "feature_analysis" not in st.session_state:
    st.session_state.feature_analysis = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "user_overrides" not in st.session_state:
    st.session_state.user_overrides = {}
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "eval_artifacts" not in st.session_state:
    st.session_state.eval_artifacts = None
if "explainability_data" not in st.session_state:
    st.session_state.explainability_data = None
if "manifest" not in st.session_state:
    st.session_state.manifest = None


# ==============================================================================
# STEP 1: Ingestion & Validation
# ==============================================================================
if selected_step.startswith("1."):
    st.header("Step 1: Ingestion & Pre-flight Validation")
    st.markdown("Upload a **CSV** or **XLSX** dataset to audit integrity, validate schema, and calculate memory footprints.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])
    with col2:
        st.markdown("**Or load built-in sample:**")
        sample_path = get_sample_path()
        if sample_path and st.button("Load Built-in Sample Dataset"):
            try:
                df = load_dataset(sample_path)
                st.session_state.raw_df = df
                st.session_state.dataset_name = "sample.csv"
                st.session_state.validation_report = validate_dataset(df)
                # Invalidate downstream state
                st.session_state.profile = None
                st.session_state.health_issues = None
                st.session_state.leakage_warnings = None
                st.session_state.target_detection = None
                st.session_state.pipeline_result = None
                st.session_state.benchmark_results = None
                st.session_state.explainability_data = None
                st.session_state.manifest = None
                st.success("Loaded built-in sample dataset successfully!")
            except Exception as e:
                st.error(f"Error loading sample: {e}")

    if uploaded_file is not None:
        try:
            suffix = pathlib.Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            df = load_dataset(tmp_path)
            st.session_state.raw_df = df
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.validation_report = validate_dataset(df)
            # Invalidate downstream state
            st.session_state.profile = None
            st.session_state.health_issues = None
            st.session_state.leakage_warnings = None
            st.session_state.target_detection = None
            st.session_state.pipeline_result = None
            st.session_state.benchmark_results = None
            st.session_state.explainability_data = None
            st.session_state.manifest = None
            st.success(f"Successfully ingested '{uploaded_file.name}'!")
        except Exception as e:
            st.error(f"Failed to ingest dataset: {e}")
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        v = st.session_state.validation_report or validate_dataset(df)

        st.subheader("Pre-flight Validation Report")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{v['n_rows']:,}")
        c2.metric("Columns", f"{v['n_cols']:,}")
        c3.metric("Numeric / Categorical", f"{v['n_numeric']} / {v['n_categorical']}")
        c4.metric("Estimated Memory", v["memory_human"])

        if v["warnings"]:
            for w in v["warnings"]:
                st.warning(f"⚠️ {w}")
        else:
            st.success("✅ Pre-flight checks passed: No duplicate headers or empty columns detected.")

        st.divider()
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)


# ==============================================================================
# STEP 2: Automated Profiler
# ==============================================================================
elif selected_step.startswith("2."):
    st.header("Step 2: Automated Dataset Profiling")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        if st.session_state.profile is None:
            st.session_state.profile = profile_dataset(df)

        profile = st.session_state.profile
        summary = profile["summary"]

        st.subheader("Dataset Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{summary['n_rows']:,}")
        c2.metric("Total Columns", f"{summary['n_cols']:,}")
        c3.metric("Duplicate Rows", f"{summary['n_duplicates']} ({summary['duplicate_pct']}%)")
        c4.metric("Total Missing Cells", f"{summary['total_missing']} ({summary['overall_missing_pct']}%)")

        st.subheader("Detailed Column Profiles & Quartiles")
        col_records = []
        for col_name, info in profile["columns"].items():
            col_records.append({
                "Column": col_name,
                "Type": info["dtype"],
                "Missing Count": info["missing_count"],
                "Missing %": f"{info['missing_pct']}%",
                "Unique Count": info["unique_count"],
                "Unique %": f"{info['unique_pct']}%",
                "Mean": info["mean"],
                "Median": info["median"],
                "Q25": info.get("q25"),
                "Q75": info.get("q75"),
                "Skewness": info["skewness"],
                "Is Constant": info.get("is_constant", False),
            })
        st.dataframe(pd.DataFrame(col_records), use_container_width=True)


# ==============================================================================
# STEP 3: Evidence-Backed Health Audit
# ==============================================================================
elif selected_step.startswith("3."):
    st.header("Step 3: Evidence-Backed Data Health Audit")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        if st.session_state.profile is None:
            st.session_state.profile = profile_dataset(df)

        if st.session_state.health_issues is None:
            st.session_state.health_issues = analyze_health(st.session_state.profile, df)

        issues = st.session_state.health_issues

        if not issues:
            st.success("✅ No critical data health issues identified! The dataset appears ready for modeling.")
        else:
            st.markdown(f"Identified **{len(issues)}** data health finding(s). Every finding includes measured evidence and actionable trade-offs.")
            for issue in issues:
                sev = issue["severity"].lower()
                badge = "🔴 HIGH" if sev == "high" else ("🟡 MEDIUM" if sev == "medium" else "🔵 LOW")
                col_label = f"Column: `{issue['column']}`" if issue["column"] else "Dataset-Level"
                approval_tag = "⚠️ Human Review Recommended" if issue.get("requires_approval") else "ℹ️ Informational"

                with st.expander(f"{badge} | {issue['issue_type'].replace('_', ' ').title()} ({col_label}) — {approval_tag}", expanded=(sev == "high")):
                    st.markdown(f"**Description:** {issue['message']}")
                    st.caption(f"**Measured Evidence:** `{issue['evidence']}`")
                    st.markdown(f"**Why It Matters:** {issue.get('why_it_matters', 'Affects model generalization.')}")
                    st.markdown(f"**Recommended Action:** {issue.get('recommendation', 'Review before training.')}")
                    st.info(f"⚖️ **Trade-offs:** {issue.get('trade_offs', 'None documented.')}")


# ==============================================================================
# STEP 4: Leakage & Reliability Warnings
# ==============================================================================
elif selected_step.startswith("4."):
    st.header("Step 4: Data Leakage & Reliability Warnings")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        if st.session_state.target_detection is None:
            st.session_state.target_detection = detect_target_and_task(df)

        target = st.session_state.selected_target or st.session_state.target_detection["target_column"]
        task = st.session_state.selected_task or st.session_state.target_detection["task_type"]

        st.session_state.leakage_warnings = audit_leakage_and_reliability(df, target_column=target, task_type=task)
        leak_warns = st.session_state.leakage_warnings

        if not leak_warns:
            st.success("✅ No immediate data leakage or extreme experimental instability detected.")
        else:
            st.markdown(f"Flagged **{len(leak_warns)}** potential data leakage or evaluation reliability concern(s):")
            for w in leak_warns:
                badge = "🔴 HIGH RISK" if w["severity"] == "high" else ("🟡 MEDIUM" if w["severity"] == "medium" else "🔵 LOW")
                feat_str = f"Feature: `{w['feature']}`" if w["feature"] else "Dataset-Level"
                with st.expander(f"{badge} | {w['warning_type'].replace('_', ' ').title()} ({feat_str})", expanded=True):
                    st.markdown(f"**Warning Notice:** {w['message']}")
                    st.caption(f"**Measured Evidence:** `{w['evidence']}`")
                    st.markdown(f"**Recommended Action:** {w['recommendation']}")


# ==============================================================================
# STEP 5: Target & Task Detection
# ==============================================================================
elif selected_step.startswith("5."):
    st.header("Step 5: Target Column & ML Task Detection")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        if st.session_state.target_detection is None:
            st.session_state.target_detection = detect_target_and_task(df)

        det = st.session_state.target_detection

        st.subheader("Automated Detection Heuristics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Inferred Target", det["target_column"] or "None detected")
        col2.metric("Inferred Task", det["task_type"].replace('_', ' ').title())
        col3.metric("Heuristic Confidence", det["confidence"].upper())

        st.markdown("**Evidence & Reasons:**")
        for reason in det["reasons"]:
            st.markdown(f"- {reason}")

        st.divider()
        st.subheader("Human-in-the-Loop Override")
        st.caption("Previa never treats automated detection as infallible. Verify or customize below:")

        all_cols = list(df.columns)
        default_target_idx = all_cols.index(det["target_column"]) if det["target_column"] in all_cols else 0
        chosen_target = st.selectbox("Designate Target Column", all_cols, index=default_target_idx)

        task_options = ["binary_classification", "multiclass_classification", "regression"]
        default_task_idx = task_options.index(det["task_type"]) if det["task_type"] in task_options else 0
        chosen_task = st.selectbox(
            "Designate Task Type",
            task_options,
            index=default_task_idx,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        st.session_state.selected_target = chosen_target
        st.session_state.selected_task = chosen_task
        st.success(f"Active Configuration: Target = **`{chosen_target}`** | Task = **`{chosen_task.replace('_', ' ').title()}`**")


# ==============================================================================
# STEP 6: Feature Intelligence
# ==============================================================================
elif selected_step.startswith("6."):
    st.header("Step 6: Feature Intelligence")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target or df.columns[-1]

        st.session_state.feature_analysis = analyze_features(df, target_column=target)
        features = st.session_state.feature_analysis

        st.markdown(f"Characterizing **{len(features)}** feature(s) individually prior to preprocessing.")
        feat_records = []
        for col_name, info in features.items():
            feat_records.append({
                "Feature": col_name,
                "Role": "Target" if info["is_target"] else "Feature",
                "Detected Type": info["feature_type"],
                "Missing %": f"{info['missing_pct']}%",
                "Unique Count": info["unique_count"],
                "Unique %": f"{info['unique_pct']}%",
                "Mean / Top Category": info["mean"] if info["mean"] is not None else info["top_value"],
                "Std / Top Freq": info["std"] if info["std"] is not None else info["top_freq"],
                "Skewness": info["skewness"],
            })
        st.dataframe(pd.DataFrame(feat_records), use_container_width=True)


# ==============================================================================
# STEP 7: Preprocessing & Recommendations
# ==============================================================================
elif selected_step.startswith("7."):
    st.header("Step 7: Explainable Preprocessing Recommendations")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target or df.columns[-1]

        if st.session_state.feature_analysis is None:
            st.session_state.feature_analysis = analyze_features(df, target_column=target)

        st.session_state.recommendations = generate_recommendations(
            st.session_state.feature_analysis, target_column=target
        )
        recs = st.session_state.recommendations

        st.markdown("Each recommendation is generated using **deterministic statistical rules**.")

        rec_table = []
        for r in recs:
            rec_table.append({
                "Feature": r["feature"],
                "Type": r["detected_type"],
                "Detected Condition": r["issue"],
                "Recommended Action": r["recommendation"],
                "Trade-offs": r.get("trade_offs", ""),
                "Confidence": r["confidence"].upper(),
            })
        st.dataframe(pd.DataFrame(rec_table), use_container_width=True)

        st.subheader("Customize or Override Actions")
        st.caption("You have full control to override Previa's default recommendation for any column:")

        overrides = st.session_state.user_overrides
        cols_to_customize = [r["feature"] for r in recs if r["recommendation"] != "keep_as_is"]
        for feat in cols_to_customize:
            default_act = next(r["recommendation"] for r in recs if r["feature"] == feat)
            opts = ["default (" + default_act + ")", "drop", "keep_as_is"]
            chosen = st.selectbox(f"Action for `{feat}`", opts, index=0, key=f"ov_{feat}")
            if chosen == "drop":
                overrides[feat] = "drop"
            elif chosen == "keep_as_is":
                overrides[feat] = "keep_as_is"
            else:
                overrides.pop(feat, None)
        st.session_state.user_overrides = overrides


# ==============================================================================
# STEP 8: Leakage-Safe Pipeline
# ==============================================================================
elif selected_step.startswith("8."):
    st.header("Step 8: Leakage-Safe Pipeline Construction")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target or df.columns[-1]

        if st.session_state.recommendations is None:
            if st.session_state.feature_analysis is None:
                st.session_state.feature_analysis = analyze_features(df, target_column=target)
            st.session_state.recommendations = generate_recommendations(
                st.session_state.feature_analysis, target_column=target
            )

        try:
            pipeline, num_feats, cat_feats, dropped_feats = build_pipeline(
                st.session_state.recommendations,
                user_overrides=st.session_state.user_overrides,
            )
            st.session_state.pipeline_result = {
                "pipeline": pipeline,
                "num_feats": num_feats,
                "cat_feats": cat_feats,
                "dropped_feats": dropped_feats,
            }

            st.success("✅ ColumnTransformer pipeline blueprint constructed successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Numeric Pipeline**")
                st.caption("Median Imputation → StandardScaler")
                st.write(num_feats if num_feats else "None")
            with col2:
                st.markdown("**Categorical Pipeline**")
                st.caption("Mode Imputation → OneHotEncoder(ignore)")
                st.write(cat_feats if cat_feats else "None")
            with col3:
                st.markdown("**Excluded Features**")
                st.caption("Dropped by recommendation or user override")
                st.write(dropped_feats if dropped_feats else "None")

            st.info(
                "🛡️ **Leakage Safety Guarantee**: This pipeline is purely a blueprint right now. "
                "Fitting occurs strictly on training folds during benchmarking."
            )
        except Exception as e:
            st.error(f"Error constructing pipeline: {e}")


# ==============================================================================
# STEP 9: Model Training & Comparison
# ==============================================================================
elif selected_step.startswith("9."):
    st.header("Step 9: Baseline Model Training & Comparison")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    elif st.session_state.pipeline_result is None:
        st.warning("Please build the pipeline in Step 8 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target or df.columns[-1]
        task = st.session_state.selected_task or "binary_classification"
        pipe_res = st.session_state.pipeline_result

        # Optional model checkboxes (core models always included)
        st.markdown("**Optional Models** *(core models always included)*")
        cb_col1, cb_col2, cb_col3, cb_col4 = st.columns(4)
        with cb_col1:
            include_lvsm = st.checkbox("Linear SVM (LVSM)", value=False)
        with cb_col2:
            include_nb = st.checkbox("Naive Bayes", value=False)
        with cb_col3:
            include_rbfn = st.checkbox("RBF Network (RBFN)", value=False)
        with cb_col4:
            include_xgb = st.checkbox("XGBoost", value=is_xgboost_available())


        if st.button("🚀 Train & Benchmark Models", type="primary"):
            with st.spinner("Executing leakage-safe training and cross-validation..."):
                try:
                    X_raw = df.drop(columns=[target])
                    y_raw = df[target]

                    valid_idx = y_raw.dropna().index
                    X_clean = X_raw.loc[valid_idx]
                    y_clean = y_raw.loc[valid_idx]

                    if task in ("binary_classification", "multiclass_classification"):
                        if not pd.api.types.is_numeric_dtype(y_clean):
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y_clean.astype(str))
                        else:
                            y_encoded = y_clean.values
                    else:
                        y_encoded = y_clean.values.astype(float)

                    pipeline = pipe_res["pipeline"]
                    X_transformed = pipeline.fit_transform(X_clean)

                    # Build models_to_evaluate list from user checkbox selections
                    from previa.models.registry import get_available_models as _get_models
                    core_names = [m["name"] for m in _get_models(task, include_optional=False)]
                    optional_selected = []
                    if include_lvsm:
                        optional_selected.append("Linear SVM (LVSM)")
                    if include_nb:
                        nb_name = "Naive Bayes" if "classification" in task else "Bayesian Ridge"
                        optional_selected.append(nb_name)
                    if include_rbfn:
                        optional_selected.append("RBF Network (RBFN)")
                    if include_xgb:
                        optional_selected.append("XGBoost")
                    models_to_evaluate = core_names + optional_selected

                    bench_res = run_benchmarks(
                        X_transformed, y_encoded, task,
                        include_optional=bool(optional_selected),
                        models_to_evaluate=models_to_evaluate,
                    )
                    st.session_state.benchmark_results = bench_res

                    split_ratio = 0.3 if len(y_encoded) < 20 else 0.2
                    if len(y_encoded) >= 10:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X_transformed, y_encoded, test_size=split_ratio, random_state=42
                        )
                    else:
                        X_tr, X_te, y_tr, y_te = X_transformed, X_transformed, y_encoded, y_encoded

                    st.session_state.eval_artifacts = {
                        "X_test": X_te,
                        "y_test": y_te,
                        "X_train": X_tr,
                        "y_train": y_tr,
                        "X_transformed": X_transformed,
                        "y_clean": y_clean,
                        "pipeline": pipeline,
                        "y_encoded": y_encoded,
                    }

                    # Assemble experiment manifest
                    prep_config = {
                        "numeric_features": pipe_res["num_feats"],
                        "categorical_features": pipe_res["cat_feats"],
                        "dropped_features": pipe_res["dropped_feats"],
                    }
                    st.session_state.manifest = create_experiment_manifest(
                        df=df,
                        target_column=target,
                        task_type=task,
                        preprocessing_config=prep_config,
                        benchmarking_results=bench_res,
                        dataset_name=st.session_state.dataset_name or "dataset.csv",
                    )

                    st.success("Benchmarking completed successfully!")
                except Exception as e:
                    st.error(f"Error during training: {e}")

        if st.session_state.benchmark_results is not None:
            res = st.session_state.benchmark_results
            if res.get("cv_warning"):
                st.warning(f"⚠️ {res['cv_warning']}")

            st.markdown(
                f"**Task:** `{res['task_type'].replace('_', ' ').title()}` | "
                f"**Samples:** `{res['n_samples']}` | "
                f"**Encoded Features:** `{res['n_features']}` | "
                f"**CV Folds:** `{res['cv_folds']}`"
            )


            st.subheader("Model Performance Comparison")
            models_df = pd.DataFrame(res["models"])
            # Reorder columns: put sensitivity/specificity after standard metrics
            desired_order = [
                "model", "accuracy", "f1", "roc_auc", "sensitivity", "specificity",
                "r2", "mae", "rmse", "mse", "train_time_s", "training_time_seconds", "status"
            ]
            display_cols = [c for c in desired_order if c in models_df.columns]
            remaining = [c for c in models_df.columns if c not in display_cols]
            models_df = models_df[display_cols + remaining]
            st.dataframe(models_df, use_container_width=True)

            st.caption(f"🏆 Winner: **{res['best_model']}** — {res['winner_explanation']}")

            # ------------------------------------------------------------------
            # Confusion Matrix of the winning model (classification only)
            # ------------------------------------------------------------------
            if "classification" in res["task_type"] and st.session_state.eval_artifacts is not None:
                art9 = st.session_state.eval_artifacts
                X_tr9, X_te9 = art9["X_train"], art9["X_test"]
                y_tr9, y_te9 = art9["y_train"], art9["y_test"]
                st.subheader(f"🏆 Winner Confusion Matrix — {res['best_model']}")
                try:
                    from previa.models.registry import create_model_instance
                    winner_model = create_model_instance(res["best_model"], res["task_type"], random_state=42)
                    winner_model.fit(X_tr9, y_tr9)
                    y_pred_winner = winner_model.predict(X_te9)
                    fig_cm_winner = plot_confusion_matrix(y_te9, y_pred_winner)
                    st.plotly_chart(fig_cm_winner, use_container_width=True)
                except Exception as cm_err:
                    st.warning(f"Could not render confusion matrix: {cm_err}")

            # ------------------------------------------------------------------
            # Download Preprocessed Dataset
            # ------------------------------------------------------------------
            if st.session_state.eval_artifacts is not None:
                st.subheader("📥 Download Preprocessed Dataset")
                art9 = st.session_state.eval_artifacts
                X_transformed_dl = art9["X_transformed"]
                y_clean_dl = art9["y_clean"]
                pipeline_dl = art9["pipeline"]
                try:
                    feat_names = [name.split("__")[-1] for name in pipeline_dl.get_feature_names_out()]
                except Exception:
                    feat_names = [f"feature_{i}" for i in range(X_transformed_dl.shape[1])]
                try:
                    preprocessed_df = pd.DataFrame(X_transformed_dl, columns=feat_names)
                    preprocessed_df["target"] = y_clean_dl.values
                    csv_bytes = preprocessed_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Preprocessed Dataset (CSV)",
                        data=csv_bytes,
                        file_name="previa_preprocessed.csv",
                        mime="text/csv",
                    )
                except Exception as dl_err:
                    st.warning(f"Could not prepare download: {dl_err}")


# ==============================================================================
# STEP 10: Diagnostics & Evaluation
# ==============================================================================
elif selected_step.startswith("10."):
    st.header("Step 10: Interactive Diagnostics & Evaluation")
    if st.session_state.benchmark_results is None or st.session_state.eval_artifacts is None:
        st.warning("Please train models in Step 9 first.")
    else:
        art = st.session_state.eval_artifacts
        task = st.session_state.selected_task or "binary_classification"
        X_tr, X_te = art["X_train"], art["X_test"]
        y_tr, y_te = art["y_train"], art["y_test"]

        st.subheader("Select Model to Inspect")
        from previa.models.registry import create_model_instance, get_available_models
        avail = get_available_models(task, include_optional=True)
        model_names = [m["name"] for m in avail]

        chosen_name = st.selectbox("Model", model_names)
        try:
            chosen_model = create_model_instance(chosen_name, task, random_state=42)
            chosen_model.fit(X_tr, y_tr)
            y_pred = chosen_model.predict(X_te)

            col1, col2 = st.columns(2)
            if "classification" in task:
                with col1:
                    st.markdown("**Confusion Matrix**")
                    fig_cm = plot_confusion_matrix(y_te, y_pred)
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    if task == "binary_classification" and hasattr(chosen_model, "predict_proba"):
                        st.markdown("**ROC Curve**")
                        y_prob = chosen_model.predict_proba(X_te)[:, 1]
                        fig_roc = plot_roc_curve(y_te, y_prob)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC curves are generated for binary classification with probability support.")
            else:
                with col1:
                    st.markdown("**Actual vs. Predicted**")
                    fig_avp = plot_actual_vs_predicted(y_te, y_pred)
                    st.plotly_chart(fig_avp, use_container_width=True)
                with col2:
                    st.markdown("**Residual Analysis**")
                    fig_res = plot_residuals(y_te, y_pred)
                    st.plotly_chart(fig_res, use_container_width=True)
        except Exception as e:
            st.error(f"Error evaluating {chosen_name}: {e}")


# ==============================================================================
# STEP 11: Explainability & SHAP
# ==============================================================================
elif selected_step.startswith("11."):
    st.header("Step 11: Model Explainability & SHAP")
    if st.session_state.eval_artifacts is None:
        st.warning("Please train models in Step 9 first.")
    else:
        art = st.session_state.eval_artifacts
        task = st.session_state.selected_task or "binary_classification"
        pipe = art["pipeline"]
        X_te, y_te = art["X_test"], art["y_test"]

        try:
            feat_names = list(pipe.get_feature_names_out())
            clean_names = [name.split("__")[-1] for name in feat_names]
        except Exception:
            clean_names = [f"feature_{i}" for i in range(X_te.shape[1])]

        best_model_name = st.session_state.benchmark_results["best_model"]
        st.markdown(f"Explaining feature contributions for winner: **{best_model_name}**")

        if st.button("🔍 Compute Feature Importance"):
            with st.spinner("Computing SHAP values and model-derived importances..."):
                try:
                    from previa.models.registry import create_model_instance
                    eval_model = create_model_instance(best_model_name, task, random_state=42)
                    eval_model.fit(art["X_train"], art["y_train"])

                    exp_res = compute_feature_importance(
                        eval_model, X_te, y_te,
                        feature_names=clean_names,
                        task_type=task,
                    )
                    st.session_state.explainability_data = exp_res
                    if st.session_state.manifest:
                        st.session_state.manifest["explainability"] = exp_res
                except Exception as e:
                    st.error(f"Error computing explainability: {e}")

        if st.session_state.explainability_data is not None:
            exp_data = st.session_state.explainability_data
            st.markdown(f"**Method Used:** `{exp_data['method_used']}`")
            fig_imp = plot_feature_importance_chart(exp_data)
            st.plotly_chart(fig_imp, use_container_width=True)

            st.warning(f"⚠️ **Scientific Causality Notice:** {exp_data['disclaimer']}")


# ==============================================================================
# STEP 12: Manifest, Report & Prediction
# ==============================================================================
elif selected_step.startswith("12."):
    st.header("Step 12: Manifest, Decision Report & Live Inference Prediction")
    if st.session_state.manifest is None:
        st.warning("Please train models in Step 9 first.")
    else:
        manifest = st.session_state.manifest
        tab_report, tab_pred = st.tabs(["📄 Decision Report & Manifest", "🔮 Live Inference Prediction Engine"])

        with tab_report:
            st.subheader("Automated Model Decision Report")
            md_rep = generate_markdown_report(
                manifest,
                health_issues=st.session_state.health_issues,
                leakage_warnings=st.session_state.leakage_warnings,
            )
            html_rep = generate_html_report(
                manifest,
                health_issues=st.session_state.health_issues,
                leakage_warnings=st.session_state.leakage_warnings,
            )

            c_down1, c_down2, c_down3 = st.columns(3)
            with c_down1:
                st.download_button(
                    "📥 Download Markdown Report",
                    data=md_rep,
                    file_name="previa_decision_report.md",
                    mime="text/markdown",
                )
            with c_down2:
                st.download_button(
                    "📥 Download HTML Report",
                    data=html_rep,
                    file_name="previa_decision_report.html",
                    mime="text/html",
                )
            with c_down3:
                manifest_json = serialize_manifest_to_json(manifest)
                st.download_button(
                    "📥 Download JSON Manifest (SHA-256)",
                    data=manifest_json,
                    file_name="previa_experiment_manifest.json",
                    mime="application/json",
                )

            with st.expander("Report Preview", expanded=True):
                st.markdown(md_rep)

        with tab_pred:
            st.subheader("Export Model Bundle & Score New Data")
            art = st.session_state.eval_artifacts
            pipe = art["pipeline"]
            task = st.session_state.selected_task or "binary_classification"
            target = st.session_state.selected_target or "target"
            best_model_name = st.session_state.benchmark_results["best_model"]

            from previa.models.registry import create_model_instance
            prod_model = create_model_instance(best_model_name, task, random_state=42)
            prod_model.fit(art["X_transformed"], art["y_encoded"])

            expected_cols = [c for c in st.session_state.raw_df.columns if c != target and c not in st.session_state.pipeline_result["dropped_feats"]]
            meta_bundle = {
                "experiment_id": manifest["experiment_id"],
                "target": target,
                "task_type": task,
                "expected_features": expected_cols,
            }
            bundle = PreviaModelBundle(pipe, prod_model, meta_bundle)

            # Download bundle
            buffer = io.BytesIO()
            import joblib
            joblib.dump(bundle, buffer)
            bundle_bytes = buffer.getvalue()
            st.download_button(
                "📦 Download Deployable Model Bundle (.joblib)",
                data=bundle_bytes,
                file_name="previa_model_bundle.joblib",
                mime="application/octet-stream",
            )

            st.divider()
            st.markdown("**Live Scoring: Upload New Unlabeled Dataset**")
            pred_file = st.file_uploader("Upload CSV with new samples", type=["csv"], key="pred_file_up")
            if pred_file is not None:
                try:
                    new_df = pd.read_csv(pred_file)
                    st.write(f"Loaded {len(new_df)} new observation(s). Generating predictions...")
                    preds_res = bundle.predict(new_df)
                    scored_df = pd.concat([new_df.reset_index(drop=True), preds_res.reset_index(drop=True)], axis=1)
                    st.dataframe(scored_df.head(10), use_container_width=True)

                    pred_csv = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Scored Predictions (CSV)",
                        data=pred_csv,
                        file_name="previa_scored_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
