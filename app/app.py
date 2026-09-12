"""
app/app.py
----------
Streamlit Application for Previa: Trustworthy Pre-Model Data Intelligence Workbench.

Guided 10-Step Workflow:
  STEP 1: Upload Dataset
  STEP 2: Dataset Overview
  STEP 3: Data Health
  STEP 4: Target & Task
  STEP 5: Feature Intelligence
  STEP 6: Preprocessing Recommendations
  STEP 7: Build Pipeline
  STEP 8: Benchmark Models
  STEP 9: Diagnostics
  STEP 10: Export
"""

import os
import sys
import tempfile
import pathlib

# Ensure project root is on Python search path for cloud deployment
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from previa.ingestion.loader import load_dataset
from previa.auditing.profiler import profile_dataset
from previa.auditing.health import analyze_health
from previa.task_detection.detector import detect_target_and_task
from previa.feature_intelligence.analyzer import analyze_features
from previa.preprocessing.recommender import generate_recommendations
from previa.preprocessing.pipeline_builder import build_pipeline
from previa.benchmarking.benchmarker import run_benchmarks
from previa.diagnostics.charts import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_actual_vs_predicted,
    plot_residuals,
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


# --- Helper to load sample dataset ---
def get_sample_path():
    base = pathlib.Path(__file__).resolve().parent.parent
    sample = base / "examples" / "sample.csv"
    return str(sample) if sample.exists() else None


# --- Sidebar Navigation ---
st.sidebar.header("Workflow Steps")
steps = [
    "1. Upload Dataset",
    "2. Dataset Overview",
    "3. Data Health",
    "4. Target & Task",
    "5. Feature Intelligence",
    "6. Preprocessing Recommendations",
    "7. Build Pipeline",
    "8. Benchmark Models",
    "9. Diagnostics",
    "10. Export Transformed Data",
]
selected_step = st.sidebar.radio("Navigate", steps)

# --- Session State Initialization ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "health_issues" not in st.session_state:
    st.session_state.health_issues = None
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
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "eval_artifacts" not in st.session_state:
    st.session_state.eval_artifacts = None


# ==============================================================================
# STEP 1: Upload Dataset
# ==============================================================================
if selected_step.startswith("1"):
    st.header("Step 1: Ingestion & Dataset Loading")
    st.markdown("Upload a **CSV** or **XLSX** file to begin auditing and preparing your data.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    with col2:
        st.markdown("**Or load the included sample:**")
        sample_path = get_sample_path()
        if sample_path and st.button("Load Built-in Sample Dataset"):
            try:
                df = load_dataset(sample_path)
                st.session_state.raw_df = df
                st.session_state.dataset_name = "sample.csv"
                # Reset downstream state
                st.session_state.profile = None
                st.session_state.health_issues = None
                st.session_state.target_detection = None
                st.session_state.pipeline_result = None
                st.session_state.benchmark_results = None
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
            # Reset downstream state on new upload
            st.session_state.profile = None
            st.session_state.health_issues = None
            st.session_state.target_detection = None
            st.session_state.pipeline_result = None
            st.session_state.benchmark_results = None
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df)} rows, {df.shape[1]} columns)")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if st.session_state.raw_df is not None:
        st.divider()
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
        st.info("Dataset is loaded. Navigate to **2. Dataset Overview** in the sidebar to proceed.")


# ==============================================================================
# STEP 2: Dataset Overview
# ==============================================================================
elif selected_step.startswith("2"):
    st.header("Step 2: Dataset Profiling & Overview")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        if st.session_state.profile is None:
            st.session_state.profile = profile_dataset(df)

        profile = st.session_state.profile
        summary = profile["summary"]

        st.subheader("Dataset Summary Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary['n_rows']:,}")
        c2.metric("Columns", f"{summary['n_cols']:,}")
        c3.metric("Duplicate Rows", f"{summary['n_duplicates']} ({summary['duplicate_pct']}%)")
        c4.metric("Total Missing Cells", f"{summary['total_missing']} ({summary['overall_missing_pct']}%)")

        st.subheader("Per-Column Profile")
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
                "Std Dev": info["std"],
                "Min": info["min"],
                "Max": info["max"],
                "Skewness": info["skewness"],
            })
        st.dataframe(pd.DataFrame(col_records), use_container_width=True)


# ==============================================================================
# STEP 3: Data Health
# ==============================================================================
elif selected_step.startswith("3"):
    st.header("Step 3: Data Health & Quality Audit")
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
            st.markdown(f"Identified **{len(issues)}** potential quality finding(s). Remember: these are evidence-based warnings, not definitive verdicts.")
            
            for issue in issues:
                sev = issue["severity"].lower()
                badge = "🔴 HIGH" if sev == "high" else ("🟡 MEDIUM" if sev == "medium" else "🔵 LOW")
                col_label = f"Column: `{issue['column']}`" if issue["column"] else "Dataset-Level"
                
                with st.expander(f"{badge} | {issue['issue_type'].replace('_', ' ').title()} ({col_label})", expanded=(sev == "high")):
                    st.write(issue["message"])
                    st.caption(f"**Evidence:** {issue['evidence']}")


# ==============================================================================
# STEP 4: Target & Task Detection
# ==============================================================================
elif selected_step.startswith("4"):
    st.header("Step 4: Target Column & ML Task Detection")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        if st.session_state.target_detection is None:
            st.session_state.target_detection = detect_target_and_task(df)

        det = st.session_state.target_detection

        st.subheader("Inferred Target & Task")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Target", det["target_column"] or "None detected")
        col2.metric("Inferred Task", det["task_type"].replace('_', ' ').title())
        col3.metric("Heuristic Confidence", det["confidence"].upper())

        st.markdown("**Evidence & Reasons:**")
        for reason in det["reasons"]:
            st.markdown(f"- {reason}")

        st.divider()
        st.subheader("Manual Override (Human in the Loop)")
        st.caption("Previa never assumes automated detection is infallible. You can confirm or override below:")

        all_cols = list(df.columns)
        default_target_idx = all_cols.index(det["target_column"]) if det["target_column"] in all_cols else 0

        chosen_target = st.selectbox("Select Target Column", all_cols, index=default_target_idx)

        task_options = ["binary_classification", "multiclass_classification", "regression"]
        default_task_idx = task_options.index(det["task_type"]) if det["task_type"] in task_options else 0
        chosen_task = st.selectbox(
            "Select Task Type",
            task_options,
            index=default_task_idx,
            format_func=lambda x: x.replace("_", " ").title()
        )

        st.session_state.selected_target = chosen_target
        st.session_state.selected_task = chosen_task

        st.success(f"Active Configuration: Target = **`{chosen_target}`** | Task = **`{chosen_task.replace('_', ' ').title()}`**")


# ==============================================================================
# STEP 5: Feature Intelligence
# ==============================================================================
elif selected_step.startswith("5"):
    st.header("Step 5: Feature Intelligence")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target
        
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
                "Mean / Top": info["mean"] if info["mean"] is not None else info["top_value"],
                "Std / Top Freq": info["std"] if info["std"] is not None else info["top_freq"],
                "Skewness": info["skewness"],
            })
        st.dataframe(pd.DataFrame(feat_records), use_container_width=True)


# ==============================================================================
# STEP 6: Preprocessing Recommendations
# ==============================================================================
elif selected_step.startswith("6"):
    st.header("Step 6: Explainable Preprocessing Recommendations")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target
        if st.session_state.feature_analysis is None:
            st.session_state.feature_analysis = analyze_features(df, target_column=target)

        st.session_state.recommendations = generate_recommendations(
            st.session_state.feature_analysis, target_column=target
        )
        recs = st.session_state.recommendations

        st.markdown("Each recommendation is generated using **deterministic rules and statistical evidence**.")

        rec_table = []
        for r in recs:
            rec_table.append({
                "Feature": r["feature"],
                "Type": r["detected_type"],
                "Condition / Issue": r["issue"],
                "Recommendation": r["recommendation"],
                "Evidence": r["evidence"],
                "Confidence": r["confidence"].upper(),
            })
        st.dataframe(pd.DataFrame(rec_table), use_container_width=True)

        st.subheader("Detailed Explanations & Rationale")
        for r in recs:
            with st.expander(f"📌 {r['feature']} → `{r['recommendation']}` (Confidence: {r['confidence'].upper()})"):
                st.markdown(f"**Issue Detected:** {r['issue']}")
                st.markdown(f"**Reason:** {r['reason']}")
                st.caption(f"**Statistical Evidence:** {r['evidence']}")


# ==============================================================================
# STEP 7: Build Leakage-Safe Pipeline
# ==============================================================================
elif selected_step.startswith("7"):
    st.header("Step 7: Leakage-Safe Pipeline Construction")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target
        if not target:
            st.warning("Please select a target in Step 4 first.")
        else:
            if st.session_state.recommendations is None:
                if st.session_state.feature_analysis is None:
                    st.session_state.feature_analysis = analyze_features(df, target_column=target)
                st.session_state.recommendations = generate_recommendations(
                    st.session_state.feature_analysis, target_column=target
                )

            try:
                pipeline, num_feats, cat_feats, dropped_feats = build_pipeline(st.session_state.recommendations)
                st.session_state.pipeline_result = {
                    "pipeline": pipeline,
                    "num_feats": num_feats,
                    "cat_feats": cat_feats,
                    "dropped_feats": dropped_feats,
                }

                st.success("✅ ColumnTransformer pipeline constructed successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Numeric Features Pipeline**")
                    st.caption("Imputation (Median) → StandardScaler")
                    st.write(num_feats if num_feats else "None")
                with col2:
                    st.markdown("**Categorical Features Pipeline**")
                    st.caption("Imputation (Most Frequent) → OneHotEncoder(ignore)")
                    st.write(cat_feats if cat_feats else "None")
                with col3:
                    st.markdown("**Excluded / Target Columns**")
                    st.caption("Target or dropped features (constants/IDs)")
                    st.write(dropped_feats if dropped_feats else "None")

                st.info(
                    "🛡️ **Leakage Safety Guarantee**: This pipeline is purely a blueprint right now. "
                    "Fitting will occur strictly on training folds/sets during benchmarking."
                )

            except Exception as e:
                st.error(f"Error constructing pipeline: {e}")


# ==============================================================================
# STEP 8: Benchmark Baseline Models
# ==============================================================================
elif selected_step.startswith("8"):
    st.header("Step 8: Baseline Model Benchmarking")
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset in Step 1 first.")
    elif st.session_state.pipeline_result is None:
        st.warning("Please construct the pipeline in Step 7 first.")
    else:
        df = st.session_state.raw_df
        target = st.session_state.selected_target
        task = st.session_state.selected_task
        pipe_res = st.session_state.pipeline_result

        if st.button("🚀 Run Baseline Benchmarks", type="primary"):
            with st.spinner("Preparing features and training baseline models..."):
                try:
                    # Prepare X and y
                    X_raw = df.drop(columns=[target])
                    y_raw = df[target]

                    # Handle missing target values if any
                    valid_idx = y_raw.dropna().index
                    X_clean = X_raw.loc[valid_idx]
                    y_clean = y_raw.loc[valid_idx]

                    # Encode non-numeric classification target if needed
                    if task in ("binary_classification", "multiclass_classification"):
                        if not pd.api.types.is_numeric_dtype(y_clean):
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y_encoded = le.fit_transform(y_clean.astype(str))
                        else:
                            y_encoded = y_clean.values
                    else:
                        y_encoded = y_clean.values.astype(float)

                    # Fit pipeline strictly on X_clean (for benchmarking, benchmarker handles train/test splitting)
                    pipeline = pipe_res["pipeline"]
                    X_transformed = pipeline.fit_transform(X_clean)

                    # Run benchmarks with small dataset adaptation
                    bench_res = run_benchmarks(X_transformed, y_encoded, task)
                    st.session_state.benchmark_results = bench_res

                    # Save evaluation artifacts for Step 9 Diagnostics
                    from sklearn.model_selection import train_test_split
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
                    }

                    st.success("Benchmarking completed!")
                except Exception as e:
                    st.error(f"Error during benchmarking: {e}")

        if st.session_state.benchmark_results is not None:
            res = st.session_state.benchmark_results

            if res.get("cv_warning"):
                st.warning(f"⚠️ {res['cv_warning']}")

            st.markdown(f"**Task Type:** `{res['task_type'].replace('_', ' ').title()}` | **Samples:** `{res['n_samples']}` | **Features After Encoding:** `{res['n_features']}` | **CV Folds:** `{res['cv_folds']}`")

            st.subheader("Model Performance Comparison")
            st.dataframe(pd.DataFrame(res["models"]), use_container_width=True)

            st.caption(
                f"🏆 Note: **{res['best_model']}** performed best among the evaluated baseline models according to `{res['primary_metric']}`. "
                "This does not imply it is objectively or universally optimal."
            )


# ==============================================================================
# STEP 9: Diagnostics
# ==============================================================================
elif selected_step.startswith("9"):
    st.header("Step 9: Model Diagnostics (Plotly)")
    if st.session_state.benchmark_results is None or st.session_state.eval_artifacts is None:
        st.warning("Please run model benchmarking in Step 8 first.")
    else:
        art = st.session_state.eval_artifacts
        task = st.session_state.selected_task
        X_tr, X_te = art["X_train"], art["X_test"]
        y_tr, y_te = art["y_train"], art["y_test"]

        st.subheader("Select Baseline Model for Inspection")
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task in ("binary_classification", "multiclass_classification"):
            model_map = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            }
        else:
            model_map = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            }

        chosen_model_name = st.selectbox("Model", list(model_map.keys()))
        selected_model = model_map[chosen_model_name]
        selected_model.fit(X_tr, y_tr)
        y_pred = selected_model.predict(X_te)

        col_diag1, col_diag2 = st.columns(2)

        if task in ("binary_classification", "multiclass_classification"):
            with col_diag1:
                st.markdown("**Confusion Matrix**")
                fig_cm = plot_confusion_matrix(y_te, y_pred)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_diag2:
                if task == "binary_classification" and hasattr(selected_model, "predict_proba"):
                    st.markdown("**ROC Curve**")
                    try:
                        y_prob = selected_model.predict_proba(X_te)[:, 1]
                        fig_roc = plot_roc_curve(y_te, y_prob)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.info(f"ROC curve could not be generated: {e}")
                else:
                    st.info("ROC curves are shown for binary classification models with probability estimates.")
        else:
            with col_diag1:
                st.markdown("**Actual vs. Predicted**")
                fig_avp = plot_actual_vs_predicted(y_te, y_pred)
                st.plotly_chart(fig_avp, use_container_width=True)

            with col_diag2:
                st.markdown("**Residual Analysis**")
                fig_res = plot_residuals(y_te, y_pred)
                st.plotly_chart(fig_res, use_container_width=True)


# ==============================================================================
# STEP 10: Export Transformed Data
# ==============================================================================
elif selected_step.startswith("10"):
    st.header("Step 10: Export Processed / Transformed Dataset")
    st.markdown(
        "⚠️ **Terminology Notice:** This is a **Processed / Transformed Dataset** "
        "(encoded, imputed, and scaled according to Previa's verified recommendations). "
        "It is distinct from raw or lightly cleaned data."
    )

    if st.session_state.eval_artifacts is None:
        st.warning("Please build the pipeline and run benchmarking in earlier steps before exporting.")
    else:
        art = st.session_state.eval_artifacts
        pipe = art["pipeline"]
        X_mat = art["X_transformed"]

        # Attempt to get readable feature names from ColumnTransformer
        try:
            feat_names = list(pipe.get_feature_names_out())
        except Exception:
            feat_names = [f"feature_{i}" for i in range(X_mat.shape[1])]

        clean_feat_names = [name.split("__")[-1] for name in feat_names]
        export_df = pd.DataFrame(X_mat, columns=clean_feat_names)

        target_name = st.session_state.selected_target
        if target_name and target_name in art["y_clean"]:
            export_df[f"TARGET_{target_name}"] = art["y_clean"].values

        st.subheader("Preview of Transformed Data")
        st.dataframe(export_df.head(10), use_container_width=True)

        fmt = st.radio("Select Export Format", ["CSV", "XLSX"], horizontal=True)

        if fmt == "CSV":
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Transformed Dataset (CSV)",
                data=csv_bytes,
                file_name="previa_transformed_dataset.csv",
                mime="text/csv",
            )
        else:
            buffer = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            buffer_path = buffer.name
            buffer.close()
            try:
                export_dataframe(export_df, buffer_path, file_format="xlsx")
                with open(buffer_path, "rb") as f:
                    xlsx_bytes = f.read()
                st.download_button(
                    label="📥 Download Transformed Dataset (XLSX)",
                    data=xlsx_bytes,
                    file_name="previa_transformed_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            finally:
                if os.path.exists(buffer_path):
                    os.remove(buffer_path)
