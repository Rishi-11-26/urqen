"""
previa/export/report_generator.py
---------------------------------
Generates comprehensive, self-contained Model Decision Reports in Markdown
and HTML formats reflecting actual experiment results.

11 Standard Sections:
1. Experiment Overview
2. Dataset Summary & Schema
3. Data-Quality Audit Findings
4. Target & Task Characterization
5. Preprocessing Decisions & Leakage Safety
6. Evaluated Models & Algorithmic Families
7. Winner Selection Criterion & Rationale
8. Hold-Out & Cross-Validation Metrics
9. Explainability & Key Influential Features
10. Known Technical Limitations
11. Recommended Next Steps
"""

import html
import pandas as pd


def generate_markdown_report(
    manifest: dict,
    health_issues: list[dict] | None = None,
    leakage_warnings: list[dict] | None = None,
) -> str:
    """Generate a clean Markdown decision report from experiment manifest."""
    m_bench = manifest.get("model_benchmarking", {})
    m_data = manifest.get("dataset_metadata", {})
    m_task = manifest.get("task_setup", {})
    m_prep = manifest.get("preprocessing_configuration", {})
    m_exp = manifest.get("explainability", {})

    lines = []
    lines.append("# URQEN — Model Decision Report")
    lines.append(f"**Experiment ID:** `{manifest.get('experiment_id')}`  ")
    lines.append(f"**Timestamp (UTC):** `{manifest.get('timestamp_utc')}`  ")
    lines.append(f"**URQEN Version:** `{manifest.get('software_versions', {}).get('previa')}`\n")
    lines.append("---\n")

    # 1. Overview
    lines.append("## 1. Experiment Overview")
    lines.append(
        f"This report records the automated audit, preprocessing, and baseline model evaluation "
        f"conducted on dataset **{m_data.get('name', 'dataset')}** (Fingerprint: `{m_data.get('fingerprint_sha256')}`).\n"
    )

    # 2. Dataset Summary
    lines.append("## 2. Dataset Summary & Schema")
    lines.append(f"- **Dimensions:** {m_data.get('n_rows'):,} rows × {m_data.get('n_columns'):,} columns")
    lines.append("- **Features & Types:**")
    for col, dtype in m_data.get("schema", {}).items():
        lines.append(f"  - `{col}`: {dtype}")
    lines.append("")

    # 3. Data Quality Findings
    lines.append("## 3. Data Quality Audit Findings")
    if health_issues:
        lines.append(f"Identified **{len(health_issues)}** data quality observation(s):")
        for h in health_issues:
            col_str = f" in column `{h['column']}`" if h.get("column") else " (dataset-wide)"
            lines.append(f"- **[{h.get('severity', '').upper()}] {h.get('issue_type')}{col_str}:** {h.get('message')}")
            if h.get("recommendation"):
                lines.append(f"  - *Recommended Action:* {h['recommendation']}")
    else:
        lines.append("No critical data quality issues flagged during initial audit.")
    lines.append("")

    # 4. Target & Task
    lines.append("## 4. Target & Task Characterization")
    lines.append(f"- **Target Column:** `{m_task.get('target_column')}`")
    lines.append(f"- **Task Type:** `{m_task.get('task_type')}`")
    lines.append(f"- **Random Seed:** `{m_task.get('random_seed')}`")
    exp_config = manifest.get("experiment_metadata", {})
    if exp_config:
        split = exp_config.get("split_configuration", {})
        lines.append(f"- **Dataset Fingerprint (SHA-256):** `{m_data.get('fingerprint_sha256')}`")
        lines.append(f"- **Split Strategy:** `{split.get('strategy', 'Unavailable')}`")
        lines.append(f"- **Training / Holdout Samples:** {split.get('training_sample_count', 'Unavailable')} / {split.get('holdout_sample_count', 'Unavailable')}")
        lines.append(f"- **Primary Metric Definition:** {exp_config.get('primary_metric_definition', 'Unavailable')}")
    lines.append("")

    # 5. Preprocessing Decisions
    lines.append("## 5. Preprocessing Decisions & Leakage Prevention")
    lines.append(f"- **Original Input Columns:** {m_prep.get('original_dataset_columns', 'Unavailable in this run.')}")
    lines.append(f"- **Candidate Features:** {m_prep.get('candidate_features', 'Unavailable in this run.')}")
    lines.append(f"- **Numeric Columns Processed:** {m_prep.get('numeric_columns_processed', m_prep.get('numeric_features', 'Unavailable in this run.'))}")
    lines.append(f"- **Categorical Columns Processed:** {m_prep.get('categorical_columns_processed', m_prep.get('categorical_features', 'Unavailable in this run.'))}")
    excluded = m_prep.get("excluded_columns")
    if excluded is None:
        excluded = m_prep.get("dropped_features", "Unavailable in this run.")
    lines.append(f"- **Excluded Columns and Reasons:** {excluded}")
    lines.append(f"- **Fitted Steps:** {m_prep.get('preprocessing_steps', 'Unavailable in this run.')}")
    lines.append(f"- **Final Transformed Feature Count:** {m_prep.get('final_transformed_feature_count', 'Unavailable in this run.')}")
    lines.append(f"- **Fit Scope:** {m_prep.get('fit_scope', 'Unavailable in this run.')}\n")

    # 6 & 7. Evaluated Models & Winner
    lines.append("## 6. Evaluated Models & Winner Selection")
    lines.append(f"- **Primary Metric:** `{m_bench.get('primary_metric')}`")
    lines.append(f"- **Selected Winner:** **{m_bench.get('best_model')}**")
    lines.append(f"- **Rationale:** {m_bench.get('winner_explanation')}\n")

    # 8. Metrics Table
    lines.append("## 7. Model Performance Comparison")
    lines.append("| Model | Status | Family | Primary Score (Holdout) | CV Mean ± Std | Training Time |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in m_bench.get("models_evaluated", []):
        score_val = m.get("primary_score")
        score_str = f"{float(score_val):.4f}" if score_val is not None else "N/A"
        cv_m = m.get("cv_mean")
        cv_s = m.get("cv_std")
        cv_str = f"{float(cv_m):.6f} ± {float(cv_s or 0):.6f}" if cv_m is not None else "N/A"
        t_val = m.get("training_time_seconds")
        t_str = f"{float(t_val):.3f}s" if t_val is not None else "N/A"
        lines.append(f"| {m.get('model')} | {m.get('status')} | {m.get('family')} | {score_str} | {cv_str} | {t_str} |")
    lines.append("")

    # Stored predictions and independent verification are represented separately
    # from cross-validation scores throughout this report.
    eval_summary = manifest.get("evaluation_summary", {})
    eval_models = eval_summary.get("models", [])
    lines.append("## 8. Holdout Evaluation Artifacts")
    if not eval_models:
        lines.append("Holdout labels/predictions were not retained for this historical run; no artifacts were reconstructed.")
    else:
        for item in eval_models:
            lines.append(f"### {item.get('model', 'Unknown')} — {item.get('status', 'unknown')}")
            holdout = item.get("holdout") or {}
            if holdout:
                lines.append(f"- **Evaluation type:** Holdout")
                lines.append(f"- **Holdout sample count:** {holdout.get('sample_count', 'Unavailable')}")
                lines.append(f"- **Holdout metric values:** {holdout.get('metrics', {})}")
                if holdout.get("confusion_matrix") is not None:
                    lines.append(f"- **Class-label order:** {holdout.get('class_labels', [])}")
                    lines.append(f"- **Confusion matrix:** `{holdout.get('confusion_matrix')}`")
                elif holdout.get("example_pairs"):
                    lines.append("- **Regression holdout actual/predicted examples (first 10):**")
                    lines.append("| Sample ID | Actual | Predicted |")
                    lines.append("| :--- | ---: | ---: |")
                    for pair in holdout["example_pairs"]:
                        lines.append(f"| {pair.get('sample_id')} | {pair.get('actual')} | {pair.get('predicted')} |")
                cv = item.get("cross_validation") or {}
                lines.append(
                    f"- **Cross-validation only:** {cv.get('fold_count', 0)} folds; "
                    f"scores `{cv.get('fold_scores', [])}`; mean {cv.get('mean')}; "
                    f"standard deviation {cv.get('standard_deviation')} "
                    f"({cv.get('metric', 'metric unavailable')})."
                )
            else:
                lines.append(f"- **Artifacts unavailable:** {item.get('error_message', 'No holdout artifacts were retained.')}")
        lines.append("")

    lines.append("## 9. Independent Metric Verification")
    if not eval_models:
        lines.append("Unavailable for this historical run because its actual holdout labels and predictions were not retained.")
    else:
        lines.append("| Model | Metric | Reported | Recalculated | Absolute Difference | Status |")
        lines.append("| :--- | :--- | ---: | ---: | ---: | :--- |")
        for item in eval_models:
            for metric, evidence in (item.get("metric_verification") or {}).items():
                if not isinstance(evidence, dict) or "status" not in evidence:
                    continue
                lines.append(
                    f"| {item.get('model')} | {metric} | {evidence.get('reported')} | "
                    f"{evidence.get('recalculated')} | {evidence.get('absolute_difference')} | "
                    f"{evidence.get('status')} |"
                )
    lines.append("")

    lines.append("## 10. Holdout Split Consistency")
    consistency = eval_summary.get("split_consistency")
    if consistency:
        lines.append(f"- **Status:** {consistency.get('status')}")
        lines.append(f"- **Models with evaluation artifacts:** {consistency.get('evaluated_model_count')}")
        lines.append(f"- **Holdout counts:** {consistency.get('holdout_sample_count_per_model')}")
        lines.append(f"- **Split configuration matches:** {consistency.get('split_configuration_matches')}")
        lines.append(f"- **Row identifiers match:** {consistency.get('holdout_row_identifiers_match')}")
        lines.append(f"- **Target labels align:** {consistency.get('target_labels_aligned')}")
        lines.append(f"- **Missing/inconsistent artifacts:** {consistency.get('missing_or_inconsistent_artifacts')}")
    else:
        lines.append("Unavailable for this historical run; equal sample counts alone are not evidence of identical holdout rows.")
    lines.append("")

    # 9. Explainability
    lines.append("## 11. Explainability & Feature Importance")
    if m_exp:
        lines.append(f"- **Method:** {m_exp.get('method_used')}")
        lines.append(f"- **Top Influential Features:** {', '.join(m_exp.get('top_features', []))}")
        lines.append(f"- **Causality Notice:** *{m_exp.get('disclaimer')}*")
    else:
        lines.append("Explainability analysis was not computed for this run.")
    lines.append("")

    # 10. Known Limitations
    lines.append("## 12. Known Technical Limitations")
    lines.append("- Baseline models only; no extensive hyperparameter tuning was applied.")
    lines.append("- Linear models assume additive linear relationships; complex nonlinear interactions may require deep trees or feature engineering.")
    lines.append("- High-cardinality categories should be monitored if deployed on unconstrained external inputs.")
    lines.append("- Repeated selection or tuning based on holdout results can cause indirect holdout overfitting.")
    lines.append("")

    # 11. Recommended Next Steps
    lines.append("## 13. Limitations and Warnings")
    warnings = eval_summary.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No evaluation warnings were recorded.")
    lines.append("")

    lines.append("## 14. Recommended Next Steps")
    lines.append("1. Validate model performance against a dedicated out-of-time test split if temporal data is available.")
    lines.append("2. Download the exported `.joblib` model bundle for local reproducible inference.")
    lines.append("3. Review flagged data quality issues with subject-matter experts before clinical or production use.")

    return "\n".join(lines)


def generate_html_report(
    manifest: dict,
    health_issues: list[dict] | None = None,
    leakage_warnings: list[dict] | None = None,
) -> str:
    """Generate a styled, self-contained HTML decision report."""
    md_content = generate_markdown_report(manifest, health_issues, leakage_warnings)

    # Convert markdown headers and tables to clean HTML
    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang='en'>")
    html_lines.append("<head>")
    html_lines.append("<meta charset='UTF-8'>")
    html_lines.append("<title>URQEN Model Decision Report</title>")
    html_lines.append("<style>")
    html_lines.append("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292f; max-width: 900px; margin: 40px auto; padding: 0 20px; }")
    html_lines.append("h1 { border-bottom: 2px solid #0969da; padding-bottom: 8px; color: #0969da; }")
    html_lines.append("h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 6px; margin-top: 30px; }")
    html_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    html_lines.append("th, td { border: 1px solid #d0d7de; padding: 10px; text-align: left; }")
    html_lines.append("th { background-color: #f6f8fa; }")
    html_lines.append("code { background-color: #f6f8fa; padding: 2px 6px; border-radius: 4px; font-size: 90%; }")
    html_lines.append(".badge { display: inline-block; padding: 2px 8px; font-size: 12px; font-weight: bold; border-radius: 12px; }")
    html_lines.append(".badge-high { background-color: #ffebe9; color: #cf222e; }")
    html_lines.append(".disclaimer { background-color: #fff8c5; border-left: 4px solid #d4a72c; padding: 12px; margin: 16px 0; }")
    html_lines.append("</style>")
    html_lines.append("</head>")
    html_lines.append("<body>")

    # Simple formatting of markdown to HTML
    in_table = False
    for line in md_content.split("\n"):
        line = line.strip()
        if not line:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            continue

        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("|") and line.endswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if "---" in parts[0]:
                continue
            if not in_table:
                html_lines.append("<table>")
                html_lines.append("<tr>" + "".join([f"<th>{html.escape(p)}</th>" for p in parts]) + "</tr>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join([f"<td>{html.escape(p)}</td>" for p in parts]) + "</tr>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            if "Causality Notice:" in line or "Notice:" in line:
                html_lines.append(f"<div class='disclaimer'>{html.escape(line)}</div>")
            else:
                html_lines.append(f"<p>{html.escape(line)}</p>")

    if in_table:
        html_lines.append("</table>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)
