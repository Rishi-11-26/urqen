"""
tests/test_workflow.py
----------------------
End-to-end workflow integration test for Previa.

Tests the full sequence:
  ingestion -> profiling -> health audit -> target/task detection ->
  feature intelligence -> preprocessing recommendations ->
  leakage-safe pipeline construction -> benchmark evaluation ->
  diagnostics generation -> export.
"""

import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
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


def test_full_previa_workflow_binary_classification():
    sample_path = pathlib.Path(__file__).parent.parent / "examples" / "sample.csv"
    assert sample_path.exists(), "sample.csv must exist"

    # Step 1: Ingestion
    df = load_dataset(str(sample_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Step 2: Profiling
    profile = profile_dataset(df)
    assert "summary" in profile
    assert "columns" in profile

    # Step 3: Health Analysis
    health_issues = analyze_health(profile, df)
    assert isinstance(health_issues, list)

    # Step 4: Target & Task Detection
    detection = detect_target_and_task(df)
    assert detection["target_column"] == "purchased"
    assert detection["task_type"] == "binary_classification"
    target = detection["target_column"]
    task = detection["task_type"]

    # Step 5: Feature Intelligence
    analysis = analyze_features(df, target_column=target)
    assert target in analysis
    assert analysis[target]["is_target"] is True

    # Step 6: Preprocessing Recommendations
    recs = generate_recommendations(analysis, target_column=target)
    assert len(recs) == df.shape[1]

    # Step 7: Leakage-Safe Pipeline Construction
    pipeline, num_feats, cat_feats, dropped_feats = build_pipeline(recs)
    assert target in dropped_feats

    # Prepare features and target
    X_raw = df.drop(columns=[target])
    y = df[target].values

    # Train / Test Split before fitting to guarantee zero leakage
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train)
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    assert not np.isnan(X_test_trans).any()

    # Step 8: Benchmarking (evaluated with adaptive CV for sample size)
    X_full_trans = pipeline.fit_transform(X_raw)
    bench_results = run_benchmarks(X_full_trans, y, task)
    assert "models" in bench_results
    assert len(bench_results["models"]) == 3
    assert bench_results["best_model"] is not None

    # Step 9: Diagnostics (Plotly)
    fig_cm = plot_confusion_matrix(y_test, y_test)
    assert fig_cm is not None
    fig_roc = plot_roc_curve(np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.2, 0.9]))
    assert fig_roc is not None

    # Step 10: Export
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_out = os.path.join(tmpdir, "exported.csv")
        export_dataframe(pd.DataFrame(X_full_trans), csv_out, file_format="csv")
        assert os.path.exists(csv_out)

        xlsx_out = os.path.join(tmpdir, "exported.xlsx")
        export_dataframe(pd.DataFrame(X_full_trans), xlsx_out, file_format="xlsx")
        assert os.path.exists(xlsx_out)


def test_full_previa_workflow_regression():
    # Synthetic regression dataset
    df = pd.DataFrame({
        "sqft": [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, None, 1800.0, 2200.0, 2800.0, 3200.0],
        "bedrooms": [2, 3, 3, 4, 4, 3, 2, 3, 4, 5],
        "neighborhood": ["Downtown", "Suburbs", "Downtown", "Suburbs", "Rural", "Suburbs", "Rural", "Downtown", "Suburbs", "Rural"],
        "price": [200000.0, 300000.0, 420000.0, 510000.0, 580000.0, 340000.0, 310000.0, 430000.0, 540000.0, 620000.0]
    })

    profile = profile_dataset(df)
    health = analyze_health(profile, df)
    detection = detect_target_and_task(df)
    assert detection["target_column"] == "price"
    assert detection["task_type"] == "regression"

    analysis = analyze_features(df, target_column="price")
    recs = generate_recommendations(analysis, target_column="price")
    pipeline, num_feats, cat_feats, dropped_feats = build_pipeline(recs)

    X_raw = df.drop(columns=["price"])
    y = df["price"].values
    X_trans = pipeline.fit_transform(X_raw)

    bench_results = run_benchmarks(X_trans, y, "regression")
    assert bench_results["primary_metric"] == "r2"
    assert len(bench_results["models"]) == 4

    # Diagnostics
    fig_avp = plot_actual_vs_predicted(y, y)
    fig_res = plot_residuals(y, y)
    assert fig_avp is not None
    assert fig_res is not None
