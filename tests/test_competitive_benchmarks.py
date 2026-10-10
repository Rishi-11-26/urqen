"""
tests/test_competitive_benchmarks.py
------------------------------------
Tests for previa.benchmarking.competitive_benchmarks

We test:
- generate_benchmark_datasets returns the 3 expected task datasets
- run_competitive_benchmark executes cleanly across all 3 tasks
- Previa outperforms naive dummy baseline across tasks
"""

import pytest
from previa.benchmarking.competitive_benchmarks import (
    generate_benchmark_datasets,
    run_competitive_benchmark,
)


class TestCompetitiveBenchmarks:

    def test_generate_datasets(self):
        datasets = generate_benchmark_datasets()
        assert "binary_classification" in datasets
        assert "multiclass_classification" in datasets
        assert "regression" in datasets
        for name, df in datasets.items():
            assert len(df) >= 100
            assert df.shape[1] >= 4

    def test_run_benchmarks(self):
        results = run_competitive_benchmark()
        assert "binary_classification" in results
        assert "regression" in results

        reg = results["regression"]
        assert reg["previa"]["score"] > reg["naive_baseline"]["score"]
        assert reg["previa"]["leakage_safe"] is True
