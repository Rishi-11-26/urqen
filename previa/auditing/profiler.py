"""
previa/auditing/profiler.py
---------------------------
Produces a structured profile of a DataFrame.

The profiler answers: "What is happening in this dataset?"

It calculates summary statistics that other modules (health analysis,
feature intelligence, recommender) can reuse — so this is computed once
and passed around.

Output structure
----------------
profile_dataset() returns a dict with two keys:

  "summary" : dict
      Dataset-level numbers: rows, columns, duplicates, memory usage, column type breakdown, etc.

  "columns" : dict[str -> dict]
      Per-column stats: dtype, missing count/%, unique count/%, quartiles, skewness,
      top categories breakdown, etc.

Design notes
------------
- No imputation or transformation happens here.
- All values are computed directly from the raw DataFrame.
- The output is plain Python dicts/lists so it is easy to serialize
  and easy to display in Streamlit.
"""

import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Compute a full profile of the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw loaded dataset.

    Returns
    -------
    dict with keys:
        "summary"  -> dataset-level statistics (dict)
        "columns"  -> per-column statistics (dict of dicts)
    """
    n_rows, n_cols = df.shape
    n_duplicates = int(df.duplicated().sum())

    # Calculate memory usage
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        mem_bytes = int(df.memory_usage().sum())

    if mem_bytes >= 1024 * 1024:
        mem_human = f"{mem_bytes / (1024 * 1024):.2f} MB"
    elif mem_bytes >= 1024:
        mem_human = f"{mem_bytes / 1024:.2f} KB"
    else:
        mem_human = f"{mem_bytes} B"

    n_numeric = 0
    n_datetime = 0
    n_categorical = 0

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_numeric += 1
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            n_datetime += 1
        else:
            n_categorical += 1

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": n_duplicates,
        "duplicate_pct": round(n_duplicates / n_rows * 100, 2) if n_rows > 0 else 0.0,
        "total_cells": n_rows * n_cols,
        "total_missing": int(df.isnull().sum().sum()),
        "overall_missing_pct": round(df.isnull().sum().sum() / (n_rows * n_cols) * 100, 2)
            if n_rows * n_cols > 0 else 0.0,
        "memory_bytes": mem_bytes,
        "memory_human": mem_human,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_datetime": n_datetime,
    }

    columns = {}
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isnull().sum())
        missing_pct = round(missing_count / n_rows * 100, 2) if n_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = round(unique_count / n_rows * 100, 2) if n_rows > 0 else 0.0

        col_info = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "is_constant": unique_count <= 1,
            "is_identifier": (unique_pct >= 95.0 and n_rows > 5),
            "is_datetime": bool(pd.api.types.is_datetime64_any_dtype(series)),
        }

        # --- Numeric-specific stats ---
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            col_info["mean"] = round(float(non_null.mean()), 4) if len(non_null) > 0 else None
            col_info["median"] = round(float(non_null.median()), 4) if len(non_null) > 0 else None
            col_info["std"] = round(float(non_null.std()), 4) if len(non_null) > 1 else None
            col_info["min"] = round(float(non_null.min()), 4) if len(non_null) > 0 else None
            col_info["max"] = round(float(non_null.max()), 4) if len(non_null) > 0 else None
            col_info["skewness"] = round(float(non_null.skew()), 4) if len(non_null) > 2 else None
            col_info["q25"] = round(float(non_null.quantile(0.25)), 4) if len(non_null) > 0 else None
            col_info["q75"] = round(float(non_null.quantile(0.75)), 4) if len(non_null) > 0 else None
            col_info["top_categories"] = None
        else:
            col_info["mean"] = None
            col_info["median"] = None
            col_info["std"] = None
            col_info["min"] = None
            col_info["max"] = None
            col_info["skewness"] = None
            col_info["q25"] = None
            col_info["q75"] = None

            # Top categories for categorical columns
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                vc = non_null.value_counts().head(5)
                top_cats = []
                for val, cnt in vc.items():
                    top_cats.append({
                        "category": str(val),
                        "count": int(cnt),
                        "pct": round(cnt / len(non_null) * 100, 2),
                    })
                col_info["top_categories"] = top_cats
            else:
                col_info["top_categories"] = []

        columns[col] = col_info

    return {"summary": summary, "columns": columns}
