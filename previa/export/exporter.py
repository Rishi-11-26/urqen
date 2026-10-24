"""
previa/export/exporter.py
--------------------------
Exports processed/transformed datasets to CSV or XLSX.

Terminology note
----------------
If a dataset has been through encoding, scaling, or imputation, it is called
a "Processed / Transformed Dataset" — NOT simply "cleaned data".
This distinction matters: the export may contain OHE-expanded columns,
scaled values, and imputed cells that do not look like the original data.

The user should understand what they are downloading.

Supported formats
-----------------
  "csv"   → comma-separated values (UTF-8, no row index by default)
  "xlsx"  → Excel workbook (no row index by default)
"""

import pathlib
import pandas as pd


SUPPORTED_FORMATS = {"csv", "xlsx"}


def export_dataframe(
    df: pd.DataFrame,
    output_path: str,
    file_format: str = "csv",
    include_index: bool = False,
) -> str:
    """
    Export a DataFrame to a file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to export.
    output_path : str
        Destination file path. The directory must exist, or this will raise.
    file_format : str
        "csv" or "xlsx". Case-insensitive.
    include_index : bool
        Whether to include the DataFrame's row index in the output.
        Default is False (index not written), which is usually what you want.

    Returns
    -------
    str
        The absolute path where the file was written.

    Raises
    ------
    ValueError
        If the format is not supported.
    ValueError
        If the DataFrame is empty.
    OSError
        If the output directory does not exist or is not writable.
    """
    fmt = file_format.lower().strip()

    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported export format: '{file_format}'. "
            f"URQEN supports: {sorted(SUPPORTED_FORMATS)}"
        )

    if df.empty:
        raise ValueError(
            "Cannot export an empty DataFrame. "
            "No data was available after preprocessing."
        )

    path = pathlib.Path(output_path)

    # Ensure the parent directory exists
    if not path.parent.exists():
        raise OSError(
            f"Export directory does not exist: '{path.parent}'\n"
            f"Please create the directory first."
        )

    try:
        if fmt == "csv":
            df.to_csv(path, index=include_index, encoding="utf-8")
        else:  # xlsx
            df.to_excel(path, index=include_index, engine="openpyxl")
    except Exception as exc:
        raise OSError(
            f"URQEN could not write the export file: '{path}'\n"
            f"Underlying error: {exc}"
        ) from exc

    return str(path.resolve())


def create_clean_preprocessed_dataframe(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    dropped_features: list[str] | None = None,
    train_indices: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Creates a clean, human-readable preprocessed dataset that preserves the original
    tabular structure, column names, and data format (NOT exploded one-hot dummy matrices).

    Applies:
    - Missing value imputation using learned training statistics (median for numeric, mode for categorical).
    - Dropping verified constant or excluded identifier columns.
    - Preserves original feature names, data types, and target representation.
    """
    clean_df = df.copy()

    # 1. Remove dropped features
    if dropped_features:
        cols_to_drop = [c for c in dropped_features if c in clean_df.columns and c != target_column]
        if cols_to_drop:
            clean_df = clean_df.drop(columns=cols_to_drop)

    # Reference data for learning imputers (prefer train partition to prevent data leakage)
    ref_df = clean_df.loc[train_indices] if train_indices is not None and len(train_indices) > 0 else clean_df

    # 2. Impute numeric features with training median
    for col in numeric_features:
        if col in clean_df.columns and col != target_column:
            med_val = ref_df[col].median(numeric_only=True)
            if pd.isna(med_val):
                med_val = 0.0
            clean_df[col] = clean_df[col].fillna(med_val)

    # 3. Impute categorical features with training mode
    for col in categorical_features:
        if col in clean_df.columns and col != target_column:
            modes = ref_df[col].dropna().mode()
            mode_val = modes.iloc[0] if len(modes) > 0 else "missing"
            clean_df[col] = clean_df[col].fillna(mode_val)

    # 4. Clean target column if present (move target to the last column for standard tabular convention)
    if target_column in clean_df.columns:
        target_series = clean_df.pop(target_column)
        clean_df[target_column] = target_series

    return clean_df
