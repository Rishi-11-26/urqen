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
            f"Previa supports: {sorted(SUPPORTED_FORMATS)}"
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
            f"Previa could not write the export file: '{path}'\n"
            f"Underlying error: {exc}"
        ) from exc

    return str(path.resolve())
