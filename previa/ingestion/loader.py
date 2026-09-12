"""
previa/ingestion/loader.py
--------------------------
Responsible for loading tabular datasets from CSV and XLSX files into
a pandas DataFrame.

Design principles:
- Accept only CSV and XLSX — do not silently support unexpected formats.
- Return a clean DataFrame on success.
- Raise clear, user-readable errors on failure.
- Never silently corrupt data (e.g., do not auto-drop columns).
"""

import pathlib
import pandas as pd


# File extensions that Previa supports.
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV or XLSX file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file extension is not supported.
    ValueError
        If the file is empty (no rows or no columns).
    RuntimeError
        If pandas fails to parse the file contents.
    """
    path = pathlib.Path(file_path)

    # --- 1. Check that the file exists ---
    if not path.exists():
        raise FileNotFoundError(
            f"Previa could not find the file: '{file_path}'\n"
            f"Please check that the path is correct."
        )

    # --- 2. Check that the extension is supported ---
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'\n"
            f"Previa currently supports: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # --- 3. Load the file with pandas ---
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        else:  # .xlsx
            df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            f"Previa could not read the file '{path.name}'.\n"
            f"Underlying error: {exc}"
        ) from exc

    # --- 4. Validate that the result has usable content ---
    if df.empty:
        raise ValueError(
            f"The file '{path.name}' was loaded but contains no data.\n"
            f"Please provide a file with at least one row and one column."
        )

    if df.shape[1] == 0:
        raise ValueError(
            f"The file '{path.name}' was loaded but has no columns.\n"
            f"Please check the file format."
        )

    return df
