"""
previa/ingestion/loader.py
--------------------------
Responsible for loading tabular datasets from CSV and XLSX files into
a pandas DataFrame, with validation against corrupted, empty, or
malformed files.

Design principles:
- Accept only CSV and XLSX — do not silently support unexpected formats.
- Return a clean DataFrame on success.
- Raise clear, user-readable errors on failure.
- Never silently corrupt data (e.g., do not auto-drop columns).
- Provide validation diagnostics (empty columns, duplicate headers, memory).
"""

import pathlib
import pandas as pd


# File extensions that Previa supports.
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
DEFAULT_MAX_FILE_SIZE_MB = 100.0


def load_dataset(file_path: str, max_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB) -> pd.DataFrame:
    """
    Load a dataset from a CSV or XLSX file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the file.
    max_size_mb : float
        Maximum allowed file size in megabytes (default 100MB).

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file extension is not supported, or file exceeds size limit,
        or file is empty (no rows or no columns).
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

    # --- 3. Check file size limit ---
    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(
            f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
        )

    # --- 4. Load the file with pandas ---
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

    # --- 5. Validate that the result has usable content ---
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


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform pre-flight validation on an ingested DataFrame.

    Checks for:
    - Empty rows or columns
    - Duplicate column headers
    - Unnamed or missing column headers
    - Completely empty columns (all NaN)
    - Small dataset advisory (< 10 rows)

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Validation report containing:
        - "is_valid": bool
        - "n_rows": int
        - "n_cols": int
        - "n_numeric": int
        - "n_categorical": int
        - "n_datetime": int
        - "duplicate_cols": list[str]
        - "unnamed_cols": list[str]
        - "completely_empty_cols": list[str]
        - "memory_bytes": int
        - "memory_human": str
        - "warnings": list[str]
    """
    n_rows, n_cols = df.shape
    warnings = []

    # Check duplicate columns
    cols = [str(c) for c in df.columns]
    seen = set()
    dup_cols = []
    for c in cols:
        if c in seen:
            dup_cols.append(c)
        seen.add(c)
    if dup_cols:
        warnings.append(f"Found {len(dup_cols)} duplicate column header(s): {dup_cols}")

    # Check unnamed / blank column headers
    unnamed_cols = [c for c in cols if c.startswith("Unnamed:") or not c.strip()]
    if unnamed_cols:
        warnings.append(f"Found {len(unnamed_cols)} unnamed or empty column header(s): {unnamed_cols}")

    # Check completely empty columns
    completely_empty_cols = [c for c in df.columns if df[c].isnull().all()]
    if completely_empty_cols:
        warnings.append(f"Found {len(completely_empty_cols)} completely empty column(s) (all null): {completely_empty_cols}")

    # Small dataset advisory
    if n_rows < 10:
        warnings.append(f"Extremely small dataset ({n_rows} rows). Statistical stability may be limited.")

    # Column type counts
    n_numeric = int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))
    n_datetime = int(sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns))
    n_categorical = n_cols - n_numeric - n_datetime

    # Memory usage
    mem_bytes = int(df.memory_usage(deep=True).sum())
    if mem_bytes >= 1024 * 1024:
        mem_human = f"{mem_bytes / (1024 * 1024):.2f} MB"
    elif mem_bytes >= 1024:
        mem_human = f"{mem_bytes / 1024:.2f} KB"
    else:
        mem_human = f"{mem_bytes} B"

    is_valid = n_rows > 0 and n_cols > 0 and (len(completely_empty_cols) < n_cols)

    return {
        "is_valid": is_valid,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_datetime": n_datetime,
        "duplicate_cols": dup_cols,
        "unnamed_cols": unnamed_cols,
        "completely_empty_cols": completely_empty_cols,
        "memory_bytes": mem_bytes,
        "memory_human": mem_human,
        "warnings": warnings,
    }
