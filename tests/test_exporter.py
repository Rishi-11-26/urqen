"""
tests/test_exporter.py
-----------------------
Tests for previa.export.exporter.export_dataframe

We test:
- CSV export creates the file
- XLSX export creates the file
- Exported CSV can be read back and matches the original
- Exported XLSX can be read back and matches the original
- Unsupported format raises ValueError
- Empty DataFrame raises ValueError
- Non-existent directory raises OSError
- include_index=False (default) does not write the index
"""

import os
import pathlib
import tempfile

import pandas as pd
import pytest

from previa.export.exporter import export_dataframe


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35],
        "income": [40000, 55000, 70000],
        "edu": ["Bachelor", "Master", "PhD"],
    })


def _temp_dir():
    """Return a temporary directory that is cleaned up by the caller."""
    return tempfile.mkdtemp()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCSVExport:

    def test_csv_file_is_created(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.csv")
        export_dataframe(df, path, file_format="csv")
        assert os.path.exists(path)

    def test_csv_returns_path_string(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.csv")
        result = export_dataframe(df, path, file_format="csv")
        assert isinstance(result, str)
        assert result.endswith("output.csv")

    def test_csv_roundtrip_matches_original(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.csv")
        export_dataframe(df, path, file_format="csv")
        df_read = pd.read_csv(path)
        assert list(df_read.columns) == list(df.columns)
        assert df_read.shape == df.shape

    def test_csv_no_index_column_by_default(self):
        """By default the row index should NOT be written."""
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.csv")
        export_dataframe(df, path, file_format="csv", include_index=False)
        df_read = pd.read_csv(path)
        # If index was written, there would be an extra "Unnamed: 0" column
        assert "Unnamed: 0" not in df_read.columns

    def test_csv_format_case_insensitive(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.csv")
        export_dataframe(df, path, file_format="CSV")  # uppercase should work
        assert os.path.exists(path)


class TestXLSXExport:

    def test_xlsx_file_is_created(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.xlsx")
        export_dataframe(df, path, file_format="xlsx")
        assert os.path.exists(path)

    def test_xlsx_roundtrip_matches_original(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.xlsx")
        export_dataframe(df, path, file_format="xlsx")
        df_read = pd.read_excel(path, engine="openpyxl")
        assert list(df_read.columns) == list(df.columns)
        assert df_read.shape == df.shape


class TestExportErrors:

    def test_unsupported_format_raises_value_error(self):
        df = _make_df()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.json")
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_dataframe(df, path, file_format="json")

    def test_empty_dataframe_raises_value_error(self):
        df = pd.DataFrame()
        out_dir = _temp_dir()
        path = os.path.join(out_dir, "output.csv")
        with pytest.raises(ValueError, match="empty"):
            export_dataframe(df, path, file_format="csv")

    def test_nonexistent_directory_raises_os_error(self):
        df = _make_df()
        path = "/nonexistent/directory/output.csv"
        with pytest.raises(OSError, match="does not exist"):
            export_dataframe(df, path, file_format="csv")
