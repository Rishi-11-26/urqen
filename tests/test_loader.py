"""
tests/test_loader.py
--------------------
Tests for previa.ingestion.loader.load_dataset

We test:
- Successful CSV loading
- Successful XLSX loading
- FileNotFoundError for missing files
- ValueError for unsupported extensions
- ValueError for empty files
- That the returned object is a DataFrame
"""

import os
import pathlib
import tempfile

import pandas as pd
import pytest

from previa.ingestion.loader import load_dataset


# ---------------------------------------------------------------------------
# Helpers: create temporary files for testing
# ---------------------------------------------------------------------------

def _write_temp_csv(content: str, suffix: str = ".csv") -> str:
    """Write content to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        f.write(content)
    return path


def _write_temp_xlsx(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temporary XLSX file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDatasetCSV:
    """Tests for CSV loading."""

    def test_load_valid_csv_returns_dataframe(self):
        """A well-formed CSV should return a DataFrame."""
        content = "a,b,c\n1,2,3\n4,5,6\n"
        path = _write_temp_csv(content)
        try:
            df = load_dataset(path)
            assert isinstance(df, pd.DataFrame)
        finally:
            os.unlink(path)

    def test_load_csv_correct_shape(self):
        """DataFrame should have the expected number of rows and columns."""
        content = "x,y\n10,20\n30,40\n50,60\n"
        path = _write_temp_csv(content)
        try:
            df = load_dataset(path)
            assert df.shape == (3, 2)
        finally:
            os.unlink(path)

    def test_load_csv_preserves_column_names(self):
        """Column names should match the CSV header exactly."""
        content = "name,age,score\nAlice,25,90\nBob,30,85\n"
        path = _write_temp_csv(content)
        try:
            df = load_dataset(path)
            assert list(df.columns) == ["name", "age", "score"]
        finally:
            os.unlink(path)

    def test_load_csv_with_missing_values(self):
        """Missing values in CSV should be preserved as NaN, not dropped."""
        content = "a,b\n1,\n,3\n"
        path = _write_temp_csv(content)
        try:
            df = load_dataset(path)
            assert df.isnull().values.any(), "Expected NaN values to be present"
        finally:
            os.unlink(path)

    def test_load_sample_csv(self):
        """The project's sample CSV should load cleanly."""
        sample_path = pathlib.Path(__file__).parent.parent / "examples" / "sample.csv"
        df = load_dataset(str(sample_path))
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0


class TestLoadDatasetXLSX:
    """Tests for XLSX loading."""

    def test_load_valid_xlsx_returns_dataframe(self):
        """A well-formed XLSX should return a DataFrame."""
        df_in = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        path = _write_temp_xlsx(df_in)
        try:
            df_out = load_dataset(path)
            assert isinstance(df_out, pd.DataFrame)
            assert df_out.shape == df_in.shape
        finally:
            os.unlink(path)

    def test_load_xlsx_preserves_column_names(self):
        """XLSX column names should be preserved."""
        df_in = pd.DataFrame({"feature_x": [1], "target": [0]})
        path = _write_temp_xlsx(df_in)
        try:
            df_out = load_dataset(path)
            assert list(df_out.columns) == ["feature_x", "target"]
        finally:
            os.unlink(path)


class TestLoadDatasetErrors:
    """Tests for error conditions."""

    def test_missing_file_raises_file_not_found(self):
        """A path that does not exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/data.csv")

    def test_unsupported_extension_raises_value_error(self):
        """An unsupported file extension should raise ValueError."""
        # Create a real file with a bad extension so we get past the existence check
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_dataset(path)
        finally:
            os.unlink(path)

    def test_empty_csv_raises_value_error(self):
        """A CSV with only a header row and no data rows should raise ValueError."""
        content = "a,b,c\n"
        path = _write_temp_csv(content)
        try:
            with pytest.raises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)
