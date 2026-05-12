"""
tests/test_data.py
==================
Unit tests for src/data/loader.py (load_and_profile) and
src/data/profiler.py (compute_stats).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_and_profile
from src.data.profiler import compute_stats


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_csv(tmp_path):
    """Write a minimal 3-row CSV with a date column and two numeric columns."""
    csv_content = (
        "date,sales,units\n"
        "2024-01-01,100.5,10\n"
        "2024-02-01,200.0,20\n"
        "2024-03-01,150.75,15\n"
    )
    csv_file = tmp_path / "test_simple.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return str(csv_file)


@pytest.fixture
def no_date_csv(tmp_path):
    """CSV with only numeric columns (no date)."""
    csv_content = "revenue,cost\n500,300\n600,350\n700,400\n"
    csv_file = tmp_path / "no_date.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return str(csv_file)


@pytest.fixture
def known_dataframe():
    """DataFrame with known statistics for assertion testing."""
    return pd.DataFrame({
        "value": [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0],  # 1000 is a clear outlier
    })


# ── load_and_profile tests ────────────────────────────────────────────────────

class TestLoadAndProfile:

    def test_returns_dict_with_required_keys(self, simple_csv):
        """load_and_profile must return all required keys."""
        profile = load_and_profile(simple_csv)
        required_keys = {
            "df", "n_rows", "n_cols", "columns",
            "date_cols", "numeric_cols", "cat_cols", "missing_pct"
        }
        assert required_keys.issubset(profile.keys())

    def test_row_and_col_counts(self, simple_csv):
        """Row and column counts should match the written CSV."""
        profile = load_and_profile(simple_csv)
        assert profile["n_rows"] == 3
        assert profile["n_cols"] == 3  # date, sales, units

    def test_date_column_detected(self, simple_csv):
        """The 'date' column must be identified as a date column."""
        profile = load_and_profile(simple_csv)
        assert "date" in profile["date_cols"], (
            f"Expected 'date' in date_cols, got {profile['date_cols']}"
        )

    def test_date_column_coerced_to_datetime(self, simple_csv):
        """After profiling, the date column dtype must be datetime."""
        profile = load_and_profile(simple_csv)
        assert pd.api.types.is_datetime64_any_dtype(profile["df"]["date"]), (
            "date column should be datetime64 after profiling"
        )

    def test_numeric_cols_detected(self, simple_csv):
        """Both 'sales' and 'units' should be in numeric_cols."""
        profile = load_and_profile(simple_csv)
        assert "sales" in profile["numeric_cols"]
        assert "units" in profile["numeric_cols"]

    def test_date_not_in_numeric_cols(self, simple_csv):
        """Date column must not appear in numeric_cols."""
        profile = load_and_profile(simple_csv)
        assert "date" not in profile["numeric_cols"]

    def test_no_date_csv_has_empty_date_cols(self, no_date_csv):
        """CSV without date columns must have empty date_cols."""
        profile = load_and_profile(no_date_csv)
        assert profile["date_cols"] == []

    def test_missing_pct_is_zero_for_complete_data(self, simple_csv):
        """Complete CSV should have 0.0% missing for all columns."""
        profile = load_and_profile(simple_csv)
        for col, pct in profile["missing_pct"].items():
            assert pct == 0.0, f"Expected 0% missing for {col}, got {pct}%"

    def test_df_is_dataframe(self, simple_csv):
        """The 'df' key must be a pandas DataFrame."""
        profile = load_and_profile(simple_csv)
        assert isinstance(profile["df"], pd.DataFrame)


# ── compute_stats tests ───────────────────────────────────────────────────────

class TestComputeStats:

    def test_returns_list_of_dicts(self, known_dataframe):
        """compute_stats must return a list of dicts."""
        result = compute_stats(known_dataframe, ["value"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_correct_mean(self, known_dataframe):
        """Mean of [10, 20, 30, 40, 50, 1000] should be approximately 191.67."""
        result = compute_stats(known_dataframe, ["value"])
        expected_mean = round(known_dataframe["value"].mean(), 4)
        assert result[0]["mean"] == expected_mean

    def test_correct_median(self, known_dataframe):
        """Median of [10, 20, 30, 40, 50, 1000] = (30 + 40) / 2 = 35.0."""
        result = compute_stats(known_dataframe, ["value"])
        assert result[0]["median"] == 35.0

    def test_correct_std(self, known_dataframe):
        """Standard deviation should match pandas computed value."""
        result = compute_stats(known_dataframe, ["value"])
        expected_std = round(float(known_dataframe["value"].std()), 4)
        assert result[0]["std"] == expected_std

    def test_min_max(self, known_dataframe):
        """Min should be 10.0, max should be 1000.0."""
        result = compute_stats(known_dataframe, ["value"])
        assert result[0]["min"] == 10.0
        assert result[0]["max"] == 1000.0

    def test_outlier_detected_by_zscore(self, known_dataframe):
        """Value 1000 in [10,20,30,40,50,1000] should be flagged as a Z-score outlier."""
        result = compute_stats(known_dataframe, ["value"])
        assert result[0]["outliers_zscore"] >= 1, (
            "Expected at least 1 Z-score outlier for value=1000 in small series"
        )

    def test_outlier_detected_by_iqr(self, known_dataframe):
        """Value 1000 should also be flagged by IQR method."""
        result = compute_stats(known_dataframe, ["value"])
        assert result[0]["outliers_iqr"] >= 1, (
            "Expected at least 1 IQR outlier for value=1000"
        )

    def test_count_equals_non_null(self, known_dataframe):
        """Count should equal the number of non-null values."""
        result = compute_stats(known_dataframe, ["value"])
        assert result[0]["count"] == 6

    def test_skips_column_with_fewer_than_3_values(self):
        """Columns with <3 non-null values must be silently skipped."""
        df = pd.DataFrame({"tiny": [1.0, 2.0, np.nan, np.nan, np.nan]})
        # Only 2 non-null values — should be skipped
        df_two = pd.DataFrame({"tiny": [1.0, 2.0]})
        result = compute_stats(df_two, ["tiny"])
        assert result == [], f"Expected empty list for <3 observations, got {result}"

    def test_missing_pct_computation(self):
        """Missing percentage should be correctly computed."""
        df = pd.DataFrame({
            "col": [1.0, 2.0, np.nan, 4.0, np.nan]  # 2/5 = 40% missing
        })
        result = compute_stats(df, ["col"])
        assert len(result) == 1
        assert result[0]["missing_pct"] == 40.0, (
            f"Expected 40.0% missing, got {result[0]['missing_pct']}"
        )

    def test_multiple_columns(self):
        """compute_stats should handle multiple columns and return one dict per column."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = compute_stats(df, ["a", "b"])
        assert len(result) == 2
        cols = [r["column"] for r in result]
        assert "a" in cols
        assert "b" in cols
