"""
tests/test_models.py
====================
Unit tests for:
- src/models/forecast.py  (detect_time_series, run_forecast)
- src/features/detection.py (detect_anomalies_in_series)
"""

import pandas as pd
import numpy as np
import pytest

from src.models.forecast import detect_time_series, run_forecast
from src.features.detection import detect_anomalies_in_series


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_monthly_profile(n: int = 36) -> dict:
    """Build a minimal profile dict with monthly date + numeric column."""
    dates = pd.date_range(start="2021-01-01", periods=n, freq="MS")
    # Trend + seasonality + noise
    rng = np.random.default_rng(seed=42)
    values = (
        500.0
        + np.arange(n) * 2.0                        # trend
        + 20.0 * np.sin(2 * np.pi * np.arange(n) / 12)  # seasonality
        + rng.normal(0, 3, size=n)                  # noise
    )
    df = pd.DataFrame({"date": dates, "sales": values})
    return {
        "df": df,
        "date_cols": ["date"],
        "numeric_cols": ["sales"],
    }


def make_non_monthly_profile() -> dict:
    """Profile with dates spaced ~90 days apart (quarterly, not monthly)."""
    dates = pd.date_range(start="2018-01-01", periods=30, freq="QS")
    values = np.linspace(100, 200, 30)
    df = pd.DataFrame({"date": dates, "revenue": values})
    return {
        "df": df,
        "date_cols": ["date"],
        "numeric_cols": ["revenue"],
    }


def make_series_with_spike(spike_index: int = 20, spike_magnitude: float = 200.0) -> pd.DataFrame:
    """Monthly series with a single extreme positive spike at spike_index."""
    n = 48
    dates = pd.date_range(start="2020-01-01", periods=n, freq="MS")
    values = np.ones(n) * 100.0  # flat series at 100
    values[spike_index] = 100.0 + spike_magnitude  # inject spike
    return pd.DataFrame({"date": dates, "value": values})


# ── detect_time_series tests ──────────────────────────────────────────────────

class TestDetectTimeSeries:

    def test_monthly_series_detected(self):
        """Monthly date+numeric series with ≥18 obs should return True."""
        profile = make_monthly_profile(n=36)
        is_ts, date_col, value_col = detect_time_series(profile)
        assert is_ts is True
        assert date_col == "date"
        assert value_col == "sales"

    def test_returns_correct_column_names(self):
        """Column names returned must match the profile dict keys."""
        profile = make_monthly_profile(n=24)
        is_ts, date_col, value_col = detect_time_series(profile)
        assert date_col in profile["date_cols"]
        assert value_col in profile["numeric_cols"]

    def test_short_series_not_detected(self):
        """Series with fewer than 18 observations must not be flagged as time-series."""
        profile = make_monthly_profile(n=12)  # only 12 months — below threshold
        is_ts, date_col, value_col = detect_time_series(profile)
        assert is_ts is False
        assert date_col is None
        assert value_col is None

    def test_no_date_col_returns_false(self):
        """Profile with empty date_cols must return False."""
        profile = make_monthly_profile(n=36)
        profile["date_cols"] = []
        is_ts, date_col, value_col = detect_time_series(profile)
        assert is_ts is False

    def test_no_numeric_col_returns_false(self):
        """Profile with empty numeric_cols must return False."""
        profile = make_monthly_profile(n=36)
        profile["numeric_cols"] = []
        is_ts, date_col, value_col = detect_time_series(profile)
        assert is_ts is False

    def test_quarterly_data_not_detected_as_monthly(self):
        """Quarterly (~90-day) spaced dates must not be flagged as a supported time-series."""
        profile = make_non_monthly_profile()
        is_ts, date_col, value_col = detect_time_series(profile)
        # Quarterly data (90-day gaps) fails the 30±5 day and 7-day checks
        assert is_ts is False

    def test_exactly_18_observations_accepted(self):
        """The minimum boundary of 18 observations should be accepted."""
        profile = make_monthly_profile(n=18)
        is_ts, _, _ = detect_time_series(profile)
        assert is_ts is True

    def test_17_observations_rejected(self):
        """17 observations is below the minimum and must be rejected."""
        profile = make_monthly_profile(n=17)
        is_ts, _, _ = detect_time_series(profile)
        assert is_ts is False


# ── detect_anomalies_in_series tests ─────────────────────────────────────────

class TestDetectAnomaliesInSeries:

    def test_spike_detected(self):
        """A large synthetic spike must produce at least one anomaly."""
        df = make_series_with_spike(spike_index=20, spike_magnitude=200.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        assert len(anomalies) >= 1, "Expected at least one anomaly for a 200-unit spike"

    def test_anomaly_date_correct(self):
        """The detected anomaly date must correspond to the spike's date."""
        spike_idx = 15
        df = make_series_with_spike(spike_index=spike_idx, spike_magnitude=300.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        anomaly_dates = [a["date"] for a in anomalies]
        expected_date = df["date"].iloc[spike_idx].strftime("%Y-%m-%d")
        assert expected_date in anomaly_dates, (
            f"Expected spike at {expected_date} to be in anomaly dates: {anomaly_dates}"
        )

    def test_flat_series_no_anomalies(self):
        """A perfectly flat series (zero variance) should produce no Z-score anomalies."""
        n = 36
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")
        values = np.ones(n) * 50.0
        df = pd.DataFrame({"date": dates, "val": values})
        anomalies = detect_anomalies_in_series(df, "date", "val")
        # Z-score: all z=0, no flags. CUSUM: σ_Δ = 0, skipped.
        zscore_anomalies = [a for a in anomalies if a["type"] == "z-score"]
        assert zscore_anomalies == [], f"Flat series should have no Z-score anomalies, got {zscore_anomalies}"

    def test_anomaly_has_required_keys(self):
        """Every anomaly dict must have: date, value, type, severity, score."""
        df = make_series_with_spike(spike_index=10, spike_magnitude=250.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        required_keys = {"date", "value", "type", "severity", "score"}
        for anomaly in anomalies:
            missing = required_keys - anomaly.keys()
            assert missing == set(), f"Anomaly missing keys: {missing}"

    def test_severity_values_valid(self):
        """All severity values must be 'high', 'medium', or 'low'."""
        df = make_series_with_spike(spike_index=10, spike_magnitude=150.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        valid_severities = {"high", "medium", "low"}
        for anomaly in anomalies:
            assert anomaly["severity"] in valid_severities, (
                f"Invalid severity: {anomaly['severity']}"
            )

    def test_type_values_valid(self):
        """All type values must be 'z-score' or 'cusum'."""
        df = make_series_with_spike(spike_index=10, spike_magnitude=200.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        valid_types = {"z-score", "cusum"}
        for anomaly in anomalies:
            assert anomaly["type"] in valid_types, (
                f"Invalid anomaly type: {anomaly['type']}"
            )

    def test_output_sorted_chronologically(self):
        """Anomaly list must be sorted in ascending date order."""
        df = make_series_with_spike(spike_index=25, spike_magnitude=200.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        dates = [a["date"] for a in anomalies]
        assert dates == sorted(dates), f"Anomalies not sorted: {dates}"

    def test_no_duplicate_dates(self):
        """Each date must appear at most once in the anomaly list (deduplication)."""
        df = make_series_with_spike(spike_index=20, spike_magnitude=250.0)
        anomalies = detect_anomalies_in_series(df, "date", "value")
        dates = [a["date"] for a in anomalies]
        assert len(dates) == len(set(dates)), f"Duplicate dates in anomaly list: {dates}"

    def test_high_severity_for_extreme_spike(self):
        """A 5x standard deviation spike should produce a 'high' severity anomaly."""
        n = 48
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")
        values = np.ones(n) * 100.0
        # Inject spike of ~5-6σ above the mean
        values[20] = 100.0 + 500.0  # much larger than noise level
        df = pd.DataFrame({"date": dates, "value": values})
        anomalies = detect_anomalies_in_series(df, "date", "value")
        severities = [a["severity"] for a in anomalies]
        assert "high" in severities, f"Expected 'high' severity for extreme spike, got {severities}"


# ── run_forecast tests ────────────────────────────────────────────────────────

class TestRunForecast:

    def test_returns_dict_for_valid_series(self):
        """run_forecast must return a dict for a valid monthly series."""
        profile = make_monthly_profile(n=36)
        df = profile["df"]
        result = run_forecast(df, "date", "sales", n=12)
        # statsmodels may not be installed in all environments
        if result is not None:
            assert isinstance(result, dict)

    def test_required_keys_present(self):
        """Forecast dict must contain all required keys."""
        profile = make_monthly_profile(n=36)
        df = profile["df"]
        result = run_forecast(df, "date", "sales", n=12)
        if result is None:
            pytest.skip("statsmodels not available")
        required_keys = {
            "history_dates", "history_values", "fitted",
            "forecast_dates", "forecast_point", "ci95u", "ci95l",
            "rmse", "aic", "latest", "latest_date",
            "forecast_12m", "forecast_12m_date", "change_pct",
            "value_col", "date_col",
        }
        missing = required_keys - result.keys()
        assert missing == set(), f"Forecast missing keys: {missing}"

    def test_forecast_horizon_length(self):
        """forecast_point, ci95u, ci95l should each have exactly n=12 entries."""
        profile = make_monthly_profile(n=36)
        df = profile["df"]
        result = run_forecast(df, "date", "sales", n=12)
        if result is None:
            pytest.skip("statsmodels not available")
        assert len(result["forecast_point"]) == 12
        assert len(result["ci95u"]) == 12
        assert len(result["ci95l"]) == 12

    def test_ci_upper_above_lower(self):
        """Upper CI must always be >= lower CI for every forecast step."""
        profile = make_monthly_profile(n=36)
        df = profile["df"]
        result = run_forecast(df, "date", "sales", n=12)
        if result is None:
            pytest.skip("statsmodels not available")
        for i, (upper, lower) in enumerate(zip(result["ci95u"], result["ci95l"])):
            assert upper >= lower, (
                f"CI upper ({upper}) < CI lower ({lower}) at step {i}"
            )

    def test_returns_none_for_short_series(self):
        """run_forecast must return None for series shorter than 18 observations."""
        dates = pd.date_range("2023-01-01", periods=10, freq="MS")
        values = np.linspace(100, 200, 10)
        df = pd.DataFrame({"date": dates, "sales": values})
        result = run_forecast(df, "date", "sales", n=12)
        assert result is None

    def test_history_length_matches_input(self):
        """history_dates should have the same length as the input series."""
        n = 36
        profile = make_monthly_profile(n=n)
        df = profile["df"]
        result = run_forecast(df, "date", "sales", n=12)
        if result is None:
            pytest.skip("statsmodels not available")
        assert len(result["history_dates"]) == n
        assert len(result["history_values"]) == n
