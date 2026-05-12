"""
src/models/forecast.py
======================
Time-series detection and Holt-Winters forecasting for the AI Strategic
Briefing pipeline.

Holt-Winters Triple Exponential Smoothing (ETS)
-----------------------------------------------
The Holt-Winters additive model — also known as ETS(A,A,A) in the state-space
taxonomy — decomposes a time series into three components:

    Level (l_t)   : l_t = α·(y_t - s_{t-m}) + (1 - α)·(l_{t-1} + b_{t-1})
    Trend (b_t)   : b_t = β·(l_t - l_{t-1})  + (1 - β)·b_{t-1}
    Season (s_t)  : s_t = γ·(y_t - l_{t-1} - b_{t-1}) + (1 - γ)·s_{t-m}

where m = 12 (months per seasonal cycle), and α, β, γ ∈ (0, 1) are
smoothing parameters estimated by maximising the likelihood of the observed
series (MLE via L-BFGS-B optimisation in statsmodels).

The h-step-ahead point forecast is:
    ŷ_{T+h} = l_T + h·b_T + s_{T+h-m}

Prediction intervals are constructed using a normal approximation, scaling
the in-sample residual standard deviation by √h to reflect growing
uncertainty over the forecast horizon.

The model is selected over ARIMA because:
- It handles trend and seasonality without differencing
- Parameters are interpretable as exponentially weighted averages
- Performance on monthly retail/business data is well-documented
  (Makridakis M3 and M4 competitions)
- It is robust to short history (≥18 months)

References
----------
- Holt, C.E. (1957). Forecasting seasonals and trends by exponentially
  weighted moving averages. ONR Research Memorandum 52.
- Winters, P.R. (1960). Forecasting sales by exponentially weighted moving
  averages. Management Science, 6(3), 324-342.
- Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and
  Practice (3rd ed.). OTexts. https://otexts.com/fpp3/
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _STATSMODELS_AVAILABLE = True
except ImportError:
    pass


def detect_time_series(
    profile: dict,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Determine whether the profiled DataFrame contains a plausible time series.

    Heuristic:
    1. The profile must contain at least one date column and one numeric column.
    2. After sorting by the date column and dropping NaT rows, at least 18
       observations must remain (Holt-Winters needs >= 2 full seasonal cycles;
       18 months is the practical minimum).
    3. The most common inter-observation gap must be approximately monthly
       (~30 ± 5 days), weekly (7 ± 1 day), or daily (1 day).

    Parameters
    ----------
    profile : dict
        Output of ``load_and_profile()``.

    Returns
    -------
    tuple[bool, str | None, str | None]
        (is_time_series, date_column_name, value_column_name)
        If not a time series, the second and third elements are None.
    """
    if not profile["date_cols"] or not profile["numeric_cols"]:
        return False, None, None

    date_col: str = profile["date_cols"][0]
    df: pd.DataFrame = (
        profile["df"]
        .sort_values(date_col)
        .dropna(subset=[date_col])
    )

    if len(df) < 18:
        return False, None, None

    diffs: pd.Series = df[date_col].diff().dropna()
    most_common_modes = diffs.mode()
    if most_common_modes.empty:
        return False, None, None

    typical = most_common_modes.iloc[0]
    days: int = typical.days

    # Accept monthly (~30 days), weekly (7 days), or daily (1 day) cadence
    if (abs(days - 30) <= 5) or (abs(days - 7) <= 1) or (days == 1):
        value_col: str = profile["numeric_cols"][0]
        return True, date_col, value_col

    return False, None, None


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    n: int = 12,
) -> Optional[dict]:
    """Fit a Holt-Winters ETS(A,A,A) model and return a 12-month forecast.

    The model uses additive trend and additive seasonality with a seasonal
    period of 12 (months).  Parameters are estimated by MLE.  Prediction
    intervals are computed as:

        CI_upper(h) = ŷ_{T+h} + 1.96 · σ · √h
        CI_lower(h) = ŷ_{T+h} - 1.96 · σ · √h

    where σ is the standard deviation of in-sample residuals.

    Parameters
    ----------
    df : pd.DataFrame
        The profiled DataFrame (must contain ``date_col`` and ``value_col``).
    date_col : str
        Name of the datetime index column.
    value_col : str
        Name of the numeric series to forecast.
    n : int, optional
        Number of periods to forecast ahead (default: 12).

    Returns
    -------
    dict | None
        Forecast payload dict ready for JSON serialisation, or None if the
        model could not be fitted (e.g., insufficient data or import error).
    """
    if not _STATSMODELS_AVAILABLE:
        return None

    try:
        ts: pd.Series = (
            df.set_index(date_col)[value_col]
            .dropna()
            .sort_index()
        )
        if len(ts) < 18:
            return None

        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        forecast: pd.Series = fitted.forecast(n)

        forecast_index = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=n,
            freq="MS",
        )

        residual_std: float = float(np.std(fitted.resid))
        horizon_factor: np.ndarray = np.sqrt(np.arange(1, n + 1))

        return {
            "history_dates": [d.strftime("%Y-%m-%d") for d in ts.index],
            "history_values": [round(float(v), 3) for v in ts],
            "fitted": [round(float(v), 3) for v in fitted.fittedvalues],
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_index],
            "forecast_point": [round(float(v), 3) for v in forecast],
            "ci95u": [
                round(float(forecast.iloc[i] + 1.96 * residual_std * horizon_factor[i]), 3)
                for i in range(n)
            ],
            "ci95l": [
                round(float(forecast.iloc[i] - 1.96 * residual_std * horizon_factor[i]), 3)
                for i in range(n)
            ],
            "rmse": round(float(np.sqrt(np.mean(fitted.resid ** 2))), 4),
            "aic": round(float(fitted.aic), 1),
            "latest": round(float(ts.iloc[-1]), 3),
            "latest_date": ts.index[-1].strftime("%B %Y"),
            "forecast_12m": round(float(forecast.iloc[-1]), 3),
            "forecast_12m_date": forecast_index[-1].strftime("%B %Y"),
            "change_pct": round(
                (float(forecast.iloc[-1]) - float(ts.iloc[-1]))
                / float(ts.iloc[-1])
                * 100,
                2,
            ),
            "value_col": value_col,
            "date_col": date_col,
        }

    except Exception as exc:
        print(f"  Forecast skipped: {exc}")
        return None
