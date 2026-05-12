"""
src/features/detection.py
=========================
Anomaly detection for time-series data in the AI Strategic Briefing pipeline.

Two complementary methods are applied:

Z-Score Method
--------------
For each observation y_t in the series, compute the standardised score:

    z_t = |y_t - μ| / σ

where μ and σ are the series mean and standard deviation.  Any observation
with z_t > 2.5 is flagged as an anomaly (threshold configurable via
configs/app/config.yaml).  Severity levels:

    |z| > 3.5  → high
    |z| > 3.0  → medium
    |z| > 2.5  → low

This method excels at detecting point outliers — individual observations
that deviate sharply from the distribution.

CUSUM (Cumulative Sum) Change-Point Detection
---------------------------------------------
Based on Page (1954), CUSUM operates on the first-differences Δy_t = y_t - y_{t-1}.
Two cumulative sums track upward and downward regime shifts:

    C_t^+ = max(0, C_{t-1}^+ + Δy_t - μ_Δ - k)
    C_t^- = max(0, C_{t-1}^- - Δy_t + μ_Δ - k)

where:
    μ_Δ  = mean of first differences
    k    = 0.5 · σ_Δ  (allowance / reference value; half the noise level)
    σ_Δ  = standard deviation of first differences

An alarm fires when C_t^+ > h or C_t^- > h, where h = 4·σ_Δ is the
decision threshold.  After an alarm the accumulators reset to zero.
Severity:

    score > 7σ → high
    score > 5σ → medium
    else       → low

CUSUM excels at detecting sustained regime shifts that Z-score misses because
no single point is extreme — only the accumulated drift is.

Deduplication
-------------
When both methods flag the same date, the higher-severity detection wins.
The final list is sorted chronologically.

References
----------
- Page, E.S. (1954). Continuous inspection schemes. Biometrika, 41(1/2), 100-115.
- Montgomery, D.C. (2009). Introduction to Statistical Quality Control (6th ed.).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def detect_anomalies_in_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> list[dict[str, Any]]:
    """Detect anomalies in a time series using Z-score and CUSUM methods.

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame containing ``date_col`` and ``value_col``.
    date_col : str
        Name of the datetime column used as the series index.
    value_col : str
        Name of the numeric column to analyse.

    Returns
    -------
    list[dict]
        Chronologically sorted list of anomaly dicts.  Each dict contains:
        date (str, YYYY-MM-DD), value (float), type ('z-score' | 'cusum'),
        severity ('high' | 'medium' | 'low'), score (float, in sigma units).
    """
    ts: pd.Series = (
        df.set_index(date_col)[value_col]
        .dropna()
        .sort_index()
    )
    anomalies: list[dict[str, Any]] = []

    # ── Z-score anomaly detection ────────────────────────────────────────────
    z_scores: np.ndarray = np.abs(stats.zscore(ts.values))
    for i, (date, val) in enumerate(ts.items()):
        z: float = float(z_scores[i])
        if z > 2.5:
            if z > 3.5:
                severity = "high"
            elif z > 3.0:
                severity = "medium"
            else:
                severity = "low"
            anomalies.append({
                "date": str(date)[:10],
                "value": round(float(val), 3),
                "type": "z-score",
                "severity": severity,
                "score": round(z, 2),
            })

    # ── CUSUM change-point detection ─────────────────────────────────────────
    diff: pd.Series = ts.diff().dropna()
    mu_d: float = float(diff.mean())
    sigma_d: float = float(diff.std())

    if sigma_d > 0:
        k: float = 0.5 * sigma_d          # reference / allowance value
        threshold: float = 4.0 * sigma_d  # decision threshold h

        cp: float = 0.0   # positive (upward shift) accumulator
        cn: float = 0.0   # negative (downward shift) accumulator
        alarm: bool = False

        for date, dv in diff.items():
            cp = max(0.0, cp + float(dv) - mu_d - k)
            cn = max(0.0, cn - float(dv) + mu_d - k)

            if (cp > threshold or cn > threshold) and not alarm:
                cusum_score: float = max(cp, cn) / sigma_d
                if cusum_score > 7:
                    severity = "high"
                elif cusum_score > 5:
                    severity = "medium"
                else:
                    severity = "low"
                anomalies.append({
                    "date": str(date)[:10],
                    "value": round(float(ts[date]), 3),
                    "type": "cusum",
                    "severity": severity,
                    "score": round(cusum_score, 2),
                })
                alarm = True
                cp, cn = 0.0, 0.0  # reset after alarm

            elif cp <= threshold and cn <= threshold:
                alarm = False

    # ── Deduplication: keep highest-severity detection per date ──────────────
    priority: dict[str, int] = {"high": 3, "medium": 2, "low": 1}
    seen: dict[str, dict[str, Any]] = {}
    for anomaly in anomalies:
        key: str = anomaly["date"]
        if key not in seen or priority[anomaly["severity"]] > priority[seen[key]["severity"]]:
            seen[key] = anomaly

    return sorted(seen.values(), key=lambda x: x["date"])
