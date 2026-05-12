"""
src/data/profiler.py
====================
Per-column descriptive statistics for the AI Strategic Briefing pipeline.

Computes mean, median, standard deviation, quartiles, missing-value rate,
and outlier counts (Z-score and IQR methods) for every numeric column in a
DataFrame.  Returns a list of dicts that is serialised to JSON and embedded
in the final HTML report.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compute_stats(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
    """Compute descriptive statistics and outlier counts for numeric columns.

    For each column the following metrics are computed on the non-null values:

    - Descriptive: count, mean, median, std, min, max, Q1, Q3
    - Missing: percentage of null rows across the full DataFrame
    - Z-score outliers: count of |z| > 2.5
    - IQR outliers: count of values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR

    Parameters
    ----------
    df : pd.DataFrame
        The full (loaded and type-coerced) DataFrame.
    numeric_cols : list[str]
        Column names to analyse.  Columns with fewer than 3 non-null values
        are silently skipped.

    Returns
    -------
    list[dict]
        One dict per column, ordered the same as ``numeric_cols``.
        Keys: column, count, mean, median, std, min, max, q1, q3,
              missing_pct, outliers_zscore, outliers_iqr.
    """
    results: list[dict[str, Any]] = []

    for col in numeric_cols:
        series: pd.Series = df[col].dropna()
        if len(series) < 3:
            continue

        z_scores: np.ndarray = np.abs(stats.zscore(series))
        q1: float = float(series.quantile(0.25))
        q3: float = float(series.quantile(0.75))
        iqr: float = q3 - q1

        results.append({
            "column": col,
            "count": int(len(series)),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "missing_pct": round(
                float(df[col].isnull().sum() / len(df) * 100), 1
            ),
            "outliers_zscore": int((z_scores > 2.5).sum()),
            "outliers_iqr": int(
                ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
            ),
        })

    return results
