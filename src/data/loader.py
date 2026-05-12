"""
src/data/loader.py
==================
CSV ingestion and schema profiling for the AI Strategic Briefing pipeline.

Responsibilities:
- Read arbitrary CSV files with pandas
- Infer date columns via heuristic parse-success rate
- Classify columns into date, numeric, and categorical buckets
- Compute per-column missing-value rates
- Return a unified profile dict consumed by downstream pipeline stages
"""

import numpy as np
import pandas as pd


def load_and_profile(path: str) -> dict:
    """Load a CSV file and return a unified profile dict.

    Performs schema detection: heuristic date parsing on object-dtype columns
    (a column is promoted to datetime if >= 80% of the first 20 non-null
    values parse successfully), then classifies remaining columns as numeric
    or categorical.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    dict with keys:
        df          : pd.DataFrame  — loaded and partially type-coerced data
        n_rows      : int           — row count
        n_cols      : int           — column count (original)
        columns     : list[str]     — original column names in order
        date_cols   : list[str]     — columns successfully coerced to datetime
        numeric_cols: list[str]     — numeric (int/float) columns
        cat_cols    : list[str]     — object columns not identified as dates
        missing_pct : dict[str,float] — percent missing per column (0-100)
    """
    df: pd.DataFrame = pd.read_csv(path)
    original_cols: list[str] = list(df.columns)

    # Detect date columns via heuristic: try parsing the first 20 non-null
    # values; if >= 80% parse cleanly, promote the entire column to datetime.
    date_cols: list[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= len(sample) * 0.8:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    date_cols.append(col)
                except Exception:
                    pass

    numeric_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: list[str] = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in date_cols
    ]

    missing_pct: dict[str, float] = (
        (df.isnull().sum() / len(df) * 100).round(1).to_dict()
    )

    return {
        "df": df,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": original_cols,
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "missing_pct": missing_pct,
    }
