# Data Dictionary

This document describes the expected CSV input format and the structure of the HTML report output.

---

## Input: CSV File Format

The pipeline accepts any well-formed CSV file. Schema detection is automatic — no configuration is required for standard inputs.

### Requirements

| Requirement | Detail |
|-------------|--------|
| Format | UTF-8 encoded CSV with a header row |
| Minimum rows | 1 (stats-only mode); ≥18 (to activate time-series mode) |
| Minimum columns | 1 numeric column |
| Date column | Optional; auto-detected if present |

### Date Column Detection

A column is promoted to datetime if:
- Its dtype is `object` (string)
- At least 80% of the first 20 non-null values parse successfully with `pd.to_datetime(..., errors="coerce")`

Any ISO 8601 format (`YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, `MM/DD/YYYY`, etc.) is supported.

### Included Sample Dataset: `data/raw/sample_retail_sales.csv`

**Source:** US Advance Monthly Retail Trade Survey (FRED series RSAFS), seasonally unadjusted, Jan 2015 – Mar 2026.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date (YYYY-MM-DD) | First calendar day of each month |
| `retail_sales_billions` | float | Total US Retail & Food Services Sales, billions of dollars |

**Key characteristics:**
- 135 monthly observations (Jan 2015 – Mar 2026)
- Upward trend: ~$428B in Jan 2015 → ~$752B in Mar 2026
- Annual seasonality: Q4 (holiday shopping) peaks each year
- Notable anomaly: April 2020 (COVID lockdown shock; $401B, the series minimum)
- Post-COVID step-change: rapid acceleration through 2021–2022

---

## Pipeline Intermediate Data Structures

### Profile Dict (output of `load_and_profile`)

```python
{
    "df":           pd.DataFrame,     # type-coerced data
    "n_rows":       int,              # number of rows
    "n_cols":       int,              # number of columns
    "columns":      list[str],        # original column names
    "date_cols":    list[str],        # columns promoted to datetime
    "numeric_cols": list[str],        # numeric (int/float) columns
    "cat_cols":     list[str],        # string columns not identified as dates
    "missing_pct":  dict[str, float], # % missing per column
}
```

### Column Stats (output of `compute_stats`)

```python
[
    {
        "column":         str,    # column name
        "count":          int,    # non-null count
        "mean":           float,
        "median":         float,
        "std":            float,
        "min":            float,
        "max":            float,
        "q1":             float,  # 25th percentile
        "q3":             float,  # 75th percentile
        "missing_pct":    float,  # 0.0–100.0
        "outliers_zscore":int,    # count of |z| > 2.5
        "outliers_iqr":   int,    # count outside Q1 - 1.5×IQR .. Q3 + 1.5×IQR
    },
    ...
]
```

### Forecast Dict (output of `run_forecast`)

```python
{
    "history_dates":    list[str],    # YYYY-MM-DD strings, training period
    "history_values":   list[float],  # observed values
    "fitted":           list[float],  # in-sample fitted values
    "forecast_dates":   list[str],    # YYYY-MM-DD strings, next 12 months
    "forecast_point":   list[float],  # point forecast
    "ci95u":            list[float],  # 95% CI upper bound
    "ci95l":            list[float],  # 95% CI lower bound
    "rmse":             float,        # root mean squared error (in-sample)
    "aic":              float,        # Akaike Information Criterion
    "latest":           float,        # most recent observed value
    "latest_date":      str,          # "Month YYYY" format
    "forecast_12m":     float,        # 12-month-ahead point forecast
    "forecast_12m_date":str,          # "Month YYYY" format
    "change_pct":       float,        # % change from latest to 12m forecast
    "value_col":        str,          # name of forecasted column
    "date_col":         str,          # name of date column
}
```

### Anomaly Dict (one entry in output of `detect_anomalies_in_series`)

```python
{
    "date":     str,   # YYYY-MM-DD
    "value":    float, # observed value at anomaly date
    "type":     str,   # "z-score" | "cusum"
    "severity": str,   # "high" | "medium" | "low"
    "score":    float, # σ-units above threshold
}
```

---

## Output: HTML Report Structure

The output is a single `.html` file named `report_<stem>_<YYYYMMDD>.html` by default.

### Sections

| Section | Condition | Content |
|---------|-----------|---------|
| Header | Always | Title, generation timestamp, row/column counts |
| AI Executive Briefing | Always | AI narrative (3 paragraphs) or placeholder if no API key |
| 12-Month Forecast | If time-series detected | KPI cards (latest, 12m forecast, % change, RMSE) + Chart.js line chart |
| Anomalies Detected | If anomalies found | Table of up to 20 anomalies with date, value, severity, method, score |
| Column Statistics | Always | One card per numeric column with all descriptive stats |
| Footer | Always | Generator attribution and source filename |

### Embedded JavaScript Data

The report embeds three JSON payloads in a `<script>` block:

```javascript
const STATS    = [...];  // array of column stat dicts
const FORECAST = {...};  // forecast dict or null
const ANOMALIES = [...]; // array of anomaly dicts
```

The `renderStats()` and `renderForecast()` functions build the statistics grid and Chart.js chart client-side from these payloads.

### Chart.js Forecast Chart

The forecast chart renders four datasets:

| Dataset | Color | Description |
|---------|-------|-------------|
| Actual (last 48 months) | Amber `#E8A020` | Historical observed values |
| Forecast (12 months) | Blue `#60a5fa` dashed | Point forecast |
| 95% CI upper | Blue transparent | Upper confidence bound |
| 95% CI lower | Blue transparent | Lower confidence bound (fills to upper) |

The chart is responsive and renders at 240px height inside a `position: relative` wrapper.
