# Architecture

This document describes the system architecture of the AI Strategic Briefing pipeline.

---

## 1. Pipeline Stages

```
CSV Input
   │
[1] load_and_profile()  → schema detection, date inference, type coercion
   │
[2] compute_stats()     → per-column descriptive statistics
   │
[3] detect_time_series() → heuristic: date col + numeric col + ≥18 monthly obs
   │
[4a] detect_anomalies() → Z-score + CUSUM, deduplication by date
[4b] run_forecast()     → Holt-Winters ETS, 12-month horizon, 95% CI
   │
[5] generate_ai_summary() → Claude Haiku API, executive brief
   │
[6] build_report()      → Self-contained HTML with Chart.js charts
   │
HTML Output
```

Stages 4a and 4b are **conditional**: they only run if stage 3 detects a time-series structure. Stage 5 is **conditional** on `ANTHROPIC_API_KEY` being set in the environment. Stages 1-3 and 6 always run.

---

## 2. Module Dependency Graph

```
src/cli/run.py
├── src/data/loader.py          load_and_profile()
├── src/data/profiler.py        compute_stats()
├── src/models/forecast.py      detect_time_series(), run_forecast()
├── src/features/detection.py   detect_anomalies_in_series()
├── src/ai/narrative.py         generate_ai_summary()
└── src/report/builder.py       build_report(), HTML_TEMPLATE

briefing.py  →  src/cli/run.main()   (compatibility shim, no logic)
```

Dependencies are strictly one-directional: the CLI orchestrates all modules; no module imports from another module at the same or lower level.

---

## 3. Data Flow

```
Stage 1 — load_and_profile(path: str) → profile: dict
    Input:  path to CSV file
    Output: {
        df: pd.DataFrame,        # type-coerced data
        n_rows: int,
        n_cols: int,
        columns: list[str],
        date_cols: list[str],
        numeric_cols: list[str],
        cat_cols: list[str],
        missing_pct: dict[str, float]
    }

Stage 2 — compute_stats(df, numeric_cols) → col_stats: list[dict]
    Input:  profile["df"], profile["numeric_cols"]
    Output: one dict per column with count, mean, std, min, max, q1, q3,
            missing_pct, outliers_zscore, outliers_iqr

Stage 3 — detect_time_series(profile) → (bool, date_col, value_col)
    Input:  profile dict
    Output: (True, "date", "sales") or (False, None, None)

Stage 4a — detect_anomalies_in_series(df, date_col, value_col) → anomalies: list[dict]
    Input:  DataFrame, column names
    Output: [{date, value, type, severity, score}, ...]

Stage 4b — run_forecast(df, date_col, value_col, n=12) → forecast: dict | None
    Input:  DataFrame, column names, forecast horizon
    Output: {history_dates, history_values, fitted, forecast_dates,
             forecast_point, ci95u, ci95l, rmse, aic, latest,
             latest_date, forecast_12m, forecast_12m_date, change_pct,
             value_col, date_col}

Stage 5 — generate_ai_summary(context: dict) → ai_summary: str
    Input:  {title, n_rows, n_cols, numeric_cols, stats, n_anomalies,
             anomalies, forecast_summary}
    Output: 3-paragraph text narrative (or "" if API unavailable)

Stage 6 — build_report(...) → html: str
    Input:  all stage outputs
    Output: complete self-contained HTML document string
```

---

## 4. Module Descriptions

### `src/data/loader.py`

**Responsibility:** Ingest a CSV and produce a unified schema profile.

Key logic:
- Reads CSV with `pd.read_csv()` using default type inference
- Heuristic date detection: samples first 20 non-null values of object columns; promotes to datetime if ≥80% parse
- Classifies remaining columns into numeric (int/float dtype) and categorical (object dtype, not date)
- Computes missing-value rates per column

No external state. Pure function: same input always yields same output.

### `src/data/profiler.py`

**Responsibility:** Compute descriptive statistics for numeric columns.

Uses `scipy.stats.zscore` for Z-score computation. Returns a list of flat dicts suitable for JSON serialisation.

### `src/models/forecast.py`

**Responsibility:** Time-series detection heuristic and Holt-Winters forecasting.

`detect_time_series()` checks date presence, minimum length (18 rows), and inter-observation frequency. `run_forecast()` wraps `statsmodels.tsa.holtwinters.ExponentialSmoothing` with additive trend and additive seasonality, MLE parameter estimation, and √h-scaled prediction intervals.

### `src/features/detection.py`

**Responsibility:** Z-score and CUSUM anomaly detection on a univariate time series.

Z-score detects point outliers; CUSUM detects sustained regime shifts. Results are merged and deduplicated by date (highest severity wins).

### `src/ai/narrative.py`

**Responsibility:** Generate executive narrative via Anthropic Claude API.

Uses prompt caching (system prompt marked with `cache_control: {type: "ephemeral"}`) to reduce API cost on repeated calls. Falls back gracefully to standard `messages.create` if beta API is unavailable. Returns empty string if `ANTHROPIC_API_KEY` is not set.

### `src/report/builder.py`

**Responsibility:** Assemble the final HTML report.

Stores the full HTML template as a module-level string constant (`HTML_TEMPLATE`). `build_report()` performs string substitution for metadata and JSON injection for statistical data. The Chart.js forecast visualisation is rendered client-side from embedded JSON.

### `src/cli/run.py`

**Responsibility:** CLI argument parsing and pipeline orchestration.

The only file that imports from all other modules. Handles file validation, title auto-generation, output path construction, and progress printing.

---

## 5. Design Decisions

### Why single-file HTML output?

The report is a self-contained HTML file with Chart.js loaded from CDN and all data embedded as inline JSON. This means:

- **Portability:** The file can be emailed, uploaded to any web host, or opened locally without a server
- **No build step:** Recipients open it directly in a browser
- **Archivability:** The file is a complete record of the analysis at a point in time

The tradeoff is that the CDN-hosted Chart.js requires internet access to render charts. For offline use, Chart.js could be inlined, but this would increase file size by ~200 KB.

### Why Holt-Winters over ARIMA?

| Criterion | Holt-Winters ETS | ARIMA |
|-----------|-----------------|-------|
| Stationarity required | No (models trend explicitly) | Yes (differencing needed) |
| Seasonal handling | Built-in (additive or multiplicative) | Requires SARIMA extension |
| Parameter interpretability | High (α, β, γ as smoothing weights) | Lower (AR/MA lag terms) |
| Data requirements | 18+ monthly obs | 36+ monthly obs recommended |
| M3/M4 competition accuracy | Competitive | Comparable |

For monthly business data with trend and seasonality and short history, Holt-Winters is the better default.

### Why Claude Haiku?

- Fastest and cheapest Anthropic model
- Sufficient capability for structured briefing from pre-processed statistical summaries
- 600 max_tokens is adequate for 3 paragraphs
- Prompt caching further reduces cost for repeat runs

### Why 18-month minimum?

Holt-Winters has m + 2 seasonal parameters (12 seasonal indices + level + trend) plus 3 smoothing parameters = 17 parameters total for monthly data. 18 observations is the absolute floor for estimation; 24 months (2 full cycles) is recommended for reliable seasonal estimates.
