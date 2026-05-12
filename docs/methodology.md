# Methodology

This document describes the statistical and machine learning methods used in the AI Strategic Briefing pipeline.

---

## 1. Statistical Profiling

### 1.1 Descriptive Statistics

For each numeric column, the pipeline computes the following on the non-null observations:

| Metric | Formula |
|--------|---------|
| Mean | μ = (1/n) Σ xᵢ |
| Median | Middle value of sorted series |
| Standard deviation | σ = √[ (1/(n-1)) Σ (xᵢ - μ)² ] (Bessel's correction) |
| Minimum / Maximum | min(x), max(x) |
| Q1, Q3 | 25th and 75th percentiles |
| Missing rate | (# null rows / total rows) × 100 |

Columns with fewer than 3 non-null values are skipped.

### 1.2 Outlier Detection (Column-Level)

Two independent outlier metrics are reported for each numeric column:

**Z-score outliers:** Count of observations where |z| > 2.5.

    z_i = |x_i - μ| / σ

Threshold 2.5 corresponds roughly to the 98.8th percentile of a standard normal distribution, capturing about 1.2% of observations under normality as "extreme."

**IQR outliers:** Count of observations outside the "inner fence."

    lower fence = Q1 - 1.5 × IQR
    upper fence = Q3 + 1.5 × IQR
    IQR = Q3 - Q1

This is Tukey's (1977) criterion, which is distribution-free and more robust to heavy tails than Z-score. A multiplier of 1.5 is used (configurable; see `configs/app/config.yaml`).

---

## 2. Anomaly Detection on Time Series

Anomaly detection is only applied when the pipeline detects a time-series structure (see Section 4 below). Two complementary methods are run and their results merged.

### 2.1 Z-Score Method

Applied to the full level series y₁, y₂, …, y_T:

    z_t = |y_t - μ| / σ

where μ and σ are computed over the entire series. Observations with z_t > 2.5 are flagged. Severity levels:

    |z| > 3.5  → high    (exceeds 99.98th percentile under normality)
    |z| > 3.0  → medium  (exceeds 99.73rd percentile)
    |z| > 2.5  → low     (exceeds 98.76th percentile)

**Strengths:** Simple, interpretable. Catches individual observations that deviate sharply from the series distribution.

**Limitations:** Sensitive to the overall distribution shape. Masking effect: multiple simultaneous outliers inflate σ, reducing z-scores and potentially missing anomalies.

### 2.2 CUSUM Change-Point Detection (Page, 1954)

CUSUM (Cumulative Sum) operates on the first-differences of the series:

    Δy_t = y_t - y_{t-1}

Let μ_Δ = mean(Δy) and σ_Δ = std(Δy). Two one-sided CUSUM statistics accumulate evidence of upward and downward regime shifts:

    C_t^+ = max(0,  C_{t-1}^+ + Δy_t - μ_Δ - k)   ← upward shift detector
    C_t^- = max(0,  C_{t-1}^- - Δy_t + μ_Δ - k)   ← downward shift detector

Parameters:
- **k (reference value / allowance):** k = 0.5 × σ_Δ. This is the standard choice (half-sigma), meaning the CUSUM only accumulates when the signal exceeds half a standard deviation above/below the mean. Smaller k makes the detector more sensitive to small shifts.
- **h (decision threshold):** h = 4 × σ_Δ. An alarm fires when C_t^+ > h or C_t^- > h. Threshold of 4σ gives a low false-alarm rate for monthly business data.

After an alarm, both accumulators reset to zero (Page's reset rule). Severity:

    score = max(C^+, C^-) / σ_Δ
    score > 7 → high
    score > 5 → medium
    else      → low

**Strengths:** Detects sustained shifts that no single observation appears extreme — for example, a step-change in mean level over several periods.

**Limitations:** Single reset point per run; may miss back-to-back structural breaks in rapid succession.

### 2.3 Deduplication

When both methods flag the same date, the higher-severity detection is retained. The final list is sorted chronologically.

---

## 3. Time-Series Detection Heuristic

Before applying forecasting or anomaly detection, the pipeline determines whether the data constitutes a time series using the following heuristic:

1. **Date column present:** At least one column must have been coerced to datetime (≥80% parse success on the first 20 non-null values).
2. **Numeric column present:** At least one numeric column must exist to serve as the measured variable.
3. **Minimum length:** After sorting by date and dropping NaT rows, the series must have ≥18 observations. This is the practical minimum for Holt-Winters parameter estimation (see Section 5).
4. **Regular frequency:** The most common inter-observation gap (modal first difference) must be approximately:
   - Monthly: 30 ± 5 days
   - Weekly: 7 ± 1 day
   - Daily: exactly 1 day

If all four conditions are met, the first detected date column and first numeric column are used.

**Design note:** The 18-month minimum is a practical lower bound. Holt-Winters needs at least two full seasonal cycles (24 months ideally) to reliably estimate seasonal parameters. At 18 months, the model will fit but seasonal estimates carry higher uncertainty.

---

## 4. Forecasting: Holt-Winters Triple Exponential Smoothing

### 4.1 Model Overview

The Holt-Winters additive model — designated ETS(A,A,A) in the Hyndman-Khandakar (2008) state-space taxonomy — decomposes a time series into three components:

**Level (ℓ_t):**
    ℓ_t = α · (y_t - s_{t-m}) + (1 - α) · (ℓ_{t-1} + b_{t-1})

**Trend (b_t):**
    b_t = β · (ℓ_t - ℓ_{t-1}) + (1 - β) · b_{t-1}

**Seasonal (s_t):**
    s_t = γ · (y_t - ℓ_{t-1} - b_{t-1}) + (1 - γ) · s_{t-m}

where:
- m = 12 (seasonal period; months per year)
- α ∈ (0, 1): level smoothing parameter
- β ∈ (0, 1): trend smoothing parameter
- γ ∈ (0, 1): seasonal smoothing parameter

**One-step-ahead forecast:**
    ŷ_{t+1} = ℓ_t + b_t + s_{t+1-m}

**h-step-ahead forecast:**
    ŷ_{T+h|T} = ℓ_T + h · b_T + s_{T+h-m}

### 4.2 Parameter Estimation

Parameters α, β, γ and the initial state vector (ℓ_0, b_0, s_{1-m}, …, s_0) are estimated jointly by **maximum likelihood estimation (MLE)** using the L-BFGS-B optimisation algorithm, as implemented in `statsmodels.tsa.holtwinters.ExponentialSmoothing` with `optimized=True` and `initialization_method="estimated"`.

**AIC (Akaike Information Criterion)** is reported as a model quality metric:
    AIC = 2k - 2 ln(L)
where k is the number of estimated parameters and L is the maximised likelihood. Lower AIC indicates better fit per unit of model complexity.

### 4.3 Prediction Intervals

Prediction intervals are computed using a **normal approximation** that scales the in-sample residual standard deviation by the square root of the forecast horizon:

    ŷ_{T+h} ± 1.96 · σ_ε · √h

where σ_ε = std(ŷ_t - y_t) for t = 1, …, T (standard deviation of in-sample residuals).

The √h scaling reflects the intuition that forecast uncertainty grows with the horizon — after h steps, errors from h independent one-step innovations accumulate.

**Limitation:** This is an approximation. Exact prediction intervals for ETS models require the state-space representation and are asymmetric; the normal approximation performs well for h ≤ 12 and large training samples.

### 4.4 Model Selection Rationale

Holt-Winters additive ETS is chosen over ARIMA for the following reasons:

1. **No stationarity requirement:** ARIMA requires differencing to achieve stationarity; Holt-Winters explicitly models trend and seasonality without transformation.
2. **Interpretability:** The α, β, γ parameters have direct business interpretation as "how much weight the recent observation gets."
3. **Robustness with short history:** MLE estimation converges reliably with ≥18 monthly observations. ARIMA with seasonal terms requires more data for stable identification.
4. **Empirical performance:** In the Makridakis M3 and M4 forecast competitions, ETS models outperformed ARIMA on a majority of monthly economic time series.

---

## 5. AI Narrative Generation

### 5.1 Prompt Structure

The pipeline sends a structured context block to `claude-haiku-4-5-20251001` via the Anthropic API. The context includes:

- Dataset title, row count, column count
- Top-5 column statistics (mean, std, outlier counts)
- Up to 5 anomalies (date, value, severity, detection method)
- Forecast summary (latest value, 12-month projection, % change)

The model is instructed to produce a **3-paragraph executive briefing**:
1. Current state — key metrics and trends
2. Risk signals — anomalies and concerning patterns
3. Forward outlook — forecast implications for decision-makers

### 5.2 Prompt Caching

The system prompt is marked with `cache_control: {type: "ephemeral"}` using the Anthropic beta prompt-caching API. This allows the Anthropic infrastructure to cache the system prompt prefix and serve it from cache on subsequent calls within the same caching window (default: 5 minutes). For batch runs over multiple CSVs, this reduces both latency and cost.

### 5.3 Fallback Behaviour

If the Anthropic API is unavailable (no `ANTHROPIC_API_KEY`, network error, or rate limit), the function returns an empty string. The HTML report is still generated with all statistical charts and tables; only the AI narrative section shows a placeholder message.

---

## 6. Limitations

1. **Stationarity assumption (implicit):** After detrending via Holt-Winters, residuals are assumed approximately normally distributed for prediction interval construction. If the residuals show heavy tails or GARCH-type volatility clustering, intervals will be too narrow.

2. **Seasonal period hardcoded to 12:** The pipeline assumes monthly data with annual seasonality. Weekly or daily data with different seasonal periods (e.g., 52 weeks, 365 days) will produce incorrect seasonal decompositions. Extend `configs/app/config.yaml` to make this configurable.

3. **No structural break handling:** CUSUM detects the first major shift, then resets. If the data contains multiple structural breaks (e.g., COVID shock followed by post-COVID regime), subsequent breaks may be missed or reported at the wrong magnitude.

4. **Single numeric column forecasted:** Only the first numeric column in the detected time series is forecasted. Multi-variate time series are not supported in v1.0.

5. **No cross-validation:** Model quality is reported via in-sample RMSE and AIC. Out-of-sample performance is not evaluated. For production use, holdout evaluation is recommended.

6. **Date frequency detection is heuristic:** The most common inter-observation gap is used. Datasets with many missing periods may misclassify as weekly or daily when they are monthly.
