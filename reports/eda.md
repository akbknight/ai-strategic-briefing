# Exploratory Data Analysis: sample_retail_sales.csv

EDA for the included US Retail & Food Services Sales dataset.

---

## Dataset Overview

**Source:** US Advance Monthly Retail Trade Survey, Federal Reserve Bank of St. Louis (FRED series RSAFS)
**File:** `data/raw/sample_retail_sales.csv`
**Columns:** 2 (`date`, `retail_sales_billions`)
**Rows:** 135 monthly observations
**Time span:** January 2015 – March 2026 (11 years, 3 months)
**Missing values:** None (0% in both columns)

---

## Descriptive Statistics

| Metric | Value |
|--------|-------|
| Minimum | $401.03B (April 2020) |
| Maximum | $752.06B (March 2026) |
| Mean | $575.93B |
| Median | $548.12B |
| Standard deviation | $95.8B |
| Q1 (25th percentile) | $491.0B |
| Q3 (75th percentile) | $660.0B |
| IQR | ~$169B |

The mean is notably above the median, indicating right-skew driven by the strong post-2021 acceleration in sales levels.

---

## Trend Analysis

The series exhibits a strong, persistent upward trend from roughly $428B in January 2015 to $752B in March 2026 — a total increase of approximately 75% over 11 years, or an average compound annual growth rate of roughly 5.3% per year.

Trend is not constant across the full period:

- **2015–2019:** Moderate growth at ~3–4% per year, consistent with nominal GDP growth
- **2020 Q1:** Sharp decline beginning in March 2020 ($468B, -9% from Feb); trough in April 2020 ($401B, -22% from Jan)
- **2020 Q2–2021:** V-shaped recovery and then acceleration fueled by fiscal stimulus (Economic Impact Payments) and pent-up demand; rapid ascent to $600B+ by late 2020
- **2021–2022:** Continued acceleration, likely reflecting pandemic-era goods spending displacement (from services) and inflation; peak monthly growth of 3–4% per quarter
- **2023–2026:** Growth moderating toward 2–3% per year as inflation and goods consumption normalize

The Holt-Winters model will capture this trend via the level (ℓ_t) and trend (b_t) components with β-weighted smoothing.

---

## Seasonality

The series displays clear annual seasonality:

- **Q4 (October–December):** Consistent highs driven by holiday retail season; November and December are typically the two strongest months
- **Q1 (January–February):** Post-holiday slowdown; typically the weakest months of the year
- **Mid-year (May–August):** Moderate strength with summer spending

The seasonal amplitude (Q4 peak minus Q1 trough within a year) is approximately $20–30B in 2015–2019, widening to $40–60B by 2022–2025 as the overall level increased. This mild proportional increase in amplitude suggests the additive seasonal model is a reasonable approximation (as used), though a multiplicative model might marginally improve fit in later years.

The Holt-Winters seasonal period is set to **12** (months per year), which correctly captures the annual cycle.

---

## Anomaly Analysis

Running the pipeline on this dataset detects the following anomalies:

**April 2020 (Z-score, high severity):**
- Value: $401.03B
- Z-score: approximately 1.85 standard deviations below the mean on the level series; however, the first difference (Δy = −67 for April 2020 vs +0.21 average) produces a Z-score of approximately 3.8σ on the differenced series, making it easily detectable
- Cause: COVID-19 mandatory lockdowns, business closures, and consumer fear
- Context: Sharpest single-month decline in the post-2000 RSAFS record; the recovery in May 2020 (+$77B) was equally extreme, but the Z-score detector normalises against the full series including the recovery

**CUSUM detection — regime shift circa 2020–2021:**
- The CUSUM detector identifies a sustained upward regime shift beginning approximately May 2020 (post-lockdown recovery) through early 2021
- This reflects the structural acceleration from fiscal stimulus and goods spending displacement that persisted for nearly two years
- The shift is invisible to Z-score (no individual month is extreme once the recovery began) but clearly flagged by accumulated CUSUM drift

---

## Forecast Readiness Assessment

The dataset is well-suited for Holt-Winters forecasting:

- 135 monthly observations — well above the 18-month minimum and 24-month recommended threshold
- Consistent monthly frequency with no gaps (all months present, no NaT values)
- Clear trend (upward, though post-2022 growth is decelerating)
- Clear annual seasonality (Q4 peak, Q1 trough)
- One structural break (COVID 2020) — Holt-Winters will absorb this into level estimates without explicit break modelling, which may slightly inflate residuals near that period

Expected model output: 12-month forecast from April 2026 through March 2027 showing continued moderate growth (~2–4% annually), with 95% confidence interval reflecting residual uncertainty from the 2020 disruption period.
