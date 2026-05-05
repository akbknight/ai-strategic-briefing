# AI Strategic Briefing Generator

Autonomous analytics pipeline: CSV input → statistical analysis → anomaly detection → Holt-Winters forecasting → Claude AI narrative → self-contained HTML report. One command, no configuration required.

Live demo: [akbknight.github.io/ai-strategic-briefing](https://akbknight.github.io/ai-strategic-briefing/)  
Sample report: [akbknight.github.io/ai-strategic-briefing/sample_report.html](https://akbknight.github.io/ai-strategic-briefing/sample_report.html)

## What it does

Point the tool at any CSV file. It automatically:

1. **Detects schema** — identifies date columns, numeric fields, missing value rates
2. **Computes statistics** — descriptive stats + IQR/Z-score outlier counts per column
3. **Detects anomalies** — Z-score (point outliers) + CUSUM (regime shifts) on time-series
4. **Forecasts** — Holt-Winters ETS (additive trend + seasonality) with 95% CI, if time-series detected
5. **Generates narrative** — sends statistical context to Claude API for a 3-paragraph executive brief
6. **Exports report** — self-contained HTML with embedded Chart.js charts, shareable without a server

## Quick start

```bash
git clone https://github.com/akbknight/ai-strategic-briefing.git
cd ai-strategic-briefing

pip install pandas numpy scipy statsmodels anthropic

# Run on the included sample data
python briefing.py --input sample_retail_sales.csv --title "US Retail Sales"

# Run on any CSV
python briefing.py --input your_data.csv --title "Q2 Analysis"
```

**With AI narrative** (requires Anthropic API key):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python briefing.py --input your_data.csv
```

Without the key, the tool runs in stats-only mode — all statistical analysis and charts are still generated.

## Sample output

The included `sample_retail_sales.csv` contains 135 months of US Retail & Food Services Sales (FRED RSAFS, Jan 2015–Mar 2026). Running the tool on it:

```
[1/5] Loading and profiling data...
  135 rows × 2 columns
  Time-series: date × retail_sales_billions

[3/5] Running anomaly detection...
  1 anomaly detected (COVID demand shock, April 2020)

[4/5] Running forecast...
  12m forecast: 781.1 (+3.9%) · RMSE 12.2

Report saved: sample_report.html
```

## Arguments

| Flag | Description | Default |
|---|---|---|
| `--input` | Path to CSV file | Required |
| `--title` | Report title | Filename (cleaned) |
| `--output` | Output HTML path | `report_[file]_[date].html` |

## How the pipeline works

**Schema detection:** Heuristic date parsing on string columns (≥80% parse success → promoted to datetime). Numeric/categorical columns classified automatically.

**Time-series detection:** Checks date column + numeric column, verifies ≥18 observations and regular spacing (daily, weekly, or monthly intervals). If detected, activates forecasting and anomaly detection.

**Anomaly detection:**
- Z-score: flags |z| > 2.5 against series mean
- CUSUM: Page (1954) on first differences, threshold 4σ, resets on alarm signal

**Forecasting:** `statsmodels ExponentialSmoothing` with additive trend and additive seasonality (12-period). Parameters estimated by maximum likelihood. Confidence intervals scale by √h with horizon.

**AI narrative:** Sends a structured context block (stats, anomalies, forecast summary) to `claude-haiku-4-5-20251001`. Returns a 3-paragraph brief: current state, risk signals, forward outlook.

**Report format:** Single self-contained HTML file with inline Chart.js 4 and JSON data. No server or build step required — open directly in a browser.

## Tech stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy |
| Statistics | scipy.stats (Z-score, IQR) |
| Anomaly detection | Custom CUSUM (Page 1954) |
| Forecasting | statsmodels Holt-Winters ETS |
| AI narrative | Anthropic Claude API (Haiku) |
| Report | Chart.js 4, HTML/CSS/JS |

## Skills demonstrated

- **Agentic pipeline design:** Multi-step automated workflow with conditional branching (forecast only if time-series, AI only if key present)
- **Statistical analysis:** Descriptive stats, outlier detection, time-series characterization
- **Anomaly detection:** Z-score, CUSUM with reset logic
- **Forecasting:** Holt-Winters ETS, uncertainty quantification
- **AI API integration:** Claude (claude-haiku-4-5) for analytical narrative generation
- **Software engineering:** Clean CLI, graceful degradation, self-contained HTML output

## Author

**Akshay Kumar**  
[linkedin.com/in/akshaykumardl](https://www.linkedin.com/in/akshaykumardl/)
