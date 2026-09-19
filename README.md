# AI Strategic Briefing Generator

> [!IMPORTANT]
> **Flagship Autonomous Intelligence Pipeline · Part of the Akshay Kumar Technical Portfolio Ecosystem**  
> 🌐 **Executive Portfolio:** [https://akbknight.github.io/](https://akbknight.github.io/) · 💼 **LinkedIn:** [linkedin.com/in/akshaykumardl](https://www.linkedin.com/in/akshaykumardl/) · 📄 **Curriculum Vitae:** [Download PDF (369 KB)](https://akbknight.github.io/assets/Akshay_Resume.pdf)

[![Live Application](https://img.shields.io/badge/Live%20Application-GitHub%20Pages-0284c7?style=flat-square&logo=github)](https://akbknight.github.io/ai-strategic-briefing/)
[![Sample Report](https://img.shields.io/badge/Sample%20Report-Executive%20HTML-10b981?style=flat-square)](https://akbknight.github.io/ai-strategic-briefing/sample_report.html)
[![AI Engine](https://img.shields.io/badge/AI%20Engine-Claude%20Haiku%20(Prompt%20Caching)-8b5cf6?style=flat-square)](https://www.anthropic.com/)
[![Forecasting](https://img.shields.io/badge/Forecasting-Holt--Winters%20ETS(A%2CA%2CA)-f59e0b?style=flat-square)](https://www.statsmodels.org/)
[![Author](https://img.shields.io/badge/Author-Akshay%20Kumar-09090b?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/akshaykumardl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

Autonomous analytics pipeline: CSV input → statistical profiling → anomaly detection (Z-Score & CUSUM) → Holt-Winters econometric forecasting → Claude AI executive narrative → self-contained interactive HTML report. One command, zero configuration required.

**Live Application:** [https://akbknight.github.io/ai-strategic-briefing/](https://akbknight.github.io/ai-strategic-briefing/)  
**Sample Generated Report:** [https://akbknight.github.io/ai-strategic-briefing/sample_report.html](https://akbknight.github.io/ai-strategic-briefing/sample_report.html)

---

## Architecture

```
CSV Input
   │
[1] load_and_profile()    → schema detection, date inference, type coercion
   │
[2] compute_stats()       → per-column descriptive statistics
   │
[3] detect_time_series()  → heuristic: date col + numeric col + ≥18 monthly obs
   │
[4a] detect_anomalies()   → Z-score + CUSUM, deduplication by date
[4b] run_forecast()       → Holt-Winters ETS, 12-month horizon, 95% CI
   │
[5] generate_ai_summary() → Claude Haiku API, executive brief
   │
[6] build_report()        → Self-contained HTML with Chart.js charts
   │
HTML Output
```

---

## Quick Start

```bash
git clone https://github.com/akbknight/ai-strategic-briefing.git
cd ai-strategic-briefing

pip install -r requirements.txt

# Run on the included sample data
python briefing.py --input data/raw/sample_retail_sales.csv --title "US Retail Sales"

# Run on any CSV
python briefing.py --input your_data.csv --title "Q2 Analysis"
```

**With AI narrative** (requires Anthropic API key):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python briefing.py --input your_data.csv
```

Without the API key the pipeline runs in stats-only mode — all statistical analysis, anomaly detection, forecasting, and charts are still generated.

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `src/data/loader.py` | Load CSV, detect date columns (≥80% parse success), classify numeric vs. categorical |
| 2 | `src/data/profiler.py` | Per-column mean, median, std, quartiles, missing rate, Z-score and IQR outlier counts |
| 3 | `src/models/forecast.py` | Heuristic time-series detection: date col + numeric col + ≥18 monthly observations |
| 4a | `src/features/detection.py` | Z-score (point outliers, threshold 2.5σ) + CUSUM (regime shifts, Page 1954) |
| 4b | `src/models/forecast.py` | Holt-Winters ETS(A,A,A), MLE parameters, 12-month horizon, 95% prediction intervals |
| 5 | `src/ai/narrative.py` | Claude Haiku API with prompt caching: 3-paragraph executive briefing |
| 6 | `src/report/builder.py` | Self-contained HTML with inline Chart.js 4, embedded JSON data |

---

## Output

The pipeline generates `report_<filename>_<date>.html` — a single self-contained HTML file containing:

- **AI Executive Briefing** — 3-paragraph narrative: current state, risk signals, forward outlook
- **12-Month Forecast Chart** — historical series + Holt-Winters forecast with 95% confidence interval
- **Anomaly Table** — detected anomalies with date, value, severity (high/medium/low), detection method, and sigma score
- **Column Statistics Cards** — descriptive stats for every numeric column

The file opens directly in any browser. No server, no build step, no internet required (except Chart.js CDN for charts).

**Sample output for `data/raw/sample_retail_sales.csv` (135 months of US Retail Sales):**

```
[1/5] Loading and profiling data...
  135 rows × 2 columns
  Numeric: ['retail_sales_billions']
  Dates:   ['date']

[2/5] Computing statistics...
  1 columns analyzed · 0 outliers flagged

[3/5] Running anomaly detection...
  Time-series detected: date × retail_sales_billions · 2 anomalies

[4/5] Running forecast...
  12m forecast: 781.1 (+3.9%) · RMSE 12.2

[5/5] Generating narrative...
  AI summary generated via Claude API

Report saved: report_sample_retail_sales_20260512.html
```

---

## Configuration

All pipeline parameters are documented in `configs/app/config.yaml`:

```yaml
anomaly_detection:
  zscore_threshold: 2.5      # |z| > 2.5 flagged as anomaly
  cusum_sigma_threshold: 4.0 # CUSUM decision threshold (h = 4σ)
  cusum_k_factor: 0.5        # CUSUM allowance (k = 0.5σ)

forecast:
  seasonal_periods: 12       # months per seasonal cycle
  confidence_interval: 0.95  # 95% prediction interval

ai_narrative:
  model: claude-haiku-4-5-20251001
  max_tokens: 600
  use_prompt_cache: true
```

---

## CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to CSV file | Required |
| `--title` | Report title | Cleaned filename |
| `--output` | Output HTML path | `report_<stem>_<date>.html` |

---

## Technical Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Data loading | pandas | ≥2.0 |
| Numerical computing | numpy | ≥1.24 |
| Statistical tests | scipy.stats | ≥1.11 |
| Forecasting | statsmodels Holt-Winters ETS | ≥0.14 |
| AI narrative | Anthropic Claude API (Haiku) | ≥0.40 |
| Report charts | Chart.js 4 | 4.4.3 (CDN) |
| Package config | setuptools / pyproject.toml | PEP 517 |

---

## Methodology

Full technical documentation is in `docs/`:

- [docs/methodology.md](docs/methodology.md) — Statistical profiling, anomaly detection (Z-score, CUSUM), Holt-Winters ETS forecasting, prediction intervals, AI narrative
- [docs/architecture.md](docs/architecture.md) — Pipeline stages, module dependency graph, data flow, design decisions
- [docs/decision_log.md](docs/decision_log.md) — Why Holt-Winters over ARIMA, why Z-score + CUSUM, why Claude Haiku, why single-file HTML output
- [docs/data_dictionary.md](docs/data_dictionary.md) — Input CSV format, intermediate data structures, output HTML structure

Supporting research is in `reports/`:
- [reports/research_notes.md](reports/research_notes.md) — Hyndman & Athanasopoulos ETS framework, Page (1954) CUSUM, M3/M4 competition evidence
- [reports/eda.md](reports/eda.md) — EDA for the included US Retail Sales dataset

---

## Example Output

The `sample_report.html` file in the repository root shows output generated from `data/raw/sample_retail_sales.csv` (135 months of US Retail & Food Services Sales, Jan 2015–Mar 2026):

- Detected time-series: monthly, 135 observations
- Anomalies: April 2020 COVID demand shock (-$115B from February)
- Forecast: continued 3–4% annual growth through 2026–2027

---

## Project Structure

```
ai-strategic-briefing/
├── briefing.py              ← compatibility shim (entry point)
├── src/
│   ├── data/
│   │   ├── loader.py        ← load_and_profile()
│   │   └── profiler.py      ← compute_stats()
│   ├── models/
│   │   └── forecast.py      ← detect_time_series(), run_forecast()
│   ├── features/
│   │   └── detection.py     ← detect_anomalies_in_series()
│   ├── ai/
│   │   └── narrative.py     ← generate_ai_summary() [prompt caching]
│   ├── report/
│   │   └── builder.py       ← build_report(), HTML_TEMPLATE
│   └── cli/
│       └── run.py           ← main() CLI orchestrator
├── data/
│   └── raw/
│       └── sample_retail_sales.csv
├── configs/app/config.yaml
├── docs/
│   ├── methodology.md
│   ├── architecture.md
│   ├── decision_log.md
│   └── data_dictionary.md
├── reports/
│   ├── research_notes.md
│   └── eda.md
├── tests/
│   ├── test_data.py
│   └── test_models.py
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## 👤 Author & Strategic Portfolio

**Akshay Kumar**  
STEM MBA Candidate · Business Analytics & AI · American University Kogod School of Business  
Former Computer Programmer · U.S. Department of State  
- **Executive Portfolio:** [https://akbknight.github.io/](https://akbknight.github.io/)  
- **LinkedIn Profile:** [linkedin.com/in/akshaykumardl](https://www.linkedin.com/in/akshaykumardl/)  
- **Direct Résumé:** [Download PDF (369 KB)](https://akbknight.github.io/assets/Akshay_Resume.pdf)  
- **Email:** [ak8335a@american.edu](mailto:ak8335a@american.edu)
