# Project Plan: ai-strategic-briefing v1.0

## What Was Built

**ai-strategic-briefing** is an autonomous analytics pipeline that transforms any CSV file into a self-contained, interactive HTML report containing statistical profiling, anomaly detection, 12-month Holt-Winters forecasting, and an AI-generated executive briefing. The pipeline requires no configuration for standard inputs — schema detection, date inference, and time-series classification are automatic.

## Upgrade Scope (v0.x → v1.0)

The upgrade transforms a 450-line monolithic script (`briefing.py`) into a production Python package with proper module separation, project configuration, documentation, and tests.

### v0.x State
- Single file: `briefing.py` (all logic in one script)
- No project configuration (`pyproject.toml`, `requirements.txt`)
- No tests
- No documentation beyond inline comments and README
- Sample data at repo root

### v1.0 Target
- Modular `src/` package (7 modules + CLI entry point)
- `pyproject.toml` and `requirements.txt` for installable package
- 20+ unit tests in `tests/`
- 4 documentation files in `docs/` (methodology, architecture, decision log, data dictionary)
- 2 report files in `reports/` (research notes, EDA)
- Config in `configs/app/config.yaml`
- Sample data in `data/raw/`
- Root `briefing.py` as backward-compatible shim

## Architecture

```
src/data/loader.py      ← load_and_profile()
src/data/profiler.py    ← compute_stats()
src/models/forecast.py  ← detect_time_series(), run_forecast()
src/features/detection.py ← detect_anomalies_in_series()
src/ai/narrative.py     ← generate_ai_summary() [with prompt caching]
src/report/builder.py   ← build_report(), HTML_TEMPLATE
src/cli/run.py          ← main() [CLI orchestrator]
briefing.py             ← compatibility shim → src/cli/run.main()
```

## Key Technical Upgrades

1. **Prompt caching** on the Claude API system prompt block — reduces cost and latency for repeat runs
2. **Type hints throughout** — all function signatures annotated with input/output types
3. **Proper docstrings** — Google-style docstrings explaining parameters, return values, and algorithms
4. **Unit tests** — `tests/test_data.py` and `tests/test_models.py` cover all pipeline stages
5. **Project installable** — `pyproject.toml` enables `pip install -e .` and `briefing` CLI command
6. **Config externalised** — all magic numbers (thresholds, model parameters) in `configs/app/config.yaml`

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| pandas | ≥2.0 | DataFrame operations, date parsing |
| numpy | ≥1.24 | Numerical computing |
| scipy | ≥1.11 | Z-score computation |
| statsmodels | ≥0.14 | Holt-Winters ETS model |
| anthropic | ≥0.40 | Claude API client |

All dependencies are widely available on PyPI and support Python 3.10+.
