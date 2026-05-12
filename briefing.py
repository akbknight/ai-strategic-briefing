"""
briefing.py — Compatibility shim
=================================
This file is the original entry point for the AI Strategic Briefing Generator.
It is kept at the repository root for backward compatibility (existing bookmarks,
documentation, and CI scripts that reference ``python briefing.py``).

All pipeline logic now lives in the modular ``src/`` package:

    src/data/loader.py      — load_and_profile()
    src/data/profiler.py    — compute_stats()
    src/models/forecast.py  — detect_time_series(), run_forecast()
    src/features/detection.py — detect_anomalies_in_series()
    src/ai/narrative.py     — generate_ai_summary()
    src/report/builder.py   — build_report(), HTML_TEMPLATE
    src/cli/run.py          — main() CLI entry point

Usage (unchanged from v0.x):
    python briefing.py --input data.csv --title "Q2 Sales Review"
    python briefing.py --input data/raw/sample_retail_sales.csv
"""

from src.cli.run import main

if __name__ == "__main__":
    main()
