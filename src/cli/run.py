"""
src/cli/run.py
==============
Command-line entry point for the AI Strategic Briefing pipeline.

Usage
-----
    python briefing.py --input data.csv --title "Q2 Sales Review"
    python briefing.py --input data.csv  # auto-generates title from filename
    python -m src.cli.run --input data.csv

    # With AI narrative (requires Anthropic API key):
    export ANTHROPIC_API_KEY=sk-ant-...
    python briefing.py --input your_data.csv

Output
------
    report_<stem>_<YYYYMMDD>.html  — self-contained HTML dashboard

Pipeline stages (in order)
---------------------------
    [1] load_and_profile()      — schema detection, date inference, type coercion
    [2] compute_stats()         — per-column descriptive statistics
    [3] detect_time_series()    — heuristic: date col + numeric col + ≥18 monthly obs
    [4a] detect_anomalies()     — Z-score + CUSUM, deduplication by date
    [4b] run_forecast()         — Holt-Winters ETS, 12-month horizon, 95% CI
    [5] generate_ai_summary()   — Claude Haiku API, executive brief
    [6] build_report()          — self-contained HTML with Chart.js charts
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from src.ai.narrative import generate_ai_summary
from src.data.loader import load_and_profile
from src.data.profiler import compute_stats
from src.features.detection import detect_anomalies_in_series
from src.models.forecast import detect_time_series, run_forecast
from src.report.builder import build_report


def main() -> None:
    """Parse CLI arguments and run the full briefing pipeline."""
    parser = argparse.ArgumentParser(
        description="AI Strategic Briefing Generator — CSV → HTML analytics report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python briefing.py --input data/raw/sample_retail_sales.csv\n"
            "  python briefing.py --input sales.csv --title 'Q2 Review' --output report.html\n"
            "\n"
            "Set ANTHROPIC_API_KEY environment variable to enable AI narrative generation."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Report title (default: cleaned filename)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: report_<stem>_<date>.html)",
    )
    args = parser.parse_args()

    # ── Validate input ───────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    title: str = (
        args.title
        or Path(args.input).stem.replace("_", " ").replace("-", " ").title()
    )
    date_slug: str = datetime.now().strftime("%Y%m%d")
    output_path: str = args.output or f"report_{Path(args.input).stem}_{date_slug}.html"

    print("AI Strategic Briefing Generator")
    print("=" * 36)
    print(f"Input:  {args.input}")
    print(f"Title:  {title}")

    # ── Stage 1: Load and profile ────────────────────────────────────────────
    print("\n[1/5] Loading and profiling data...")
    profile: dict = load_and_profile(args.input)
    print(f"  {profile['n_rows']} rows × {profile['n_cols']} columns")
    print(f"  Numeric: {profile['numeric_cols']}")
    print(f"  Dates:   {profile['date_cols']}")

    # ── Stage 2: Compute statistics ──────────────────────────────────────────
    print("[2/5] Computing statistics...")
    col_stats: list = compute_stats(profile["df"], profile["numeric_cols"])
    total_outliers: int = sum(s["outliers_zscore"] for s in col_stats)
    print(f"  {len(col_stats)} columns analyzed · {total_outliers} outliers flagged")

    # ── Stage 3 + 4a: Time-series detection and anomaly detection ────────────
    print("[3/5] Running anomaly detection...")
    is_ts: bool
    date_col: str | None
    value_col: str | None
    is_ts, date_col, value_col = detect_time_series(profile)

    anomalies: list = []
    if is_ts:
        anomalies = detect_anomalies_in_series(profile["df"], date_col, value_col)
        print(
            f"  Time-series detected: {date_col} × {value_col} "
            f"· {len(anomalies)} anomalies"
        )
    else:
        print("  No time-series structure detected")

    # ── Stage 4b: Forecast ───────────────────────────────────────────────────
    print("[4/5] Running forecast...")
    forecast: dict | None = None
    if is_ts:
        forecast = run_forecast(profile["df"], date_col, value_col)
        if forecast:
            print(
                f"  12m forecast: {forecast['forecast_12m']} "
                f"({forecast['change_pct']:+.1f}%) · RMSE {forecast['rmse']}"
            )
        else:
            print("  Forecast skipped")
    else:
        print("  Skipped (no time-series)")

    # ── Stage 5: AI narrative ────────────────────────────────────────────────
    print("[5/5] Generating narrative...")
    ai_context: dict = {
        "title": title,
        "n_rows": profile["n_rows"],
        "n_cols": profile["n_cols"],
        "numeric_cols": profile["numeric_cols"],
        "stats": col_stats,
        "n_anomalies": len(anomalies),
        "anomalies": anomalies[:5],
        "forecast_summary": (
            {
                "latest": forecast["latest"],
                "forecast_12m": forecast["forecast_12m"],
                "change_pct": forecast["change_pct"],
            }
            if forecast
            else None
        ),
    }
    ai_summary: str = generate_ai_summary(ai_context)
    if ai_summary:
        print("  AI summary generated via Claude API")
    else:
        print("  AI summary skipped (set ANTHROPIC_API_KEY to enable)")

    # ── Stage 6: Build and write HTML report ─────────────────────────────────
    html: str = build_report(
        title=title,
        profile=profile,
        col_stats=col_stats,
        forecast=forecast,
        anomalies=anomalies,
        ai_summary=ai_summary,
        source_file=args.input,
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"\nReport saved: {output_path}")
    print("Open in a browser to view the interactive dashboard.")


if __name__ == "__main__":
    main()
