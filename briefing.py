"""
AI Strategic Briefing Generator
=================================
Autonomous pipeline: CSV input → statistical analysis → anomaly detection
→ forecasting (if time-series) → Claude AI narrative → self-contained HTML report.

Usage:
    pip install pandas numpy scipy anthropic statsmodels
    python briefing.py --input data.csv --title "Q2 Sales Review"
    python briefing.py --input data.csv  # auto-generates title from filename

Set ANTHROPIC_API_KEY in your environment for AI-generated insights.
Without the key the report runs in stats-only mode (no AI narrative).

Output: report_[title]_[date].html  (self-contained, shareable)
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    pass

STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    pass


# ── Data Loading ─────────────────────────────────────────────────────────

def load_and_profile(path: str) -> dict:
    df = pd.read_csv(path)
    original_cols = list(df.columns)

    # Detect date columns
    date_cols = []
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

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in date_cols]

    return {
        "df": df,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": original_cols,
        "date_cols": date_cols,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(1).to_dict(),
    }


# ── Statistical Analysis ──────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, numeric_cols: list) -> list:
    results = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        z = np.abs(stats.zscore(s))
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        results.append({
            "column": col,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "missing_pct": round(float(df[col].isnull().sum() / len(df) * 100), 1),
            "outliers_zscore": int((z > 2.5).sum()),
            "outliers_iqr": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
        })
    return results


def detect_time_series(profile: dict) -> tuple[bool, str | None, str | None]:
    """Return (is_ts, date_col, value_col) if a plausible time-series structure exists."""
    if not profile["date_cols"] or not profile["numeric_cols"]:
        return False, None, None

    date_col = profile["date_cols"][0]
    df = profile["df"].sort_values(date_col).dropna(subset=[date_col])

    if len(df) < 18:
        return False, None, None

    diffs = df[date_col].diff().dropna()
    most_common = diffs.mode()
    if most_common.empty:
        return False, None, None

    typical = most_common.iloc[0]
    if abs(typical.days - 30) <= 5 or abs(typical.days - 7) <= 1 or typical.days == 1:
        value_col = profile["numeric_cols"][0]
        return True, date_col, value_col

    return False, None, None


def run_forecast(df: pd.DataFrame, date_col: str, value_col: str, n: int = 12) -> dict | None:
    if not STATSMODELS_AVAILABLE:
        return None
    try:
        ts = df.set_index(date_col)[value_col].dropna().sort_index()
        if len(ts) < 18:
            return None

        model = ExponentialSmoothing(
            ts, trend="add", seasonal="add", seasonal_periods=12,
            initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(n)
        forecast_index = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1), periods=n, freq="MS"
        )
        std = float(np.std(fitted.resid))
        hf = np.sqrt(np.arange(1, n + 1))

        return {
            "history_dates": [d.strftime("%Y-%m-%d") for d in ts.index],
            "history_values": [round(float(v), 3) for v in ts],
            "fitted": [round(float(v), 3) for v in fitted.fittedvalues],
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_index],
            "forecast_point": [round(float(v), 3) for v in forecast],
            "ci95u": [round(float(forecast.iloc[i] + 1.96 * std * hf[i]), 3) for i in range(n)],
            "ci95l": [round(float(forecast.iloc[i] - 1.96 * std * hf[i]), 3) for i in range(n)],
            "rmse": round(float(np.sqrt(np.mean(fitted.resid ** 2))), 4),
            "aic": round(float(fitted.aic), 1),
            "latest": round(float(ts.iloc[-1]), 3),
            "latest_date": ts.index[-1].strftime("%B %Y"),
            "forecast_12m": round(float(forecast.iloc[-1]), 3),
            "forecast_12m_date": forecast_index[-1].strftime("%B %Y"),
            "change_pct": round((float(forecast.iloc[-1]) - float(ts.iloc[-1])) / float(ts.iloc[-1]) * 100, 2),
            "value_col": value_col,
            "date_col": date_col,
        }
    except Exception as e:
        print(f"  Forecast skipped: {e}")
        return None


# ── Anomaly Detection ─────────────────────────────────────────────────────

def detect_anomalies_in_series(df: pd.DataFrame, date_col: str, value_col: str) -> list:
    ts = df.set_index(date_col)[value_col].dropna().sort_index()
    anomalies = []

    z = np.abs(stats.zscore(ts.values))
    for i, (date, val) in enumerate(ts.items()):
        if z[i] > 2.5:
            sev = "high" if z[i] > 3.5 else "medium" if z[i] > 3.0 else "low"
            anomalies.append({
                "date": str(date)[:10], "value": round(float(val), 3),
                "type": "z-score", "severity": sev, "score": round(float(z[i]), 2)
            })

    diff = ts.diff().dropna()
    mu, sigma = diff.mean(), diff.std()
    if sigma > 0:
        k = 0.5 * sigma
        threshold = 4.0 * sigma
        cp, cn = 0.0, 0.0
        alarm = False
        for date, dv in diff.items():
            cp = max(0.0, cp + dv - mu - k)
            cn = max(0.0, cn - dv + mu - k)
            if (cp > threshold or cn > threshold) and not alarm:
                score = max(cp, cn) / sigma
                sev = "high" if score > 7 else "medium" if score > 5 else "low"
                anomalies.append({
                    "date": str(date)[:10], "value": round(float(ts[date]), 3),
                    "type": "cusum", "severity": sev, "score": round(score, 2)
                })
                alarm = True
                cp, cn = 0.0, 0.0
            elif cp <= threshold and cn <= threshold:
                alarm = False

    seen: dict[str, dict] = {}
    pri = {"high": 3, "medium": 2, "low": 1}
    for a in anomalies:
        k = a["date"]
        if k not in seen or pri[a["severity"]] > pri[seen[k]["severity"]]:
            seen[k] = a
    return sorted(seen.values(), key=lambda x: x["date"])


# ── AI Narrative ──────────────────────────────────────────────────────────

def generate_ai_summary(context: dict) -> str:
    if not ANTHROPIC_AVAILABLE:
        return ""

    client = anthropic.Anthropic()
    prompt = f"""You are an expert data analyst generating an executive briefing. Analyze the following dataset profile and produce a concise, business-focused summary.

Dataset: {context['title']}
Rows: {context['n_rows']} | Columns: {context['n_cols']}
Numeric columns: {', '.join(context['numeric_cols'][:8])}

Key statistics:
{json.dumps(context['stats'][:5], indent=2)}

Anomalies detected: {context['n_anomalies']} total
{json.dumps(context['anomalies'][:5], indent=2) if context['anomalies'] else 'None'}

Forecast (if available):
{json.dumps(context.get('forecast_summary'), indent=2) if context.get('forecast_summary') else 'Not applicable'}

Write a 3-paragraph executive briefing:
1. Current state: What does this data show? Key metrics and trends.
2. Risk signals: What anomalies or concerning patterns exist?
3. Forward outlook: What does the forecast suggest? What should decision-makers watch?

Be specific, reference actual numbers, avoid generic filler. Write for a business executive, not a data scientist."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        print(f"  AI summary skipped: {e}")
        return ""


# ── HTML Report ───────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}} — Strategic Briefing</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e5e5e5; font-size: 14px; line-height: 1.6; padding: 40px 20px; }
  .container { max-width: 960px; margin: 0 auto; }
  .report-header { border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 24px; margin-bottom: 32px; }
  .report-title { font-size: 24px; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 6px; }
  .report-meta { font-size: 12px; color: #606060; font-family: monospace; }
  .section { margin-bottom: 36px; }
  .section-label { font-size: 10px; font-family: monospace; text-transform: uppercase; letter-spacing: 0.12em; color: #606060; margin-bottom: 12px; }
  .ai-brief { background: #161616; border: 1px solid rgba(232,160,32,0.2); border-left: 3px solid #E8A020; border-radius: 8px; padding: 20px 24px; font-size: 13px; color: #c0c0c0; line-height: 1.75; white-space: pre-wrap; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
  .stat-card { background: #161616; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 14px; }
  .stat-name { font-size: 11px; font-family: monospace; color: #606060; margin-bottom: 8px; }
  .stat-row { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }
  .stat-key { color: #808080; }
  .stat-val { color: #e5e5e5; font-family: monospace; }
  .outlier-warn { color: #f87171; }
  .chart-card { background: #161616; border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 20px; margin-bottom: 16px; }
  .chart-title { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .chart-sub { font-size: 11px; color: #808080; margin-bottom: 16px; }
  .chart-wrap { position: relative; height: 240px; }
  .anomaly-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .anomaly-table th { font-family: monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: #606060; padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.07); text-align: left; }
  .anomaly-table td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #a0a0a0; }
  .sev-high { color: #f87171; font-weight: 600; }
  .sev-medium { color: #fbbf24; font-weight: 600; }
  .sev-low { color: #4ade80; font-weight: 600; }
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }
  .kpi { background: #161616; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 14px; }
  .kpi-label { font-size: 10px; font-family: monospace; color: #606060; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .kpi-value { font-size: 20px; font-weight: 700; color: #e5e5e5; }
  .kpi-sub { font-size: 11px; color: #808080; margin-top: 3px; }
  footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px; color: #505050; }
</style>
</head>
<body>
<div class="container">
  <div class="report-header">
    <div class="report-title">{{TITLE}}</div>
    <div class="report-meta">Strategic Briefing · Generated {{DATE}} · {{N_ROWS}} records · {{N_COLS}} fields</div>
  </div>

  {{AI_SECTION}}

  {{FORECAST_SECTION}}

  {{ANOMALY_SECTION}}

  <div class="section">
    <div class="section-label">Column Statistics</div>
    <div class="stats-grid" id="statsGrid"></div>
  </div>

  <footer>
    Generated by <strong>AI Strategic Briefing Generator</strong> · <a href="https://akbknight.github.io" style="color:#808080">akbknight.github.io</a>
    · Source: {{SOURCE_FILE}}
  </footer>
</div>

<script>
const STATS = {{STATS_JSON}};
const FORECAST = {{FORECAST_JSON}};
const ANOMALIES = {{ANOMALIES_JSON}};

function esc(s) { const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }

function renderStats() {
  const grid = document.getElementById('statsGrid');
  STATS.forEach(s => {
    const card = document.createElement('div');
    card.className = 'stat-card';
    const warnClass = (s.outliers_zscore > 0 || s.outliers_iqr > 0) ? ' outlier-warn' : '';
    card.appendChild(Object.assign(document.createElement('div'), { className: 'stat-name', textContent: s.column }));
    const rows = [
      ['mean', s.mean], ['median', s.median], ['std', s.std],
      ['min', s.min], ['max', s.max], ['missing', s.missing_pct + '%'],
      ['outliers (z)', s.outliers_zscore], ['outliers (IQR)', s.outliers_iqr],
    ];
    rows.forEach(([k, v]) => {
      const row = document.createElement('div');
      row.className = 'stat-row';
      const keyEl = Object.assign(document.createElement('span'), { className: 'stat-key', textContent: k });
      const valEl = Object.assign(document.createElement('span'), { className: 'stat-val' + (k.startsWith('outlier') && v > 0 ? ' outlier-warn' : ''), textContent: String(v) });
      row.appendChild(keyEl);
      row.appendChild(valEl);
      card.appendChild(row);
    });
    grid.appendChild(card);
  });
}

function renderForecast() {
  if (!FORECAST) return;
  const canvas = document.getElementById('forecastCanvas');
  if (!canvas) return;
  new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: [...FORECAST.history_dates.slice(-48), ...FORECAST.forecast_dates].map(d => {
        const dt = new Date(d); return dt.toLocaleDateString('en-US', {year:'numeric',month:'short'});
      }),
      datasets: [
        { label: FORECAST.value_col + ' (actual)', data: [...FORECAST.history_values.slice(-48), ...new Array(FORECAST.forecast_point.length).fill(null)], borderColor: '#E8A020', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false },
        { label: 'Forecast', data: [...new Array(48).fill(null), ...FORECAST.forecast_point], borderColor: '#60a5fa', borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, tension: 0.3, fill: false },
        { label: '95% CI upper', data: [...new Array(48).fill(null), ...FORECAST.ci95u], borderColor: 'rgba(96,165,250,0.15)', backgroundColor: 'rgba(96,165,250,0.08)', borderWidth: 0, pointRadius: 0, fill: '+1' },
        { label: '95% CI lower', data: [...new Array(48).fill(null), ...FORECAST.ci95l], borderColor: 'rgba(96,165,250,0.15)', borderWidth: 0, pointRadius: 0, fill: false },
      ]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#606060', font: { size: 10 }, maxTicksLimit: 10, maxRotation: 0 }, grid: { color: 'rgba(255,255,255,0.03)' }, border: { display: false } }, y: { ticks: { color: '#606060', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.03)' }, border: { display: false } } } }
  });
}

renderStats();
renderForecast();
</script>
</body>
</html>"""


def build_report(title: str, profile: dict, col_stats: list, forecast: dict | None,
                 anomalies: list, ai_summary: str, source_file: str) -> str:

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = HTML_TEMPLATE

    # Substitutions
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{DATE}}", date_str)
    html = html.replace("{{N_ROWS}}", str(profile["n_rows"]))
    html = html.replace("{{N_COLS}}", str(profile["n_cols"]))
    html = html.replace("{{SOURCE_FILE}}", Path(source_file).name)

    # AI section
    if ai_summary:
        ai_html = f'<div class="section"><div class="section-label">AI Executive Briefing</div><div class="ai-brief">{ai_summary}</div></div>'
    else:
        ai_html = '<div class="section"><div class="section-label">Executive Briefing</div><div class="ai-brief" style="border-color:rgba(255,255,255,0.1);border-left-color:#808080">Set ANTHROPIC_API_KEY to enable AI-generated executive insights.</div></div>'
    html = html.replace("{{AI_SECTION}}", ai_html)

    # Forecast section
    if forecast:
        change_color = "#4ade80" if forecast["change_pct"] >= 0 else "#f87171"
        change_sign = "+" if forecast["change_pct"] >= 0 else ""
        forecast_html = f"""<div class="section">
  <div class="section-label">12-Month Forecast — {forecast['value_col']}</div>
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-label">Latest</div><div class="kpi-value">{forecast['latest']}</div><div class="kpi-sub">{forecast['latest_date']}</div></div>
    <div class="kpi"><div class="kpi-label">12m Forecast</div><div class="kpi-value">{forecast['forecast_12m']}</div><div class="kpi-sub">{forecast['forecast_12m_date']}</div></div>
    <div class="kpi"><div class="kpi-label">Projected Change</div><div class="kpi-value" style="color:{change_color}">{change_sign}{forecast['change_pct']}%</div><div class="kpi-sub">Holt-Winters ETS</div></div>
    <div class="kpi"><div class="kpi-label">RMSE</div><div class="kpi-value">{forecast['rmse']}</div><div class="kpi-sub">In-sample fit</div></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">{forecast['value_col']} — History + 12-Month Forecast</div>
    <div class="chart-sub">Holt-Winters additive trend + seasonality · AIC {forecast['aic']}</div>
    <div class="chart-wrap"><canvas id="forecastCanvas"></canvas></div>
  </div>
</div>"""
    else:
        forecast_html = ""
    html = html.replace("{{FORECAST_SECTION}}", forecast_html)

    # Anomaly section
    if anomalies:
        rows = ""
        for a in anomalies[:20]:
            sev_cls = f"sev-{a['severity']}"
            rows += f"<tr><td style='font-family:monospace'>{a['date']}</td><td>{a['value']}</td><td class='{sev_cls}'>{a['severity']}</td><td style='font-family:monospace;color:#606060'>{a['type']}</td><td style='color:#808080'>{a['score']}&sigma;</td></tr>"
        anomaly_html = f"""<div class="section">
  <div class="section-label">{len(anomalies)} Anomalies Detected</div>
  <div class="chart-card" style="padding:0">
    <table class="anomaly-table">
      <thead><tr><th>Date</th><th>Value</th><th>Severity</th><th>Method</th><th>Score</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""
    else:
        anomaly_html = ""
    html = html.replace("{{ANOMALY_SECTION}}", anomaly_html)

    # JSON data
    html = html.replace("{{STATS_JSON}}", json.dumps(col_stats))
    html = html.replace("{{FORECAST_JSON}}", json.dumps(forecast) if forecast else "null")
    html = html.replace("{{ANOMALIES_JSON}}", json.dumps(anomalies))

    return html


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Strategic Briefing Generator")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--title", default=None, help="Report title (default: filename)")
    parser.add_argument("--output", default=None, help="Output HTML path (default: auto)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    title = args.title or Path(args.input).stem.replace("_", " ").replace("-", " ").title()
    date_slug = datetime.now().strftime("%Y%m%d")
    output_path = args.output or f"report_{Path(args.input).stem}_{date_slug}.html"

    print(f"AI Strategic Briefing Generator")
    print(f"{'=' * 36}")
    print(f"Input:  {args.input}")
    print(f"Title:  {title}")

    print("\n[1/5] Loading and profiling data...")
    profile = load_and_profile(args.input)
    print(f"  {profile['n_rows']} rows × {profile['n_cols']} columns")
    print(f"  Numeric: {profile['numeric_cols']}")
    print(f"  Dates:   {profile['date_cols']}")

    print("[2/5] Computing statistics...")
    col_stats = compute_stats(profile["df"], profile["numeric_cols"])
    total_outliers = sum(s["outliers_zscore"] for s in col_stats)
    print(f"  {len(col_stats)} columns analyzed · {total_outliers} outliers flagged")

    print("[3/5] Running anomaly detection...")
    is_ts, date_col, value_col = detect_time_series(profile)
    anomalies = []
    if is_ts:
        anomalies = detect_anomalies_in_series(profile["df"], date_col, value_col)
        print(f"  Time-series detected: {date_col} × {value_col} · {len(anomalies)} anomalies")
    else:
        print("  No time-series structure detected")

    print("[4/5] Running forecast...")
    forecast = None
    if is_ts:
        forecast = run_forecast(profile["df"], date_col, value_col)
        if forecast:
            print(f"  12m forecast: {forecast['forecast_12m']} ({forecast['change_pct']:+.1f}%) · RMSE {forecast['rmse']}")
        else:
            print("  Forecast skipped")
    else:
        print("  Skipped (no time-series)")

    print("[5/5] Generating narrative...")
    ai_context = {
        "title": title,
        "n_rows": profile["n_rows"],
        "n_cols": profile["n_cols"],
        "numeric_cols": profile["numeric_cols"],
        "stats": col_stats,
        "n_anomalies": len(anomalies),
        "anomalies": anomalies[:5],
        "forecast_summary": {
            "latest": forecast["latest"],
            "forecast_12m": forecast["forecast_12m"],
            "change_pct": forecast["change_pct"],
        } if forecast else None,
    }
    ai_summary = generate_ai_summary(ai_context)
    if ai_summary:
        print("  AI summary generated via Claude API")
    else:
        print("  AI summary skipped (set ANTHROPIC_API_KEY to enable)")

    html = build_report(title, profile, col_stats, forecast, anomalies, ai_summary, args.input)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport saved: {output_path}")
    print("Open in a browser to view the interactive dashboard.")


if __name__ == "__main__":
    main()
