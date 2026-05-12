"""
src/ai/narrative.py
===================
AI narrative generation for the AI Strategic Briefing pipeline.

Uses the Anthropic Claude API (claude-haiku-4-5-20251001) to produce a
3-paragraph executive briefing from pre-computed statistical context.

Prompt Caching
--------------
The system prompt is marked with ``cache_control: {type: "ephemeral"}`` so
that the Anthropic API can cache the prompt prefix across repeated calls to
the same pipeline session.  For a single report run this saves latency on
the API round-trip; for batch mode (multiple CSVs) it yields measurable
token-cost reduction since the system prompt is shared across all requests.

The ``anthropic`` SDK exposes cache_control at the content-block level when
using the beta ``prompt-caching-2024-07-31`` feature flag.

Error Handling
--------------
All API errors are caught and logged; the function returns an empty string
on failure.  The pipeline degrades gracefully — the report is still produced
with all statistics and charts, just without the AI narrative section.

References
----------
- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

import json
import os
from typing import Any, Optional

_ANTHROPIC_AVAILABLE: bool = False
try:
    import anthropic as _anthropic_module
    _ANTHROPIC_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    pass


def generate_ai_summary(context: dict[str, Any]) -> str:
    """Generate a 3-paragraph executive briefing via the Claude API.

    Sends a structured context payload (dataset size, statistics, anomalies,
    forecast summary) to ``claude-haiku-4-5-20251001`` and returns the model's
    response text.

    The system prompt is marked for prompt caching (``cache_control`` with
    ``type: "ephemeral"``), which reduces cost and latency when the same
    pipeline runs on multiple datasets in the same session.

    Parameters
    ----------
    context : dict
        Must contain:
            title        (str)  — report title
            n_rows       (int)  — number of data rows
            n_cols       (int)  — number of columns
            numeric_cols (list) — names of numeric columns
            stats        (list) — output of compute_stats()
            n_anomalies  (int)  — total anomaly count
            anomalies    (list) — list of anomaly dicts (top 5)
            forecast_summary (dict | None) — {latest, forecast_12m, change_pct}

    Returns
    -------
    str
        Formatted executive briefing text, or empty string if the API is
        unavailable, the key is missing, or an error occurs.
    """
    if not _ANTHROPIC_AVAILABLE:
        return ""

    system_prompt: str = (
        "You are an expert data analyst generating concise executive briefings "
        "for business decision-makers.  Your analysis is specific, numbers-driven, "
        "and free of generic filler.  You always reference actual figures from the "
        "provided data.  You avoid jargon that a non-technical executive would not "
        "understand.  Your output is structured as three clear paragraphs."
    )

    forecast_block: str = (
        json.dumps(context.get("forecast_summary"), indent=2)
        if context.get("forecast_summary")
        else "Not applicable — no time-series structure detected."
    )
    anomaly_block: str = (
        json.dumps(context["anomalies"][:5], indent=2)
        if context["anomalies"]
        else "None detected."
    )

    user_prompt: str = f"""Analyze the following dataset profile and produce a concise, business-focused executive briefing.

Dataset: {context['title']}
Rows: {context['n_rows']} | Columns: {context['n_cols']}
Numeric columns: {', '.join(context['numeric_cols'][:8])}

Key statistics (top columns):
{json.dumps(context['stats'][:5], indent=2)}

Anomalies detected: {context['n_anomalies']} total
{anomaly_block}

Forecast (12-month Holt-Winters ETS):
{forecast_block}

Write a 3-paragraph executive briefing:
1. Current state: What does this data show? Key metrics, trends, and scale.
2. Risk signals: What anomalies or concerning patterns exist? Reference dates and values.
3. Forward outlook: What does the forecast suggest? What should decision-makers monitor?

Be specific — reference actual numbers.  Write for a business executive, not a data scientist."""

    try:
        client = _anthropic_module.Anthropic()

        message = client.beta.prompt_caching.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        return message.content[0].text

    except Exception as exc:
        # Fallback: try standard messages.create without caching
        try:
            client = _anthropic_module.Anthropic()
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            return message.content[0].text
        except Exception as fallback_exc:
            print(f"  AI summary skipped: {fallback_exc}")
            return ""
