# Decision Log

Key design decisions made during development of the AI Strategic Briefing pipeline, with rationale.

---

## Decision 1: Holt-Winters over ARIMA

**Decision:** Use Holt-Winters Triple Exponential Smoothing (ETS) as the default forecasting model rather than ARIMA.

**Rationale:**

Holt-Winters handles trend and seasonality natively without requiring the data to be differenced to stationarity, which is the first requirement ARIMA imposes. For monthly business data — retail sales, revenue, headcount — the series typically exhibits both an upward trend and annual seasonality. Fitting ARIMA to such data requires identifying the correct order of differencing (d) and seasonal differencing (D), which adds complexity and requires more data for reliable identification.

Holt-Winters parameters (α for level smoothing, β for trend smoothing, γ for seasonal smoothing) have direct business interpretations: α controls how quickly the model forgets old levels, β controls trend responsiveness, and γ controls seasonal adaptation. This makes the model debuggable by non-technical stakeholders.

Empirically, ETS models matched or outperformed ARIMA on 67% of monthly series in the Makridakis M3 competition and on approximately 70% in M4, particularly for series with strong annual seasonality.

The practical lower bound for reliable Holt-Winters estimation is 18 monthly observations. ARIMA with seasonal terms typically requires 36+ observations for stable order identification.

---

## Decision 2: Z-score + CUSUM Together

**Decision:** Run both Z-score and CUSUM anomaly detectors rather than choosing one.

**Rationale:**

The two methods detect fundamentally different failure modes and are genuinely complementary:

**Z-score** flags individual observations that deviate sharply from the series mean in terms of standard deviations. It catches single-point spikes — for example, one month where sales dropped 40% due to a supply chain disruption, then immediately recovered. Z-score will flag that month; CUSUM will not (because it operates on differences, and a recovery is a large positive difference that cancels the detection).

**CUSUM** accumulates evidence over multiple periods. It catches sustained regime shifts — for example, a step-change in growth rate following a product launch or a major recession. No individual observation in such a shift is necessarily extreme (each month looks normal), but the accumulated drift exceeds the threshold. CUSUM will flag this; Z-score will not (because σ adapts to the new level over time).

Running both and merging with highest-severity deduplication gives broad coverage without double-counting. The combined approach mirrors standard statistical process control practice in industrial and financial monitoring.

---

## Decision 3: Claude Haiku as the AI Model

**Decision:** Use `claude-haiku-4-5-20251001` rather than a more capable but more expensive model (e.g., claude-sonnet).

**Rationale:**

The AI narrative task is **structured generation from pre-processed data**, not open-ended reasoning. The pipeline feeds a complete, structured context block — descriptive statistics, anomaly list, forecast summary — to the model and asks for a formatted 3-paragraph brief. The model does not need to reason about raw data or write code.

Claude Haiku is Anthropic's fastest and lowest-cost model. For this use case:
- Latency: Haiku returns in ~1-2 seconds; Sonnet ~3-5 seconds
- Cost: Haiku is approximately 25× cheaper than Sonnet per output token
- Quality: Haiku produces coherent, specific business text from structured prompts without hallucinating values (because all values are provided in the prompt)

The structured briefing format (current state → risk signals → forward outlook) further constrains the task, leaving little room for model capability to matter. Haiku is cost-appropriate for this use case.

Prompt caching (system prompt marked `cache_control: {type: "ephemeral"}`) further reduces cost for batch runs by ~90% on the system prompt token cost.

---

## Decision 4: Self-Contained HTML Output

**Decision:** Output a single `.html` file with all data embedded inline, rather than a web application or structured data export (JSON/CSV/PDF).

**Rationale:**

The primary use case is **analyst-to-executive reporting**. The report needs to be:
1. Shareable via email or Slack (no server required)
2. Viewable without installing software (every modern device has a browser)
3. Archived as a point-in-time snapshot of the analysis

A single self-contained HTML file satisfies all three requirements. All statistical data is embedded as inline JSON in a `<script>` block; charts are rendered client-side by Chart.js loaded from CDN. The output is fully functional when opened directly from the filesystem (`file://` protocol).

Alternatives considered:
- **PDF:** Requires a headless browser (Playwright/Puppeteer) as a dependency; not interactive
- **Jupyter notebook:** Requires Jupyter to be installed; not shareable as a single file
- **JSON + separate HTML template:** Splits the output into two files, reducing portability
- **Web server with live database:** Overkill for the single-report use case; adds operational complexity

The only limitation is that Chart.js requires internet access to render. For airgapped environments, Chart.js could be inlined at the cost of ~200 KB additional file size.

---

## Decision 5: 18-Month Minimum for Time-Series Detection

**Decision:** Require ≥18 monthly observations before activating time-series mode (anomaly detection + forecasting).

**Rationale:**

Holt-Winters with additive seasonality (period = 12) must estimate the following parameters:
- 3 smoothing parameters: α, β, γ
- Initial level: ℓ_0
- Initial trend: b_0
- 12 initial seasonal indices: s_{-11}, s_{-10}, …, s_0

Total: 17 parameters estimated by MLE. With only 12 observations (one seasonal cycle), the model is essentially over-parameterised — the MLE optimiser can fit the training data perfectly while producing wildly inaccurate forecasts.

With 18 observations (1.5 seasonal cycles), the model has enough data to begin distinguishing the seasonal pattern from noise. The estimates carry high uncertainty, but the forecast is usable as a directional indicator.

The recommended threshold for production use is 24+ months (2 full seasonal cycles). The 18-month minimum is a practical lower bound appropriate for this tool's use case (portfolio analytics, business reviews), where a directional forecast is valuable even under uncertainty. The threshold is documented in `configs/app/config.yaml` as `min_series_length` for clarity.
