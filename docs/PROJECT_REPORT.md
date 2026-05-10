# SeeWeeS Multi-Agent Dispatch Intelligence — Project Report

**UCLA MSBA AI Agents Project Challenge 2026**
**Version 1.0** · Submission deadline: Sunday May 10 2026 12:00 PM PST

---

## 1. Executive Summary

### Stakeholder
The **Director of Specialty Distribution Operations** at SeeWeeS — accountable
for on-time delivery of time-critical specialty medicines (oncology
biologics, antivirals, insulins, and clinical-trial drugs) from the New
Jersey distribution centre to hospitals along the I-95 corridor (NJ→Boston)
and the new I-95 South corridor (NJ→Philadelphia).

### Operational pain point
The director receives a daily dispatch plan that is generated linearly and
without self-correction. In a one-pass pipeline, three failure modes all hit
the leadership inbox unfiltered:

1. **Hallucinated or non-compliant plans** — the LLM planner can apply the
   wrong buffer percentage for a given weather risk score, or skip an
   escalation that the playbook mandates, with no automatic check.
2. **Quietly excluded shipments** — rows with missing `unique_item_id`,
   unknown `item_id`, or duplicate IDs (Data Quality issues DQ-01–DQ-04) are
   silently dropped, masking real demand and skewing capacity decisions.
3. **High-risk decisions without human oversight** — even when one corridor
   has a weather risk of 3 (precipitation ≥ 15 mm/day, wind gusts ≥ 45 km/h,
   freezing temp), the system would rubber-stamp a plan and email it.

The pain manifests as **late Tier-1 shipments**, **cold-chain breaches**,
and **executive distrust** of the automated report.

### Solution
We extended the linear LangGraph prototype into a 9-node multi-agent system
that:

- runs three independent data-gathering steps in **parallel**,
- self-corrects via a **cyclic audit loop** (Python hard rules + LLM soft
  checks, max 3 retries),
- allocates scarce resources with a **deterministic penalty model** that
  overrides the LLM allocator when it over-allocates cold-chain trucks,
- **interrupts execution** for manager approval when any corridor has a risk
  score of 3, and
- produces a colour-coded HTML report with explicit warnings when violations
  could not be resolved automatically.

The result is an "executive-ready" dispatch plan: anything that reaches
leadership has either passed every rule check or carries a clear
"unresolved violations" banner.

---

## 2. Key Assumptions

### 2.1 Logistics constraints
| Assumption | Source | Value |
|---|---|---|
| Truck capacity | Playbook §6 | 10 units / truck |
| Daily packing buffer | Playbook §6 | +10 % over actual demand |
| Driver pool / day | `Resource_availability_48h.csv` | 6 drivers |
| Standard trucks / day | `Resource_availability_48h.csv` | 4 trucks |
| Cold-chain trucks / day | `Resource_availability_48h.csv` | 2 trucks |
| Planning horizon | Playbook §5 | 48 hours (Day0 + Day1) |
| Weather forecast horizon | Open-Meteo | 2 forecast days |

### 2.2 Risk thresholds (Playbook §5.2)
| Risk Score | Trigger | Required buffer | Escalation |
|---|---|---|---|
| 0 | No flags | 0 % | No |
| 1 | Any one flag | 10 % | No |
| 2 | Any two flags | 25 % | No |
| 3 | All three flags | 40 % | **Yes** |

Flags: `precipitation_sum ≥ 15 mm/day`, `wind_gusts_10m_max ≥ 45 km/h`,
`temperature_2m_min ≤ 0 °C`. Corridor risk = max waypoint score across the
corridor's 4–5 waypoints; route risk = max corridor risk across the 48-hour
horizon.

### 2.3 Penalty model (Playbook §7)
| Event | Penalty (pts/unit) |
|---|---|
| Tier 1 SLA violation (life-critical) | 100 |
| Tier 2 SLA violation | 40 |
| Cold-chain breach (wrong truck type) | +80 (stacks with SLA) |
| Non-SLA delay | 10 |

Tier-1 SKUs in the item master: Antiviral, Oncology Biologic, Clinical
Trial. All Tier-1 SKUs in our domain are also cold-chain.

### 2.4 Data assumptions
- The 14-day shipment CSV (`Incoming_shipments_14d_multi_corridor.csv`)
  carries all four DQ patterns (DQ-01 missing UID, DQ-02 unknown item_id,
  DQ-03 name mismatch, DQ-04 duplicate UID) at realistic rates (~5 % of
  rows).
- Weather data is real-time from Open-Meteo (no API key required, no rate
  limit at our volume).
- LLM calls use `gpt-4.1-mini` at temperature 0.2 — we trade some creativity
  for deterministic structured output that the audit step can parse.
- The PDF playbook is the single source of truth for business rules; the
  agents are **grounded** in PDF excerpts via RAG so they cannot invent
  rules that aren't in the document.

### 2.5 Synthetic data we generated
Original challenge data did not include enough variation to stress-test the
allocator. We added (`generate_synthetic_data.py`):
- **synthetic_baseline.csv** — control group, no anomalies
- **synthetic_volume_spike.csv** — sudden 2× volume burst on Day0
- **synthetic_growth_trend.csv** — 7 days of week-over-week growth
- **synthetic_dq_heavy.csv** — 30 % DQ-01 / DQ-04 rows for reconciliation
  stress
- **synthetic_tier1_surge.csv** — Tier-1 oncology surge triggering cold-chain
  cap clipping
- **synthetic_rich_60d.csv** — 60-day history for trend analysis

---

## 3. Technical Methodology

### 3.1 Architectural enhancements

The original prototype was a 6-node linear graph. We expanded it to a
**9-node graph with parallel fan-out, a conditional cyclic edge, and an
interrupt-based human checkpoint**:

```
START ─┬─→ pdf_context  ────┐
       ├─→ csv_analysis ────┼─→ planner ─→ audit ┐
       └─→ weather     ─────┘                    │
                                ┌──────[ FAIL ]──┘
                                ↓               
                          (back to planner)      
                                ↓               
                     [ PASS ] → allocator ─→ human_checkpoint
                                                ↓ (interrupt if risk=3)
                                              report ─→ email ─→ END
```

Concrete additions to `src/graph.py`:

1. **Parallel fan-out** — `add_edge(START, "pdf_context")`,
   `add_edge(START, "csv_analysis")`, `add_edge(START, "weather")` lets
   LangGraph's Pregel runner execute all three independent data-gathering
   nodes concurrently. The planner is reached only after all three converge.

2. **Conditional cyclic edge** — `_route_after_audit(state)` returns either
   `"planner"` (FAIL) or `"allocator"` (PASS); `add_conditional_edges`
   routes accordingly. This is the only cycle in the graph.

3. **Interrupt-based checkpoint** — `node_human_checkpoint` calls
   `langgraph.types.interrupt(...)` with the allocation plan and weather
   risk; the call suspends the graph until the caller resumes with
   `Command(resume=<decision>)`. The Streamlit UI binds this to an
   Approve / Reject button; the CLI auto-resumes for unattended runs.

### 3.2 Agent design

Six GPT agents share one `ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)`
instance. Each agent has a dedicated `ChatPromptTemplate` in `src/prompts.py`
with explicit data contracts and refusal clauses.

| Agent | Role | Inputs | Outputs | New / changed |
|---|---|---|---|---|
| **ContextAgent** | Extract structured business rules from playbook | k=6 PDF chunks | 7-section bullets (KPIs, SLAs, weather thresholds, escalation, DQ rules, capacity, penalty) | Expanded categories from 4 to 7 |
| **OpsDataAgent** | Interpret per-corridor KPIs and DQ violations | Reconciled summary, KPIs, anomalies | Findings per corridor, root causes, actions | Now corridor-aware |
| **PlannerAgent** | 48-hour dispatch plan + structured JSON for audit | Rules, ops, weather, KPIs, resources, scenario, audit_feedback | Prose plan + ` ```json``` ` block | **NEW**: emits structured JSON for downstream rule check; honours `audit_feedback` |
| **AuditAgent** | Soft check the plan after Python hard checks pass | Plan, rules, weather | `PASS` or `FAIL: <reasons>` | **NEW** |
| **AllocatorAgent** | Allocate scarce resources to minimise penalty | KPIs, resources, weather, rules | JSON allocation block | **NEW** |
| **ReportAgent** | Produce executive HTML report | Everything | HTML with 9 sections | Section list expanded; required to use only fields present in the data (no hallucinated metrics) |

### 3.3 Defence-in-depth: why deterministic post-corrections

LLMs occasionally produce numerically incorrect allocations (e.g. allocating
3 cold-chain trucks per corridor when the daily pool has 2 trucks total).
Trusting the LLM here would cascade a wrong penalty score into the
executive report. We therefore wrap the AllocatorAgent's output in two
deterministic correction steps in `src/graph.py`:

- **`_clip_cold_chain_allocation`** (lines 253-296) — greedy reduction:
  while the per-day cold-chain truck total exceeds `RESOURCE_POOL`, take
  one truck from the highest-allocated corridor. Logs the clip count.

- **`_recompute_penalty`** (lines 305-380) — replaces the LLM's
  `total_penalty_score` and `deferred_units` with a deterministic
  computation that fulfils Tier 1 cold-chain first, then Tier 2 cold-chain,
  then standard demand, against truck supply × capacity. Adds a transparent
  rationale showing both the new and the LLM-reported numbers.

This pattern — **LLM proposes, Python verifies, Python corrects** — is the
core of the system's reliability story.

### 3.4 RAG and grounding

`src/tools/pdf_tools.py` implements a minimal RAG pipeline: `PyPDFLoader →
RecursiveCharacterTextSplitter (chunk=900, overlap=150) → OpenAI embeddings
→ Chroma persistent store`. A SHA-256 fingerprint of the PDF
(`path|size|mtime`) is used as a marker file so re-runs against the same
PDF skip re-embedding (sub-second startup once cached).

The ContextAgent is told to extract only what's in the snippets ("Be
precise and exhaustive") and the ReportAgent is forbidden from mentioning
metrics not present in `weather_risk` ("Do NOT mention snowfall,
visibility, or any metric not present in the data") — both are anti-
hallucination guardrails.

### 3.5 KPI definitions (operational metrics)

| KPI | How calculated | Source data | Thresholds |
|---|---|---|---|
| **Exclusion rate** | `excluded_rows / total_rows × 100` per corridor / day | Reconciled CSV | < 5 % healthy, ≥ 10 % warn, ≥ 20 % alert |
| **Cold-chain demand** | Count of valid rows where `product_class` ∈ COLD_CHAIN_CLASSES | Reconciled CSV + Item Master | Compared against daily cold-chain truck supply (2 trucks × 10 units = 20 units/day) |
| **Tier-1 mix** | `tier1_units / valid_rows × 100` per corridor | Item Master `SLA_TIER` map | Higher mix → higher penalty if deferred |
| **Trucks needed (cold-chain)** | `ceil(cold_chain_units × 1.10 / 10)` | Playbook §6 capacity model | Compared against `RESOURCE_POOL["DayN"]["truck_temp_controlled"]` |
| **Route risk score** | `max(waypoint risk scores)` per corridor | Open-Meteo daily aggregates | 0 / 1 / 2 / 3 → buffer 0 % / 10 % / 25 % / 40 % |
| **Required buffer %** | `BUFFER_POLICY[route_risk_score]` | Playbook §5.2 | Audit FAIL if planner deviates |
| **Total penalty score** | `Σ(deferred Tier-1) × 100 + Σ(deferred Tier-2) × 40` | Truck supply × demand | Lower is better; reported per scenario |
| **Audit attempts** | `audit_attempts` counter on AppState | Graph state | ≥ 3 → force-pass with banner |

---

## 4. Results & Validation

### 4.1 Business insights (from a representative live run)

A single run on the 14-day multi-corridor CSV plus live weather:

```
[Weather] Corridor scores: {'C1_I95_NJ_BOS': 1, 'C2_NJ_PHL': 0}
[Audit] PASS — plan is compliant.
[Email] REPORT_EMAIL_TO not set — skipping email send.
Report saved to report.html
```

Behavioural observations:
- **C1 risk = 1** drove a 10 % buffer per the policy table; the planner
  applied this correctly on the first attempt.
- **DQ-01 / DQ-04 reconciliation** excluded N rows from the planning
  window across both corridors; these were surfaced in `anomalies_md` and
  in the report's DQ section so leadership sees the demand they cannot
  fulfil.
- **Allocator over-allocated cold-chain trucks** (LLM proposed 3 cold-chain
  trucks for C1-Day0 when the cap is 2/day); `_clip_cold_chain_allocation`
  reduced it to 2 and logged the rationale. `_recompute_penalty` rebuilt
  the score from supply × demand. Both corrections are visible in the
  final HTML report.

### 4.2 Stress-test results (`stress_test_scenarios.py`)

Six adversarial scenarios were run against the same 14-day CSV. The
table below shows representative behaviour (penalties depend on live
weather):

| # | Scenario | Audit attempts | Penalty | Deferred | Notes |
|---|---|---:|---:|---:|---|
| 0 | Baseline | 1 | low | low | Plan passes first try |
| 1 | 20 % demand spike on C2 | 1–2 | medium | medium | Allocator shifts cold-chain trucks toward Tier-1 |
| 2 | Driver shortage (3 instead of 6) | 1–2 | medium-high | medium | Tier-2 deferrals dominate; Tier-1 protected |
| 3 | Cold-chain truck breakdown (1 instead of 2) | 1–2 | high | high | Clipping fires aggressively; banner on report |
| 4 | I-95 partial closure (C1 +4 h) | 1 | low | low | Buffer increased; SLA still met |
| 5 | Combined demand spike + driver shortage | 2–3 | very high | very high | Audit loop visible; force-pass triggers if 3 retries |

The audit loop **demonstrably re-prompts the planner** with specific
violations on stressed scenarios; the deterministic penalty recomputation
**catches LLM under-counting** in scenarios 3 and 5.

### 4.3 Validation strategy

We use three complementary validation layers:

1. **Unit tests (`tests/`)** — pytest suite covering AppState schema,
   weather risk scoring (boundary tests at 14.9 mm vs 15.0 mm,
   44.9 km/h vs 45.0 km/h, 0.1 °C vs 0.0 °C), CSV reconciliation
   (alias / legacy / DQ paths), audit routing, penalty recomputation
   (Tier-1 protection invariant), cold-chain clipping convergence, and
   human checkpoint trigger condition. **LLM calls are mocked** with
   pytest fixtures so the suite runs offline and does not consume
   credits.

2. **Integration / stress tests (`stress_test_scenarios.py`)** — six
   live runs against the real OpenAI + Open-Meteo APIs. Each scenario
   asserts: graph completes (or interrupts as expected), audit verdict
   eventually `PASS`, allocator output schema is intact, penalty score
   is non-negative, report HTML is non-empty.

3. **Manual smoke run (`python src/main.py`)** — full live pipeline
   against today's weather. Used to spot-check that: corridor risk
   scores look plausible vs the actual forecast, the report HTML
   renders in a browser, and the email node skips cleanly when SMTP
   is unset.

---

## 5. Limitations & Next Steps

### 5.1 Current limitations

- **Single LLM provider, single model.** All six agents use
  `gpt-4.1-mini`. Falling back to a different model on rate-limit or
  timing out is not implemented. For production we would route the
  expensive `ReportAgent` through Sonnet and keep the audit loop on a
  smaller model.
- **Audit loop has no introspection.** The loop counts attempts and
  records feedback but does not learn across runs — the same violation
  pattern triggers the same retry every day. A persistence layer
  (e.g. a violations history) could let the planner pre-empt common
  failure modes.
- **Penalty model is calibrated, not learned.** The point values
  (100 / 40 / +80) come from the playbook. In practice the relative
  weighting of cold-chain breach vs. SLA violation should be tuned
  against historical incidents.
- **Interrupt path requires synchronous resume.** The CLI auto-resumes
  with "YES" so unattended runs work, but a real ops workflow needs a
  durable approval queue (Slack thread, paging integration). The
  Streamlit UI is a step toward that.
- **RAG is single-PDF.** If SeeWeeS adds e.g. a separate compliance
  manual, we would need a multi-collection retriever and per-question
  routing.
- **Trend window is 7 days at most.** History rows in the CSV are
  bounded by what the operations system exports; period-over-period
  comparisons across longer horizons would need a data-warehouse
  source.
- **No A/B comparison of plans.** When the audit loop fires, we keep
  only the latest plan. Storing all attempts would let leadership see
  *what changed* across iterations.

### 5.2 If we had more time / production data access

1. **Hook the email node up to a real ops mailbox** — currently the
   pipeline writes `report.html` to disk and optionally emails it.
   The next step is signed sender domains (DKIM / SPF) and an
   unsubscribe / digest preference per recipient.
2. **Replace synthetic resource pool with a live API call** — today
   `RESOURCE_POOL` is hardcoded from `Resource_availability_48h.csv`.
   Pulling it from the WMS (warehouse management system) at run time
   would close the only manual-update loop in the system.
3. **Promote the Streamlit UI to a multi-tenant dashboard** — single-
   user app today; production would need auth (SSO), per-user audit
   logs, and persistent run history.
4. **Add a feedback channel from leadership back into the planner** —
   if a manager rejects a plan, we currently only know the boolean.
   Capturing free-text rejection reasons and feeding them into the
   next planner attempt would meaningfully improve plan quality.
5. **Real-time traffic integration** — Open-Meteo gives weather but
   not road conditions. INRIX or HERE traffic APIs would let us turn
   the I-95 closure scenario from a simulated input into a live
   trigger.
6. **Cost-aware allocation** — the current penalty model penalises
   SLA misses but ignores transport cost. A two-objective optimiser
   (penalty + cost) would give leadership a Pareto frontier to choose
   from.

---

## Appendix A — File map (graders' guide)

| Concern | File |
|---|---|
| Multi-agent orchestration / cyclic graph | `src/graph.py` |
| Agent prompt contracts | `src/prompts.py` |
| LLM agent functions | `src/agents.py` |
| RAG over playbook PDF | `src/tools/pdf_tools.py` |
| Item master + reconciliation + per-corridor KPIs | `src/tools/csv_tools.py` |
| Open-Meteo weather + risk scoring | `src/tools/weather_tools.py` |
| SMTP report dispatch | `src/tools/email_tools.py` |
| What-if stress harness | `stress_test_scenarios.py` |
| Synthetic data generator | `generate_synthetic_data.py` |
| Streamlit UI (HITL approval) | `app.py` |
| Test suite | `tests/` |
| Authoritative inputs (PDF) | `data/SeeWeeS Specialty distribution.pdf` |
| Multi-corridor 14-day CSV | `data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv` |
| Resource constraints | `data-for-enhancement/Resource_availability_48h.csv` |
| Markdown playbook (dev reference) | `data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md` |

---

## Appendix B — Reproducibility checklist

- [x] `requirements.txt` covers every import in the code base
- [x] `.env.example` lists every environment variable with comments
- [x] `python src/main.py` runs end-to-end against the committed data files
- [x] `pytest` passes against a clean clone with mocked LLMs
- [x] `streamlit run app.py` opens the UI without code edits
- [x] All paths in code are relative to the project root, no hardcoded
      `/Users/...` paths
- [x] `chroma_db/` and `.env` are gitignored; no secrets or generated
      artifacts in the repo
