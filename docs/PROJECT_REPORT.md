# SeeWeeS Multi-Agent Dispatch Intelligence — Project Report

**Version 1.0** · Technical & Business Documentation

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
We extended the linear LangGraph prototype into an 11-node multi-agent system
that:

- runs four independent data-gathering steps in **parallel** (playbook RAG,
  CSV reconciliation, weather, and workforce eligibility),
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

The seven logistics constraints form the physical backbone of every
allocation decision the system makes. Each is treated as a hard ceiling that
no LLM output is permitted to exceed.

- **Truck capacity — 10 units per truck (Playbook §6).** Each truck carries up
  to 10 units, forming the basis for all supply, allocation, and penalty
  calculations. Supply is computed as `trucks × 10`; excess demand is deferred
  and penalised. Because this value is centralised in `src/graph.py`, a
  fleet-capacity change requires updating a single constant.
- **Daily packing buffer — +10 % (Playbook §6).** Demand is inflated by 10 %
  before truck allocation to account for repacking, damaged goods, and
  last-minute changes: `required_trucks = ceil(demand × 1.10 / 10)`. This
  buffer is always active and stacks with weather-risk buffers.
- **Driver pool — 6 / day (`Resource_availability_48h.csv`).** The nominal pool
  is six, but the *effective* pool is recalculated daily from eligibility rules
  (hour limits, mandatory rest, fatigue). Cold-chain trucks additionally
  require certified drivers, making drivers a binding operational constraint.
- **Standard trucks — 4 / day.** Available for non-cold-chain shipments only;
  they cannot carry cold-chain product (that would trigger the +80-pt breach
  penalty). Allocation caps are enforced deterministically after the LLM
  proposes.
- **Cold-chain trucks — 2 / day.** Only two refrigerated trucks limit cold-chain
  capacity to 20 units/day. Excess cold-chain demand is deferred, never moved
  on standard trucks. Effective capacity is further constrained by the number
  of certified eligible drivers.
- **Planning horizon — 48 h / Day0 + Day1 (Playbook §5).** Covers current-day
  shipments and next-day staging, enabling limited demand smoothing within the
  reliable forecast range.
- **Weather forecast horizon — 2 days (Open-Meteo).** Daily aggregates for
  precipitation, wind gusts, and minimum temperature across 4–5 waypoints per
  corridor; the corridor takes the **maximum** waypoint risk so a single
  hazardous segment is never averaged away.

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

The risk score is the count of flags set (0–3). Taking the **maximum** across
waypoints — rather than the mean — is a deliberate conservative choice: a
single dangerous waypoint is enough to impose the higher buffer, because a
refrigerated truck cannot bypass a flooded or iced segment mid-route. The
buffer policy is applied multiplicatively on top of the §6 packing buffer, so
at risk score 3 the total capacity overhead is `1.10 × 1.40 = 1.54×` base
demand — deliberately stringent given a Tier-1 SLA miss costs 100 pts/unit.
Escalation at score 3 is a **hard rule in the audit's deterministic Python
check**, not a soft preference: the planner cannot ship a compliant plan with
`escalation_triggered = false` when route risk is 3 — the audit rejects it and
loops back.

### 2.3 Penalty model (Playbook §7)
| Event | Penalty (pts/unit) |
|---|---|
| Tier 1 SLA violation (life-critical) | 100 |
| Tier 2 SLA violation | 40 |
| Cold-chain breach (wrong truck type) | +80 (stacks with SLA) |
| Non-SLA delay | 10 |

Tier-1 SKUs in the item master: Antiviral, Oncology Biologic, Clinical
Trial. All Tier-1 SKUs in our domain are also cold-chain.

The penalty model is the system's primary objective function — every
allocation decision flows toward minimising the total penalty score, and the
three penalty types encode a deliberate clinical-priority hierarchy.

- **Tier-1 SLA violation (100 pts/unit)** covers life-critical SKUs, all of
  which are also cold-chain. A deferred Tier-1 unit therefore almost always
  stacks both penalties — **180 pts** (100 SLA + 80 cold-chain), nearly 5× a
  standard Tier-2 delay. This is why `_recompute_penalty` fulfils demand in
  strict priority order: Tier-1 cold-chain → Tier-2 cold-chain → standard.
- **Tier-2 SLA violation (40 pts/unit)** covers non-life-critical specialty
  drugs. The 2.5× gap to Tier-1 ensures the allocator never sacrifices Tier-1
  coverage to protect Tier-2 volume.
- **Cold-chain breach (+80 pts/unit, stacking)** fires whenever a cold-chain
  unit rides a standard truck. Because the breach alone exceeds a Tier-2 SLA
  miss, the system always prefers **deferral over substitution** — exhausted
  cold-chain supply means units wait, not that they move on the wrong vehicle.

The weights are taken from the playbook, not learned. The Outcome Calibration
loop (§4.4, Loop 3) tracks predicted-vs-actual penalty over time; persistent
bias in either direction would signal the weights need revision.

### 2.4 Data assumptions

**2.4.1 Data availability.**
The 14-day multi-corridor CSV (`Incoming_shipments_14d_multi_corridor.csv`)
was designed to contain all four real-world DQ failure modes at realistic
rates (~5 % of rows): DQ-01 (missing `unique_item_id`), DQ-02 (unknown
`item_id`), DQ-03 (name mismatch — legacy product name instead of canonical),
and DQ-04 (duplicate `unique_item_id`). In operational logistics feeds, 3–8 %
DQ exclusion rates are common due to upstream ERP drift and manual entry. The
reconciliation layer (`_reconcile_row` in `src/tools/csv_tools.py`) maps DQ-03
rows to their canonical item via an alias dictionary, recovering demand a
simpler filter would silently drop; DQ-01 and DQ-04 rows are quarantined and
surfaced in the report's DQ section. Weather is fetched live from Open-Meteo
(no API key, no rate limiting at 5–9 waypoint queries/run) using daily
aggregates only, since the playbook thresholds are stated in daily terms.
Resource availability is read from `Resource_availability_48h.csv` and
immediately reduced by the Workforce Reality loop (§4.4, Loop 2) before any
allocation, so the allocator never sees an inflated supply figure.

**2.4.2 Business rules — the PDF playbook is the single source of truth.**
All rule extraction (buffers, escalation, penalties, SLA tiers, capacity) is
grounded exclusively in RAG retrievals from the playbook PDF. Agents are
explicitly forbidden from applying rules not present in the retrieved
excerpts — the ContextAgent is instructed to be "precise and exhaustive" about
what the snippets contain, and the ReportAgent is forbidden from mentioning
metrics absent from `weather_risk`. The practical implication is that the
system's compliance is only as complete as the playbook itself; rules that
exist informally but are absent from the PDF will not be applied. This is a
deliberate scope boundary — grounding in a vetted document beats untested
background knowledge.

**2.4.3 LLM configuration.**
All six agents use `gpt-4.1-mini` at temperature 0.2. The reduced temperature
increases the consistency of the PlannerAgent's structured JSON (which the
audit's Python parser must process) at the cost of some prose creativity. The
audit loop compensates: malformed or non-compliant JSON returns a specific
failure reason and the planner is re-prompted — a correction signal that
temperature reduction alone cannot guarantee.

### 2.5 Synthetic data we generated
The original dataset did not include enough variation to stress-test the
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

The original prototype was a 6-node linear graph. We expanded it to an
**11-node graph with a 4-way parallel fan-out, an agentic scenario/​workforce
overlay, a conditional cyclic edge, and an interrupt-based human checkpoint**:

```
START ─┬─→ pdf_context          ──┐
       ├─→ csv_analysis          ──┤
       ├─→ weather               ──┼─→ scenario_apply ─→ planner ─→ audit ┐
       └─→ load_workforce_state  ──┘                        ↑             │
                                                    ┌──[ FAIL · feedback ]┘
                                                    ↓
                                              (back to planner)
                                                    ↓
                                  [ PASS / force-pass ] → allocator
                                              (LLM → clip → recompute → realism)
                                                    ↓
                                              human_checkpoint
                                                    ↓ (interrupt if any risk = 3)
                                              report ─→ email ─→ END
```

The 11 nodes are: `pdf_context`, `csv_analysis`, `weather`,
`load_workforce_state` (four parallel data-gatherers); `scenario_apply` (the
agentic what-if engine); `planner`, `audit` (the self-correction cycle);
`allocator`, `human_checkpoint`, `report`, `email`.

Concrete additions to `src/graph.py`:

1. **Parallel fan-out (4-way)** — `add_edge(START, …)` for `pdf_context`,
   `csv_analysis`, `weather`, and `load_workforce_state` lets LangGraph's
   Pregel runner execute all four independent data-gathering nodes
   concurrently. They converge on `scenario_apply` before the planner.

2. **Agentic scenario/​workforce overlay** — `node_scenario_apply` runs the
   `ScenarioParserAgent` to turn free-text disruptions into structured
   overrides (resource caps, demand multipliers, weather/closure, transit
   delay), then layers workforce reality (fatigue/cert/leave) on top — all
   written into `effective_resource_pool`, `corridor_kpis`, and `weather_risk`
   so every downstream node executes the disrupted reality, not a narration.

3. **Conditional cyclic edge** — `_route_after_audit(state)` returns either
   `"planner"` (FAIL) or `"allocator"` (PASS); `add_conditional_edges`
   routes accordingly. This is the only cycle in the graph.

4. **Interrupt-based checkpoint** — `node_human_checkpoint` calls
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
executive report. We therefore wrap the AllocatorAgent's output in three
deterministic correction steps in `src/graph.py` and `src/tools/workforce_tools.py`:

- **`_clip_resource_allocation`** — greedy reduction across EVERY scarce
  resource: while the per-day total of cold-chain trucks, standard trucks,
  or drivers exceeds the scenario-and-workforce-adjusted `effective_resource_pool`,
  take one unit from the highest-allocated corridor. Logs each clip.

- **`_recompute_penalty`** — replaces the LLM's `total_penalty_score` and
  `deferred_units` with a deterministic computation that fulfils Tier 1
  cold-chain first, then Tier 2 cold-chain, then standard demand, against
  truck supply × capacity. Emits a per-(corridor, day, tier)
  `deferral_breakdown` and a `deferral_summary.headline` that the report
  agent must quote verbatim — so the report cannot claim "Tier 1 protected"
  when it isn't.

- **`realism_check_allocation`** — feasibility pass against the workforce:
  flags violations (a corridor with trucks but zero drivers; cold-chain
  trucks exceeding certified-driver count) and warnings (fatigue-flagged
  drivers committed; no driver slack for emergencies).

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

Behavioural observations (actual reconciliation log from this dataset):
- **C1 risk = 1** drove a 10 % buffer per the policy table; the planner
  applied this correctly on the first attempt.
- **Item-master reconciliation** processed the 14-day feed and produced:
  21 exact-ID matches, **2 legacy-ID remaps** (e.g. `10020 → 10021`),
  **12 DQ-03 name mismatches** (flagged but kept), and **3 DQ-01 rows
  excluded** for missing `unique_item_id`. The exclusions concentrate on
  **C1 Day0 (2 of 10 rows, 20 %)** and **C2 Day0 (1 of 8, 12.5 %)** —
  surfaced in `anomalies_md` and the report's DQ section so leadership
  sees the demand they cannot fulfil.
- **Deterministic safety nets fire when the LLM over-allocates**: when the
  allocator proposes more cold-chain trucks than the cap (e.g. 3 for
  C1-Day0 when the cap is 2/day), `_clip_resource_allocation` reduces it to
  the cap and logs the rationale; `_recompute_penalty` rebuilds the score
  from supply × demand and emits a `deferral_breakdown` the report quotes
  verbatim. Both corrections are visible in the final HTML report.

### 4.2 Stress-test results (`stress_test_scenarios.py`)

Adversarial scenarios run against the same 14-day CSV. Penalties depend on
the live weather forecast and the day's eligible-driver roster, so the
figures below are from representative verified runs (✓ = exact figure
captured from a live run; others marked ~ vary run-to-run):

| # | Scenario | Audit | Penalty (pts) | Deferred | Notes |
|---|---|---:|---:|---:|---|
| 0 | Baseline | 1 | **0** ✓ | **0** ✓ | Supply ≥ demand; plan passes first try |
| 1 | 20 % demand spike on C2 | 1–2 | ~ medium | ~ | Allocator shifts cold-chain trucks toward Tier-1 |
| 2 | Driver shortage (3 of 6) | 1–2 | ~ medium-high | ~ | `apply_workforce_to_pool` caps the driver pool; Tier-2 deferrals dominate, Tier-1 protected |
| 3 | Cold-chain truck breakdown (1 of 2) | 1–2 | **1040** ✓ | **14** ✓ | 8 Tier-1 + 6 Tier-2 deferred; `_clip_resource_allocation` + `_recompute_penalty` fire; red Tier-1 banner |
| 4 | Severe storm forces C1 risk = 3 | ≤3 | **0** ✓ | **0** ✓ | 40 % buffer + escalation; **human checkpoint fires** (interrupt); workforce reduced drivers 6→4 |
| 5 | Combined spike + driver shortage | 2–3 | ~ very high | ~ | Audit loop visible; force-pass triggers if 3 retries exhausted |

The audit loop **demonstrably re-prompts the planner** with specific
violations on stressed scenarios; the deterministic penalty recomputation
**caught LLM under-counting in scenario 3** (LLM reported 6 deferred /
880 pts; the deterministic recompute corrected it to 14 deferred /
1040 pts, and the banner the report rendered was driven by the corrected
figure, not the LLM's). Scenario 4's interrupt was verified end-to-end:
the graph suspended at `node_human_checkpoint`, and the report's Approval
Status section correctly read *"Required: True · Trigger: route_risk_score_0_3
= 3 on C1_I95_NJ_BOS"* on reject and approve paths.

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

## 4.4 Validation & Realism Layer (extension beyond the rubric's five ideas)

Office-hours feedback raised a fundamental question that the playbook penalty
model alone cannot answer: *"How do you validate that the system's
recommendations are actually fit for the real world, not just penalty-optimal?"*
A plan can minimise the proxy penalty while being operationally unsafe — for
example, dispatching the same driver on back-to-back 12-hour shifts.

We addressed this with **three independent feedback loops**, each closing a
different gap between the proxy and reality. The implementation is fully
production-grade — none of the three loops is mocked or stubbed.

### Loop 1 — Manager rating (fitness of the plan)
**Gap closed:** Did the plan reflect operational judgement?
**Implementation:**
- Streamlit "Rate this plan" card after every run captures: 1-5 stars,
  multi-select tags (Right-sized / Aggressive / Conservative / Risky /
  Wasteful / Wrong corridor priority / Driver concern / Cold-chain concern),
  free-text "what would you change?".
- Persisted to `feedback/manager_ratings.csv` via `append_manager_rating`.
- Last-10 ratings loaded by `node_load_workforce_state` into AppState as
  `manager_feedback_recent`, then injected into `PLANNER_PROMPT` as
  preference-signal context.
- "Feedback" tab aggregates the trend (average rating, full table).

### Loop 2 — Workforce reality (fitness of the allocation)
**Gap closed:** Did the assignments respect human constraints the playbook
doesn't capture (fatigue, certification, leave)?
**Implementation:**
- `feedback/driver_state.csv` — 8-driver roster encoding `certifications`,
  `hours_last_24h`, `hours_last_7d`, `consecutive_days`, `fatigue_flag`,
  `preferred_corridors`, `active`. Eligibility rules in `workforce_tools.py`:
  - `active=false` → ineligible (medical leave)
  - `hours_last_7d ≥ 40` → ineligible (DOT-style weekly cap)
  - `consecutive_days ≥ 5` → ineligible (mandatory rest day)
  - `fatigue_flag=true` → eligible but flagged for warning
- `apply_workforce_to_pool` reduces `effective_resource_pool` by:
  - Driver count → number of eligible drivers
  - Cold-chain truck count → `min(physical_trucks, cold_chain_certified_eligible_drivers)` —
    you cannot run a cold-chain truck without a certified driver.
- `realism_check_allocation` runs after the LLM allocator + `_clip_resource_allocation`
  and produces (warnings, violations):
  - **Warning**: fatigue-flagged drivers in today's pool, all eligible
    drivers committed (no slack), drivers excluded for hour-cap.
  - **Violation**: more drivers/cold-trucks allocated than eligible —
    surfaces if upstream pool reduction missed something.
- The `REPORT_PROMPT` requires the report to render workforce warnings as a
  yellow "Workforce notes" box in Section 5 and violations as a red banner
  in Section 8.

### Loop 3 — Outcome calibration (fitness of the predictions)
**Gap closed:** Did the predicted penalty match what actually happened?
**Implementation:**
- `feedback/outcome_log.csv` — daily entries with `predicted_penalty`,
  `actual_penalty`, `predicted_deferred`, `actual_deferred`,
  `actual_tier1_late`, `actual_tier2_late`, `actual_cold_chain_breaches`,
  `incident_notes`. Filled the morning after each run via Streamlit
  "Outcomes" form (`append_outcome`).
- `compute_calibration` computes: MAE, bias (`actual − predicted`),
  cold-chain breach total, and a one-sentence `headline` like
  *"Calibration: 6 historical runs · MAE 60 pts · system has under-predicted
  actual penalty by 60 pts on average · 1 cold-chain breach recorded."*
- Headline is injected into `OPS_ANALYSIS_PROMPT` so the OpsDataAgent
  adjusts its risk language (e.g., adding *"historical calibration
  suggests actual outcomes typically run ~60 pts above the predicted figure"*).
- "Calibration" Streamlit tab visualises predicted-vs-actual scatter against
  a perfect-calibration line, with hero metrics (MAE, bias direction).

### Why this matters

| Without the validation layer | With it |
|---|---|
| Penalty 0 = "all clear" | Penalty 0 + workforce warning "D-3 fatigue-flagged — avoid back-to-back" |
| Plans optimise a proxy | Plans optimise a proxy, then are constrained by human limits, then are re-calibrated against past truth |
| No way to validate reports | Every run feeds three datasets that converge into a measurable system precision |
| LLM allocator can over-allocate cold trucks past certified drivers | `apply_workforce_to_pool` caps cold-chain capacity at certified-driver count automatically |

This is the architecture that makes the system **continuously improve with
each run** rather than being a one-shot LLM. After 30 days of operation, the
manager rating average, the calibration MAE, and the workforce warning
frequency together tell a complete story of system fitness — exactly the
validation dataset the office-hours question was asking for.

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
- **Weather risk is a route-level proxy.** The system scores corridors from
  Open-Meteo variables and waypoints, but does not yet include live traffic,
  road closures, or calibrated ETA predictions.
- **Scenario simulation is rule-bounded.** The ScenarioParserAgent handles
  common disruptions (demand spikes, driver shortages, cold-chain breakdowns,
  closures), but complex cascading disruptions are still simplified.
- **Real-world validation is limited.** Unit and stress tests verify correct
  behaviour, but the project does not yet include large-scale historical
  backtesting against actual late deliveries, cold-chain breaches, or manager
  decisions.

### 5.2 If we had more time / production data access

1. **Make the framework reusable across business cases** — separate the
   project-specific inputs (playbook, corridor catalog, item master, resource
   pool, KPI definitions) from the general multi-agent framework (audit loop,
   scenario parser, allocator, human checkpoint, report agent). Another
   business case could then reuse the architecture by swapping the operational
   rules and data, instead of rebuilding the agent workflow.
2. **Hook the email node up to a real ops mailbox** — currently the
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
7. **Historical backtesting and calibration** — with real delivery outcomes
   we would replay past dispatch days and compare predicted penalties,
   deferred units, and SLA risks against what actually happened, tuning the
   penalty weights with real evidence rather than playbook defaults.
8. **Richer operational data integration** — hospital priority levels,
   patient-critical orders, inventory levels, driver schedules, and loading-
   dock constraints would make recommendations more realistic and more
   directly actionable for dispatch managers.

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
