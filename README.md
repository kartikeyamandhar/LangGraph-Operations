# SeeWeeS Multi-Agent Dispatch Intelligence

A LangGraph-based multi-agent system that turns SeeWeeS' specialty-medicine
operations data into an executive-ready dispatch report. The pipeline ingests
the operations playbook (PDF), 14 days of multi-corridor shipment data (CSV),
and live weather forecasts; runs a six-agent reasoning pipeline with a
self-correcting audit loop, deterministic resource allocation, and a
human-in-the-loop checkpoint; and returns a colour-coded HTML report.

Built for the **UCLA MSBA AI Agents Project Challenge 2026**.

---

## What it does

```
┌─────────────────────────── PARALLEL DATA GATHERING ──────────────────────────┐
│  pdf_context  ─┐         csv_analysis  ─┐         weather  ─┐                │
│   (RAG over    │          (per-corridor │          (5 + 4   │                │
│    playbook)   │           KPIs, item    │          waypoints│                │
│                │           reconciliation│          via Open │                │
│                │           anomaly check)│          -Meteo)  │                │
└────────────────┴─────────────────────────┴───────────────────┘                │
                                  ↓
                              planner ──→ audit ──┬─ FAIL → planner (loop)
                                                  │
                                                  └─ PASS → allocator
                                                              ↓
                                                       human_checkpoint
                                                       (interrupt if any
                                                        corridor risk = 3)
                                                              ↓
                                                            report
                                                              ↓
                                                            email (optional)
```

---

## Enhancements implemented (all 5)

| # | Idea | Where it lives |
|---|---|---|
| 1 | **Self-Correction & Audit Loop** | `node_audit` + `_route_after_audit` cyclic edge in `src/graph.py`; Python hard-rule checks (buffer policy, escalation) followed by `AuditAgent` GPT soft-check. Max 3 retries before force-pass with violation flag. |
| 2 | **What-if Scenario Simulation** | `scenario` field on `AppState`; `stress_test_scenarios.py` runs six pre-built disruptions (demand spike, driver shortage, cold-chain truck breakdown, I-95 closure, dual disruption, baseline). |
| 3 | **Deep-Dive Trend & Item Master Reconciliation** | `_reconcile_row` + `ReconciliationLog` in `src/tools/csv_tools.py` map legacy IDs and name aliases to a canonical item master; `_compute_corridor_kpis` produces per-corridor / per-day Tier-1/Tier-2 mix, cold-chain demand, and 7-day trend. |
| 4 | **Human-in-the-Loop Workflow** | `node_human_checkpoint` calls `langgraph.types.interrupt(...)` when `max(route_risk_score) >= 3`. CLI runs auto-approve via `Command(resume="YES")`; the Streamlit UI surfaces an Approve / Reject button. |
| 5 | **Multi-Region Resource Planning** | `node_allocator` + `AllocatorAgent` with deterministic post-corrections: `_clip_cold_chain_allocation` clips over-allocation to the daily cap, `_recompute_penalty` rebuilds the penalty score from truck-supply × demand instead of trusting the LLM. |

---

## Project structure

```
.
├── src/
│   ├── main.py                 # CLI entry point — runs the full graph
│   ├── graph.py                # LangGraph StateGraph, AppState, all 9 nodes
│   ├── agents.py               # 6 GPT agent functions (gpt-4.1-mini, T=0.2)
│   ├── prompts.py              # ChatPromptTemplate for each agent
│   ├── tracing.py              # LangSmith init
│   └── tools/
│       ├── pdf_tools.py        # RAG: PyPDFLoader + Chroma + OpenAI embeddings
│       ├── csv_tools.py        # Reconciliation, per-corridor KPIs, anomalies
│       ├── weather_tools.py    # Open-Meteo client + risk scoring
│       └── email_tools.py      # SMTP report sender
│
├── app.py                      # Streamlit UI — interactive run + approval
├── stress_test_scenarios.py    # Batch what-if simulator (6 scenarios)
├── generate_synthetic_data.py  # Synthetic-shipment generator (6 profiles)
│
├── data/                       # Authoritative inputs (PDF playbook + sample)
├── data-for-enhancement/       # Multi-corridor CSV, resource constraints,
│                               # synthetic CSVs, markdown playbook source
├── docs/                       # Technical & business documentation
├── tests/                      # pytest suite (mocked LLMs)
├── chroma_db/                  # Local vector store (gitignored)
│
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Setup

```bash
# 1. Clone and enter the repo
cd MSBA_AI_Agents_Demo

# 2. Create and activate a Python 3.11 virtualenv
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate            # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
#  → open .env and paste your OPENAI_API_KEY
#  → optionally enable LangSmith tracing or SMTP email

# 5. Verify the install
pytest                             # all tests should pass
```

---

## Running the system

### A. Single end-to-end run (CLI)
```bash
python src/main.py
```
Reads `data/SeeWeeS Specialty distribution.pdf` and
`data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv`, runs the full
graph, and writes `report.html` to the project root.

### B. What-if scenario stress test
```bash
python stress_test_scenarios.py                # all 6 scenarios
python stress_test_scenarios.py --scenario 2   # just scenario index 2
```
Saves one HTML report per scenario (`report_scenario_<n>.html`) and prints a
penalty-score summary table.

### C. Interactive Streamlit UI
```bash
streamlit run app.py
```
Opens an in-browser dashboard with a corridor risk map, KPI tiles, the
allocation table, and an **Approve / Reject** button when the human checkpoint
fires. Lets you swap data files (real, synthetic, or DQ-heavy) from the
sidebar.

### D. Generate synthetic shipment data
```bash
python generate_synthetic_data.py
```
Writes 6 profile-specific CSVs to `data-for-enhancement/synthetic/`
(baseline, volume spike, growth trend, DQ-heavy, Tier-1 surge, 60-day rich).
Useful when you want fresh data for the Streamlit UI or a new stress test.

### E. Run the test suite
```bash
pytest                             # full suite, mocked LLMs (no API cost)
pytest -v tests/test_graph.py      # one file
pytest -k "audit"                  # one keyword
```

---

## Data files

| Path | Role |
|---|---|
| `data/SeeWeeS Specialty distribution.pdf` | Playbook v0.2 — authoritative business rules, RAG source |
| `data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md` | Same playbook in markdown — easier to read while developing |
| `data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv` | 14-day shipment feed across C1 (NJ→Boston) + C2 (NJ→Philadelphia), with intentional DQ issues |
| `data-for-enhancement/Resource_availability_48h.csv` | Daily driver / truck / cold-chain truck capacity (mirrored in `RESOURCE_POOL` in `graph.py`) |
| `data-for-enhancement/synthetic/*.csv` | Six profiles for stress testing (volume spike, DQ heavy, Tier-1 surge, etc.) |
| `data/Incoming_shipment_02_08.csv` | Single-corridor sample (legacy demo data) |

---

## Configuration knobs

Defined at the top of `src/graph.py`:

- `CORRIDOR_WAYPOINTS` — hardcoded from Playbook v0.2 §3.2; if you add a
  corridor, register its waypoints here (lat/lon) so the weather node fans out
  across them.
- `BUFFER_POLICY` — `{0: 0%, 1: 10%, 2: 25%, 3: 40%}` (Playbook §5.2).
- `RESOURCE_POOL` — daily driver / truck / cold-chain truck availability
  (Playbook §6).
- `MAX_AUDIT_ATTEMPTS` — set to 3; after that the audit force-passes and
  flags violations on the report.

Penalty model in `_recompute_penalty` (Playbook §7):
- Tier 1 SLA violation: **100 pts/unit** · Tier 2: **40 pts/unit**
- Cold-chain breach: **+80 pts/unit** · Truck capacity: **10 units/truck**

---

## Architecture: how the audit loop works

1. **Planner** generates a prose plan plus a JSON block:
   `{buffer_pct_c1, buffer_pct_c2, escalation_triggered, tier1_sla_at_risk, estimated_penalty_score}`.
2. **Audit (deterministic Python)** verifies the JSON against Playbook rules
   (buffer % == policy[risk_score], escalation iff risk == 3).
3. **Audit (LLM soft check)** runs only if Python checks pass; checks for
   vague / non-actionable plans.
4. On `FAIL`, control loops back to **planner** with the specific violations
   in `audit_feedback`. The planner is told to fix only those violations.
5. After `MAX_AUDIT_ATTEMPTS` (3), the system force-passes with a banner on
   the final report so leadership sees the failure mode.

This is the only cyclic edge in the graph; all other edges are linear.

---

## Validation strategy

- **Unit tests** (`tests/`): cover the AppState schema, weather risk scoring,
  CSV reconciliation, audit routing, penalty recomputation, cold-chain
  clipping, and human-checkpoint trigger logic. LLM calls are mocked, so the
  suite runs offline and does not consume OpenAI credits.
- **Stress tests** (`stress_test_scenarios.py`): six adversarial what-if
  inputs (demand spike, driver shortage, cold-chain breakdown, I-95 closure,
  dual disruption) verify that the planner respects audit feedback and that
  the allocator's penalty model degrades gracefully.
- **Manual run** (`python src/main.py`): smoke test against live OpenAI +
  Open-Meteo APIs.

See `docs/PROJECT_REPORT.md` for the full technical & business write-up.

---

## Common issues

| Symptom | Fix |
|---|---|
| `403 Forbidden` from LangSmith | Set `LANGCHAIN_TRACING_V2=false` in `.env` until you have a valid key |
| `SMTPAuthenticationError` | Either fill real `SMTP_*` values or blank `REPORT_EMAIL_TO=` — the email node now skips on missing creds and never crashes the run |
| `Failed to find OPENAI_API_KEY` | Confirm `.env` exists in the project root and `OPENAI_API_KEY=` is filled in |
| Slow first run | First call to `pdf_tools.PdfRag.build` chunks + embeds the PDF (~30 s); subsequent runs reuse `chroma_db/` |

---

## Submission

This repository is the deliverable for the UCLA MSBA AI Agents Project
Challenge 2026. See `docs/PROJECT_REPORT.md` for the technical & business
documentation. Submission deadline: **Sunday May 10 2026, 12:00 PM PST**.
