from __future__ import annotations

import os
from typing import TypedDict, Dict, Any, List, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from tools.pdf_tools import PdfRag
from tools.csv_tools import analyze_csv
from tools.weather_tools import get_weather_forecast, derive_dispatch_weather_risk
from tools.email_tools import send_email_smtp
from agents import (
    run_context_agent, run_ops_agent,
    run_planner_agent, run_audit_agent,
    run_allocator_agent, run_report_agent,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Corridor waypoints (authoritative — from Playbook v0.2 Section 3.2)
# Hardcoded here so we never depend on fragile PDF regex parsing.
# ---------------------------------------------------------------------------
CORRIDOR_WAYPOINTS: Dict[str, List[Dict[str, Any]]] = {
    "C1_I95_NJ_BOS": [
        {"id": "C1_W1", "city": "Newark",      "state": "NJ", "lat": 40.7357, "lon": -74.1724},
        {"id": "C1_W2", "city": "Bronx",        "state": "NY", "lat": 40.8448, "lon": -73.8648},
        {"id": "C1_W3", "city": "New Haven",    "state": "CT", "lat": 41.3083, "lon": -72.9279},
        {"id": "C1_W4", "city": "Providence",   "state": "RI", "lat": 41.8240, "lon": -71.4128},
        {"id": "C1_W5", "city": "Boston",       "state": "MA", "lat": 42.3601, "lon": -71.0589},
    ],
    "C2_NJ_PHL": [
        {"id": "C2_W1", "city": "Newark",        "state": "NJ", "lat": 40.7357, "lon": -74.1724},
        {"id": "C2_W2", "city": "New Brunswick", "state": "NJ", "lat": 40.4862, "lon": -74.4518},
        {"id": "C2_W3", "city": "Trenton",       "state": "NJ", "lat": 40.2204, "lon": -74.7643},
        {"id": "C2_W4", "city": "Philadelphia",  "state": "PA", "lat": 39.9526, "lon": -75.1652},
    ],
}

# Buffer policy (Playbook Section 5.2)
BUFFER_POLICY: Dict[int, int] = {0: 0, 1: 10, 2: 25, 3: 40}

# Resource availability (from Resource_availability_48h.csv)
RESOURCE_POOL: Dict[str, Dict[str, int]] = {
    "Day0": {"driver": 6, "truck_standard": 4, "truck_temp_controlled": 2},
    "Day1": {"driver": 6, "truck_standard": 4, "truck_temp_controlled": 2},
}

MAX_AUDIT_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Shared state — the notepad every node reads and writes
# ---------------------------------------------------------------------------
class AppState(TypedDict, total=False):
    # Inputs
    pdf_path: str
    csv_path: str
    scenario: Optional[str]           # what-if scenario description

    # Step 1 outputs (parallel)
    business_context: str
    csv_summary: Dict[str, Any]
    csv_kpis: Dict[str, Any]
    anomalies_md: str
    ops_insights: str
    corridor_kpis: List[Dict[str, Any]]
    trend_summary: Dict[str, Any]
    weather_risk: Dict[str, Any]      # keyed by corridor_id

    # Planner
    dispatch_plan: str                # narrative plan (prose)
    plan_structured: Dict[str, Any]   # structured JSON for audit checks

    # Audit loop
    audit_verdict: str                # "PASS" or "FAIL"
    audit_feedback: str               # specific violations for planner retry
    audit_attempts: int

    # Resource allocation
    allocation_plan: Dict[str, Any]

    # Human checkpoint
    human_approved: bool

    # Report
    report_html: str


# ---------------------------------------------------------------------------
# NODE 1A — PDF context (runs in parallel)
# ---------------------------------------------------------------------------
def node_pdf_context(state: AppState) -> AppState:
    rag = PdfRag(persist_dir="chroma_db")
    vectordb = rag.build(state["pdf_path"])
    retriever = rag.retriever(vectordb, k=6)
    query = "Extract KPI definitions, thresholds, SLAs, constraints, dispatch rules, buffer policy, escalation rules."
    docs = retriever.invoke(query)
    snippets = "\n\n---\n\n".join(d.page_content for d in docs)
    return {"business_context": run_context_agent(snippets)}


# ---------------------------------------------------------------------------
# NODE 1B — CSV analysis (runs in parallel)
# ---------------------------------------------------------------------------
def node_csv_analysis(state: AppState) -> AppState:
    res = analyze_csv(state["csv_path"])
    ops_insights = run_ops_agent(
        summary=res.summary,
        kpis=res.kpis,
        anomalies_md=res.anomalies_md,
    )
    return {
        "csv_summary": res.summary,
        "csv_kpis": res.kpis,
        "anomalies_md": res.anomalies_md,
        "ops_insights": ops_insights,
        "corridor_kpis": [k.__dict__ for k in res.corridor_kpis],
        "trend_summary": res.trend_summary,
    }


# ---------------------------------------------------------------------------
# NODE 1C — Weather for ALL corridors (runs in parallel)
# ---------------------------------------------------------------------------
def node_weather(__: AppState) -> AppState:
    tz = os.getenv("WEATHER_TZ", "America/New_York")
    weather_risk: Dict[str, Any] = {}

    for corridor_id, waypoints in CORRIDOR_WAYPOINTS.items():
        per_waypoint: List[Dict[str, Any]] = []
        max_score = -1
        worst = None

        for wp in waypoints:
            forecast = get_weather_forecast(str(wp["lat"]), str(wp["lon"]), tz)
            risk = derive_dispatch_weather_risk(forecast)
            enriched = {
                "waypoint": wp["id"],
                "city": wp["city"],
                "state": wp["state"],
                **risk,
            }
            per_waypoint.append(enriched)
            if enriched["risk_score_0_3"] > max_score:
                max_score = enriched["risk_score_0_3"]
                worst = enriched

        weather_risk[corridor_id] = {
            "corridor_id": corridor_id,
            "route_risk_score_0_3": max_score,
            "worst_waypoint": worst,
            "per_waypoint": per_waypoint,
            "required_buffer_pct": BUFFER_POLICY.get(max_score, 0),
            "escalation_required": max_score == 3,
        }

    print(f"\n[Weather] Corridor scores: { {k: v['route_risk_score_0_3'] for k, v in weather_risk.items()} }\n")
    return {"weather_risk": weather_risk}


# ---------------------------------------------------------------------------
# NODE 2 — Planner (waits for all 3 parallel nodes to finish)
# ---------------------------------------------------------------------------
def node_planner(state: AppState) -> AppState:
    attempts = state.get("audit_attempts", 0)
    feedback = state.get("audit_feedback", "")
    scenario = state.get("scenario", "")

    plan_text, plan_structured = run_planner_agent(
        business_context=state.get("business_context", ""),
        ops_insights=state.get("ops_insights", ""),
        weather_risk=state.get("weather_risk", {}),
        corridor_kpis=state.get("corridor_kpis", []),
        resource_pool=RESOURCE_POOL,
        audit_feedback=feedback,
        scenario=scenario,
    )

    return {
        "dispatch_plan": plan_text,
        "plan_structured": plan_structured,
        "audit_attempts": attempts + 1,
        "audit_feedback": "",
    }


# ---------------------------------------------------------------------------
# NODE 3 — Audit (Python rule checks + GPT reasoning check)
# ---------------------------------------------------------------------------
def node_audit(state: AppState) -> AppState:
    plan = state.get("plan_structured", {})
    weather_risk = state.get("weather_risk", {})
    attempts = state.get("audit_attempts", 1)
    violations: List[str] = []

    # ── Hard rule checks (Python, not GPT) ──────────────────────────────────
    for corridor_id, w in weather_risk.items():
        score = w.get("route_risk_score_0_3", 0)
        expected_buffer = BUFFER_POLICY.get(score, 0)

        corridor_key = "c1" if "BOS" in corridor_id else "c2"
        applied_buffer = plan.get(f"buffer_pct_{corridor_key}")

        if applied_buffer is not None and applied_buffer != expected_buffer:
            violations.append(
                f"{corridor_id}: buffer is {applied_buffer}% but risk score {score} requires {expected_buffer}%"
            )

        if score == 3 and not plan.get("escalation_triggered", False):
            violations.append(
                f"{corridor_id}: risk score 3 requires escalation — not triggered in plan"
            )

    # Note: cold-chain truck count is enforced deterministically in node_allocator
    # via _clip_cold_chain_allocation, so we don't audit the planner for it here.

    # ── If hard checks pass, run GPT soft check ──────────────────────────────
    if not violations:
        gpt_verdict, gpt_feedback = run_audit_agent(
            dispatch_plan=state.get("dispatch_plan", ""),
            business_context=state.get("business_context", ""),
            weather_risk=weather_risk,
        )
        if gpt_verdict == "FAIL":
            violations.append(gpt_feedback)

    if violations:
        # Force pass after max attempts to avoid infinite loop
        if attempts >= MAX_AUDIT_ATTEMPTS:
            print(f"\n[Audit] Max attempts ({MAX_AUDIT_ATTEMPTS}) reached. Force-passing with violation flags.\n")
            return {
                "audit_verdict": "PASS",
                "audit_feedback": f"UNRESOLVED VIOLATIONS (max retries): {'; '.join(violations)}",
            }

        feedback = "Fix the following violations:\n" + "\n".join(f"- {v}" for v in violations)
        print(f"\n[Audit] FAIL — attempt {attempts}/{MAX_AUDIT_ATTEMPTS}\n{feedback}\n")
        return {"audit_verdict": "FAIL", "audit_feedback": feedback}

    print(f"\n[Audit] PASS — plan is compliant.\n")
    return {"audit_verdict": "PASS", "audit_feedback": ""}


def _route_after_audit(state: AppState) -> str:
    return "planner" if state.get("audit_verdict") == "FAIL" else "allocator"


# ---------------------------------------------------------------------------
# NODE 4 — Resource allocator (penalty-minimising truck assignment)
# ---------------------------------------------------------------------------
def _clip_cold_chain_allocation(allocation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce the cold-chain truck cap (truck_temp_controlled per day) on the
    allocator's output. The LLM allocator sometimes proposes more cold-chain
    trucks than physically exist; this is a post-correction that clips the
    largest-count corridor first until the per-day total respects RESOURCE_POOL.

    Each clipped truck represents ~10 units that cannot be cold-chain dispatched,
    so we increment deferred_units accordingly and prepend a note to the
    rationale so the report agent shows the constraint was enforced.
    """
    if not isinstance(allocation, dict) or allocation.get("raw"):
        return allocation

    total_clipped = 0
    for day in ("Day0", "Day1"):
        day_alloc = allocation.get(day)
        if not isinstance(day_alloc, dict):
            continue
        max_avail = RESOURCE_POOL[day]["truck_temp_controlled"]

        # Greedy reduction: repeatedly take 1 truck from the highest-count corridor
        while True:
            counts = {
                c: int(v.get("truck_temp_controlled", 0) or 0)
                for c, v in day_alloc.items()
                if isinstance(v, dict)
            }
            total = sum(counts.values())
            if total <= max_avail:
                break
            top_corridor = max(counts, key=counts.get)
            day_alloc[top_corridor]["truck_temp_controlled"] = counts[top_corridor] - 1
            total_clipped += 1

    if total_clipped > 0:
        existing_rationale = allocation.get("rationale", "") or ""
        allocation["rationale"] = (
            f"[Auto-corrected: clipped {total_clipped} cold-chain truck(s) to "
            f"respect daily cap of {RESOURCE_POOL['Day0']['truck_temp_controlled']}/day.] "
            + existing_rationale
        )

    return allocation


# Penalty model from playbook Section 7
PENALTY_TIER1 = 100   # pts per Tier 1 unit deferred (SLA violation)
PENALTY_TIER2 = 40    # pts per Tier 2 unit deferred
UNITS_PER_TRUCK = 10  # capacity per truck (playbook truck capacity model)


def _recompute_penalty(
    allocation: Dict[str, Any],
    corridor_kpis: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministically recompute total_penalty_score and deferred_units from
    the truck allocation in `allocation` and the demand in `corridor_kpis`.

    Strategy (matches playbook priority order):
      1. Tier 1 demand is fulfilled first within available cold-chain capacity.
         (In our item master, all Tier 1 SKUs are cold-chain: antiviral,
         oncology biologic, clinical trial.)
      2. Tier 2 cold-chain (insulin) takes any cold-chain capacity left.
      3. Tier 2 standard demand (room-temp + controlled) is fulfilled within
         standard truck capacity.
      4. Anything that doesn't fit is a deferred unit, scored at tier-weighted
         penalty points.

    The LLM allocator's reported penalty/deferred numbers are replaced with
    this deterministic computation so the published report reflects the
    actual constraint outcome.
    """
    if not isinstance(allocation, dict) or allocation.get("raw"):
        return allocation

    total_penalty = 0
    total_deferred = 0

    for kpi in corridor_kpis:
        corridor = kpi.get("corridor_id")
        day = kpi.get("day")
        day_alloc = allocation.get(day, {}) if isinstance(allocation.get(day), dict) else {}
        c_alloc = day_alloc.get(corridor, {}) if isinstance(day_alloc.get(corridor), dict) else {}

        cold_supply = int(c_alloc.get("truck_temp_controlled", 0) or 0) * UNITS_PER_TRUCK
        std_supply  = int(c_alloc.get("truck_standard", 0) or 0) * UNITS_PER_TRUCK

        # In our domain, all Tier 1 SKUs are cold-chain
        tier1_cold_demand = int(kpi.get("tier1_units", 0))
        tier2_cold_demand = max(0, int(kpi.get("cold_chain_units", 0)) - tier1_cold_demand)
        # All non-cold-chain demand is Tier 2 by construction
        std_demand = int(kpi.get("room_temp_units", 0)) + int(kpi.get("controlled_units", 0))

        # Tier 1 cold-chain dispatched first
        t1_cold_disp = min(tier1_cold_demand, cold_supply)
        t1_cold_def  = tier1_cold_demand - t1_cold_disp
        cold_left    = cold_supply - t1_cold_disp

        # Tier 2 cold-chain takes remainder of cold supply
        t2_cold_disp = min(tier2_cold_demand, cold_left)
        t2_cold_def  = tier2_cold_demand - t2_cold_disp

        # Tier 2 standard against standard truck supply
        std_disp = min(std_demand, std_supply)
        std_def  = std_demand - std_disp

        total_penalty  += t1_cold_def * PENALTY_TIER1 + (t2_cold_def + std_def) * PENALTY_TIER2
        total_deferred += t1_cold_def + t2_cold_def + std_def

    llm_penalty  = int(allocation.get("total_penalty_score", 0) or 0)
    llm_deferred = int(allocation.get("deferred_units", 0) or 0)

    allocation["total_penalty_score"] = total_penalty
    allocation["deferred_units"]      = total_deferred

    if total_penalty != llm_penalty or total_deferred != llm_deferred:
        existing = allocation.get("rationale", "") or ""
        allocation["rationale"] = (
            f"[Penalty recomputed deterministically from truck allocation × demand: "
            f"{total_deferred} units deferred → {total_penalty} pts "
            f"(Tier 1 × {PENALTY_TIER1} + Tier 2 × {PENALTY_TIER2}). "
            f"LLM had reported {llm_deferred} deferred / {llm_penalty} pts.] "
            + existing
        )

    return allocation


def node_allocator(state: AppState) -> AppState:
    allocation = run_allocator_agent(
        corridor_kpis=state.get("corridor_kpis", []),
        resource_pool=RESOURCE_POOL,
        weather_risk=state.get("weather_risk", {}),
        business_context=state.get("business_context", ""),
    )
    allocation = _clip_cold_chain_allocation(allocation)
    allocation = _recompute_penalty(allocation, state.get("corridor_kpis", []))
    return {"allocation_plan": allocation}


# ---------------------------------------------------------------------------
# NODE 5 — Human checkpoint (interrupt if any corridor risk = 3)
# ---------------------------------------------------------------------------
def node_human_checkpoint(state: AppState) -> AppState:
    weather_risk = state.get("weather_risk", {})
    max_score = max(
        (v.get("route_risk_score_0_3", 0) for v in weather_risk.values()),
        default=0,
    )

    if max_score >= 3 and not state.get("human_approved", False):
        from langgraph.types import interrupt
        print("\n[CHECKPOINT] Risk score 3 detected — awaiting manager approval.\n")
        decision = interrupt({
            "message": "Risk score 3 detected on one or more corridors. Review the allocation plan and approve or reject.",
            "allocation_plan": state.get("allocation_plan", {}),
            "weather_risk": weather_risk,
        })
        approved = str(decision).strip().upper() in ("YES", "APPROVE", "Y")
        return {"human_approved": approved}

    return {"human_approved": True}


# ---------------------------------------------------------------------------
# NODE 6 — Report
# ---------------------------------------------------------------------------
def node_report(state: AppState) -> AppState:
    html = run_report_agent(
        business_context=state.get("business_context", ""),
        kpis=state.get("csv_kpis", {}),
        corridor_kpis=state.get("corridor_kpis", []),
        trend_summary=state.get("trend_summary", {}),
        anomaly_highlights=state.get("anomalies_md", "(none)"),
        weather_risk=state.get("weather_risk", {}),
        dispatch_plan=state.get("dispatch_plan", ""),
        allocation_plan=state.get("allocation_plan", {}),
        audit_feedback=state.get("audit_feedback", ""),
        scenario=state.get("scenario", ""),
        human_approved=state.get("human_approved", True),
    )
    return {"report_html": html}


# ---------------------------------------------------------------------------
# NODE 7 — Email (optional)
# ---------------------------------------------------------------------------
def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        print("[Email] REPORT_EMAIL_TO not set — skipping email send.")
        return {}

    required = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD")
    missing = [k for k in required if not os.getenv(k, "").strip()]
    if missing:
        print(f"[Email] Missing SMTP vars {missing} — skipping email send.")
        return {}

    try:
        send_email_smtp(
            subject="SeeWeeS Multi-Agent Dispatch Report",
            html_body=state["report_html"],
            to_email=to_email,
        )
        print(f"[Email] Report sent to {to_email}.")
    except Exception as exc:
        print(f"[Email] Send failed ({type(exc).__name__}): {exc} — pipeline continues.")
    return {}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    g = StateGraph(AppState)

    # Register nodes
    g.add_node("pdf_context",       node_pdf_context)
    g.add_node("csv_analysis",      node_csv_analysis)
    g.add_node("weather",           node_weather)
    g.add_node("planner",           node_planner)
    g.add_node("audit",             node_audit)
    g.add_node("allocator",         node_allocator)
    g.add_node("human_checkpoint",  node_human_checkpoint)
    g.add_node("report",            node_report)
    g.add_node("email",             node_email)

    # Parallel fan-out from START → three independent data-gathering nodes
    g.add_edge(START, "pdf_context")
    g.add_edge(START, "csv_analysis")
    g.add_edge(START, "weather")

    # All three feed into planner (LangGraph waits for all three to finish)
    g.add_edge("pdf_context",  "planner")
    g.add_edge("csv_analysis", "planner")
    g.add_edge("weather",      "planner")

    # Audit loop — conditional edge routes back to planner on FAIL
    g.add_edge("planner", "audit")
    g.add_conditional_edges("audit", _route_after_audit, {"planner": "planner", "allocator": "allocator"})

    # Linear after audit passes
    g.add_edge("allocator",        "human_checkpoint")
    g.add_edge("human_checkpoint", "report")
    g.add_edge("report",           "email")
    g.add_edge("email",            END)

    return g.compile(checkpointer=checkpointer)
