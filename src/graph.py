from __future__ import annotations

import os
from typing import TypedDict, Dict, Any, List, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from tools.pdf_tools import PdfRag
from tools.csv_tools import analyze_csv
from tools.weather_tools import get_weather_forecast, derive_dispatch_weather_risk
from tools.email_tools import send_email_smtp
from tools.workforce_tools import (
    load_workforce_state, load_manager_ratings, compute_calibration,
    apply_workforce_to_pool, realism_check_allocation,
    WorkforceState,
)
from agents import (
    run_scenario_parser_agent,
    run_context_agent, run_ops_agent,
    run_planner_agent, run_audit_agent,
    run_allocator_agent, run_report_agent,
)
import copy

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

# Base resource availability (from Resource_availability_48h.csv).
# This is the DEFAULT pool. The scenario_apply node may override individual
# fields based on the user's free-text scenario; downstream nodes always read
# `effective_resource_pool` from state, never this constant directly.
BASE_RESOURCE_POOL: Dict[str, Dict[str, int]] = {
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
    scenario: Optional[str]           # what-if scenario description (free text)

    # Step 1 outputs (parallel)
    business_context: str
    csv_summary: Dict[str, Any]
    csv_kpis: Dict[str, Any]
    anomalies_md: str
    ops_insights: str
    corridor_kpis: List[Dict[str, Any]]
    trend_summary: Dict[str, Any]
    weather_risk: Dict[str, Any]      # keyed by corridor_id

    # Workforce + feedback layer (the validation/realism engine)
    workforce_state: Dict[str, Any]          # eligible drivers, certs, fatigue
    manager_feedback_recent: List[Dict[str, Any]]  # last 10 manager ratings
    calibration_history: Dict[str, Any]      # predicted vs actual stats

    # Scenario apply (the agentic what-if engine)
    scenario_overrides: Dict[str, Any]      # parsed structured overrides
    effective_resource_pool: Dict[str, Any]  # base pool ⊕ resource_overrides ⊕ workforce
    scenario_diff: Dict[str, Any]            # human-readable "what changed"

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
    human_approval_required: bool   # True only when interrupt actually fired
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

    # Load calibration history directly here (cheap file read) so the ops
    # agent can adjust its language for past systematic miscalibration
    # without waiting on the parallel workforce node.
    calibration = compute_calibration("feedback/outcome_log.csv")

    ops_insights = run_ops_agent(
        summary=res.summary,
        kpis=res.kpis,
        anomalies_md=res.anomalies_md,
        calibration_history=calibration,
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
# NODE 1D — Workforce + feedback loader (parallel with PDF/CSV/weather)
#
# Loads the four feedback streams from `feedback/` and exposes:
#   • workforce_state  — driver pool + certifications + fatigue, derived
#   • manager_feedback_recent — last 10 manager ratings (prompt-injected)
#   • calibration_history — predicted-vs-actual aggregates
# ---------------------------------------------------------------------------
def node_load_workforce_state(__: AppState) -> AppState:
    workforce = load_workforce_state("feedback/driver_state.csv")
    manager_recent = load_manager_ratings("feedback/manager_ratings.csv")
    calibration = compute_calibration("feedback/outcome_log.csv")

    summary = workforce.to_dict().get("summary", "")
    print(f"\n[Workforce] {summary}")
    if calibration.get("runs_n"):
        print(f"[Workforce] {calibration.get('headline')}")
    if manager_recent:
        avg_star = sum(int(r.get("star_rating", 0)) for r in manager_recent) / len(manager_recent)
        print(f"[Workforce] Manager rating trend: {avg_star:.1f}/5 over last {len(manager_recent)} runs")

    return {
        "workforce_state": workforce.to_dict(),
        "manager_feedback_recent": manager_recent,
        "calibration_history": calibration,
    }


# ---------------------------------------------------------------------------
# NODE 1E — Scenario apply (waits for the 4 parallel nodes; runs before planner)
#
# This is the heart of the agentic what-if engine. It:
#   1. Calls ScenarioParserAgent on the free-text scenario.
#   2. Builds an `effective_resource_pool` by overlaying the parsed resource
#      overrides on top of BASE_RESOURCE_POOL.
#   3. Applies demand multipliers to corridor_kpis (every numeric field scales).
#   4. Applies weather overrides / closures to weather_risk so downstream
#      audit + planner see the disrupted reality.
#   5. Records a scenario_diff that the report agent can show to the user.
# ---------------------------------------------------------------------------
def _apply_resource_overrides(
    base: Dict[str, Dict[str, int]],
    overrides: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    out = copy.deepcopy(base)
    for day, day_overrides in (overrides or {}).items():
        if day not in out or not isinstance(day_overrides, dict):
            continue
        for resource, value in day_overrides.items():
            if resource in out[day] and value is not None:
                out[day][resource] = max(0, int(value))
    return out


def _apply_demand_multipliers(
    corridor_kpis: List[Dict[str, Any]],
    multipliers: Dict[str, float],
) -> List[Dict[str, Any]]:
    if not multipliers:
        return corridor_kpis
    scaled_keys = (
        "valid_rows", "tier1_units", "tier2_units",
        "cold_chain_units", "room_temp_units", "controlled_units",
        "trucks_needed_standard", "trucks_needed_cold_chain",
    )
    out: List[Dict[str, Any]] = []
    for kpi in corridor_kpis:
        new = dict(kpi)
        m = multipliers.get(kpi.get("corridor_id"))
        if m and m != 1.0:
            for k in scaled_keys:
                if k in new and isinstance(new[k], (int, float)):
                    new[k] = int(round(new[k] * float(m)))
        out.append(new)
    return out


def _apply_weather_overrides(
    weather_risk: Dict[str, Any],
    overrides: Dict[str, Dict[str, Any]],
    closures: List[str],
) -> Dict[str, Any]:
    out = copy.deepcopy(weather_risk) or {}
    for corridor_id, override in (overrides or {}).items():
        if corridor_id not in out:
            continue
        forced_score = int(override.get("route_risk_score_0_3", out[corridor_id].get("route_risk_score_0_3", 0)))
        forced_score = max(0, min(3, forced_score))
        out[corridor_id]["route_risk_score_0_3"] = forced_score
        out[corridor_id]["risk_score_0_3"] = forced_score
        out[corridor_id]["required_buffer_pct"] = BUFFER_POLICY.get(forced_score, 0)
        out[corridor_id]["escalation_required"] = forced_score == 3
        out[corridor_id]["scenario_forced"] = True
    for corridor_id in (closures or []):
        if corridor_id not in out:
            continue
        out[corridor_id]["route_risk_score_0_3"] = 3
        out[corridor_id]["risk_score_0_3"] = 3
        out[corridor_id]["required_buffer_pct"] = BUFFER_POLICY[3]
        out[corridor_id]["escalation_required"] = True
        out[corridor_id]["closed"] = True
        out[corridor_id]["scenario_forced"] = True
    return out


def node_scenario_apply(state: AppState) -> AppState:
    scenario_text = (state.get("scenario") or "").strip()

    overrides = run_scenario_parser_agent(scenario_text, BASE_RESOURCE_POOL)

    # Step 1: scenario overrides on top of base
    effective_pool = _apply_resource_overrides(
        BASE_RESOURCE_POOL, overrides.get("resource_overrides", {})
    )

    # Step 2: workforce reality on top of the scenario-adjusted pool.
    # Drivers excluded for fatigue / rest / leave reduce the cap further;
    # cold-chain trucks are capped by both physical trucks AND certified drivers.
    workforce_dict = state.get("workforce_state") or {}
    workforce_obj = WorkforceState(
        drivers=[]  # rebuild a lightweight WorkforceState from the dict
    )
    # Reconstruct properties needed by apply_workforce_to_pool.
    # We avoid re-reading the CSV — the dict carries everything we need.
    from tools.workforce_tools import DriverProfile
    workforce_obj.drivers = [
        DriverProfile(
            driver_id=d["driver_id"],
            name=d.get("name", ""),
            certifications=d.get("certifications", []),
            hours_last_24h=0.0,  # not needed for pool reduction
            hours_last_7d=0.0,
            consecutive_days=0,
            fatigue_flag=bool(d.get("fatigue_flag", False)),
            preferred_corridors=d.get("preferred_corridors", []),
            active=bool(d.get("eligible_today", False)),  # treat eligible_today as 'active for pool sizing'
        )
        for d in workforce_dict.get("drivers", [])
    ]
    effective_pool, workforce_pool_changes = apply_workforce_to_pool(effective_pool, workforce_obj)

    new_corridor_kpis = _apply_demand_multipliers(
        state.get("corridor_kpis", []) or [],
        overrides.get("demand_multipliers", {}) or {},
    )
    new_weather = _apply_weather_overrides(
        state.get("weather_risk", {}) or {},
        overrides.get("weather_overrides", {}) or {},
        overrides.get("corridor_closures", []) or [],
    )

    # Compute a transparent diff so the report can explain "what changed"
    diff: Dict[str, Any] = {"summary": overrides.get("summary", "")}
    pool_changes = []
    for day in BASE_RESOURCE_POOL:
        for resource, base_value in BASE_RESOURCE_POOL[day].items():
            new_value = effective_pool[day][resource]
            if new_value != base_value:
                # Tag the cause (scenario vs workforce) so the report can
                # surface them separately.
                cause = "scenario"
                for wc in workforce_pool_changes:
                    if wc["day"] == day and wc["resource"] == resource and wc["to"] == new_value:
                        cause = "workforce"
                        break
                pool_changes.append({
                    "day": day, "resource": resource,
                    "from": base_value, "to": new_value, "cause": cause,
                })
    if pool_changes:
        diff["resource_changes"] = pool_changes
    if workforce_pool_changes:
        diff["workforce_pool_reductions"] = workforce_pool_changes
    if overrides.get("demand_multipliers"):
        diff["demand_multipliers"] = overrides["demand_multipliers"]
    if overrides.get("corridor_closures"):
        diff["corridor_closures"] = overrides["corridor_closures"]
    if overrides.get("weather_overrides"):
        diff["weather_overrides"] = overrides["weather_overrides"]
    if overrides.get("transit_delay_hours"):
        diff["transit_delay_hours"] = overrides["transit_delay_hours"]

    if scenario_text:
        print(f"\n[Scenario] {overrides.get('summary')}")
        if pool_changes:
            print(f"[Scenario] Resource changes: {pool_changes}")
        if overrides.get("demand_multipliers"):
            print(f"[Scenario] Demand multipliers: {overrides['demand_multipliers']}")
        if overrides.get("weather_overrides") or overrides.get("corridor_closures"):
            print(f"[Scenario] Weather/closure overrides applied.")
    else:
        print("\n[Scenario] No scenario supplied — using base resource pool.")

    if workforce_pool_changes:
        print(f"[Workforce] Pool reduced by workforce reality: {workforce_pool_changes}")

    return {
        "scenario_overrides": overrides,
        "effective_resource_pool": effective_pool,
        "corridor_kpis": new_corridor_kpis,
        "weather_risk": new_weather,
        "scenario_diff": diff,
    }


# ---------------------------------------------------------------------------
# NODE 2 — Planner (uses scenario-adjusted resources + demand + weather)
# ---------------------------------------------------------------------------
def node_planner(state: AppState) -> AppState:
    attempts = state.get("audit_attempts", 0)
    feedback = state.get("audit_feedback", "")
    scenario = state.get("scenario", "")
    pool = state.get("effective_resource_pool") or BASE_RESOURCE_POOL

    plan_text, plan_structured = run_planner_agent(
        business_context=state.get("business_context", ""),
        ops_insights=state.get("ops_insights", ""),
        weather_risk=state.get("weather_risk", {}),
        corridor_kpis=state.get("corridor_kpis", []),
        resource_pool=pool,
        audit_feedback=feedback,
        scenario=scenario,
        workforce_state=state.get("workforce_state") or {},
        manager_feedback_recent=state.get("manager_feedback_recent") or [],
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
VAGUE_PHRASES = (
    "monitoring is essential",
    "vigilant",
    "close oversight",
    "as appropriate",
    "to be defined",
    "careful coordination",
    "to be determined",
    "where feasible",
    "as needed",
    "as required",
)


def _count_vague_phrases(prose: str) -> List[str]:
    """Return the vague phrases found in prose (case-insensitive). Used by audit."""
    lower = prose.lower()
    return [p for p in VAGUE_PHRASES if p in lower]


def node_audit(state: AppState) -> AppState:
    plan = state.get("plan_structured", {})
    plan_prose = state.get("dispatch_plan", "") or ""
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

    # ── Vague-phrase guard (deterministic, runs before any LLM call) ────────
    vague_hits = _count_vague_phrases(plan_prose)
    if len(vague_hits) >= 3:
        violations.append(
            f"plan prose contains {len(vague_hits)} vague phrases ({', '.join(vague_hits[:3])}...) — "
            f"replace each with concrete trigger / action / owner"
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
# Resource fields we enforce caps on. Anything in the allocator output that
# matches these keys gets clipped to the corresponding pool entry per day.
# The driver/drivers naming inconsistency from the LLM is handled below.
CAPPED_RESOURCES = ("truck_temp_controlled", "truck_standard", "driver")


def _clip_resource_allocation(
    allocation: Dict[str, Any],
    resource_pool: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """
    Enforce the per-day cap on EVERY scarce resource (cold-chain trucks,
    standard trucks, drivers) using the scenario-adjusted resource pool.

    The LLM allocator can over-allocate any of these; we greedily reduce the
    highest-count corridor for each (day, resource) pair until the per-day
    total respects the cap. Rationale is annotated so the constraint is
    visible in the executive report.
    """
    if not isinstance(allocation, dict) or allocation.get("raw"):
        return allocation

    clip_log: List[str] = []

    for day in ("Day0", "Day1"):
        day_alloc = allocation.get(day)
        if not isinstance(day_alloc, dict):
            continue
        # Normalise the LLM's occasional 'drivers' (plural) → 'driver'
        for c, v in list(day_alloc.items()):
            if isinstance(v, dict) and "drivers" in v and "driver" not in v:
                v["driver"] = v.pop("drivers")

        for resource in CAPPED_RESOURCES:
            max_avail = int(resource_pool.get(day, {}).get(resource, 0) or 0)
            clipped_here = 0
            while True:
                counts = {
                    c: int((v or {}).get(resource, 0) or 0)
                    for c, v in day_alloc.items()
                    if isinstance(v, dict)
                }
                total = sum(counts.values())
                if total <= max_avail:
                    break
                top_corridor = max(counts, key=counts.get)
                day_alloc[top_corridor][resource] = counts[top_corridor] - 1
                clipped_here += 1
            if clipped_here:
                pretty = resource.replace("truck_temp_controlled", "cold-chain truck")\
                                 .replace("truck_standard", "standard truck")\
                                 .replace("driver", "driver")
                clip_log.append(f"{day}: clipped {clipped_here} {pretty}(s) (cap {max_avail})")

    if clip_log:
        existing_rationale = allocation.get("rationale", "") or ""
        allocation["rationale"] = (
            f"[Auto-corrected — {'; '.join(clip_log)}.] "
            + existing_rationale
        )

    return allocation


# Backwards-compatible alias (other callers / older tests may still import this name)
_clip_cold_chain_allocation = _clip_resource_allocation


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
    # Per (corridor, day) breakdown — used by the report agent so it cannot
    # hallucinate "Tier 1 fully covered" when units actually deferred.
    breakdown: List[Dict[str, Any]] = []
    tier1_def_total = 0
    tier2_def_total = 0

    for kpi in corridor_kpis:
        corridor = kpi.get("corridor_id")
        day = kpi.get("day")
        day_alloc = allocation.get(day, {}) if isinstance(allocation.get(day), dict) else {}
        c_alloc = day_alloc.get(corridor, {}) if isinstance(day_alloc.get(corridor), dict) else {}

        cold_supply = int(c_alloc.get("truck_temp_controlled", 0) or 0) * UNITS_PER_TRUCK
        std_supply  = int(c_alloc.get("truck_standard", 0) or 0) * UNITS_PER_TRUCK

        tier1_cold_demand = int(kpi.get("tier1_units", 0))
        tier2_cold_demand = max(0, int(kpi.get("cold_chain_units", 0)) - tier1_cold_demand)
        std_demand = int(kpi.get("room_temp_units", 0)) + int(kpi.get("controlled_units", 0))

        t1_cold_disp = min(tier1_cold_demand, cold_supply)
        t1_cold_def  = tier1_cold_demand - t1_cold_disp
        cold_left    = cold_supply - t1_cold_disp

        t2_cold_disp = min(tier2_cold_demand, cold_left)
        t2_cold_def  = tier2_cold_demand - t2_cold_disp

        std_disp = min(std_demand, std_supply)
        std_def  = std_demand - std_disp

        corridor_penalty = (
            t1_cold_def * PENALTY_TIER1
            + (t2_cold_def + std_def) * PENALTY_TIER2
        )
        total_penalty  += corridor_penalty
        total_deferred += t1_cold_def + t2_cold_def + std_def
        tier1_def_total += t1_cold_def
        tier2_def_total += t2_cold_def + std_def

        breakdown.append({
            "corridor_id": corridor,
            "day": day,
            "tier1_cold_deferred": t1_cold_def,
            "tier2_cold_deferred": t2_cold_def,
            "tier2_standard_deferred": std_def,
            "deferred_total": t1_cold_def + t2_cold_def + std_def,
            "penalty_pts": corridor_penalty,
            "tier1_cold_demand": tier1_cold_demand,
            "tier1_cold_dispatched": t1_cold_disp,
            "tier2_cold_demand": tier2_cold_demand,
            "tier2_cold_dispatched": t2_cold_disp,
            "standard_demand": std_demand,
            "standard_dispatched": std_disp,
        })

    llm_penalty  = int(allocation.get("total_penalty_score", 0) or 0)
    llm_deferred = int(allocation.get("deferred_units", 0) or 0)

    allocation["total_penalty_score"] = total_penalty
    allocation["deferred_units"]      = total_deferred
    allocation["deferral_breakdown"]  = breakdown
    allocation["deferral_summary"]    = {
        "tier1_units_deferred": tier1_def_total,
        "tier2_units_deferred": tier2_def_total,
        "tier1_protected": tier1_def_total == 0,
        "headline": (
            f"{tier1_def_total} Tier-1 + {tier2_def_total} Tier-2 deferred "
            f"= {total_deferred} units, {total_penalty} pts."
        ),
    }

    if total_penalty != llm_penalty or total_deferred != llm_deferred:
        existing = allocation.get("rationale", "") or ""
        allocation["rationale"] = (
            f"[Penalty recomputed deterministically: {tier1_def_total} Tier-1 + "
            f"{tier2_def_total} Tier-2 = {total_deferred} units deferred → "
            f"{total_penalty} pts. LLM had reported "
            f"{llm_deferred} deferred / {llm_penalty} pts.] "
            + existing
        )

    return allocation


def node_allocator(state: AppState) -> AppState:
    pool = state.get("effective_resource_pool") or BASE_RESOURCE_POOL
    workforce_dict = state.get("workforce_state") or {}

    allocation = run_allocator_agent(
        corridor_kpis=state.get("corridor_kpis", []),
        resource_pool=pool,
        weather_risk=state.get("weather_risk", {}),
        business_context=state.get("business_context", ""),
        workforce_state=workforce_dict,
    )
    allocation = _clip_resource_allocation(allocation, pool)
    allocation = _recompute_penalty(allocation, state.get("corridor_kpis", []))

    # Deterministic realism check — produces workforce_warnings + violations
    # on the final allocation, after clipping. Anything in violations means
    # the upstream pool reduction failed to catch a workforce constraint.
    if isinstance(allocation, dict) and not allocation.get("raw"):
        from tools.workforce_tools import DriverProfile
        workforce_obj = WorkforceState(drivers=[
            DriverProfile(
                driver_id=d["driver_id"],
                name=d.get("name", ""),
                certifications=d.get("certifications", []),
                hours_last_24h=0.0,
                hours_last_7d=0.0,
                consecutive_days=0,
                fatigue_flag=bool(d.get("fatigue_flag", False)),
                preferred_corridors=d.get("preferred_corridors", []),
                active=bool(d.get("eligible_today", False)),
            )
            for d in workforce_dict.get("drivers", [])
        ])
        warnings, violations = realism_check_allocation(allocation, workforce_obj, pool)
        allocation["workforce_warnings"] = warnings
        allocation["workforce_violations"] = violations
        if warnings:
            print(f"[Realism] Warnings: {warnings}")
        if violations:
            print(f"[Realism] VIOLATIONS: {violations}")

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
        return {"human_approval_required": True, "human_approved": approved}

    # No corridor reached the critical threshold — checkpoint did NOT fire.
    return {"human_approval_required": False, "human_approved": True}


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
        scenario_diff=state.get("scenario_diff", {}),
        effective_resource_pool=state.get("effective_resource_pool") or BASE_RESOURCE_POOL,
        workforce_state=state.get("workforce_state") or {},
        manager_feedback_recent=state.get("manager_feedback_recent") or [],
        calibration_history=state.get("calibration_history") or {},
        human_approval_required=state.get("human_approval_required", False),
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
    g.add_node("pdf_context",            node_pdf_context)
    g.add_node("csv_analysis",           node_csv_analysis)
    g.add_node("weather",                node_weather)
    g.add_node("load_workforce_state",   node_load_workforce_state)
    g.add_node("scenario_apply",         node_scenario_apply)
    g.add_node("planner",                node_planner)
    g.add_node("audit",                  node_audit)
    g.add_node("allocator",              node_allocator)
    g.add_node("human_checkpoint",       node_human_checkpoint)
    g.add_node("report",                 node_report)
    g.add_node("email",                  node_email)

    # Parallel fan-out from START → four independent data-gathering nodes
    g.add_edge(START, "pdf_context")
    g.add_edge(START, "csv_analysis")
    g.add_edge(START, "weather")
    g.add_edge(START, "load_workforce_state")

    # All four converge into scenario_apply (the agentic what-if + realism engine)
    g.add_edge("pdf_context",          "scenario_apply")
    g.add_edge("csv_analysis",         "scenario_apply")
    g.add_edge("weather",              "scenario_apply")
    g.add_edge("load_workforce_state", "scenario_apply")

    # scenario_apply → planner so the planner sees scenario-adjusted state
    g.add_edge("scenario_apply", "planner")

    # Audit loop — conditional edge routes back to planner on FAIL
    g.add_edge("planner", "audit")
    g.add_conditional_edges("audit", _route_after_audit, {"planner": "planner", "allocator": "allocator"})

    # Linear after audit passes
    g.add_edge("allocator",        "human_checkpoint")
    g.add_edge("human_checkpoint", "report")
    g.add_edge("report",           "email")
    g.add_edge("email",            END)

    return g.compile(checkpointer=checkpointer)
