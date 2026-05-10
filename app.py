"""
SeeWeeS Dispatch Intelligence — Streamlit cockpit.

Designed for an operations director, not a developer. The flow is:
  1. Sidebar: pick the data file, optionally pick or type a scenario.
  2. Click Run.
  3. Hero panel updates with the four numbers leadership cares about:
     route risk, total penalty, deferred units, approval status.
  4. Corridor cards show NJ→Boston and NJ→Philadelphia side by side, with a
     real geographic map of the I-95 corridor.
  5. Tabs below for the live execution log, the executive HTML report, the
     deeper analytics, and the scenario-impact diff.
"""
from __future__ import annotations

import os
import sys
import uuid
import json
import time

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from graph import build_graph, CORRIDOR_WAYPOINTS, BASE_RESOURCE_POOL
from tools.workforce_tools import (
    append_manager_rating, append_outcome,
    load_manager_ratings, load_outcome_log, compute_calibration,
)


# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SeeWeeS Dispatch Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
        h1, h2, h3 { color: #0e2a47; font-weight: 600; }
        .hero-card {
            background: white;
            border: 1px solid #e3e8ef;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }
        .hero-label {
            font-size: 11px;
            font-weight: 600;
            color: #64748b;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .hero-value { font-size: 30px; font-weight: 700; color: #0e2a47; line-height: 1.1; }
        .hero-sub { font-size: 12px; color: #64748b; margin-top: 4px; }
        .pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .pill-green  { background: #dcfce7; color: #166534; }
        .pill-amber  { background: #fef3c7; color: #92400e; }
        .pill-red    { background: #fee2e2; color: #991b1b; }
        .pill-neutral{ background: #e2e8f0; color: #334155; }
        .corridor-card {
            background: white; border: 1px solid #e3e8ef; border-radius: 12px;
            padding: 16px 20px; height: 100%;
        }
        .scenario-chip > button {
            background: #f1f5f9 !important; color: #0e2a47 !important;
            border: 1px solid #cbd5e1 !important; border-radius: 999px !important;
            padding: 6px 14px !important; font-size: 13px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "checkpointer": MemorySaver(),
        "awaiting_approval": False,
        "approval_payload": None,
        "run_complete": False,
        "final_state": None,
        "execution_log": [],
        "audit_attempts": 0,
        "scenario_input": "",
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

if "app" not in st.session_state:
    st.session_state.app = build_graph(checkpointer=st.session_state.checkpointer)
app = st.session_state.app


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
RISK_COLOUR = {
    0: ("#16a34a", "pill-green",   "Low"),
    1: ("#eab308", "pill-amber",   "Moderate"),
    2: ("#f97316", "pill-amber",   "High"),
    3: ("#dc2626", "pill-red",     "Critical"),
}

CORRIDOR_LABEL = {
    "C1_I95_NJ_BOS": "NJ → Boston (I-95)",
    "C2_NJ_PHL":     "NJ → Philadelphia",
}

NODE_LABELS = {
    "pdf_context":      ("📖", "Reading playbook rules"),
    "csv_analysis":     ("📦", "Analysing shipment data"),
    "weather":          ("🌦️", "Fetching corridor weather"),
    "scenario_apply":   ("🎯", "Applying what-if overrides"),
    "planner":          ("🧠", "Drafting dispatch plan"),
    "audit":            ("🔍", "Auditing plan compliance"),
    "allocator":        ("🚛", "Allocating trucks & drivers"),
    "human_checkpoint": ("👤", "Manager approval checkpoint"),
    "report":           ("📝", "Generating executive report"),
    "email":            ("✉️",  "Sending email"),
}


def _hero_card(label: str, value: str, sub: str = "", pill: str = ""):
    pill_html = f"<span class='pill {pill}'>{value}</span>" if pill else f"<div class='hero-value'>{value}</div>"
    st.markdown(
        f"""
        <div class='hero-card'>
            <div class='hero-label'>{label}</div>
            {pill_html}
            <div class='hero-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(state: dict):
    weather = state.get("weather_risk", {}) or {}
    alloc = state.get("allocation_plan", {}) or {}
    diff = state.get("scenario_diff", {}) or {}

    max_score = max((v.get("route_risk_score_0_3", 0) for v in weather.values()), default=0)
    risk_color, risk_pill, risk_label = RISK_COLOUR[max_score]

    penalty = alloc.get("total_penalty_score", "—") if isinstance(alloc, dict) else "—"
    deferred = alloc.get("deferred_units", "—") if isinstance(alloc, dict) else "—"

    if max_score >= 3:
        approval_text = "Approved" if state.get("human_approved") else "Awaiting"
        approval_pill = "pill-green" if state.get("human_approved") else "pill-amber"
    else:
        approval_text = "Not required"
        approval_pill = "pill-neutral"

    scenario_active = bool(diff.get("resource_changes") or diff.get("demand_multipliers")
                          or diff.get("corridor_closures") or diff.get("weather_overrides"))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _hero_card("Worst Corridor Risk", risk_label, f"Score {max_score} of 3", risk_pill)
    with c2:
        _hero_card("Penalty Score", str(penalty), "Lower is better")
    with c3:
        _hero_card("Deferred Units", str(deferred), "Shipments not dispatched")
    with c4:
        _hero_card("Manager Approval", approval_text, "", approval_pill)
    with c5:
        _hero_card(
            "Scenario",
            "Active" if scenario_active else "Baseline",
            diff.get("summary", "")[:60] + ("…" if len(diff.get("summary", "")) > 60 else ""),
            "pill-amber" if scenario_active else "pill-neutral",
        )


def _render_corridor_card(corridor_id: str, weather_entry: dict, kpi_entries: list, alloc: dict):
    label = CORRIDOR_LABEL.get(corridor_id, corridor_id)
    score = weather_entry.get("route_risk_score_0_3", 0) if weather_entry else 0
    color, pill, risk_label = RISK_COLOUR[score]
    buffer_pct = weather_entry.get("required_buffer_pct", 0) if weather_entry else 0
    closed = weather_entry.get("closed", False) if weather_entry else False
    forced = weather_entry.get("scenario_forced", False) if weather_entry else False

    badges = [f"<span class='pill {pill}'>{risk_label} (score {score})</span>"]
    if closed:
        badges.append("<span class='pill pill-red'>CLOSED</span>")
    elif forced:
        badges.append("<span class='pill pill-amber'>Scenario-forced</span>")
    if score == 3:
        badges.append("<span class='pill pill-red'>+40% buffer · escalate</span>")
    elif score > 0:
        badges.append(f"<span class='pill pill-amber'>+{buffer_pct}% buffer</span>")

    worst = weather_entry.get("worst_waypoint") if weather_entry else None
    worst_html = ""
    if worst:
        worst_html = (
            f"<div style='font-size:13px; color:#475569; margin-top:8px;'>"
            f"Worst waypoint: <b>{worst.get('city')}, {worst.get('state')}</b> · "
            f"🌧 {worst.get('max_precip_mm_day', 0):.1f} mm/day · "
            f"💨 {worst.get('max_wind_gust_kmh', 0):.0f} km/h</div>"
        )

    # Demand vs supply summary
    demand_html = ""
    if kpi_entries:
        total_cold = sum(k.get("cold_chain_units", 0) for k in kpi_entries)
        total_std  = sum(k.get("room_temp_units", 0) + k.get("controlled_units", 0) for k in kpi_entries)
        total_t1   = sum(k.get("tier1_units", 0) for k in kpi_entries)
        # Allocator's allocation for THIS corridor across both days
        c_alloc_total = 0
        s_alloc_total = 0
        if isinstance(alloc, dict):
            for day in ("Day0", "Day1"):
                day_alloc = alloc.get(day, {}).get(corridor_id, {}) if isinstance(alloc.get(day), dict) else {}
                c_alloc_total += int(day_alloc.get("truck_temp_controlled", 0) or 0)
                s_alloc_total += int(day_alloc.get("truck_standard", 0) or 0)
        cold_capacity = c_alloc_total * 10
        std_capacity = s_alloc_total * 10
        demand_html = (
            f"<div style='font-size:13px; color:#475569; margin-top:10px; line-height:1.7;'>"
            f"<b>Demand (48 h):</b> {total_t1} Tier-1 · {total_cold} cold-chain · {total_std} standard<br>"
            f"<b>Allocated:</b> {c_alloc_total} cold trucks ({cold_capacity} u capacity) · "
            f"{s_alloc_total} std trucks ({std_capacity} u capacity)"
            f"</div>"
        )

    st.markdown(
        f"""
        <div class='corridor-card'>
            <div style='display:flex; align-items:center; justify-content:space-between;'>
                <h3 style='margin:0;'>{label}</h3>
            </div>
            <div style='margin-top:8px;'>{' '.join(badges)}</div>
            {worst_html}
            {demand_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_corridor_map(weather_risk: dict):
    """Plot waypoints on a real US map, colour-coded by risk."""
    rows = []
    for cid, w in weather_risk.items():
        for wp in w.get("per_waypoint", []):
            rows.append({
                "Corridor": CORRIDOR_LABEL.get(cid, cid),
                "Waypoint": f"{wp.get('waypoint', '')} {wp.get('city', '')}",
                "Lat": wp.get("lat") or _lookup_lat(cid, wp.get("waypoint")),
                "Lon": wp.get("lon") or _lookup_lon(cid, wp.get("waypoint")),
                "Risk": wp.get("risk_score_0_3", 0),
                "City": f"{wp.get('city')}, {wp.get('state')}",
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.scatter_geo(
        df,
        lat="Lat", lon="Lon",
        color="Risk",
        size=[12] * len(df),
        hover_name="Waypoint",
        hover_data={"City": True, "Risk": True, "Corridor": True, "Lat": False, "Lon": False},
        color_continuous_scale=[(0, "#16a34a"), (0.34, "#eab308"), (0.67, "#f97316"), (1, "#dc2626")],
        range_color=[0, 3],
        scope="usa",
    )
    fig.update_geos(
        center=dict(lat=40.5, lon=-73.5),
        projection_scale=8,
        showsubunits=True, subunitcolor="#cbd5e1",
        showland=True, landcolor="#f8fafc",
    )
    fig.update_layout(
        height=320, margin=dict(t=20, b=0, l=0, r=0),
        coloraxis_colorbar=dict(title="Risk", tickvals=[0, 1, 2, 3], thickness=12, len=0.6),
    )
    st.plotly_chart(fig, width="stretch")


def _lookup_lat(corridor_id: str, waypoint_id: str):
    for wp in CORRIDOR_WAYPOINTS.get(corridor_id, []):
        if wp["id"] == waypoint_id:
            return wp["lat"]
    return None


def _lookup_lon(corridor_id: str, waypoint_id: str):
    for wp in CORRIDOR_WAYPOINTS.get(corridor_id, []):
        if wp["id"] == waypoint_id:
            return wp["lon"]
    return None


def _render_pipeline_log(entries: list):
    if not entries:
        st.info("No pipeline run yet — click ▶ Run Analysis in the sidebar.")
        return
    for e in entries:
        kind = e.get("kind", "info")
        text = e.get("text", "")
        if kind == "audit_fail":
            st.warning(text)
        elif kind == "warning":
            st.warning(text)
        elif kind == "error":
            st.error(text)
        else:
            st.success(text)


def _render_corridor_kpis_table(corridor_kpis: list, multipliers: dict | None = None):
    if not corridor_kpis:
        st.info("No corridor KPIs available.")
        return
    rows = []
    for k in corridor_kpis:
        cid = k["corridor_id"]
        m = (multipliers or {}).get(cid)
        label = CORRIDOR_LABEL.get(cid, cid)
        if m and m != 1.0:
            label = f"{label} (× {m:.2f})"
        rows.append({
            "Corridor": label,
            "Day": k["day"],
            "Valid": k["valid_rows"],
            "Excluded": k["excluded_rows"],
            "Excl %": f"{k['exclusion_rate_pct']}%",
            "Tier 1": k["tier1_units"],
            "Tier 2": k["tier2_units"],
            "Cold": k["cold_chain_units"],
            "Cold trucks": k["trucks_needed_cold_chain"],
            "Std trucks": k["trucks_needed_standard"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_allocation_table(allocation_plan: dict):
    if not allocation_plan or allocation_plan.get("raw"):
        st.write(allocation_plan.get("narrative", "(no allocation produced)"))
        return
    rows = []
    for day in ("Day0", "Day1"):
        for corridor, alloc in allocation_plan.get(day, {}).items():
            rows.append({
                "Day": day,
                "Corridor": CORRIDOR_LABEL.get(corridor, corridor),
                "Cold-chain trucks": alloc.get("truck_temp_controlled", "—"),
                "Standard trucks": alloc.get("truck_standard", "—"),
                "Drivers": alloc.get("drivers", "—"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    rationale = allocation_plan.get("rationale", "")
    if rationale:
        st.caption(rationale)


def _render_resource_pool_diff(state: dict):
    """Show base vs effective resource pool when scenario applied a change."""
    diff = state.get("scenario_diff", {}) or {}
    changes = diff.get("resource_changes", [])
    if not changes:
        return
    st.markdown("**Resource pool — what the scenario changed**")
    rows = []
    for c in changes:
        rows.append({
            "Day": c["day"],
            "Resource": c["resource"].replace("truck_temp_controlled", "Cold-chain trucks")
                                      .replace("truck_standard", "Standard trucks")
                                      .replace("driver", "Drivers"),
            "Base": c["from"],
            "Effective": c["to"],
            "Δ": c["to"] - c["from"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_trend_chart(trend_summary: dict):
    if not trend_summary or "daily_volume_by_corridor" not in trend_summary:
        st.info("No trend data available.")
        return
    vol = trend_summary["daily_volume_by_corridor"]
    avg = vol.get("avg_daily_units", {})
    corridors = list(avg.keys())
    labels = [CORRIDOR_LABEL.get(c, c) for c in corridors]
    values = [avg[c] for c in corridors]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#3b82f6", "#f97316"],
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Average daily units · last {trend_summary.get('history_days', 0)} days",
        yaxis_title="Units / day",
        height=320, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")


def _render_waypoint_chart(weather_risk: dict):
    rows = []
    for cid, w in weather_risk.items():
        for wp in w.get("per_waypoint", []):
            rows.append({
                "Corridor": CORRIDOR_LABEL.get(cid, cid),
                "Waypoint": f"{wp['waypoint']} {wp['city']}",
                "Risk": wp["risk_score_0_3"],
                "Precipitation (mm)": wp["max_precip_mm_day"],
                "Wind (km/h)": wp["max_wind_gust_kmh"],
            })
    if not rows:
        return
    fig = px.bar(
        pd.DataFrame(rows),
        x="Waypoint", y="Risk", color="Corridor", barmode="group",
        color_discrete_map={
            "NJ → Boston (I-95)": "#3b82f6",
            "NJ → Philadelphia": "#f97316",
        },
        range_y=[0, 3.5],
    )
    fig.update_layout(height=320, margin=dict(t=10, b=20),
                      title="Risk score per waypoint")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
SCENARIO_PRESETS = [
    ("Cold-chain truck breakdown",
     "One temperature-controlled truck broke down. Only 1 cold-chain truck is "
     "available per day instead of 2. Identify which cold-chain medicines must "
     "be deferred and the penalty impact."),
    ("Driver shortage (3 of 6)",
     "Driver shortage today: only 3 drivers available instead of the normal 6. "
     "Adjust allocation accordingly and flag any SLA risk this creates."),
    ("20% demand spike on C2",
     "20% demand spike on the NJ→Philadelphia corridor (C2). All existing "
     "orders must still be fulfilled; the spike adds units on top."),
    ("Severe storm on C1",
     "Severe winter storm forecast across the I-95 corridor — force the C1 "
     "(NJ→Boston) route risk score to 3."),
    ("I-95 partial closure",
     "I-95 is partially closed due to an accident north of New Haven. "
     "Estimated transit time for C1 (NJ→Boston) is extended by 4 hours. "
     "Assess Tier-1 SLA compliance and adjust buffers."),
    ("Combined: spike + driver shortage",
     "Dual disruption: 30% demand spike across both corridors AND a driver "
     "shortage with only 2 drivers available per day."),
]

with st.sidebar:
    st.title("🚚 SeeWeeS")
    st.caption("Multi-Agent Dispatch Intelligence")
    st.divider()

    st.markdown("**Operations data**")
    csv_path = st.selectbox(
        "Shipment file",
        [
            "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
            "data-for-enhancement/synthetic/synthetic_baseline.csv",
            "data-for-enhancement/synthetic/synthetic_volume_spike.csv",
            "data-for-enhancement/synthetic/synthetic_dq_heavy.csv",
            "data-for-enhancement/synthetic/synthetic_tier1_surge.csv",
            "data-for-enhancement/synthetic/synthetic_growth_trend.csv",
            "data-for-enhancement/synthetic/synthetic_rich_60d.csv",
            "data/Incoming_shipment_02_08.csv",
        ],
        help="Real ops feed or one of six synthetic profiles",
    )
    pdf_path = "data/SeeWeeS Specialty distribution.pdf"

    st.divider()
    st.markdown("**What-if scenario** *(optional)*")
    st.caption("Pick a preset or describe your own disruption in plain English.")

    preset_cols = st.columns(2)
    for i, (preset_label, preset_text) in enumerate(SCENARIO_PRESETS):
        col = preset_cols[i % 2]
        with col:
            if st.button(preset_label, key=f"preset_{i}", width="stretch"):
                st.session_state.scenario_input = preset_text

    scenario = st.text_area(
        "Or write your own:",
        value=st.session_state.scenario_input,
        height=100,
        placeholder="e.g. Cold-chain truck broke down. Only 1 cold-chain truck per day.",
        key="scenario_textarea",
    )
    st.session_state.scenario_input = scenario

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", width="stretch",
                        disabled=st.session_state.running)

    if st.session_state.run_complete:
        if st.button("↺  New Run", width="stretch"):
            for k in ["thread_id", "awaiting_approval", "approval_payload",
                      "run_complete", "final_state", "execution_log",
                      "audit_attempts", "running"]:
                st.session_state.pop(k, None)
            st.session_state.checkpointer = MemorySaver()
            st.session_state.app = build_graph(checkpointer=st.session_state.checkpointer)
            _init_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Title + hero strip
# ---------------------------------------------------------------------------
st.title("Dispatch Intelligence Cockpit")
st.caption("48-hour outlook · NJ Distribution Centre → I-95 corridor hospitals")

if st.session_state.run_complete and st.session_state.final_state:
    _render_hero(st.session_state.final_state)
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _hero_card("Worst Corridor Risk", "—", "Run pending")
    with c2: _hero_card("Penalty Score", "—", "")
    with c3: _hero_card("Deferred Units", "—", "")
    with c4: _hero_card("Manager Approval", "—", "")
    with c5: _hero_card("Scenario", "—", "")

st.write("")  # spacer


# ---------------------------------------------------------------------------
# Corridor cards + map (always shown, populated when run completes)
# ---------------------------------------------------------------------------
if st.session_state.run_complete and st.session_state.final_state:
    fs = st.session_state.final_state
    weather = fs.get("weather_risk", {}) or {}
    kpis = fs.get("corridor_kpis", []) or []
    alloc = fs.get("allocation_plan", {}) or {}

    col_left, col_right = st.columns(2)
    with col_left:
        _render_corridor_card(
            "C1_I95_NJ_BOS",
            weather.get("C1_I95_NJ_BOS", {}),
            [k for k in kpis if k.get("corridor_id") == "C1_I95_NJ_BOS"],
            alloc,
        )
    with col_right:
        _render_corridor_card(
            "C2_NJ_PHL",
            weather.get("C2_NJ_PHL", {}),
            [k for k in kpis if k.get("corridor_id") == "C2_NJ_PHL"],
            alloc,
        )

    st.write("")
    with st.container():
        st.markdown("**Corridor map · waypoint risk scores**")
        _render_corridor_map(weather)

    # Workforce status strip
    workforce = fs.get("workforce_state") or {}
    if workforce.get("drivers"):
        st.write("")
        st.markdown("**Workforce status · today**")
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        with wcol1:
            _hero_card("Eligible drivers",
                       f"{workforce.get('eligible_count', 0)}/{workforce.get('total_roster', 0)}",
                       "active and within hour limits")
        with wcol2:
            _hero_card("Cold-chain certified",
                       str(workforce.get("cold_chain_eligible_count", 0)),
                       "eligible AND certified")
        with wcol3:
            flagged = workforce.get("fatigue_flagged_count", 0)
            pill = "pill-amber" if flagged else "pill-green"
            _hero_card("Fatigue flags",
                       str(flagged),
                       "eligible w/ warning",
                       pill if flagged else "")
        with wcol4:
            wf_warns = (fs.get("allocation_plan") or {}).get("workforce_warnings", []) or []
            _hero_card("Workforce notes",
                       str(len(wf_warns)),
                       "items raised this run",
                       "pill-amber" if wf_warns else "")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_pipeline, tab_scenario, tab_report, tab_analytics, tab_feedback, tab_outcomes, tab_calibration = st.tabs([
    "🔄  Live pipeline",
    "🎯  Scenario impact",
    "📄  Executive report",
    "📊  Deep dive",
    "⭐  Feedback",
    "📈  Outcomes",
    "🎯  Calibration",
])


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
def _log(kind: str, text: str):
    st.session_state.execution_log.append({"kind": kind, "text": text})


def _stream_graph(stream_input, cfg, status_box):
    for event in app.stream(stream_input, cfg, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                interrupt_data = node_output[0].value if node_output else {}
                st.session_state.awaiting_approval = True
                st.session_state.approval_payload = interrupt_data
                _log("warning", "👤 Manager approval checkpoint reached — risk score 3 detected.")
                with status_box: _render_pipeline_log(st.session_state.execution_log)
                return True

            emoji, label = NODE_LABELS.get(node_name, ("⚙️", node_name))

            if node_name == "audit":
                verdict = node_output.get("audit_verdict", "")
                if verdict == "FAIL":
                    attempts = st.session_state.audit_attempts + 1
                    st.session_state.audit_attempts = attempts
                    feedback = node_output.get("audit_feedback", "")
                    snippet = feedback[:140].replace("\n", " ")
                    _log("audit_fail",
                         f"🔍 Audit FAIL (attempt {attempts}/3) — looping back to planner. "
                         f"Reason: {snippet}…")
                else:
                    _log("info", f"{emoji} Audit PASS — plan compliant.")
            elif node_name == "scenario_apply":
                diff = node_output.get("scenario_diff", {})
                summary = diff.get("summary", "")
                if diff.get("resource_changes") or diff.get("demand_multipliers") or diff.get("corridor_closures") or diff.get("weather_overrides"):
                    _log("info", f"{emoji} Scenario applied: {summary}")
                else:
                    _log("info", f"{emoji} No scenario — using base resource pool.")
            elif node_name == "allocator":
                a = node_output.get("allocation_plan", {})
                penalty = a.get("total_penalty_score", "?") if isinstance(a, dict) else "?"
                deferred = a.get("deferred_units", "?") if isinstance(a, dict) else "?"
                _log("info", f"{emoji} {label} — penalty {penalty}, {deferred} deferred.")
            elif node_name not in ("email",):
                _log("info", f"{emoji} {label} — done.")

            with status_box: _render_pipeline_log(st.session_state.execution_log)
    return False


def _finalise(cfg):
    snapshot = app.get_state(cfg)
    st.session_state.final_state = snapshot.values
    st.session_state.run_complete = True
    st.session_state.running = False
    html = snapshot.values.get("report_html", "")
    if html:
        try:
            with open("report.html", "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1 — Start a brand-new run
# ---------------------------------------------------------------------------
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if run_btn:
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.awaiting_approval = False
    st.session_state.approval_payload = None
    st.session_state.run_complete = False
    st.session_state.final_state = None
    st.session_state.execution_log = []
    st.session_state.audit_attempts = 0
    st.session_state.running = True
    st.session_state.pop("_pending_decision", None)
    st.session_state.pop("feedback_submitted", None)

    initial_state = {
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "scenario": scenario.strip() if scenario.strip() else None,
    }

    with tab_pipeline:
        st.subheader("Live execution")
        status_box = st.container()
        try:
            interrupted = _stream_graph(initial_state, config, status_box)
            if not interrupted:
                _finalise(config)
                st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.running = False
            raise


# ---------------------------------------------------------------------------
# Phase 2 — Awaiting manager approval
# ---------------------------------------------------------------------------
if st.session_state.awaiting_approval and not st.session_state.run_complete:
    interrupt_data = st.session_state.approval_payload or {}
    with tab_pipeline:
        st.subheader("Live execution")
        st.error("⚠️ **RISK SCORE 3 DETECTED — Manager Approval Required**")
        st.write("**Critical weather risk on one or more corridors. Review and decide.**")

        weather = interrupt_data.get("weather_risk", {})
        cols = st.columns(len(weather)) if weather else [st]
        for col, (cid, w) in zip(cols, weather.items()):
            with col:
                _render_corridor_card(cid, w, [], interrupt_data.get("allocation_plan", {}))

        alloc = interrupt_data.get("allocation_plan", {})
        if alloc and not alloc.get("raw"):
            st.write(f"**Proposed allocation — Penalty: {alloc.get('total_penalty_score', '?')}**")
            _render_allocation_table(alloc)

        col_yes, col_no = st.columns(2)
        if col_yes.button("✅  Approve & generate report", type="primary",
                          key="hitl_approve", width="stretch"):
            st.session_state._pending_decision = "YES"
            st.session_state.awaiting_approval = False
            st.rerun()
        if col_no.button("❌  Reject plan", key="hitl_reject", width="stretch"):
            st.session_state._pending_decision = "NO"
            st.session_state.awaiting_approval = False
            st.rerun()


# ---------------------------------------------------------------------------
# Phase 3 — Resume after approval decision
# ---------------------------------------------------------------------------
if "_pending_decision" in st.session_state and not st.session_state.run_complete:
    decision = st.session_state.pop("_pending_decision")
    with tab_pipeline:
        st.subheader("Live execution")
        st.info(f"Decision recorded: **{'APPROVED' if decision == 'YES' else 'REJECTED'}**")
        status_box = st.container()
        try:
            _stream_graph(Command(resume=decision), config, status_box)
            _finalise(config)
            st.rerun()
        except Exception as e:
            st.error(f"Resume error: {e}")
            st.session_state.running = False
            raise


# ---------------------------------------------------------------------------
# Tab: Live pipeline (when not actively running)
# ---------------------------------------------------------------------------
with tab_pipeline:
    if not run_btn and not st.session_state.awaiting_approval:
        st.subheader("Live execution")
        _render_pipeline_log(st.session_state.execution_log)


# ---------------------------------------------------------------------------
# Tab: Scenario impact
# ---------------------------------------------------------------------------
with tab_scenario:
    if not st.session_state.run_complete or not st.session_state.final_state:
        st.info("Run the pipeline first using the sidebar.")
    else:
        fs = st.session_state.final_state
        diff = fs.get("scenario_diff", {}) or {}
        scenario_text = fs.get("scenario") or "(no scenario supplied)"

        if not (diff.get("resource_changes") or diff.get("demand_multipliers")
                or diff.get("corridor_closures") or diff.get("weather_overrides")):
            st.info("**Baseline run** — no scenario was applied. Pick a preset in the sidebar to see the impact.")
            if scenario_text != "(no scenario supplied)":
                st.caption(f"Scenario submitted: \"{scenario_text}\" — but the parser determined it didn't require any structured override.")
        else:
            st.markdown(f"**Scenario:** *{scenario_text}*")
            st.markdown(f"_{diff.get('summary', '')}_")
            st.divider()

            _render_resource_pool_diff(fs)

            if diff.get("demand_multipliers"):
                st.markdown("**Demand multipliers**")
                rows = [{"Corridor": CORRIDOR_LABEL.get(k, k), "Multiplier": f"× {v:.2f}"}
                        for k, v in diff["demand_multipliers"].items()]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            if diff.get("weather_overrides"):
                st.markdown("**Weather overrides**")
                for cid, override in diff["weather_overrides"].items():
                    st.write(f"- **{CORRIDOR_LABEL.get(cid, cid)}**: forced risk score = {override.get('route_risk_score_0_3')}")

            if diff.get("corridor_closures"):
                st.markdown("**Closures**")
                for cid in diff["corridor_closures"]:
                    st.error(f"❌  {CORRIDOR_LABEL.get(cid, cid)} is CLOSED")

            if diff.get("transit_delay_hours"):
                st.markdown("**Transit delays**")
                for cid, hrs in diff["transit_delay_hours"].items():
                    st.write(f"- **{CORRIDOR_LABEL.get(cid, cid)}**: +{hrs} hours")

            st.divider()
            st.markdown("**Resulting impact**")
            alloc = fs.get("allocation_plan", {}) or {}
            penalty = alloc.get("total_penalty_score", 0) if isinstance(alloc, dict) else 0
            deferred = alloc.get("deferred_units", 0) if isinstance(alloc, dict) else 0
            c1, c2 = st.columns(2)
            with c1:
                _hero_card("Penalty score", str(penalty), "after scenario applied")
            with c2:
                _hero_card("Deferred units", str(deferred), "shipments not dispatched")


# ---------------------------------------------------------------------------
# Tab: Executive report
# ---------------------------------------------------------------------------
with tab_report:
    if st.session_state.run_complete and st.session_state.final_state:
        html = st.session_state.final_state.get("report_html", "")
        if html:
            import streamlit.components.v1 as components
            components.html(html, height=900, scrolling=True)
        else:
            st.warning("Report not yet generated.")
    else:
        st.info("Run the pipeline first using the sidebar.")


# ---------------------------------------------------------------------------
# Tab: Deep dive
# ---------------------------------------------------------------------------
with tab_analytics:
    if not st.session_state.run_complete or not st.session_state.final_state:
        st.info("Run the pipeline first using the sidebar.")
    else:
        fs = st.session_state.final_state
        diff = fs.get("scenario_diff", {}) or {}

        st.subheader("Per-waypoint weather risk")
        _render_waypoint_chart(fs.get("weather_risk", {}))

        st.divider()
        st.subheader("Shipment KPIs by corridor and day")
        _render_corridor_kpis_table(fs.get("corridor_kpis", []),
                                    diff.get("demand_multipliers"))

        st.divider()
        st.subheader("Historical trend")
        _render_trend_chart(fs.get("trend_summary", {}))

        st.divider()
        st.subheader("Resource allocation")
        _render_allocation_table(fs.get("allocation_plan", {}))

        st.divider()
        st.subheader("Audit & data quality")
        af = fs.get("audit_feedback", "")
        if af and af not in ("None — plan passed audit.", ""):
            st.warning(af)
        else:
            st.success("Plan passed all audit checks.")
        dq_md = fs.get("anomalies_md", "")
        if dq_md and dq_md != "(none)":
            st.markdown(dq_md)


# ---------------------------------------------------------------------------
# Tab: Feedback — manager rating card + history
# ---------------------------------------------------------------------------
with tab_feedback:
    st.subheader("Manager feedback on this run")

    if not st.session_state.run_complete or not st.session_state.final_state:
        st.info("Run the pipeline first, then come back to rate the plan.")
    elif st.session_state.get("feedback_submitted"):
        st.success("✅ Thanks — your rating has been recorded. It will inform future plans.")
    else:
        with st.form("manager_rating_form"):
            colA, colB = st.columns([1, 2])
            with colA:
                stars = st.select_slider(
                    "Rating",
                    options=[1, 2, 3, 4, 5],
                    value=4,
                    help="1 = poor, 5 = exactly what I would have done",
                )
                manager_id = st.text_input("Your ID", value="M-Director")
            with colB:
                tags = st.multiselect(
                    "Tags",
                    ["Right-sized", "Aggressive", "Conservative", "Risky",
                     "Wasteful", "Wrong corridor priority", "Driver concern",
                     "Cold-chain concern"],
                    default=["Right-sized"],
                )
                comment = st.text_area(
                    "Comment (optional)",
                    placeholder="What would you change? Anything to flag?",
                    height=100,
                )
            submitted = st.form_submit_button("Submit feedback", type="primary")

        if submitted:
            run_id = st.session_state.thread_id
            scenario = (st.session_state.final_state.get("scenario") or "baseline").strip()
            try:
                append_manager_rating(
                    run_id=run_id,
                    manager_id=manager_id or "M-Anonymous",
                    scenario=scenario,
                    star_rating=int(stars),
                    tags=tags,
                    comment=comment,
                )
                st.session_state.feedback_submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")

    # Always show history below
    st.divider()
    st.markdown("### Feedback history (last 10 runs)")
    try:
        ratings = load_manager_ratings("feedback/manager_ratings.csv", last_n=10)
    except Exception:
        ratings = []

    if not ratings:
        st.caption("No ratings yet.")
    else:
        avg = sum(int(r.get("star_rating", 0)) for r in ratings) / max(1, len(ratings))
        st.metric("Average rating (last 10)", f"{avg:.1f} / 5")
        df_ratings = pd.DataFrame(ratings)[
            ["timestamp", "scenario", "star_rating", "tags", "comment"]
        ]
        st.dataframe(df_ratings, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Outcomes — log what actually happened (next morning)
# ---------------------------------------------------------------------------
with tab_outcomes:
    st.subheader("Yesterday's outcomes")
    st.caption("Once a delivery cycle completes, log the actual results here so the system can self-calibrate over time.")

    df_outcomes = load_outcome_log("feedback/outcome_log.csv")

    with st.form("outcome_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            run_id_input = st.text_input("Run ID", placeholder="e.g. run-20260510-am")
            scenario_input = st.text_input("Scenario tag", value="baseline")
        with col2:
            predicted_penalty = st.number_input("Predicted penalty (pts)", min_value=0, value=0, step=10)
            actual_penalty = st.number_input("Actual penalty (pts)", min_value=0, value=0, step=10)
            predicted_deferred = st.number_input("Predicted deferred", min_value=0, value=0, step=1)
            actual_deferred = st.number_input("Actual deferred", min_value=0, value=0, step=1)
        with col3:
            actual_t1_late = st.number_input("Tier-1 actually late", min_value=0, value=0, step=1)
            actual_t2_late = st.number_input("Tier-2 actually late", min_value=0, value=0, step=1)
            actual_breaches = st.number_input("Cold-chain breaches", min_value=0, value=0, step=1)

        incident_notes = st.text_area("Incident notes",
                                      placeholder="What actually happened? Any surprises?",
                                      height=80)

        outcome_submit = st.form_submit_button("Log outcome", type="primary")

    if outcome_submit:
        if not run_id_input.strip():
            st.error("Run ID is required.")
        else:
            try:
                append_outcome(
                    run_id=run_id_input.strip(),
                    scenario=scenario_input or "baseline",
                    predicted_penalty=int(predicted_penalty),
                    actual_penalty=int(actual_penalty),
                    predicted_deferred=int(predicted_deferred),
                    actual_deferred=int(actual_deferred),
                    actual_tier1_late=int(actual_t1_late),
                    actual_tier2_late=int(actual_t2_late),
                    actual_cold_chain_breaches=int(actual_breaches),
                    incident_notes=incident_notes,
                )
                st.success("✅ Outcome logged. The next run's OpsDataAgent will use this to recalibrate.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save outcome: {e}")

    st.divider()
    st.markdown("### Logged outcomes (most recent first)")
    if df_outcomes.empty:
        st.caption("No outcomes logged yet.")
    else:
        df_show = df_outcomes.sort_values("timestamp", ascending=False).head(20)
        st.dataframe(df_show, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Calibration — predicted vs actual chart + metrics
# ---------------------------------------------------------------------------
with tab_calibration:
    st.subheader("System calibration")
    st.caption("How well do the system's predicted penalties match what actually happens? "
               "Bias > 0 means the system under-predicts; bias < 0 means over-predicts.")

    cal = compute_calibration("feedback/outcome_log.csv")

    if cal.get("runs_n", 0) == 0:
        st.info("No outcome history yet. Log a few outcomes in the Outcomes tab to see calibration.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: _hero_card("Runs scored", str(cal["runs_n"]), "")
        with c2: _hero_card("Penalty MAE", f"{cal['penalty_mae']:.0f}", "lower is better")
        with c3:
            bias = cal["penalty_bias"]
            label = "Under-pred" if bias > 0 else "Over-pred" if bias < 0 else "Matched"
            pill = "pill-amber" if abs(bias) > 50 else "pill-green"
            _hero_card("Penalty bias", f"{bias:+.0f}", label, pill)
        with c4: _hero_card("Cold-chain breaches", str(cal["cold_chain_breach_total"]),
                           "all-time recorded")

        st.divider()

        # Predicted vs actual scatter
        df_cal = load_outcome_log("feedback/outcome_log.csv")
        if not df_cal.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_cal["predicted_penalty"], y=df_cal["actual_penalty"],
                mode="markers+text",
                marker=dict(size=12, color="#3b82f6"),
                text=df_cal["scenario"].fillna("baseline").str[:20],
                textposition="top center",
                hovertemplate="Run: %{customdata}<br>Predicted: %{x}<br>Actual: %{y}<extra></extra>",
                customdata=df_cal["run_id"],
                name="Runs",
            ))
            # Perfect calibration line
            mx = max(df_cal["predicted_penalty"].max(), df_cal["actual_penalty"].max(), 100) * 1.1
            fig.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx],
                mode="lines", line=dict(color="#94a3b8", dash="dash"),
                name="Perfect calibration",
            ))
            fig.update_layout(
                title="Predicted vs actual penalty",
                xaxis_title="Predicted penalty (pts)",
                yaxis_title="Actual penalty (pts)",
                height=420, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("**Headline:** " + cal.get("headline", ""))

            st.markdown("### Full outcome log")
            st.dataframe(df_cal.sort_values("timestamp", ascending=False),
                         width="stretch", hide_index=True)
