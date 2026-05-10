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
from graph import build_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SeeWeeS Dispatch Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Build graph once and cache on session
if "app" not in st.session_state:
    st.session_state.app = build_graph(checkpointer=st.session_state.checkpointer)

app = st.session_state.app

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/delivery-truck.png", width=64)
    st.title("SeeWeeS Ops Intelligence")
    st.caption("Multi-Agent Dispatch Planning System")
    st.divider()

    pdf_path = st.selectbox(
        "Playbook",
        ["data/SeeWeeS Specialty distribution.pdf"],
        disabled=True,
    )
    csv_path = st.selectbox(
        "Shipment data",
        [
            "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
            "data/Incoming_shipment_02_08.csv",
            "data-for-enhancement/synthetic/synthetic_baseline.csv",
            "data-for-enhancement/synthetic/synthetic_volume_spike.csv",
            "data-for-enhancement/synthetic/synthetic_dq_heavy.csv",
            "data-for-enhancement/synthetic/synthetic_tier1_surge.csv",
            "data-for-enhancement/synthetic/synthetic_growth_trend.csv",
            "data-for-enhancement/synthetic/synthetic_rich_60d.csv",
        ],
    )
    st.divider()

    st.subheader("What-if Scenario")
    scenario = st.text_area(
        "Describe a disruption (optional)",
        placeholder=(
            "e.g. 20% demand spike on C2\n"
            "e.g. Warehouse closure at Newark DC\n"
            "e.g. Driver shortage — only 3 drivers available"
        ),
        height=110,
    )

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    if st.session_state.run_complete:
        if st.button("↺  New Run", use_container_width=True):
            for k in ["thread_id", "awaiting_approval", "approval_payload",
                      "run_complete", "final_state", "execution_log", "audit_attempts"]:
                del st.session_state[k]
            st.session_state.checkpointer = MemorySaver()
            st.session_state.app = build_graph(checkpointer=st.session_state.checkpointer)
            st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_pipeline, tab_report, tab_analytics = st.tabs([
    "🔄  Live Pipeline", "📄  Report", "📊  Analytics"
])

# ---------------------------------------------------------------------------
# Node label map for display
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "pdf_context":      ("📖", "Reading playbook rules"),
    "csv_analysis":     ("📦", "Analysing shipment data"),
    "weather":          ("🌦️", "Fetching corridor weather"),
    "planner":          ("🧠", "Writing dispatch plan"),
    "audit":            ("🔍", "Auditing plan compliance"),
    "allocator":        ("🚛", "Allocating trucks & drivers"),
    "human_checkpoint": ("👤", "Human approval checkpoint"),
    "report":           ("📝", "Generating HTML report"),
    "email":            ("✉️",  "Sending email"),
}

RISK_COLOUR = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

# ---------------------------------------------------------------------------
# Helper: render weather corridor cards
# ---------------------------------------------------------------------------
def _render_weather(weather_risk: dict):
    if not weather_risk:
        return
    cols = st.columns(len(weather_risk))
    for col, (corridor_id, w) in zip(cols, weather_risk.items()):
        score = w.get("route_risk_score_0_3", 0)
        emoji = RISK_COLOUR.get(score, "⚪")
        label = "NJ → Boston" if "BOS" in corridor_id else "NJ → Philadelphia"
        with col:
            st.metric(
                label=f"{emoji} {label}",
                value=f"Risk {score}/3",
                delta=f"{w.get('required_buffer_pct', 0)}% buffer",
                delta_color="inverse",
            )
            worst = w.get("worst_waypoint", {})
            if worst:
                st.caption(
                    f"Worst: {worst.get('city')} — "
                    f"🌧️ {worst.get('max_precip_mm_day', 0):.1f}mm  "
                    f"💨 {worst.get('max_wind_gust_kmh', 0):.1f}km/h"
                )

# ---------------------------------------------------------------------------
# Helper: render corridor KPI table
# ---------------------------------------------------------------------------
def _render_corridor_kpis(corridor_kpis: list):
    if not corridor_kpis:
        return
    rows = []
    for k in corridor_kpis:
        rows.append({
            "Corridor": "NJ→Boston" if "BOS" in k["corridor_id"] else "NJ→Philadelphia",
            "Day": k["day"],
            "Valid": k["valid_rows"],
            "Excluded": k["excluded_rows"],
            "Excl %": f"{k['exclusion_rate_pct']}%",
            "Tier 1": k["tier1_units"],
            "Tier 2": k["tier2_units"],
            "Cold Chain": k["cold_chain_units"],
            "Cold Trucks Needed": k["trucks_needed_cold_chain"],
            "Std Trucks Needed": k["trucks_needed_standard"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Helper: render resource allocation
# ---------------------------------------------------------------------------
def _render_allocation(allocation_plan: dict):
    if not allocation_plan or allocation_plan.get("raw"):
        st.write(allocation_plan.get("narrative", allocation_plan))
        return

    st.metric("Total Penalty Score", allocation_plan.get("total_penalty_score", "—"))
    st.metric("Deferred Units", allocation_plan.get("deferred_units", "—"))
    st.write(allocation_plan.get("rationale", ""))

    rows = []
    for day in ["Day0", "Day1"]:
        for corridor, alloc in allocation_plan.get(day, {}).items():
            label = "NJ→Boston" if "BOS" in corridor else "NJ→Philadelphia"
            rows.append({
                "Day": day,
                "Corridor": label,
                "Cold-Chain Trucks": alloc.get("truck_temp_controlled", "—"),
                "Standard Trucks": alloc.get("truck_standard", "—"),
                "Drivers": alloc.get("drivers", "—"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Helper: trend chart
# ---------------------------------------------------------------------------
def _render_trend_chart(trend_summary: dict):
    if not trend_summary or "daily_volume_by_corridor" not in trend_summary:
        st.info("No trend data available.")
        return

    vol = trend_summary["daily_volume_by_corridor"]
    avg = vol.get("avg_daily_units", {})
    corridors = list(avg.keys())
    labels = ["NJ→Boston" if "BOS" in c else "NJ→Philadelphia" for c in corridors]
    values = [avg[c] for c in corridors]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#4A90E2", "#E25C4A"],
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Avg Daily Units — {trend_summary.get('history_days', 0)}-day history",
        yaxis_title="Units / day",
        height=350,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Helper: per-waypoint risk chart
# ---------------------------------------------------------------------------
def _render_waypoint_chart(weather_risk: dict):
    if not weather_risk:
        return
    rows = []
    for corridor_id, w in weather_risk.items():
        label = "NJ→Boston" if "BOS" in corridor_id else "NJ→Philadelphia"
        for wp in w.get("per_waypoint", []):
            rows.append({
                "Corridor": label,
                "Waypoint": f"{wp['waypoint']} {wp['city']}",
                "Risk Score": wp["risk_score_0_3"],
                "Precip (mm)": wp["max_precip_mm_day"],
                "Wind (km/h)": wp["max_wind_gust_kmh"],
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Waypoint", y="Risk Score", color="Corridor",
        barmode="group",
        color_discrete_map={"NJ→Boston": "#4A90E2", "NJ→Philadelphia": "#E25C4A"},
        title="Waypoint Risk Scores — Both Corridors",
        range_y=[0, 3.5],
    )
    fig.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Config is always derived from the current thread_id in session state
# ---------------------------------------------------------------------------
config = {"configurable": {"thread_id": st.session_state.thread_id}}


# ---------------------------------------------------------------------------
# Helper: stream a graph run and display each node in tab_pipeline
# Breaks on __interrupt__ and saves payload to session_state.
# Returns True if interrupted, False if run completed normally.
# ---------------------------------------------------------------------------
def _stream_graph(stream_input, cfg):
    for event in app.stream(stream_input, cfg, stream_mode="updates"):
        for node_name, node_output in event.items():

            if node_name == "__interrupt__":
                interrupt_data = node_output[0].value if node_output else {}
                st.session_state.awaiting_approval = True
                st.session_state.approval_payload = interrupt_data
                return True  # caller must stop here

            emoji, label = NODE_LABELS.get(node_name, ("⚙️", node_name))

            if node_name == "audit":
                verdict = node_output.get("audit_verdict", "")
                if verdict == "FAIL":
                    attempts = st.session_state.audit_attempts + 1
                    st.session_state.audit_attempts = attempts
                    st.warning(
                        f"🔍 **Audit FAIL** (attempt {attempts}/3) — "
                        f"{node_output.get('audit_feedback', '')[:120]}..."
                    )
                else:
                    st.success(f"{emoji} **{label}** — PASS ✓")
            elif node_name == "weather":
                with st.expander(f"{emoji} **{label}** — done", expanded=True):
                    _render_weather(node_output.get("weather_risk", {}))
            elif node_name == "allocator":
                alloc = node_output.get("allocation_plan", {})
                penalty = alloc.get("total_penalty_score", "?") if isinstance(alloc, dict) else "?"
                st.success(f"{emoji} **{label}** — Penalty score: {penalty}")
            elif node_name not in ("email",):
                st.success(f"{emoji} **{label}** — done")

    return False  # completed without interrupt


def _finalise(cfg):
    """Grab final state, mark complete, save HTML."""
    snapshot = app.get_state(cfg)
    st.session_state.final_state = snapshot.values
    st.session_state.run_complete = True
    html = snapshot.values.get("report_html", "")
    if html:
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(html)
    st.success("✅ **Pipeline complete.** Switch to the Report or Analytics tab.")


# ---------------------------------------------------------------------------
# Phase 1 — Start a brand-new run
# ---------------------------------------------------------------------------
if run_btn:
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.awaiting_approval = False
    st.session_state.approval_payload = None
    st.session_state.run_complete = False
    st.session_state.final_state = None
    st.session_state.execution_log = []
    st.session_state.audit_attempts = 0
    st.session_state.pop("_pending_decision", None)

    initial_state = {
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "scenario": scenario.strip() if scenario.strip() else None,
    }

    with tab_pipeline:
        st.subheader("Pipeline Execution")
        try:
            interrupted = _stream_graph(initial_state, config)
            if not interrupted:
                _finalise(config)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            raise


# ---------------------------------------------------------------------------
# Phase 2 — Awaiting manager approval (Streamlit re-render cycle)
# ---------------------------------------------------------------------------
if st.session_state.awaiting_approval and not st.session_state.run_complete:
    interrupt_data = st.session_state.approval_payload or {}
    with tab_pipeline:
        st.subheader("Pipeline Execution")
        st.error("⚠️ RISK SCORE 3 DETECTED — Manager Approval Required")
        st.write("**Weather risk is critical on one or more corridors. Review and decide.**")

        alloc = interrupt_data.get("allocation_plan", {})
        weather = interrupt_data.get("weather_risk", {})
        _render_weather(weather)
        if alloc and not alloc.get("raw"):
            st.write(f"**Proposed allocation — Penalty Score: {alloc.get('total_penalty_score', '?')}**")
            _render_allocation(alloc)

        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Approve & Generate Report", type="primary", key="hitl_approve"):
            st.session_state._pending_decision = "YES"
            st.session_state.awaiting_approval = False
            st.rerun()
        if col_no.button("❌ Reject Plan", key="hitl_reject"):
            st.session_state._pending_decision = "NO"
            st.session_state.awaiting_approval = False
            st.rerun()


# ---------------------------------------------------------------------------
# Phase 3 — Resume graph after approval decision
# ---------------------------------------------------------------------------
if "_pending_decision" in st.session_state and not st.session_state.run_complete:
    decision = st.session_state.pop("_pending_decision")
    with tab_pipeline:
        st.subheader("Pipeline Execution")
        st.info(f"Decision recorded: **{'APPROVED' if decision == 'YES' else 'REJECTED'}**")
        try:
            _stream_graph(Command(resume=decision), config)
            _finalise(config)
        except Exception as e:
            st.error(f"Resume error: {e}")
            raise

# ---------------------------------------------------------------------------
# Report tab
# ---------------------------------------------------------------------------
with tab_report:
    if st.session_state.run_complete and st.session_state.final_state:
        html = st.session_state.final_state.get("report_html", "")
        if html:
            import streamlit.components.v1 as components
            components.html(html, height=900, scrolling=True)
    elif not st.session_state.run_complete:
        st.info("Run the pipeline first using the sidebar.")

# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------
with tab_analytics:
    if st.session_state.run_complete and st.session_state.final_state:
        fs = st.session_state.final_state

        st.subheader("🌦️ Weather — Corridor & Waypoint Risk")
        _render_waypoint_chart(fs.get("weather_risk", {}))

        st.divider()
        st.subheader("📦 Shipment KPIs — By Corridor & Day")
        _render_corridor_kpis(fs.get("corridor_kpis", []))

        st.divider()
        st.subheader("📈 Historical Trend")
        _render_trend_chart(fs.get("trend_summary", {}))

        st.divider()
        st.subheader("🚛 Resource Allocation")
        _render_allocation(fs.get("allocation_plan", {}))

        st.divider()
        st.subheader("🔍 Audit & Data Quality")
        af = fs.get("audit_feedback", "")
        if af and af != "None — plan passed audit.":
            st.warning(af)
        else:
            st.success("Plan passed all audit checks.")
        dq_md = fs.get("anomalies_md", "")
        if dq_md and dq_md != "(none)":
            st.markdown(dq_md)

    elif not st.session_state.run_complete:
        st.info("Run the pipeline first using the sidebar.")
