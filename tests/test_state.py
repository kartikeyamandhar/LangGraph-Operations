"""
AppState schema tests.

The graph passes a single TypedDict through every node. If a key is renamed
or dropped without updating downstream readers, runs silently produce an
incomplete report. These tests pin the contract.
"""
from __future__ import annotations

from typing import get_type_hints

import graph as graph_mod


REQUIRED_KEYS = {
    # Inputs
    "pdf_path", "csv_path", "scenario",
    # Step 1 outputs (parallel data gathering)
    "business_context", "csv_summary", "csv_kpis", "anomalies_md",
    "ops_insights", "corridor_kpis", "trend_summary", "weather_risk",
    # Planner
    "dispatch_plan", "plan_structured",
    # Audit loop
    "audit_verdict", "audit_feedback", "audit_attempts",
    # Allocation
    "allocation_plan",
    # Human checkpoint
    "human_approved",
    # Final output
    "report_html",
}


def test_appstate_has_all_required_keys():
    keys = set(get_type_hints(graph_mod.AppState).keys())
    missing = REQUIRED_KEYS - keys
    assert not missing, f"AppState is missing keys: {missing}"


def test_appstate_total_is_false():
    """All keys must be optional so individual nodes can write subsets."""
    assert graph_mod.AppState.__total__ is False


def test_corridor_waypoints_define_both_corridors():
    assert "C1_I95_NJ_BOS" in graph_mod.CORRIDOR_WAYPOINTS
    assert "C2_NJ_PHL" in graph_mod.CORRIDOR_WAYPOINTS
    # Each waypoint must have lat/lon for the weather node
    for cid, wps in graph_mod.CORRIDOR_WAYPOINTS.items():
        assert len(wps) >= 4, f"{cid} should have at least 4 waypoints"
        for wp in wps:
            assert {"id", "city", "state", "lat", "lon"} <= set(wp.keys())
            assert isinstance(wp["lat"], (int, float))
            assert isinstance(wp["lon"], (int, float))


def test_buffer_policy_matches_playbook():
    """Buffer policy is non-negotiable per Playbook §5.2."""
    assert graph_mod.BUFFER_POLICY == {0: 0, 1: 10, 2: 25, 3: 40}


def test_resource_pool_has_both_planning_days():
    assert "Day0" in graph_mod.RESOURCE_POOL
    assert "Day1" in graph_mod.RESOURCE_POOL
    for day, pool in graph_mod.RESOURCE_POOL.items():
        assert pool["truck_temp_controlled"] >= 1
        assert pool["truck_standard"] >= 1
        assert pool["driver"] >= 1


def test_max_audit_attempts_is_bounded():
    """Must be small enough to bound LLM cost and avoid infinite loops."""
    assert 1 <= graph_mod.MAX_AUDIT_ATTEMPTS <= 10
