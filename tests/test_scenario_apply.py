"""
Scenario apply tests — the agentic what-if engine.

These tests pin the contract that scenarios actually MODIFY downstream state.
Without them, a regression could turn the scenario field back into theatre.

The LLM call inside `run_scenario_parser_agent` is mocked so the suite stays
offline; the override-application logic itself is pure Python and is tested
directly.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

import graph as graph_mod
from graph import (
    BASE_RESOURCE_POOL, BUFFER_POLICY,
    _apply_resource_overrides,
    _apply_demand_multipliers,
    _apply_weather_overrides,
    node_scenario_apply,
)


# ---------------------------------------------------------------------------
# _apply_resource_overrides
# ---------------------------------------------------------------------------
def test_no_overrides_returns_deep_copy_of_base():
    out = _apply_resource_overrides(BASE_RESOURCE_POOL, {})
    assert out == BASE_RESOURCE_POOL
    # Must be a deep copy — mutations don't bleed into the constant
    out["Day0"]["truck_temp_controlled"] = 999
    assert BASE_RESOURCE_POOL["Day0"]["truck_temp_controlled"] != 999


def test_resource_override_updates_one_day():
    overrides = {"Day0": {"truck_temp_controlled": 1}}
    out = _apply_resource_overrides(BASE_RESOURCE_POOL, overrides)
    assert out["Day0"]["truck_temp_controlled"] == 1
    # Day1 untouched
    assert out["Day1"]["truck_temp_controlled"] == BASE_RESOURCE_POOL["Day1"]["truck_temp_controlled"]
    # Other resources on Day0 also untouched
    assert out["Day0"]["truck_standard"] == BASE_RESOURCE_POOL["Day0"]["truck_standard"]


def test_resource_override_clamps_to_zero():
    """Negative resource counts make no physical sense — must clamp to 0."""
    overrides = {"Day0": {"truck_temp_controlled": -5}}
    out = _apply_resource_overrides(BASE_RESOURCE_POOL, overrides)
    assert out["Day0"]["truck_temp_controlled"] == 0


def test_unknown_day_or_resource_is_ignored():
    overrides = {"Day9": {"truck_temp_controlled": 1}, "Day0": {"unicorn_truck": 7}}
    out = _apply_resource_overrides(BASE_RESOURCE_POOL, overrides)
    assert out == BASE_RESOURCE_POOL
    assert "Day9" not in out
    assert "unicorn_truck" not in out["Day0"]


# ---------------------------------------------------------------------------
# _apply_demand_multipliers
# ---------------------------------------------------------------------------
def test_no_multipliers_is_a_passthrough():
    kpis = [{"corridor_id": "C1_I95_NJ_BOS", "valid_rows": 10, "tier1_units": 4}]
    out = _apply_demand_multipliers(kpis, {})
    assert out == kpis


def test_multiplier_scales_numeric_demand_fields():
    kpis = [{
        "corridor_id": "C1_I95_NJ_BOS",
        "day": "Day0",
        "valid_rows": 10, "tier1_units": 4, "tier2_units": 6,
        "cold_chain_units": 8, "room_temp_units": 2, "controlled_units": 0,
        "trucks_needed_standard": 1, "trucks_needed_cold_chain": 1,
        "exclusion_rate_pct": 0.0,        # NOT in scaled_keys → must stay
    }]
    out = _apply_demand_multipliers(kpis, {"C1_I95_NJ_BOS": 1.20})
    assert out[0]["valid_rows"] == 12      # 10 × 1.2 = 12
    assert out[0]["tier1_units"] == 5      # 4 × 1.2 = 4.8 → 5 (rounded)
    assert out[0]["cold_chain_units"] == 10  # 8 × 1.2 = 9.6 → 10
    assert out[0]["exclusion_rate_pct"] == 0.0  # not in scaled_keys


def test_multiplier_only_affects_target_corridor():
    kpis = [
        {"corridor_id": "C1_I95_NJ_BOS", "valid_rows": 10},
        {"corridor_id": "C2_NJ_PHL",     "valid_rows": 10},
    ]
    out = _apply_demand_multipliers(kpis, {"C2_NJ_PHL": 2.0})
    assert out[0]["valid_rows"] == 10
    assert out[1]["valid_rows"] == 20


def test_multiplier_of_one_is_a_noop():
    kpis = [{"corridor_id": "C1_I95_NJ_BOS", "valid_rows": 10}]
    out = _apply_demand_multipliers(kpis, {"C1_I95_NJ_BOS": 1.0})
    assert out[0]["valid_rows"] == 10


# ---------------------------------------------------------------------------
# _apply_weather_overrides
# ---------------------------------------------------------------------------
def test_weather_override_forces_score_and_buffer():
    weather = {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 0, "required_buffer_pct": 0}}
    out = _apply_weather_overrides(
        weather, {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 3}}, []
    )
    assert out["C1_I95_NJ_BOS"]["route_risk_score_0_3"] == 3
    assert out["C1_I95_NJ_BOS"]["required_buffer_pct"] == BUFFER_POLICY[3]
    assert out["C1_I95_NJ_BOS"]["escalation_required"] is True
    assert out["C1_I95_NJ_BOS"]["scenario_forced"] is True


def test_weather_override_clamps_score_to_valid_range():
    weather = {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 0}}
    out = _apply_weather_overrides(
        weather, {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 99}}, []
    )
    assert out["C1_I95_NJ_BOS"]["route_risk_score_0_3"] == 3


def test_corridor_closure_forces_max_risk_and_marks_closed():
    weather = {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 0}}
    out = _apply_weather_overrides(weather, {}, ["C1_I95_NJ_BOS"])
    assert out["C1_I95_NJ_BOS"]["route_risk_score_0_3"] == 3
    assert out["C1_I95_NJ_BOS"]["closed"] is True
    assert out["C1_I95_NJ_BOS"]["escalation_required"] is True


def test_unknown_corridor_in_override_is_ignored():
    weather = {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 0}}
    out = _apply_weather_overrides(
        weather, {"NEVER_HEARD_OF_IT": {"route_risk_score_0_3": 3}}, []
    )
    assert out == weather   # unchanged


# ---------------------------------------------------------------------------
# node_scenario_apply (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------
def _stub_parser(overrides: Dict[str, Any]):
    """Return a callable that mimics run_scenario_parser_agent."""
    def _f(scenario, base_pool):
        return {
            "resource_overrides": overrides.get("resource_overrides", {}),
            "demand_multipliers": overrides.get("demand_multipliers", {}),
            "corridor_closures": overrides.get("corridor_closures", []),
            "weather_overrides": overrides.get("weather_overrides", {}),
            "transit_delay_hours": overrides.get("transit_delay_hours", {}),
            "summary": overrides.get("summary", "stub summary"),
        }
    return _f


def test_no_scenario_text_keeps_base_pool(monkeypatch):
    monkeypatch.setattr(graph_mod, "run_scenario_parser_agent",
                        _stub_parser({}))
    out = node_scenario_apply({"scenario": "", "corridor_kpis": [], "weather_risk": {}})
    assert out["effective_resource_pool"] == BASE_RESOURCE_POOL
    assert out["scenario_diff"]["summary"]


def test_cold_chain_breakdown_scenario_actually_changes_pool(monkeypatch):
    """The integration test that proves the architecture is no longer theatre."""
    monkeypatch.setattr(
        graph_mod, "run_scenario_parser_agent",
        _stub_parser({
            "resource_overrides": {
                "Day0": {"truck_temp_controlled": 1},
                "Day1": {"truck_temp_controlled": 1},
            },
            "summary": "Cold-chain truck breakdown — capacity halved.",
        }),
    )
    out = node_scenario_apply({
        "scenario": "Cold-chain truck broke down",
        "corridor_kpis": [],
        "weather_risk": {},
    })

    # The scenario actually halved the cap
    assert out["effective_resource_pool"]["Day0"]["truck_temp_controlled"] == 1
    assert out["effective_resource_pool"]["Day1"]["truck_temp_controlled"] == 1
    # The diff records what changed in human-readable form
    changes = out["scenario_diff"]["resource_changes"]
    assert any(c["resource"] == "truck_temp_controlled" and c["from"] == 2 and c["to"] == 1
               for c in changes)


def test_demand_spike_scenario_scales_corridor_kpis(monkeypatch):
    monkeypatch.setattr(
        graph_mod, "run_scenario_parser_agent",
        _stub_parser({
            "demand_multipliers": {"C2_NJ_PHL": 1.20},
            "summary": "20% demand spike on NJ→Philadelphia.",
        }),
    )
    out = node_scenario_apply({
        "scenario": "20% spike on C2",
        "corridor_kpis": [
            {"corridor_id": "C1_I95_NJ_BOS", "valid_rows": 10, "tier1_units": 4,
             "tier2_units": 6, "cold_chain_units": 5, "room_temp_units": 5,
             "controlled_units": 0, "trucks_needed_standard": 1,
             "trucks_needed_cold_chain": 1},
            {"corridor_id": "C2_NJ_PHL", "valid_rows": 10, "tier1_units": 4,
             "tier2_units": 6, "cold_chain_units": 5, "room_temp_units": 5,
             "controlled_units": 0, "trucks_needed_standard": 1,
             "trucks_needed_cold_chain": 1},
        ],
        "weather_risk": {},
    })
    c1, c2 = out["corridor_kpis"]
    assert c1["valid_rows"] == 10           # untouched
    assert c2["valid_rows"] == 12           # scaled by 1.20
    assert out["scenario_diff"]["demand_multipliers"] == {"C2_NJ_PHL": 1.20}


def test_severe_weather_scenario_forces_route_risk(monkeypatch):
    monkeypatch.setattr(
        graph_mod, "run_scenario_parser_agent",
        _stub_parser({
            "weather_overrides": {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 3}},
            "summary": "Severe storm forecast on I-95.",
        }),
    )
    out = node_scenario_apply({
        "scenario": "Severe storm on C1",
        "corridor_kpis": [],
        "weather_risk": {
            "C1_I95_NJ_BOS": {"route_risk_score_0_3": 0, "required_buffer_pct": 0},
            "C2_NJ_PHL":     {"route_risk_score_0_3": 0, "required_buffer_pct": 0},
        },
    })
    assert out["weather_risk"]["C1_I95_NJ_BOS"]["route_risk_score_0_3"] == 3
    assert out["weather_risk"]["C1_I95_NJ_BOS"]["required_buffer_pct"] == 40
    assert out["weather_risk"]["C1_I95_NJ_BOS"]["escalation_required"] is True
    # The other corridor is untouched
    assert out["weather_risk"]["C2_NJ_PHL"]["route_risk_score_0_3"] == 0


def test_combined_scenario_applies_all_layers(monkeypatch):
    """A multi-faceted scenario must apply all override types together."""
    monkeypatch.setattr(
        graph_mod, "run_scenario_parser_agent",
        _stub_parser({
            "resource_overrides": {"Day0": {"driver": 3}},
            "demand_multipliers": {"C2_NJ_PHL": 1.30},
            "summary": "Driver shortage AND demand spike on C2.",
        }),
    )
    out = node_scenario_apply({
        "scenario": "3 drivers only, plus 30% spike on C2",
        "corridor_kpis": [
            {"corridor_id": "C2_NJ_PHL", "valid_rows": 10, "tier1_units": 3,
             "tier2_units": 7, "cold_chain_units": 5, "room_temp_units": 5,
             "controlled_units": 0, "trucks_needed_standard": 1,
             "trucks_needed_cold_chain": 1},
        ],
        "weather_risk": {},
    })
    assert out["effective_resource_pool"]["Day0"]["driver"] == 3
    assert out["corridor_kpis"][0]["valid_rows"] == 13
    assert "resource_changes" in out["scenario_diff"]
    assert "demand_multipliers" in out["scenario_diff"]
