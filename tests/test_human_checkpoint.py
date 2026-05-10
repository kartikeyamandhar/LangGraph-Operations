"""
Human-in-the-loop checkpoint tests.

The checkpoint must:
  - skip silently when no corridor has a critical risk score,
  - call interrupt(...) exactly once when any corridor reaches risk score 3,
  - return the manager's decision as a boolean human_approved flag,
  - never re-trigger after a prior approval.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

import graph as graph_mod
from graph import node_human_checkpoint


# ---------------------------------------------------------------------------
# Skip path: no risk score reaches the threshold
# ---------------------------------------------------------------------------
def test_no_interrupt_when_all_corridors_safe(fake_weather_risk_safe):
    out = node_human_checkpoint({"weather_risk": fake_weather_risk_safe})
    # The flag must distinguish 'auto-cleared' (required=False) from
    # 'fired and approved' (required=True, approved=True).
    assert out == {"human_approval_required": False, "human_approved": True}


def test_no_interrupt_on_risk_score_two():
    weather = {
        "C1_I95_NJ_BOS": {"route_risk_score_0_3": 2},
        "C2_NJ_PHL":     {"route_risk_score_0_3": 1},
    }
    out = node_human_checkpoint({"weather_risk": weather})
    assert out == {"human_approval_required": False, "human_approved": True}


def test_no_interrupt_on_empty_weather_risk():
    """Defensive: empty dict shouldn't crash or trigger an interrupt."""
    out = node_human_checkpoint({"weather_risk": {}})
    assert out == {"human_approval_required": False, "human_approved": True}


# ---------------------------------------------------------------------------
# Interrupt path: at least one corridor at risk 3
# ---------------------------------------------------------------------------
def test_interrupt_fires_when_any_corridor_at_score_three(monkeypatch):
    """Mock interrupt() to capture the call payload and return an approval."""
    captured: Dict[str, Any] = {}

    def fake_interrupt(payload):
        captured["payload"] = payload
        return "YES"

    # Patch the import that node_human_checkpoint does inline
    import langgraph.types as lt
    monkeypatch.setattr(lt, "interrupt", fake_interrupt)

    weather = {
        "C1_I95_NJ_BOS": {"route_risk_score_0_3": 3},
        "C2_NJ_PHL":     {"route_risk_score_0_3": 1},
    }
    out = node_human_checkpoint({
        "weather_risk": weather,
        "allocation_plan": {"summary": "test"},
        "human_approved": False,
    })

    assert out["human_approved"] is True
    assert out["human_approval_required"] is True
    assert "message" in captured["payload"]
    assert "allocation_plan" in captured["payload"]
    assert captured["payload"]["weather_risk"] == weather


@pytest.mark.parametrize("response, expected", [
    ("YES", True),
    ("yes", True),
    ("Y", True),
    ("approve", True),
    ("Approve", True),
    ("NO", False),
    ("reject", False),
    ("", False),
    ("maybe", False),
])
def test_decision_parsing_is_case_insensitive(monkeypatch, response, expected):
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: response)

    out = node_human_checkpoint({
        "weather_risk": {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 3}},
        "allocation_plan": {},
        "human_approved": False,
    })
    assert out["human_approved"] is expected


def test_already_approved_state_skips_interrupt(monkeypatch):
    """If the manager already approved on a prior pass, don't re-prompt."""
    called = {"count": 0}

    def boom(payload):
        called["count"] += 1
        return "NO"

    monkeypatch.setattr("langgraph.types.interrupt", boom)

    out = node_human_checkpoint({
        "weather_risk": {"C1_I95_NJ_BOS": {"route_risk_score_0_3": 3}},
        "allocation_plan": {},
        "human_approved": True,    # already approved
    })

    assert called["count"] == 0
    assert out["human_approved"] is True
