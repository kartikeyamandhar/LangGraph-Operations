"""
Pytest configuration: makes src/ importable and provides shared fixtures
that mock every LLM call so the suite runs offline without OpenAI credits.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# Make `src/` importable as if we were running from the project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Make sure tests never accidentally hit a real LLM or LangSmith
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_weather_risk_safe() -> Dict[str, Any]:
    """All corridors clear (score 0)."""
    return {
        "C1_I95_NJ_BOS": {
            "corridor_id": "C1_I95_NJ_BOS",
            "route_risk_score_0_3": 0,
            "required_buffer_pct": 0,
            "escalation_required": False,
            "per_waypoint": [],
            "worst_waypoint": None,
        },
        "C2_NJ_PHL": {
            "corridor_id": "C2_NJ_PHL",
            "route_risk_score_0_3": 0,
            "required_buffer_pct": 0,
            "escalation_required": False,
            "per_waypoint": [],
            "worst_waypoint": None,
        },
    }


@pytest.fixture
def fake_weather_risk_critical() -> Dict[str, Any]:
    """C1 has a maximum risk score; C2 is mild."""
    return {
        "C1_I95_NJ_BOS": {
            "corridor_id": "C1_I95_NJ_BOS",
            "route_risk_score_0_3": 3,
            "required_buffer_pct": 40,
            "escalation_required": True,
            "per_waypoint": [],
            "worst_waypoint": None,
        },
        "C2_NJ_PHL": {
            "corridor_id": "C2_NJ_PHL",
            "route_risk_score_0_3": 1,
            "required_buffer_pct": 10,
            "escalation_required": False,
            "per_waypoint": [],
            "worst_waypoint": None,
        },
    }


@pytest.fixture
def fake_corridor_kpis() -> List[Dict[str, Any]]:
    """Two corridors × two days of demand for allocator tests."""
    return [
        {
            "corridor_id": "C1_I95_NJ_BOS", "day": "Day0",
            "valid_rows": 25, "tier1_units": 8, "tier2_units": 17,
            "cold_chain_units": 12, "room_temp_units": 8, "controlled_units": 5,
        },
        {
            "corridor_id": "C1_I95_NJ_BOS", "day": "Day1",
            "valid_rows": 20, "tier1_units": 5, "tier2_units": 15,
            "cold_chain_units": 8, "room_temp_units": 7, "controlled_units": 5,
        },
        {
            "corridor_id": "C2_NJ_PHL", "day": "Day0",
            "valid_rows": 15, "tier1_units": 3, "tier2_units": 12,
            "cold_chain_units": 5, "room_temp_units": 6, "controlled_units": 4,
        },
        {
            "corridor_id": "C2_NJ_PHL", "day": "Day1",
            "valid_rows": 12, "tier1_units": 2, "tier2_units": 10,
            "cold_chain_units": 4, "room_temp_units": 5, "controlled_units": 3,
        },
    ]


@pytest.fixture
def planner_pass_first_try() -> Tuple[str, Dict[str, Any]]:
    """Mock planner output that respects the buffer policy and escalation rule."""
    return (
        "Prose dispatch plan with buffers applied.",
        {
            "buffer_pct_c1": 40,
            "buffer_pct_c2": 10,
            "escalation_triggered": True,
            "tier1_sla_at_risk": False,
            "estimated_penalty_score": 200,
        },
    )


@pytest.fixture
def planner_violates_buffer() -> Tuple[str, Dict[str, Any]]:
    """Mock planner output with the wrong buffer for risk score 3."""
    return (
        "Plan with incorrect buffer.",
        {
            "buffer_pct_c1": 10,           # WRONG — should be 40 for risk=3
            "buffer_pct_c2": 10,
            "escalation_triggered": True,
            "tier1_sla_at_risk": False,
            "estimated_penalty_score": 300,
        },
    )


@pytest.fixture
def planner_missing_escalation() -> Tuple[str, Dict[str, Any]]:
    """Mock planner output with correct buffer but missing escalation flag."""
    return (
        "Plan that forgot to escalate.",
        {
            "buffer_pct_c1": 40,
            "buffer_pct_c2": 10,
            "escalation_triggered": False,  # WRONG — required when score=3
            "tier1_sla_at_risk": False,
            "estimated_penalty_score": 250,
        },
    )
