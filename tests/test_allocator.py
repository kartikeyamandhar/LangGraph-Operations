"""
Allocator post-correction tests.

The LLM allocator can over-allocate cold-chain trucks (or miscount the
penalty); these post-corrections are the safety net. They have to be
deterministic, terminate, and produce numbers that match the playbook
penalty model.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from graph import (
    _clip_cold_chain_allocation,
    _recompute_penalty,
    RESOURCE_POOL,
    PENALTY_TIER1,
    PENALTY_TIER2,
    UNITS_PER_TRUCK,
)


# ---------------------------------------------------------------------------
# Cold-chain clipping
# ---------------------------------------------------------------------------
def test_clipping_within_cap_is_a_noop():
    cap = RESOURCE_POOL["Day0"]["truck_temp_controlled"]
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": cap, "truck_standard": 2, "drivers": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0,   "truck_standard": 1, "drivers": 1},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 2, "drivers": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 1, "drivers": 1},
        },
        "rationale": "ok",
    }
    out = _clip_cold_chain_allocation(allocation)
    assert out["Day0"]["C1_I95_NJ_BOS"]["truck_temp_controlled"] == cap
    assert "Auto-corrected" not in out.get("rationale", "")


def test_clipping_reduces_to_cap_and_takes_from_largest_first():
    cap = RESOURCE_POOL["Day0"]["truck_temp_controlled"]   # default 2
    over = cap + 3                                          # 3 too many
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": over, "truck_standard": 0, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1,    "truck_standard": 0, "drivers": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 0, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 0, "drivers": 0},
        },
        "rationale": "original",
    }
    out = _clip_cold_chain_allocation(allocation)

    day0_total = sum(v["truck_temp_controlled"] for v in out["Day0"].values())
    assert day0_total == cap
    # C1 was the over-allocator, so C1 absorbs the cuts (greedy: highest first)
    assert out["Day0"]["C1_I95_NJ_BOS"]["truck_temp_controlled"] < over
    assert out["Day0"]["C2_NJ_PHL"]["truck_temp_controlled"] == 1
    assert "Auto-corrected" in out["rationale"]


def test_clipping_preserves_existing_rationale():
    over = RESOURCE_POOL["Day0"]["truck_temp_controlled"] + 1
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": over, "truck_standard": 0, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0,    "truck_standard": 0, "drivers": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "drivers": 0},
        },
        "rationale": "original explanation",
    }
    out = _clip_cold_chain_allocation(allocation)
    assert "original explanation" in out["rationale"]


def test_clipping_handles_raw_passthrough():
    """When the LLM produced un-parseable output, allocator returns
    {'narrative': ..., 'raw': True} — clipping must not touch it."""
    raw = {"narrative": "free text", "raw": True}
    out = _clip_cold_chain_allocation(raw)
    assert out is raw


# ---------------------------------------------------------------------------
# Penalty recomputation
# ---------------------------------------------------------------------------
def _full_supply(corridor_kpis):
    """Build an allocation that covers all demand."""
    by_day_corridor = {}
    for k in corridor_kpis:
        by_day_corridor.setdefault(k["day"], {})[k["corridor_id"]] = {
            "truck_temp_controlled": (k["cold_chain_units"] + UNITS_PER_TRUCK - 1) // UNITS_PER_TRUCK,
            "truck_standard":        ((k["room_temp_units"] + k["controlled_units"]) + UNITS_PER_TRUCK - 1) // UNITS_PER_TRUCK,
            "drivers": 2,
        }
    return {**by_day_corridor, "rationale": "full supply"}


def test_penalty_zero_when_supply_meets_demand(fake_corridor_kpis):
    allocation = _full_supply(fake_corridor_kpis)
    out = _recompute_penalty(allocation, fake_corridor_kpis)
    assert out["total_penalty_score"] == 0
    assert out["deferred_units"] == 0


def test_penalty_uses_tier1_rate_for_tier1_deferrals(fake_corridor_kpis):
    """If we starve cold-chain capacity, Tier-1 cold-chain demand defers
    first and is scored at PENALTY_TIER1."""
    allocation = {
        "Day0": {
            # No cold-chain trucks at all on Day0
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 99, "drivers": 99},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 99, "drivers": 99},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 99, "truck_standard": 99, "drivers": 99},
            "C2_NJ_PHL":     {"truck_temp_controlled": 99, "truck_standard": 99, "drivers": 99},
        },
        "rationale": "starved cold chain",
    }
    out = _recompute_penalty(allocation, fake_corridor_kpis)

    # Day0 cold-chain demand: C1=12 (tier1=8, t2=4), C2=5 (tier1=3, t2=2)
    # Day1 should be fully met
    expected_t1_def = 8 + 3                       # day0 only
    expected_t2_cold_def = 4 + 2                  # day0 only
    expected_penalty = expected_t1_def * PENALTY_TIER1 + expected_t2_cold_def * PENALTY_TIER2

    assert out["deferred_units"] == expected_t1_def + expected_t2_cold_def
    assert out["total_penalty_score"] == expected_penalty


def test_penalty_protects_tier1_first_when_partial_supply(fake_corridor_kpis):
    """With cold-chain capacity below total demand but above Tier-1 demand,
    Tier-1 must be fully fulfilled (zero T1 deferral), Tier-2 absorbs the gap."""
    # Day0 Tier-1 cold demand = 8 + 3 = 11; total cold = 12 + 5 = 17
    # Allocate exactly enough cold-chain for Tier-1 only on Day0
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 99, "drivers": 99},  # 10 units
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 99, "drivers": 99},  # 10 units
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 99, "truck_standard": 99, "drivers": 99},
            "C2_NJ_PHL":     {"truck_temp_controlled": 99, "truck_standard": 99, "drivers": 99},
        },
        "rationale": "barely covers tier1",
    }
    out = _recompute_penalty(allocation, fake_corridor_kpis)

    # C1 Day0: cold supply 10, tier1 cold demand 8 → 0 T1 def, 2 leftover; tier2 cold demand 4 → 2 T2 cold def
    # C2 Day0: cold supply 10, tier1 cold demand 3 → 0 T1 def, 7 leftover; tier2 cold demand 2 → 0 T2 cold def
    # Day1 fully fulfilled
    expected_t2_cold_def = 2  # only C1 day0 has unmet tier2 cold
    assert out["total_penalty_score"] == expected_t2_cold_def * PENALTY_TIER2
    assert out["deferred_units"] == expected_t2_cold_def


def test_penalty_overwrites_llm_reported_values(fake_corridor_kpis):
    allocation = _full_supply(fake_corridor_kpis)
    allocation["total_penalty_score"] = 999_999  # LLM lied
    allocation["deferred_units"] = 999

    out = _recompute_penalty(allocation, fake_corridor_kpis)

    assert out["total_penalty_score"] == 0
    assert out["deferred_units"] == 0
    assert "Penalty recomputed deterministically" in out["rationale"]


def test_penalty_handles_raw_passthrough(fake_corridor_kpis):
    raw = {"narrative": "free text", "raw": True}
    out = _recompute_penalty(raw, fake_corridor_kpis)
    assert out is raw


def test_penalty_constants_match_playbook():
    """Pin the constants — changing them is a business decision, not a refactor."""
    assert PENALTY_TIER1 == 100
    assert PENALTY_TIER2 == 40
    assert UNITS_PER_TRUCK == 10
