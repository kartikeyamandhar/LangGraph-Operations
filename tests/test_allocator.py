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
    _clip_resource_allocation,
    _recompute_penalty,
    BASE_RESOURCE_POOL,
    PENALTY_TIER1,
    PENALTY_TIER2,
    UNITS_PER_TRUCK,
)


# ---------------------------------------------------------------------------
# Cold-chain clipping (now takes a pool argument — driven by scenario_apply)
# ---------------------------------------------------------------------------
def test_clipping_within_cap_is_a_noop():
    pool = BASE_RESOURCE_POOL
    cap = pool["Day0"]["truck_temp_controlled"]
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
    out = _clip_cold_chain_allocation(allocation, pool)
    assert out["Day0"]["C1_I95_NJ_BOS"]["truck_temp_controlled"] == cap
    assert "Auto-corrected" not in out.get("rationale", "")


def test_clipping_reduces_to_cap_and_takes_from_largest_first():
    pool = BASE_RESOURCE_POOL
    cap = pool["Day0"]["truck_temp_controlled"]   # default 2
    over = cap + 3                                # 3 too many
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
    out = _clip_cold_chain_allocation(allocation, pool)

    day0_total = sum(v["truck_temp_controlled"] for v in out["Day0"].values())
    assert day0_total == cap
    assert out["Day0"]["C1_I95_NJ_BOS"]["truck_temp_controlled"] < over
    assert out["Day0"]["C2_NJ_PHL"]["truck_temp_controlled"] == 1
    assert "Auto-corrected" in out["rationale"]


def test_clipping_uses_scenario_pool_not_base_pool():
    """Critical: when a scenario halves cold-chain capacity, clipping must
    enforce the scenario cap, not the base cap."""
    scenario_pool = {
        "Day0": {"driver": 6, "truck_standard": 4, "truck_temp_controlled": 1},
        "Day1": {"driver": 6, "truck_standard": 4, "truck_temp_controlled": 1},
    }
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 2, "truck_standard": 0, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "drivers": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 0, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "drivers": 0},
        },
        "rationale": "",
    }
    out = _clip_cold_chain_allocation(allocation, scenario_pool)

    # Day0 had 2 cold-chain trucks but the scenario cap is 1 → must clip 1
    day0_total = sum(v["truck_temp_controlled"] for v in out["Day0"].values())
    assert day0_total == 1
    # New format: "[Auto-corrected — Day0: clipped 1 cold-chain truck(s) (cap 1).]"
    assert "Auto-corrected" in out["rationale"]
    assert "cold-chain truck" in out["rationale"]
    assert "cap 1" in out["rationale"]


def test_clipping_preserves_existing_rationale():
    pool = BASE_RESOURCE_POOL
    over = pool["Day0"]["truck_temp_controlled"] + 1
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
    out = _clip_cold_chain_allocation(allocation, pool)
    assert "original explanation" in out["rationale"]


def test_clipping_handles_raw_passthrough():
    """When the LLM produced un-parseable output, allocator returns
    {'narrative': ..., 'raw': True} — clipping must not touch it."""
    raw = {"narrative": "free text", "raw": True}
    out = _clip_cold_chain_allocation(raw, BASE_RESOURCE_POOL)
    assert out is raw


def test_clipping_also_caps_standard_trucks():
    """The bug we just fixed: clipping must enforce the standard-truck cap,
    not just the cold-chain cap."""
    pool = BASE_RESOURCE_POOL  # std cap = 4 per day
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 4, "driver": 5},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
        },
        "rationale": "",
    }
    # Day0 std total = 4 + 1 = 5 (over cap of 4) → must clip 1 std truck from C1
    out = _clip_resource_allocation(allocation, pool)
    day0_std_total = sum(v["truck_standard"] for v in out["Day0"].values())
    assert day0_std_total == 4
    assert "standard truck" in out["rationale"].lower()


def test_clipping_caps_drivers():
    """Drivers are also a scarce resource that must be capped."""
    pool = BASE_RESOURCE_POOL  # driver cap = 6 per day
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 2, "driver": 5},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 2, "driver": 5},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 2, "driver": 3},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 2, "driver": 3},
        },
        "rationale": "",
    }
    # Day0 drivers total = 10 (over cap of 6) → must clip 4
    out = _clip_resource_allocation(allocation, pool)
    day0_driver_total = sum(v["driver"] for v in out["Day0"].values())
    assert day0_driver_total == 6


def test_clipping_normalises_drivers_plural_to_singular():
    """The LLM occasionally writes 'drivers' instead of 'driver' — the
    clipping function must normalise so the cap still applies."""
    pool = BASE_RESOURCE_POOL
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 2, "drivers": 8},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 2, "drivers": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 2, "drivers": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 2, "drivers": 0},
        },
        "rationale": "",
    }
    out = _clip_resource_allocation(allocation, pool)
    # After normalisation+clip: Day0 driver total ≤ 6
    day0_driver_total = sum(int(v.get("driver", 0)) for v in out["Day0"].values())
    assert day0_driver_total == 6


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


# ---------------------------------------------------------------------------
# Deferral breakdown — the structured payload the report agent must quote
# ---------------------------------------------------------------------------
def test_deferral_breakdown_contains_per_corridor_per_day_rows(fake_corridor_kpis):
    allocation = _full_supply(fake_corridor_kpis)
    out = _recompute_penalty(allocation, fake_corridor_kpis)
    assert "deferral_breakdown" in out
    assert len(out["deferral_breakdown"]) == len(fake_corridor_kpis)
    for row in out["deferral_breakdown"]:
        for k in ("corridor_id", "day", "tier1_cold_deferred", "tier2_cold_deferred",
                  "tier2_standard_deferred", "deferred_total", "penalty_pts",
                  "tier1_cold_demand", "tier1_cold_dispatched"):
            assert k in row, f"breakdown row missing {k}"


def test_deferral_summary_flags_tier1_protected_correctly(fake_corridor_kpis):
    allocation = _full_supply(fake_corridor_kpis)
    out = _recompute_penalty(allocation, fake_corridor_kpis)
    summary = out["deferral_summary"]
    assert summary["tier1_protected"] is True
    assert summary["tier1_units_deferred"] == 0


def test_deferral_summary_flags_tier1_unprotected_when_starved(fake_corridor_kpis):
    """When cold-chain capacity is starved, Tier-1 units defer and the summary
    must say so — this is the safeguard against the 'Tier 1 fully covered' lie."""
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 99, "driver": 99},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 99, "driver": 99},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 99, "truck_standard": 99, "driver": 99},
            "C2_NJ_PHL":     {"truck_temp_controlled": 99, "truck_standard": 99, "driver": 99},
        },
        "rationale": "",
    }
    out = _recompute_penalty(allocation, fake_corridor_kpis)
    summary = out["deferral_summary"]

    assert summary["tier1_protected"] is False
    assert summary["tier1_units_deferred"] > 0
    # Day0 Tier-1 cold demand = 8 + 3 = 11
    assert summary["tier1_units_deferred"] == 11
    assert "Tier-1" in summary["headline"]
    assert "deferred" in summary["headline"]
