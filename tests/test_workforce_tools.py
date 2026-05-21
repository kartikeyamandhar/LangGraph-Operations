"""
Workforce tools tests — eligibility, certification counts, fatigue logic,
and the deterministic realism check on the final allocation.

The persistence helpers (append_manager_rating, append_outcome) are tested
against a tmp_path fixture so we never touch the real feedback/ files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from tools.workforce_tools import (
    DriverProfile, WorkforceState,
    MAX_HOURS_LAST_7D, MAX_CONSECUTIVE_DAYS,
    load_workforce_state, load_manager_ratings, load_outcome_log,
    append_manager_rating, append_outcome,
    apply_workforce_to_pool, realism_check_allocation,
    compute_calibration,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------
def _driver(driver_id: str, **kwargs) -> DriverProfile:
    defaults = {
        "name": f"Test {driver_id}",
        "certifications": ["cdl"],
        "hours_last_24h": 8.0,
        "hours_last_7d": 30.0,
        "consecutive_days": 2,
        "fatigue_flag": False,
        "preferred_corridors": ["C1_I95_NJ_BOS"],
        "active": True,
        "notes": "",
    }
    defaults.update(kwargs)
    return DriverProfile(driver_id=driver_id, **defaults)


def _pool(driver=6, std=4, cold=2):
    return {
        "Day0": {"driver": driver, "truck_standard": std, "truck_temp_controlled": cold},
        "Day1": {"driver": driver, "truck_standard": std, "truck_temp_controlled": cold},
    }


# ---------------------------------------------------------------------------
# DriverProfile eligibility
# ---------------------------------------------------------------------------
def test_eligible_baseline_driver():
    d = _driver("D-1")
    assert d.eligible_today is True
    assert d.status_label == "Eligible"


def test_inactive_driver_is_ineligible():
    d = _driver("D-1", active=False)
    assert d.eligible_today is False
    assert d.status_label == "On leave"


def test_driver_at_or_over_weekly_cap_is_ineligible():
    d = _driver("D-1", hours_last_7d=MAX_HOURS_LAST_7D)
    assert d.eligible_today is False
    assert d.status_label == "Over weekly cap"


def test_driver_just_under_weekly_cap_is_eligible():
    d = _driver("D-1", hours_last_7d=MAX_HOURS_LAST_7D - 0.1)
    assert d.eligible_today is True


def test_driver_over_consecutive_day_limit_is_ineligible():
    d = _driver("D-1", consecutive_days=MAX_CONSECUTIVE_DAYS)
    assert d.eligible_today is False
    assert d.status_label == "Mandatory rest day"


def test_fatigue_flag_does_not_remove_eligibility_but_warns():
    d = _driver("D-1", fatigue_flag=True)
    assert d.eligible_today is True
    assert "Fatigue flag" in d.status_label


def test_cold_chain_certification_check():
    d_cold = _driver("D-1", certifications=["cdl", "cold_chain"])
    d_std  = _driver("D-2", certifications=["cdl"])
    assert d_cold.cold_chain_certified is True
    assert d_std.cold_chain_certified is False


# ---------------------------------------------------------------------------
# WorkforceState aggregation
# ---------------------------------------------------------------------------
def test_workforce_state_counts_eligible_correctly():
    ws = WorkforceState(drivers=[
        _driver("D-1"),
        _driver("D-2", active=False),
        _driver("D-3", hours_last_7d=42.0),
        _driver("D-4", certifications=["cdl", "cold_chain"]),
    ])
    assert ws.eligible_count == 2          # D-1, D-4
    assert ws.cold_chain_eligible_count == 1   # D-4 only
    assert len(ws.ineligible) == 2


def test_workforce_state_to_dict_is_serialisable():
    ws = WorkforceState(drivers=[_driver("D-1"), _driver("D-2", certifications=["cdl", "cold_chain"])])
    d = ws.to_dict()
    import json; json.dumps(d)  # must not raise
    assert d["eligible_count"] == 2
    assert d["cold_chain_eligible_count"] == 1
    assert "summary" in d


# ---------------------------------------------------------------------------
# CSV loaders against the seeded feedback/ data
# ---------------------------------------------------------------------------
def test_load_workforce_state_parses_eligibility_rules(tmp_path: Path):
    """Decoupled from the live feedback/ CSV (which is an operational file the
    user edits). Writes a known roster to tmp_path and asserts the parser
    applies every eligibility rule correctly."""
    csv = tmp_path / "driver_state.csv"
    csv.write_text(
        "driver_id,name,certifications,hours_last_24h,hours_last_7d,"
        "consecutive_days,fatigue_flag,preferred_corridors,active,notes\n"
        "D-1,Alice,cold_chain;cdl,8,28,2,false,C1_I95_NJ_BOS,true,senior\n"
        "D-2,Bob,cdl,10,35,3,false,C1_I95_NJ_BOS,true,std\n"
        "D-3,Carla,cold_chain;cdl,11,38,4,true,C2_NJ_PHL,true,fatigued\n"
        "D-7,Grace,cold_chain;cdl,0,0,0,false,C1_I95_NJ_BOS,false,on leave\n"
        "D-8,Henry,cdl,12,42,5,true,C2_NJ_PHL,true,over cap\n"
    )
    ws = load_workforce_state(str(csv))
    assert len(ws.drivers) == 5

    eligible_ids = {d.driver_id for d in ws.eligible}
    assert "D-7" not in eligible_ids   # active=false
    assert "D-8" not in eligible_ids   # 42h ≥ 40h cap
    assert "D-3" in eligible_ids       # fatigued but eligible

    cold_eligible = {d.driver_id for d in ws.eligible if d.cold_chain_certified}
    assert cold_eligible == {"D-1", "D-3"}


def test_load_workforce_state_reads_live_file_without_crashing():
    """Smoke test: the live operational file must always parse, whatever the
    user has edited it to. We assert structure, not specific driver IDs."""
    ws = load_workforce_state("feedback/driver_state.csv")
    assert len(ws.drivers) >= 1
    # Eligibility partitions must be consistent
    assert ws.eligible_count == len(ws.eligible)
    assert ws.cold_chain_eligible_count <= ws.eligible_count
    d = ws.to_dict()
    import json; json.dumps(d)  # serialisable for AppState


def test_load_manager_ratings_from_seeded_data():
    ratings = load_manager_ratings("feedback/manager_ratings.csv", last_n=10)
    assert len(ratings) >= 1
    # Most recent first
    if len(ratings) >= 2:
        assert ratings[0]["timestamp"] >= ratings[-1]["timestamp"]


def test_load_outcome_log_from_seeded_data():
    df = load_outcome_log("feedback/outcome_log.csv")
    assert not df.empty
    assert {"predicted_penalty", "actual_penalty"} <= set(df.columns)


# ---------------------------------------------------------------------------
# apply_workforce_to_pool — the workforce-reality reduction
# ---------------------------------------------------------------------------
def test_apply_workforce_to_pool_reduces_drivers_when_short():
    # 3 eligible drivers vs base pool of 6
    ws = WorkforceState(drivers=[
        _driver("D-1"), _driver("D-2"), _driver("D-3"),
        _driver("D-4", active=False),
        _driver("D-5", hours_last_7d=42),
        _driver("D-6", consecutive_days=5),
    ])
    pool = _pool(driver=6, cold=2)
    new_pool, changes = apply_workforce_to_pool(pool, ws)

    assert new_pool["Day0"]["driver"] == 3
    assert new_pool["Day1"]["driver"] == 3
    assert any(c["resource"] == "driver" and c["from"] == 6 and c["to"] == 3 for c in changes)


def test_apply_workforce_to_pool_caps_cold_trucks_by_certified_drivers():
    # Only 1 cold-chain certified eligible driver — must cap cold trucks at 1
    ws = WorkforceState(drivers=[
        _driver("D-1", certifications=["cdl", "cold_chain"]),
        _driver("D-2"),
        _driver("D-3"),
        _driver("D-4", certifications=["cdl", "cold_chain"], active=False),  # ineligible
    ])
    pool = _pool(driver=6, cold=2)
    new_pool, changes = apply_workforce_to_pool(pool, ws)

    assert new_pool["Day0"]["truck_temp_controlled"] == 1
    assert new_pool["Day1"]["truck_temp_controlled"] == 1
    assert any(c["resource"] == "truck_temp_controlled" and "certified" in c["reason"].lower()
               for c in changes)


def test_apply_workforce_to_pool_no_change_when_supply_sufficient():
    # Plenty of eligible + certified drivers → pool unchanged
    ws = WorkforceState(drivers=[
        _driver(f"D-{i}", certifications=["cdl", "cold_chain"]) for i in range(1, 9)
    ])
    pool = _pool(driver=6, cold=2)
    new_pool, changes = apply_workforce_to_pool(pool, ws)

    assert new_pool == pool
    assert changes == []


def test_apply_workforce_to_pool_handles_empty_workforce():
    ws = WorkforceState(drivers=[])
    pool = _pool()
    new_pool, changes = apply_workforce_to_pool(pool, ws)
    assert new_pool == pool
    assert changes == []


# ---------------------------------------------------------------------------
# realism_check_allocation
# ---------------------------------------------------------------------------
def test_realism_passes_when_allocation_respects_eligible_count():
    ws = WorkforceState(drivers=[
        _driver("D-1", certifications=["cdl", "cold_chain"]),
        _driver("D-2", certifications=["cdl", "cold_chain"]),
        _driver("D-3"), _driver("D-4"), _driver("D-5"), _driver("D-6"),
    ])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
        },
        "rationale": "",
    }
    warnings, violations = realism_check_allocation(allocation, ws, pool)
    assert violations == []  # no violations
    # No "all eligible committed" warning because cap > usage


def test_realism_violates_when_more_drivers_allocated_than_eligible():
    ws = WorkforceState(drivers=[
        _driver("D-1"), _driver("D-2", active=False), _driver("D-3", active=False),
    ])  # only 1 eligible
    pool = {
        "Day0": {"driver": 1, "truck_standard": 4, "truck_temp_controlled": 0},
        "Day1": {"driver": 1, "truck_standard": 4, "truck_temp_controlled": 0},
    }
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 3},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 0},
        },
        "rationale": "",
    }
    _, violations = realism_check_allocation(allocation, ws, pool)
    # Day0 allocates 3 drivers but cap = 1 → violation
    assert any("3 drivers allocated" in v for v in violations)


def test_realism_violates_cold_trucks_without_certified_drivers():
    ws = WorkforceState(drivers=[
        _driver("D-1"),  # not cold-chain certified
        _driver("D-2"),
    ])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 0, "driver": 1},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "rationale": "",
    }
    _, violations = realism_check_allocation(allocation, ws, pool)
    assert any("cold-chain trucks allocated" in v and "certified" in v for v in violations)


def test_realism_warns_about_fatigued_drivers_in_eligible_pool():
    ws = WorkforceState(drivers=[
        _driver("D-1"),
        _driver("D-2", fatigue_flag=True),
    ])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "rationale": "",
    }
    warnings, _ = realism_check_allocation(allocation, ws, pool)
    assert any("fatigue" in w.lower() and "D-2" in w for w in warnings)


def test_realism_warns_when_all_eligible_drivers_committed():
    ws = WorkforceState(drivers=[_driver(f"D-{i}") for i in range(1, 4)])  # 3 eligible
    pool = _pool(driver=3)
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "rationale": "",
    }
    warnings, _ = realism_check_allocation(allocation, ws, pool)
    assert any("no slack" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Trucks-need-drivers constraint (new)
# ---------------------------------------------------------------------------
def test_realism_violates_when_corridor_has_trucks_but_no_drivers():
    """A corridor allocated 1+ trucks but 0 drivers is infeasible —
    trucks cannot dispatch without a driver."""
    ws = WorkforceState(drivers=[
        _driver("D-1", certifications=["cdl", "cold_chain"]),
        _driver("D-2", certifications=["cdl", "cold_chain"]),
    ])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 0},  # broken
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "rationale": "",
    }
    _, violations = realism_check_allocation(allocation, ws, pool)
    assert any("C2_NJ_PHL" in v and "0 drivers" in v and "truck" in v.lower() for v in violations)


def test_realism_passes_when_each_corridor_with_trucks_has_a_driver():
    ws = WorkforceState(drivers=[_driver(f"D-{i}") for i in range(1, 7)])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 1, "driver": 1},
        },
        "rationale": "",
    }
    _, violations = realism_check_allocation(allocation, ws, pool)
    # No 'trucks without drivers' violation
    assert not any("truck" in v.lower() and "without a driver" in v.lower() for v in violations)


def test_realism_warns_when_drivers_short_of_truck_count_on_corridor():
    """If a corridor has 3 trucks but 1 driver, the trucks share a driver —
    flag for confirmation (warning, not violation)."""
    ws = WorkforceState(drivers=[_driver(f"D-{i}") for i in range(1, 7)])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 2, "driver": 1},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "rationale": "",
    }
    warnings, _ = realism_check_allocation(allocation, ws, pool)
    assert any("share" in w.lower() and "driver" in w.lower() for w in warnings)


def test_realism_no_warning_when_corridor_has_no_trucks_and_no_drivers():
    """Zero trucks AND zero drivers is fine — corridor not active that day."""
    ws = WorkforceState(drivers=[_driver(f"D-{i}") for i in range(1, 7)])
    pool = _pool()
    allocation = {
        "Day0": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 1, "truck_standard": 1, "driver": 2},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "Day1": {
            "C1_I95_NJ_BOS": {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
            "C2_NJ_PHL":     {"truck_temp_controlled": 0, "truck_standard": 0, "driver": 0},
        },
        "rationale": "",
    }
    warnings, violations = realism_check_allocation(allocation, ws, pool)
    # C2 has 0 trucks AND 0 drivers — should not trigger any truck-driver warning/violation
    assert not any("0 drivers" in v for v in violations)
    assert not any("share" in w.lower() and "driver" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Persistence helpers (use tmp_path so we never touch real feedback/)
# ---------------------------------------------------------------------------
def test_append_manager_rating_creates_file_with_header(tmp_path: Path):
    target = tmp_path / "ratings.csv"
    append_manager_rating(
        run_id="run-1", manager_id="M-1", scenario="test",
        star_rating=4, tags=["Right-sized"], comment="ok",
        path=str(target),
    )
    assert target.exists()
    df = pd.read_csv(target)
    assert len(df) == 1
    assert df.iloc[0]["star_rating"] == 4


def test_append_manager_rating_appends_to_existing(tmp_path: Path):
    target = tmp_path / "ratings.csv"
    append_manager_rating("run-1", "M-1", "s", 3, ["a"], "first", path=str(target))
    append_manager_rating("run-2", "M-1", "s", 5, ["b"], "second", path=str(target))
    df = pd.read_csv(target)
    assert len(df) == 2


def test_append_outcome_creates_file(tmp_path: Path):
    target = tmp_path / "outcomes.csv"
    append_outcome(
        run_id="run-1", scenario="baseline",
        predicted_penalty=100, actual_penalty=120,
        predicted_deferred=2, actual_deferred=3,
        actual_tier1_late=0, actual_tier2_late=2,
        actual_cold_chain_breaches=0, incident_notes="",
        path=str(target),
    )
    df = pd.read_csv(target)
    assert df.iloc[0]["actual_penalty"] == 120


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def test_compute_calibration_with_no_data_returns_empty_headline(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    out = compute_calibration(path=str(empty))
    assert out["runs_n"] == 0
    assert "not available" in out["headline"].lower() or "no" in out["headline"].lower()


def test_compute_calibration_from_seeded_outcomes():
    out = compute_calibration("feedback/outcome_log.csv")
    assert out["runs_n"] >= 5
    assert "penalty_mae" in out
    assert "penalty_bias" in out
    assert "headline" in out


def test_compute_calibration_detects_under_prediction(tmp_path: Path):
    target = tmp_path / "outcomes.csv"
    # System consistently under-predicted → bias > 0
    rows = [
        {"timestamp": f"2026-05-{i:02d}", "run_id": f"r{i}", "scenario": "x",
         "predicted_penalty": 100, "actual_penalty": 200,
         "predicted_deferred": 1, "actual_deferred": 2,
         "actual_tier1_late": 0, "actual_tier2_late": 1,
         "actual_cold_chain_breaches": 0, "incident_notes": ""}
        for i in range(1, 6)
    ]
    pd.DataFrame(rows).to_csv(target, index=False)
    out = compute_calibration(path=str(target))
    assert out["penalty_bias"] == 100
    assert "under-predicted" in out["headline"]
