"""
CSV reconciliation + KPI tests.

Every decision rule from the Item Master Appendix (D1-D6, DQ-01..04) is
exercised. We test the row-level _reconcile_row directly with synthetic
pd.Series rows so the tests are deterministic and don't need a CSV file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.csv_tools import (
    CANONICAL_MASTER, NAME_ALIASES, LEGACY_ID_MAP,
    SLA_TIER, COLD_CHAIN_CLASSES,
    ReconciliationLog,
    _reconcile_row,
    _compute_corridor_kpis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row(item_id=None, item_name="", uid=None) -> pd.Series:
    return pd.Series({
        "item_id": item_id,
        "item_name": item_name,
        "unique_item_id": uid,
        "corridor_id": "C1_I95_NJ_BOS",
        "planning_day": "Day0",
    })


# ---------------------------------------------------------------------------
# D3: Exact match on item_id
# ---------------------------------------------------------------------------
def test_exact_match_canonical_id_and_name():
    log = ReconciliationLog()
    out = _reconcile_row(_row(10021, "Remdesivir", "U-1"), set(), log)

    assert log.exact_match == 1
    assert out["reconcile_status"] == "exact_match"
    assert out["canonical_item_id"] == "RMD"
    assert out["product_class"] == "Antiviral"
    assert out["sla_tier"] == 1
    assert out["needs_cold_chain"] is True
    assert out["excluded"] is False


def test_exact_match_with_name_mismatch_is_flagged_not_excluded():
    """DQ-03 — keep the row but log the name mismatch."""
    log = ReconciliationLog()
    out = _reconcile_row(_row(10021, "Bogus Brand X", "U-2"), set(), log)

    assert log.dq03_name_mismatch == 1
    assert out["reconcile_status"] == "exact_match_name_mismatch"
    assert out["excluded"] is False
    assert out["product_class"] == "Antiviral"


# ---------------------------------------------------------------------------
# D4: Alias match on item_name
# ---------------------------------------------------------------------------
def test_alias_resolves_to_canonical_id():
    log = ReconciliationLog()
    # "remdesivir 100 mg" is an alias for 10021 in NAME_ALIASES
    out = _reconcile_row(_row(None, "Remdesivir 100 mg", "U-3"), set(), log)

    assert log.alias_match == 1
    assert out["item_id"] == 10021
    assert out["reconcile_status"] == "alias_match"
    assert out["canonical_item_id"] == "RMD"


def test_alias_match_is_case_insensitive():
    log = ReconciliationLog()
    out = _reconcile_row(_row(None, "PEMBROLIZUMAB (KEYTRUDA)", "U-4"), set(), log)

    assert log.alias_match == 1
    assert out["item_id"] == 10035
    assert out["product_class"] == "Oncology Biologic"


# ---------------------------------------------------------------------------
# D5: Legacy ID map
# ---------------------------------------------------------------------------
def test_legacy_id_remaps_then_resolves():
    log = ReconciliationLog()
    # 10020 → 10021 in LEGACY_ID_MAP
    out = _reconcile_row(_row(10020, "Remdesivir", "U-5"), set(), log)

    assert log.legacy_id_map == 1
    assert log.exact_match == 1
    assert out["item_id"] == 10021
    assert out["canonical_item_id"] == "RMD"


# ---------------------------------------------------------------------------
# DQ-01: missing unique_item_id → exclude
# ---------------------------------------------------------------------------
def test_dq01_missing_uid_excludes_row():
    log = ReconciliationLog()
    out = _reconcile_row(_row(10021, "Remdesivir", None), set(), log)

    assert log.dq01_missing_uid == 1
    assert log.excluded == 1
    assert out["excluded"] is True
    assert out["exclusion_reason"] == "DQ-01"


def test_dq01_empty_string_uid_excludes_row():
    log = ReconciliationLog()
    out = _reconcile_row(_row(10021, "Remdesivir", "   "), set(), log)

    assert log.dq01_missing_uid == 1
    assert out["excluded"] is True


# ---------------------------------------------------------------------------
# DQ-02: unknown item_id → flag, not exclude
# ---------------------------------------------------------------------------
def test_dq02_unknown_item_id_flags_but_keeps_row():
    log = ReconciliationLog()
    out = _reconcile_row(_row(99000, "MysteryDrug", "U-6"), set(), log)

    assert log.dq02_unknown_item_id == 1
    assert out["reconcile_status"] == "dq02_unknown"
    # Unknown — no canonical mapping was applied
    assert out["canonical_item_id"] is None
    # Per current behaviour DQ-02 alone does not exclude (only DQ-01/04 do)
    assert out["excluded"] is False


# ---------------------------------------------------------------------------
# DQ-04: duplicate unique_item_id → exclude the second occurrence
# ---------------------------------------------------------------------------
def test_dq04_duplicate_uid_excludes_second_occurrence():
    log = ReconciliationLog()
    seen = set()

    first = _reconcile_row(_row(10021, "Remdesivir", "DUP-1"), seen, log)
    second = _reconcile_row(_row(10022, "Insulin Lispro", "DUP-1"), seen, log)

    assert first["excluded"] is False
    assert second["excluded"] is True
    assert second["exclusion_reason"] == "DQ-04"
    assert log.dq04_duplicate_uid == 1


# ---------------------------------------------------------------------------
# Tier and cold-chain attribution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("item_id, expected_tier, expected_cold", [
    (10021, 1, True),    # Antiviral  — Tier 1, cold
    (10035, 1, True),    # Oncology   — Tier 1, cold
    (99999, 1, True),    # Clinical Trial — Tier 1, cold
    (10022, 2, True),    # Insulin    — Tier 2 but still cold
    (10040, 2, False),   # EpiPen     — Tier 2, room temp
    (10050, 2, False),   # Heparin    — Tier 2, room temp
])
def test_sla_tier_and_cold_chain_attribution(item_id, expected_tier, expected_cold):
    log = ReconciliationLog()
    canonical_name = CANONICAL_MASTER[item_id]["canonical_item_name"]
    out = _reconcile_row(_row(item_id, canonical_name, "U-x"), set(), log)
    assert out["sla_tier"] == expected_tier
    assert out["needs_cold_chain"] is expected_cold


# ---------------------------------------------------------------------------
# Static invariants on the embedded item master
# ---------------------------------------------------------------------------
def test_sla_tier_table_covers_every_product_class():
    classes_in_master = {info["product_class"] for info in CANONICAL_MASTER.values()}
    assert classes_in_master <= set(SLA_TIER.keys()), (
        f"SLA_TIER missing entries for: {classes_in_master - set(SLA_TIER.keys())}"
    )


def test_cold_chain_classes_subset_of_known_classes():
    classes_in_master = {info["product_class"] for info in CANONICAL_MASTER.values()}
    assert COLD_CHAIN_CLASSES <= classes_in_master


def test_legacy_ids_resolve_to_known_canonicals():
    for legacy, canonical in LEGACY_ID_MAP.items():
        assert canonical in CANONICAL_MASTER, (
            f"Legacy ID {legacy} maps to unknown canonical {canonical}"
        )


def test_aliases_resolve_to_known_canonicals():
    for alias, canonical in NAME_ALIASES.items():
        assert canonical in CANONICAL_MASTER, (
            f"Alias {alias!r} maps to unknown canonical {canonical}"
        )


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------
def _reconciled_df(rows: list) -> pd.DataFrame:
    """Build the dataframe shape that _compute_corridor_kpis expects."""
    return pd.DataFrame(rows)


def test_kpi_truck_calculation_uses_10pct_buffer_and_capacity_10():
    """Playbook §6: trucks = ceil(units * 1.10 / 10)."""
    df = _reconciled_df([
        # 10 cold-chain valid units → ceil(10*1.10/10) = 2 trucks
        *[{"excluded": False, "needs_cold_chain": True,
           "temp_control": "Cold (2-8C)", "sla_tier": 1, "product_class": "Antiviral"}
          for _ in range(10)],
        # 5 room-temp valid units → ceil(5*1.10/10) = 1 truck
        *[{"excluded": False, "needs_cold_chain": False,
           "temp_control": "Room Temp (20-25C)", "sla_tier": 2, "product_class": "Emergency"}
          for _ in range(5)],
    ])
    kpi = _compute_corridor_kpis(df, "C1_I95_NJ_BOS", "Day0", {})

    assert kpi.cold_chain_units == 10
    assert kpi.room_temp_units == 5
    assert kpi.trucks_needed_cold_chain == 2
    assert kpi.trucks_needed_standard == 1
    assert kpi.tier1_units == 10
    assert kpi.tier2_units == 5


def test_kpi_excluded_rows_dont_count_toward_demand():
    df = _reconciled_df([
        {"excluded": False, "needs_cold_chain": True,
         "temp_control": "Cold (2-8C)", "sla_tier": 1, "product_class": "Antiviral"},
        {"excluded": True, "needs_cold_chain": True,
         "temp_control": "Cold (2-8C)", "sla_tier": 1, "product_class": "Antiviral"},
    ])
    kpi = _compute_corridor_kpis(df, "C1_I95_NJ_BOS", "Day0", {})

    assert kpi.total_rows == 2
    assert kpi.valid_rows == 1
    assert kpi.excluded_rows == 1
    assert kpi.exclusion_rate_pct == 50.0
    assert kpi.cold_chain_units == 1


def test_kpi_zero_demand_returns_zero_trucks():
    """All-excluded rows means no valid demand → 0 trucks."""
    df = _reconciled_df([
        {"excluded": True, "needs_cold_chain": False,
         "temp_control": "Room Temp (20-25C)", "sla_tier": 2, "product_class": "Emergency"},
    ])
    kpi = _compute_corridor_kpis(df, "C1_I95_NJ_BOS", "Day0", {})
    assert kpi.valid_rows == 0
    assert kpi.trucks_needed_cold_chain == 0
    assert kpi.trucks_needed_standard == 0
