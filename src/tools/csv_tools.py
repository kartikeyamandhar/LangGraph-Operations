from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Appendix A — Item Master (embedded so agents don't need to re-read the PDF)
# ---------------------------------------------------------------------------

# A.1 Canonical master: item_id → canonical info
CANONICAL_MASTER: Dict[int, Dict[str, str]] = {
    10021: {"canonical_item_id": "RMD",    "canonical_item_name": "Remdesivir",             "temp_control": "Cold (2-8C)",          "product_class": "Antiviral"},
    10022: {"canonical_item_id": "INS-LIS","canonical_item_name": "Insulin Lispro",          "temp_control": "Cold (2-8C)",          "product_class": "Endocrine"},
    10023: {"canonical_item_id": "INS-ASP","canonical_item_name": "Insulin Aspart",          "temp_control": "Cold (2-8C)",          "product_class": "Endocrine"},
    10035: {"canonical_item_id": "PMB-KEY","canonical_item_name": "Pembrolizumab",           "temp_control": "Cold (2-8C)",          "product_class": "Oncology Biologic"},
    10040: {"canonical_item_id": "EPI-AI", "canonical_item_name": "Epinephrine Auto-Injector","temp_control": "Room Temp (20-25C)",   "product_class": "Emergency"},
    10050: {"canonical_item_id": "HEP-SOD","canonical_item_name": "Heparin Sodium",          "temp_control": "Room Temp (20-25C)",   "product_class": "Anticoagulant"},
    10060: {"canonical_item_id": "MOR-SUL","canonical_item_name": "Morphine Sulfate",        "temp_control": "Controlled Storage",   "product_class": "Controlled"},
    10070: {"canonical_item_id": "ALB-INH","canonical_item_name": "Albuterol Inhaler",       "temp_control": "Room Temp (20-25C)",   "product_class": "Respiratory"},
    10071: {"canonical_item_id": "LEV-INH","canonical_item_name": "Levalbuterol Inhaler",    "temp_control": "Room Temp (20-25C)",   "product_class": "Respiratory"},
    99999: {"canonical_item_id": "EXP-ONC","canonical_item_name": "Experimental Oncology Drug","temp_control": "Strict Cold Chain (-20C)","product_class": "Clinical Trial"},
}

# A.2 Name aliases → canonical item_id
NAME_ALIASES: Dict[str, int] = {
    "remdesivir 100 mg": 10021,
    "remdesivir 200 mg": 10021,
    "pembrolizumab (keytruda)": 10035,
    "epipen auto injector": 10040,
    "heparin na": 10050,
    "morphine sulphate": 10060,
    "albuterol inhaler 90mcg": 10070,
}

# A.3 Legacy item_id → canonical item_id
LEGACY_ID_MAP: Dict[int, int] = {
    10020: 10021,
    20021: 10021,
    1070:  10070,
}

# SLA tier by product class
SLA_TIER: Dict[str, int] = {
    "Antiviral": 1,
    "Oncology Biologic": 1,
    "Clinical Trial": 1,
    "Endocrine": 2,
    "Emergency": 2,
    "Anticoagulant": 2,
    "Controlled": 2,
    "Respiratory": 2,
}

COLD_CHAIN_CLASSES = {"Antiviral", "Endocrine", "Oncology Biologic", "Clinical Trial"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationLog:
    exact_match: int = 0
    alias_match: int = 0
    legacy_id_map: int = 0
    dq01_missing_uid: int = 0
    dq02_unknown_item_id: int = 0
    dq03_name_mismatch: int = 0
    dq04_duplicate_uid: int = 0
    excluded: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "exact_match": self.exact_match,
            "alias_match": self.alias_match,
            "legacy_id_map": self.legacy_id_map,
            "DQ-01 missing unique_item_id": self.dq01_missing_uid,
            "DQ-02 unknown item_id": self.dq02_unknown_item_id,
            "DQ-03 name mismatch": self.dq03_name_mismatch,
            "DQ-04 duplicate unique_item_id": self.dq04_duplicate_uid,
            "total_excluded": self.excluded,
        }


@dataclass
class CorridorKPIs:
    corridor_id: str
    day: str
    total_rows: int
    valid_rows: int
    excluded_rows: int
    exclusion_rate_pct: float
    cold_chain_units: int
    room_temp_units: int
    controlled_units: int
    tier1_units: int
    tier2_units: int
    trucks_needed_standard: int
    trucks_needed_cold_chain: int
    dq_breakdown: Dict[str, int]


@dataclass
class CsvAnalysisResult:
    # Raw data splits
    planning_df: pd.DataFrame          # Day0 + Day1, reconciled, valid only
    history_df: pd.DataFrame           # History rows for trend analysis
    all_reconciled_df: pd.DataFrame    # Full reconciled dataframe

    # Per-corridor per-day KPIs
    corridor_kpis: List[CorridorKPIs]

    # Trend: corridor-level aggregates over history
    trend_summary: Dict[str, Any]

    # Reconciliation log
    recon_log: ReconciliationLog

    # Human-readable summary strings for agents
    summary: Dict[str, Any]
    kpis: Dict[str, Any]
    anomalies_md: str


# ---------------------------------------------------------------------------
# Core reconciliation logic (Appendix A decision rules D1–D6)
# ---------------------------------------------------------------------------

def _reconcile_row(row: pd.Series, seen_uids: set, log: ReconciliationLog) -> pd.Series:
    item_id = row.get("item_id")
    item_name = str(row.get("item_name", "")).strip()
    uid = row.get("unique_item_id")

    row = row.copy()
    row["reconcile_status"] = "unresolved"
    row["canonical_item_id"] = None
    row["canonical_item_name"] = None
    row["temp_control"] = None
    row["product_class"] = None
    row["sla_tier"] = None
    row["needs_cold_chain"] = False
    row["excluded"] = False
    row["exclusion_reason"] = None

    # D5: Legacy ID map
    if pd.notna(item_id) and int(item_id) in LEGACY_ID_MAP:
        item_id = LEGACY_ID_MAP[int(item_id)]
        row["item_id"] = item_id
        log.legacy_id_map += 1

    # D3: Exact match on item_id
    item_id_int = int(item_id) if pd.notna(item_id) else None
    if item_id_int and item_id_int in CANONICAL_MASTER:
        master = CANONICAL_MASTER[item_id_int]
        # Check name mismatch (D3 — flag but don't exclude)
        canonical_name = master["canonical_item_name"].lower()
        if item_name.lower() not in (canonical_name, NAME_ALIASES.get(item_name.lower(), canonical_name)):
            log.dq03_name_mismatch += 1
            row["reconcile_status"] = "exact_match_name_mismatch"
        else:
            row["reconcile_status"] = "exact_match"
            log.exact_match += 1
        row.update(master)
    # D4: Alias match on item_name
    elif item_name.lower() in NAME_ALIASES:
        resolved_id = NAME_ALIASES[item_name.lower()]
        master = CANONICAL_MASTER[resolved_id]
        row["item_id"] = resolved_id
        row["reconcile_status"] = "alias_match"
        row.update(master)
        log.alias_match += 1
    else:
        # DQ-02: unknown item_id
        log.dq02_unknown_item_id += 1
        row["reconcile_status"] = "dq02_unknown"

    # Assign SLA tier and cold chain flag
    pc = row.get("product_class")
    if pc:
        row["sla_tier"] = SLA_TIER.get(pc, 2)
        row["needs_cold_chain"] = pc in COLD_CHAIN_CLASSES

    # DQ-01: missing unique_item_id — exclude from dispatch
    if pd.isna(uid) or str(uid).strip() == "":
        log.dq01_missing_uid += 1
        log.excluded += 1
        row["excluded"] = True
        row["exclusion_reason"] = "DQ-01"
        return row

    # DQ-04: duplicate unique_item_id
    if uid in seen_uids:
        log.dq04_duplicate_uid += 1
        log.excluded += 1
        row["excluded"] = True
        row["exclusion_reason"] = "DQ-04"
        return row

    seen_uids.add(uid)
    return row


# ---------------------------------------------------------------------------
# KPI computation per corridor per day
# ---------------------------------------------------------------------------

def _compute_corridor_kpis(df: pd.DataFrame, corridor_id: str, day: str, dq_log: Dict[str, int]) -> CorridorKPIs:
    valid = df[~df["excluded"]]
    total = len(df)
    n_valid = len(valid)
    n_excluded = total - n_valid

    cold = valid[valid["needs_cold_chain"] == True]
    room = valid[valid["temp_control"] == "Room Temp (20-25C)"]
    ctrl = valid[valid["temp_control"] == "Controlled Storage"]
    tier1 = valid[valid["sla_tier"] == 1]
    tier2 = valid[valid["sla_tier"] == 2]

    cold_units = len(cold)
    room_units = len(room)
    ctrl_units = len(ctrl)

    # Truck calculation: ceil(volume * 1.10 / 10)
    trucks_cold = int(np.ceil((cold_units * 1.10) / 10)) if cold_units > 0 else 0
    trucks_std = int(np.ceil(((room_units + ctrl_units) * 1.10) / 10)) if (room_units + ctrl_units) > 0 else 0

    return CorridorKPIs(
        corridor_id=corridor_id,
        day=day,
        total_rows=total,
        valid_rows=n_valid,
        excluded_rows=n_excluded,
        exclusion_rate_pct=round((n_excluded / total * 100) if total > 0 else 0, 1),
        cold_chain_units=cold_units,
        room_temp_units=room_units,
        controlled_units=ctrl_units,
        tier1_units=len(tier1),
        tier2_units=len(tier2),
        trucks_needed_standard=trucks_std,
        trucks_needed_cold_chain=trucks_cold,
        dq_breakdown=dq_log,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_csv(csv_path: str) -> CsvAnalysisResult:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    # Detect if this is the enhanced multi-corridor CSV
    is_enhanced = "corridor_id" in df.columns and "is_planning_window" in df.columns

    if not is_enhanced:
        # Fallback: treat entire CSV as single-corridor Day0 planning window
        df["corridor_id"] = "C1_I95_NJ_BOS"
        df["planning_day"] = "Day0"
        df["is_planning_window"] = 1
        df["shipment_date"] = pd.NaT

    # Split history vs planning window
    planning_mask = df["is_planning_window"] == 1
    planning_raw = df[planning_mask].copy()
    history_raw = df[~planning_mask].copy()

    # Reconcile planning rows
    log = ReconciliationLog()
    seen_uids: set = set()
    reconciled_rows = [_reconcile_row(row, seen_uids, log) for _, row in planning_raw.iterrows()]
    planning_reconciled = pd.DataFrame(reconciled_rows) if reconciled_rows else pd.DataFrame()

    # Reconcile history rows (for trend — relaxed, no exclusion tracking)
    history_log = ReconciliationLog()
    history_seen: set = set()
    history_rows = [_reconcile_row(row, history_seen, history_log) for _, row in history_raw.iterrows()]
    history_reconciled = pd.DataFrame(history_rows) if history_rows else pd.DataFrame()

    all_reconciled = pd.concat([planning_reconciled, history_reconciled], ignore_index=True)

    # Compute per-corridor per-day KPIs
    corridor_kpis: List[CorridorKPIs] = []
    if not planning_reconciled.empty:
        for corridor_id in planning_reconciled["corridor_id"].unique():
            for day in planning_reconciled["planning_day"].unique():
                subset = planning_reconciled[
                    (planning_reconciled["corridor_id"] == corridor_id) &
                    (planning_reconciled["planning_day"] == day)
                ]
                if len(subset) > 0:
                    kpi = _compute_corridor_kpis(subset, corridor_id, day, log.to_dict())
                    corridor_kpis.append(kpi)

    # Trend: volume per corridor per day over history
    trend_summary: Dict[str, Any] = {}
    if not history_reconciled.empty and "shipment_date" in history_reconciled.columns:
        hist = history_reconciled[~history_reconciled["excluded"]].copy()
        if not hist.empty:
            by_corridor_date = (
                hist.groupby(["corridor_id", "shipment_date"])
                .size()
                .reset_index(name="units")
            )
            trend_summary["daily_volume_by_corridor"] = (
                by_corridor_date.groupby("corridor_id")["units"]
                .agg(["mean", "min", "max"])
                .rename(columns={"mean": "avg_daily_units", "min": "min_daily_units", "max": "max_daily_units"})
                .to_dict()
            )
            trend_summary["history_days"] = int(hist["shipment_date"].nunique()) if "shipment_date" in hist.columns else 0
            trend_summary["history_rows"] = len(hist)

    # Build summary dict for agents
    valid_planning = planning_reconciled[~planning_reconciled["excluded"]] if not planning_reconciled.empty else pd.DataFrame()
    summary = {
        "total_planning_rows": len(planning_reconciled),
        "valid_planning_rows": len(valid_planning),
        "excluded_planning_rows": int(log.excluded),
        "corridors": list(planning_reconciled["corridor_id"].unique()) if not planning_reconciled.empty else [],
        "planning_days": list(planning_reconciled["planning_day"].unique()) if not planning_reconciled.empty else [],
        "reconciliation": log.to_dict(),
        "is_enhanced_csv": is_enhanced,
    }

    # KPIs dict for agents
    kpis: Dict[str, Any] = {}
    for k in corridor_kpis:
        key = f"{k.corridor_id}__{k.day}"
        kpis[key] = {
            "valid_units": k.valid_rows,
            "excluded_units": k.excluded_rows,
            "exclusion_rate_pct": k.exclusion_rate_pct,
            "tier1_units": k.tier1_units,
            "tier2_units": k.tier2_units,
            "cold_chain_units": k.cold_chain_units,
            "trucks_needed_cold_chain": k.trucks_needed_cold_chain,
            "trucks_needed_standard": k.trucks_needed_standard,
        }

    # Anomalies markdown: DQ violations in planning window
    anomalies_md = "(none)"
    if not planning_reconciled.empty:
        excluded_df = planning_reconciled[planning_reconciled["excluded"]][
            ["corridor_id", "planning_day", "item_id", "item_name", "unique_item_id", "exclusion_reason", "reconcile_status"]
        ]
        flagged_df = planning_reconciled[
            planning_reconciled["reconcile_status"].isin(["dq02_unknown", "exact_match_name_mismatch"])
            & ~planning_reconciled["excluded"]
        ][["corridor_id", "planning_day", "item_id", "item_name", "reconcile_status"]]

        parts = []
        if not excluded_df.empty:
            parts.append("**Excluded rows (DQ-01, DQ-04):**\n" + excluded_df.to_markdown(index=False))
        if not flagged_df.empty:
            parts.append("**Flagged rows (DQ-02, DQ-03 — not excluded):**\n" + flagged_df.to_markdown(index=False))
        if parts:
            anomalies_md = "\n\n".join(parts)

    return CsvAnalysisResult(
        planning_df=valid_planning,
        history_df=history_reconciled,
        all_reconciled_df=all_reconciled,
        corridor_kpis=corridor_kpis,
        trend_summary=trend_summary,
        recon_log=log,
        summary=summary,
        kpis=kpis,
        anomalies_md=anomalies_md,
    )
