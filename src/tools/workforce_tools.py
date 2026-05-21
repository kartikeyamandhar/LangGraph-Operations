"""
Workforce + feedback layer.

This module is the bridge between the agentic planner and the real world. It
loads four CSV streams from the feedback/ directory and turns them into:

  • An effective driver pool (eligibility-filtered)
  • Certification counts (cold-chain certified driver count)
  • Recent manager ratings (for prompt injection)
  • Outcome calibration (predicted vs actual stats)

All numerical / boolean parsing happens here so the graph nodes never have to
re-parse the CSV themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — eligibility rules
# ---------------------------------------------------------------------------
MAX_HOURS_LAST_7D = 40              # DOT-style weekly cap
MAX_CONSECUTIVE_DAYS = 5            # mandatory rest after 5 days
COLD_CHAIN_CERT = "cold_chain"
HISTORY_WINDOW_RUNS = 10            # how many recent runs to feed prompts


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DriverProfile:
    driver_id: str
    name: str
    certifications: List[str]
    hours_last_24h: float
    hours_last_7d: float
    consecutive_days: int
    fatigue_flag: bool
    preferred_corridors: List[str]
    active: bool
    notes: str = ""

    @property
    def eligible_today(self) -> bool:
        if not self.active:
            return False
        if self.hours_last_7d >= MAX_HOURS_LAST_7D:
            return False
        if self.consecutive_days >= MAX_CONSECUTIVE_DAYS:
            return False
        return True

    @property
    def cold_chain_certified(self) -> bool:
        return COLD_CHAIN_CERT in self.certifications

    @property
    def status_label(self) -> str:
        if not self.active:
            return "On leave"
        if self.hours_last_7d >= MAX_HOURS_LAST_7D:
            return "Over weekly cap"
        if self.consecutive_days >= MAX_CONSECUTIVE_DAYS:
            return "Mandatory rest day"
        if self.fatigue_flag:
            return "Fatigue flag (eligible w/ warning)"
        return "Eligible"


@dataclass
class WorkforceState:
    """Computed once per run, carried in AppState as a serializable dict."""
    drivers: List[DriverProfile] = field(default_factory=list)

    @property
    def eligible(self) -> List[DriverProfile]:
        return [d for d in self.drivers if d.eligible_today]

    @property
    def ineligible(self) -> List[DriverProfile]:
        return [d for d in self.drivers if not d.eligible_today]

    @property
    def fatigue_flagged(self) -> List[DriverProfile]:
        return [d for d in self.eligible if d.fatigue_flag]

    @property
    def eligible_count(self) -> int:
        return len(self.eligible)

    @property
    def cold_chain_eligible_count(self) -> int:
        return sum(1 for d in self.eligible if d.cold_chain_certified)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_roster": len(self.drivers),
            "eligible_count": self.eligible_count,
            "cold_chain_eligible_count": self.cold_chain_eligible_count,
            "fatigue_flagged_count": len(self.fatigue_flagged),
            "drivers": [
                {
                    "driver_id": d.driver_id,
                    "name": d.name,
                    "certifications": d.certifications,
                    "hours_last_7d": d.hours_last_7d,
                    "consecutive_days": d.consecutive_days,
                    "fatigue_flag": d.fatigue_flag,
                    "active": d.active,
                    "eligible_today": d.eligible_today,
                    "cold_chain_certified": d.cold_chain_certified,
                    "status": d.status_label,
                    "preferred_corridors": d.preferred_corridors,
                }
                for d in self.drivers
            ],
            "summary": (
                f"{self.eligible_count} of {len(self.drivers)} drivers eligible today "
                f"({self.cold_chain_eligible_count} cold-chain certified, "
                f"{len(self.fatigue_flagged)} flagged for fatigue)."
            ),
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _split_semi(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(";") if s.strip()]


def _coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_workforce_state(path: str = "feedback/driver_state.csv") -> WorkforceState:
    p = Path(path)
    if not p.exists():
        return WorkforceState(drivers=[])

    df = pd.read_csv(p)
    drivers: List[DriverProfile] = []
    for _, row in df.iterrows():
        drivers.append(DriverProfile(
            driver_id=str(row["driver_id"]),
            name=str(row.get("name", "")),
            certifications=_split_semi(row.get("certifications", "")),
            hours_last_24h=float(row.get("hours_last_24h", 0) or 0),
            hours_last_7d=float(row.get("hours_last_7d", 0) or 0),
            consecutive_days=int(row.get("consecutive_days", 0) or 0),
            fatigue_flag=_coerce_bool(row.get("fatigue_flag")),
            preferred_corridors=_split_semi(row.get("preferred_corridors", "")),
            active=_coerce_bool(row.get("active", True)),
            notes=str(row.get("notes", "")) if pd.notna(row.get("notes")) else "",
        ))
    return WorkforceState(drivers=drivers)


def load_driver_post_shift_feedback(
    path: str = "feedback/driver_post_shift_feedback.csv",
    last_n: int = HISTORY_WINDOW_RUNS,
) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if df.empty:
        return []
    # Most recent first, capped to last_n records
    df = df.sort_values("timestamp", ascending=False).head(last_n)
    return df.to_dict(orient="records")


def load_manager_ratings(
    path: str = "feedback/manager_ratings.csv",
    last_n: int = HISTORY_WINDOW_RUNS,
) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if df.empty:
        return []
    df = df.sort_values("timestamp", ascending=False).head(last_n)
    return df.to_dict(orient="records")


def append_manager_rating(
    run_id: str,
    manager_id: str,
    scenario: str,
    star_rating: int,
    tags: List[str],
    comment: str,
    path: str = "feedback/manager_ratings.csv",
) -> None:
    """Append a single rating row, preserving existing data."""
    from datetime import datetime
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "run_id": run_id,
        "manager_id": manager_id,
        "scenario": scenario or "baseline",
        "star_rating": int(star_rating),
        "tags": ";".join(tags or []),
        "comment": comment or "",
    }])
    if p.exists():
        existing = pd.read_csv(p)
        out = pd.concat([existing, new_row], ignore_index=True)
    else:
        out = new_row
    out.to_csv(p, index=False)


def append_outcome(
    run_id: str,
    scenario: str,
    predicted_penalty: int,
    actual_penalty: int,
    predicted_deferred: int,
    actual_deferred: int,
    actual_tier1_late: int,
    actual_tier2_late: int,
    actual_cold_chain_breaches: int,
    incident_notes: str,
    path: str = "feedback/outcome_log.csv",
) -> None:
    from datetime import datetime
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "run_id": run_id,
        "scenario": scenario or "baseline",
        "predicted_penalty": int(predicted_penalty),
        "actual_penalty": int(actual_penalty),
        "predicted_deferred": int(predicted_deferred),
        "actual_deferred": int(actual_deferred),
        "actual_tier1_late": int(actual_tier1_late),
        "actual_tier2_late": int(actual_tier2_late),
        "actual_cold_chain_breaches": int(actual_cold_chain_breaches),
        "incident_notes": incident_notes or "",
    }])
    if p.exists():
        existing = pd.read_csv(p)
        out = pd.concat([existing, new_row], ignore_index=True)
    else:
        out = new_row
    out.to_csv(p, index=False)


def load_outcome_log(
    path: str = "feedback/outcome_log.csv",
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Calibration metrics — used for prompt injection + calibration tab
# ---------------------------------------------------------------------------
def compute_calibration(
    path: str = "feedback/outcome_log.csv",
) -> Dict[str, Any]:
    """
    Compare predicted vs actual outcomes across the full history.

    Returns a dict with:
      • penalty_mae      — mean absolute error
      • penalty_bias     — mean(actual - predicted); positive = under-predicted
      • deferred_bias    — same for deferred units
      • cold_chain_breach_total
      • runs_n
      • headline         — one-liner the LLM can quote
    """
    df = load_outcome_log(path)
    if df.empty:
        return {
            "runs_n": 0,
            "headline": "No historical outcomes yet — calibration not available.",
        }

    df = df.dropna(subset=["predicted_penalty", "actual_penalty"])
    if df.empty:
        return {
            "runs_n": 0,
            "headline": "No comparable historical outcomes yet.",
        }

    penalty_mae = float((df["actual_penalty"] - df["predicted_penalty"]).abs().mean())
    penalty_bias = float((df["actual_penalty"] - df["predicted_penalty"]).mean())
    deferred_bias = float((df["actual_deferred"] - df["predicted_deferred"]).mean())
    breaches = int(df["actual_cold_chain_breaches"].sum())

    direction = "under-predicted" if penalty_bias > 0 else "over-predicted" if penalty_bias < 0 else "matched"

    return {
        "runs_n": int(len(df)),
        "penalty_mae": round(penalty_mae, 1),
        "penalty_bias": round(penalty_bias, 1),
        "deferred_bias": round(deferred_bias, 2),
        "cold_chain_breach_total": breaches,
        "headline": (
            f"Calibration: {len(df)} historical runs · MAE {penalty_mae:.0f} pts · "
            f"system has {direction} actual penalty by {abs(penalty_bias):.0f} pts on average · "
            f"{breaches} cold-chain breaches recorded over history."
        ),
    }


# ---------------------------------------------------------------------------
# Workforce-aware effective resource pool
# ---------------------------------------------------------------------------
def apply_workforce_to_pool(
    pool: Dict[str, Dict[str, int]],
    workforce: WorkforceState,
) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    """
    Reduce a resource pool's driver and cold-chain-truck counts by the
    workforce reality. The cold-chain truck cap is min(physical_trucks,
    cold_chain_certified_eligible_drivers) — you cannot run a cold-chain
    truck without a certified driver.

    Returns:
      (new_pool, change_log)
    where change_log is a list of {day, resource, from, to, reason} dicts
    suitable for displaying in the report.
    """
    import copy
    new_pool = copy.deepcopy(pool)
    changes: List[Dict[str, Any]] = []

    if not workforce.drivers:
        return new_pool, changes

    eligible_n = workforce.eligible_count
    cold_n = workforce.cold_chain_eligible_count

    for day in new_pool:
        # Driver cap
        base_drivers = int(new_pool[day].get("driver", 0))
        if eligible_n < base_drivers:
            new_pool[day]["driver"] = eligible_n
            changes.append({
                "day": day, "resource": "driver",
                "from": base_drivers, "to": eligible_n,
                "reason": (
                    f"Only {eligible_n} of {len(workforce.drivers)} drivers "
                    f"eligible today (others on leave / over weekly cap / mandatory rest)."
                ),
            })

        # Cold-chain truck cap (physical trucks AND certified drivers both required)
        base_cold = int(new_pool[day].get("truck_temp_controlled", 0))
        if cold_n < base_cold:
            new_pool[day]["truck_temp_controlled"] = cold_n
            changes.append({
                "day": day, "resource": "truck_temp_controlled",
                "from": base_cold, "to": cold_n,
                "reason": (
                    f"Only {cold_n} cold-chain certified drivers eligible today; "
                    f"each cold-chain truck requires a certified driver."
                ),
            })

    return new_pool, changes


# ---------------------------------------------------------------------------
# Realism check — produces warnings and violations on the final allocation
# ---------------------------------------------------------------------------
def realism_check_allocation(
    allocation: Dict[str, Any],
    workforce: WorkforceState,
    effective_pool: Dict[str, Dict[str, int]],
) -> Tuple[List[str], List[str]]:
    """
    Produce (warnings, violations) for the allocation given the workforce reality.

    Warnings are non-blocking informational notes (yellow box on report).
    Violations are hard failures (red banner — should not happen if pool was
    correctly reduced upstream, but we double-check).
    """
    warnings: List[str] = []
    violations: List[str] = []

    if not isinstance(allocation, dict) or allocation.get("raw") or not workforce.drivers:
        return warnings, violations

    # 1. Total driver demand vs eligible pool (per day)
    for day in ("Day0", "Day1"):
        day_alloc = allocation.get(day)
        if not isinstance(day_alloc, dict):
            continue
        drivers_used = sum(
            int((v or {}).get("driver", 0) or 0)
            for v in day_alloc.values()
            if isinstance(v, dict)
        )
        cap = effective_pool.get(day, {}).get("driver", workforce.eligible_count)
        if drivers_used > cap:
            violations.append(
                f"{day}: {drivers_used} drivers allocated but only {cap} eligible drivers available "
                f"({workforce.eligible_count} eligible total)."
            )
        elif drivers_used == workforce.eligible_count and workforce.eligible_count > 0:
            warnings.append(
                f"{day}: All {workforce.eligible_count} eligible drivers committed — no slack for emergencies."
            )

        # Cold-chain certified driver count vs cold-chain trucks
        cold_trucks = sum(
            int((v or {}).get("truck_temp_controlled", 0) or 0)
            for v in day_alloc.values()
            if isinstance(v, dict)
        )
        if cold_trucks > workforce.cold_chain_eligible_count:
            violations.append(
                f"{day}: {cold_trucks} cold-chain trucks allocated but only "
                f"{workforce.cold_chain_eligible_count} cold-chain certified drivers eligible."
            )

        # Per-corridor consistency: a corridor with trucks but zero drivers
        # is operationally infeasible — the trucks cannot move themselves.
        for corridor_id, v in day_alloc.items():
            if not isinstance(v, dict):
                continue
            trucks_on_corridor = (
                int(v.get("truck_temp_controlled", 0) or 0)
                + int(v.get("truck_standard", 0) or 0)
            )
            drivers_on_corridor = int(v.get("driver", 0) or 0)
            if trucks_on_corridor > 0 and drivers_on_corridor < 1:
                violations.append(
                    f"{day} {corridor_id}: {trucks_on_corridor} truck(s) allocated "
                    f"with 0 drivers — trucks cannot dispatch without a driver."
                )
            elif trucks_on_corridor > 0 and drivers_on_corridor < trucks_on_corridor:
                warnings.append(
                    f"{day} {corridor_id}: {trucks_on_corridor} trucks share "
                    f"{drivers_on_corridor} driver(s) — confirm driver swap plan."
                )

    # 2. Fatigue-flagged drivers in the eligible pool
    if workforce.fatigue_flagged:
        flagged_ids = ", ".join(d.driver_id for d in workforce.fatigue_flagged)
        warnings.append(
            f"Fatigue-flagged drivers in today's eligible pool: {flagged_ids}. "
            f"Avoid back-to-back assignments where possible."
        )

    # 3. Drivers over weekly cap (excluded already, but call out to manager)
    over_cap = [d for d in workforce.drivers if d.hours_last_7d >= MAX_HOURS_LAST_7D]
    if over_cap:
        ids = ", ".join(d.driver_id for d in over_cap)
        warnings.append(
            f"Excluded from today's pool (over {MAX_HOURS_LAST_7D}h/7d cap): {ids}."
        )

    return warnings, violations
