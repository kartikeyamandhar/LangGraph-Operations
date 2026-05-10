"""
Synthetic shipment data generator for SeeWeeS pipeline testing.

Produces a library of CSVs (matching the schema of
data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv) rich enough to
answer 40-50 different operational scenarios. Each profile encodes a distinct
real-world narrative — system glitches, FDA recalls, seasonal surges, weekly
patterns, facility imbalances, etc. — so the pipeline can be exercised
dynamically and analytical questions have real signal to detect.

Schema:
    shipment_date, planning_day, is_planning_window, corridor_id,
    item_id, item_name, unique_item_id, dispatch_location

Run:
    python generate_synthetic_data.py
"""
from __future__ import annotations
import csv
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

OUT_DIR = Path("data-for-enhancement/synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Item master (matches csv_tools.CANONICAL_MASTER) + realistic name aliases
# ---------------------------------------------------------------------------
ITEMS = [
    # (item_id, canonical_name, aliases, product_class, sla_tier, cold_chain)
    (10021, "Remdesivir 100mg",
     ["Remdesivir 100mg", "Remdesivir 100 mg", "Remdesivir 200mg"],
     "Antiviral", 1, True),
    (10022, "Insulin Lispro",
     ["Insulin Lispro"], "Endocrine", 2, True),
    (10023, "Insulin Aspart",
     ["Insulin Aspart"], "Endocrine", 2, True),
    (10035, "Pembrolizumab",
     ["Pembrolizumab", "Pembrolizumab (Keytruda)"],
     "Oncology Biologic", 1, True),
    (10040, "Epinephrine Auto-Injector",
     ["Epinephrine Auto-Injector", "EpiPen Auto Injector"],
     "Emergency", 2, False),
    (10050, "Heparin Sodium",
     ["Heparin Sodium", "Heparin Na"], "Anticoagulant", 2, False),
    (10060, "Morphine Sulfate",
     ["Morphine Sulfate", "Morphine Sulphate"], "Controlled", 2, False),
    (10070, "Albuterol Inhaler",
     ["Albuterol Inhaler", "Albuterol Inhaler 90mcg"],
     "Respiratory", 2, False),
    (10071, "Levalbuterol Inhaler",
     ["Levalbuterol Inhaler"], "Respiratory", 2, False),
    (99999, "Experimental Oncology Drug",
     ["Experimental Oncology Drug"], "Clinical Trial", 1, True),
]

LEGACY_IDS = {10020: 10021, 20021: 10021, 1070: 10070}

# Real-ish facility specialisations — used by some profiles to create
# realistic per-facility drug mixes
FACILITY_SPECIALTIES = {
    "Boston-MGH":          {"Antiviral": 0.30, "Endocrine": 0.20, "Anticoagulant": 0.20, "Controlled": 0.15, "Other": 0.15},
    "Boston-BWH":          {"Endocrine": 0.30, "Anticoagulant": 0.20, "Antiviral": 0.20, "Other": 0.30},
    "Boston-DanaFarber":   {"Oncology Biologic": 0.55, "Clinical Trial": 0.20, "Antiviral": 0.10, "Other": 0.15},
    "Boston-Children":     {"Respiratory": 0.30, "Emergency": 0.25, "Endocrine": 0.20, "Other": 0.25},
    "Philadelphia-UPenn":  {"Oncology Biologic": 0.30, "Endocrine": 0.20, "Antiviral": 0.20, "Other": 0.30},
    "Philadelphia-CHOP":   {"Respiratory": 0.30, "Emergency": 0.25, "Endocrine": 0.25, "Other": 0.20},
    "Philadelphia-Jefferson":{"Anticoagulant": 0.30, "Controlled": 0.25, "Endocrine": 0.20, "Other": 0.25},
}

CORRIDOR_LOCATIONS = {
    "C1_I95_NJ_BOS": ["Boston-MGH", "Boston-BWH", "Boston-DanaFarber", "Boston-Children"],
    "C2_NJ_PHL": ["Philadelphia-UPenn", "Philadelphia-CHOP", "Philadelphia-Jefferson"],
}

CORRIDORS = list(CORRIDOR_LOCATIONS.keys())

# Item categorisation helpers
COLD_CHAIN_ITEMS = [it for it in ITEMS if it[5]]
TIER1_ITEMS      = [it for it in ITEMS if it[4] == 1]
TIER2_ITEMS      = [it for it in ITEMS if it[4] == 2]
ROOM_TEMP_ITEMS  = [it for it in ITEMS if not it[5]]
ANTIVIRAL_ITEMS  = [it for it in ITEMS if it[3] == "Antiviral"]


# ---------------------------------------------------------------------------
# Row builder helpers
# ---------------------------------------------------------------------------
def _make_uid(item_id: int, corridor: str, seq: int) -> str:
    prefix = {
        10021: "RMD", 10022: "INS", 10023: "INS", 10035: "PBR",
        10040: "EPI", 10050: "HEP", 10060: "MOR", 10070: "ALB",
        10071: "LEV", 99999: "EXP",
    }.get(item_id, "GEN")
    corridor_block = "0" if corridor == "C1_I95_NJ_BOS" else "1"
    return f"{prefix}-2026-{corridor_block}{seq:04d}"


def _row(date_str: str, planning_day: str, is_planning: int, corridor: str,
         item_id: int, item_name: str, uid: str, location: str) -> dict:
    return {
        "shipment_date": date_str,
        "planning_day": planning_day,
        "is_planning_window": is_planning,
        "corridor_id": corridor,
        "item_id": item_id,
        "item_name": item_name,
        "unique_item_id": uid,
        "dispatch_location": location,
    }


def _pick_item(rng: random.Random, pool: List[Tuple]) -> Tuple[int, str]:
    item = rng.choice(pool)
    return item[0], rng.choice(item[2])


def _pick_facility_item(rng: random.Random, location: str) -> Tuple[int, str]:
    """Pick an item weighted by facility specialty."""
    specialties = FACILITY_SPECIALTIES.get(location, {"Other": 1.0})
    classes, weights = zip(*specialties.items())
    chosen_class = rng.choices(classes, weights=weights)[0]
    if chosen_class == "Other":
        item = rng.choice(ITEMS)
    else:
        candidates = [it for it in ITEMS if it[3] == chosen_class]
        item = rng.choice(candidates) if candidates else rng.choice(ITEMS)
    return item[0], rng.choice(item[2])


def _maybe_corrupt(rng: random.Random, row: dict, dq_rate: float) -> dict:
    if rng.random() >= dq_rate:
        return row
    fault = rng.choice(["dq01_blank_uid", "dq02_unknown_id", "dq03_name_mismatch", "dq05_legacy_id"])
    if fault == "dq01_blank_uid":
        row["unique_item_id"] = ""
    elif fault == "dq02_unknown_id":
        row["item_id"] = rng.choice([11111, 22222, 88888])
    elif fault == "dq03_name_mismatch":
        row["item_name"] = "Generic " + row["item_name"]
    elif fault == "dq05_legacy_id":
        row["item_id"] = rng.choice(list(LEGACY_IDS.keys()))
    return row


def _inject_duplicate(rows: List[dict], rng: random.Random, count: int):
    if not rows or count == 0:
        return rows
    for _ in range(count):
        valid = [r for r in rows if r["unique_item_id"]]
        if len(valid) < 2:
            break
        donor, victim = rng.sample(valid, 2)
        victim["unique_item_id"] = donor["unique_item_id"]
    return rows


def _dow_multiplier(d: date) -> float:
    """Mon-Fri busy, Sat half, Sun quiet — realistic hospital ordering pattern."""
    dow = d.weekday()  # 0=Mon, 6=Sun
    return [1.10, 1.05, 1.00, 1.05, 1.15, 0.50, 0.30][dow]


# ---------------------------------------------------------------------------
# Profile generators
# ---------------------------------------------------------------------------
def gen_baseline(seed: int = 42) -> List[dict]:
    """
    Baseline: 30 days history + Day0/Day1, day-of-week patterns, ~5% DQ rate.
    Realistic everyday operations across both corridors.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    base = date(2026, 2, 1)

    for d_offset in range(30):
        ship_date = base + timedelta(days=d_offset)
        dow_mult = _dow_multiplier(ship_date)
        for corridor in CORRIDORS:
            base_n = 5 if corridor == "C1_I95_NJ_BOS" else 4
            n_rows = max(1, int(round(base_n * dow_mult * rng.uniform(0.85, 1.15))))
            for _ in range(n_rows):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                row = _row(ship_date.isoformat(), "History", 0, corridor,
                           item_id, name, uid, loc)
                rows.append(_maybe_corrupt(rng, row, dq_rate=0.05))

    planning_base = base + timedelta(days=30)
    for d_idx, day_label in enumerate(["Day0", "Day1"]):
        ship_date = planning_base + timedelta(days=d_idx)
        for corridor in CORRIDORS:
            n_rows = rng.randint(8, 12)
            for _ in range(n_rows):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                row = _row(ship_date.isoformat(), day_label, 1, corridor,
                           item_id, name, uid, loc)
                rows.append(_maybe_corrupt(rng, row, dq_rate=0.05))

    _inject_duplicate(rows, rng, count=2)
    return rows


def gen_volume_spike(seed: int = 7) -> List[dict]:
    """
    3-day cold-chain surge climaxing on Day0. History shows 6 days of
    elevated cold-chain volume preceding the planning window — gives the
    trend agent something to detect ('cold-chain volume up 60% over last week').
    Day0 needs ~3 cold-chain trucks but only 2 available → forced deferral.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    base = date(2026, 2, 10)

    # 14 days normal history
    for d_offset in range(14):
        ship_date = base + timedelta(days=d_offset)
        for corridor in CORRIDORS:
            for _ in range(rng.randint(3, 5)):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                rows.append(_row(ship_date.isoformat(), "History", 0, corridor,
                                 item_id, name, uid, loc))

    # 6 days of escalating cold-chain volume (the run-up to the spike)
    for d_offset in range(14, 20):
        ship_date = base + timedelta(days=d_offset)
        cold_n = 4 + (d_offset - 14)  # 4, 5, 6, 7, 8, 9
        for corridor in CORRIDORS:
            for _ in range(cold_n):
                item_id, name = _pick_item(rng, COLD_CHAIN_ITEMS)
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                rows.append(_row(ship_date.isoformat(), "History", 0, corridor,
                                 item_id, name, uid, loc))
            for _ in range(rng.randint(2, 3)):
                item_id, name = _pick_item(rng, ROOM_TEMP_ITEMS)
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                rows.append(_row(ship_date.isoformat(), "History", 0, corridor,
                                 item_id, name, uid, loc))

    # Day0: PEAK — 14 cold-chain per corridor (28 total → 4 trucks needed)
    planning_base = base + timedelta(days=20)
    for corridor in CORRIDORS:
        for _ in range(14):
            item_id, name = _pick_item(rng, COLD_CHAIN_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(planning_base.isoformat(), "Day0", 1, corridor,
                             item_id, name, uid, loc))
        for _ in range(rng.randint(3, 5)):
            item_id, name = _pick_item(rng, ROOM_TEMP_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(planning_base.isoformat(), "Day0", 1, corridor,
                             item_id, name, uid, loc))

    # Day1: tapering off but still elevated
    day1 = planning_base + timedelta(days=1)
    for corridor in CORRIDORS:
        for _ in range(8):
            item_id, name = _pick_item(rng, COLD_CHAIN_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(day1.isoformat(), "Day1", 1, corridor,
                             item_id, name, uid, loc))
        for _ in range(rng.randint(3, 5)):
            item_id, name = _pick_item(rng, ROOM_TEMP_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(day1.isoformat(), "Day1", 1, corridor,
                             item_id, name, uid, loc))

    return rows


def gen_dq_heavy(seed: int = 13) -> List[dict]:
    """
    Data quality crisis: 35% planning-window DQ rate, plus a 'system glitch
    day' in history where DQ rate jumps to 60% for one day (trains the eye
    to spot anomaly bursts). Multiple DQ-04 duplicate UIDs injected.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    base = date(2026, 2, 1)

    for d_offset in range(28):
        ship_date = base + timedelta(days=d_offset)
        # Glitch day: day 19 (one Tuesday with the order system bug)
        glitch = (d_offset == 19)
        dq_rate = 0.60 if glitch else 0.10
        for corridor in CORRIDORS:
            for _ in range(rng.randint(3, 5)):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                row = _row(ship_date.isoformat(), "History", 0, corridor,
                           item_id, name, uid, loc)
                rows.append(_maybe_corrupt(rng, row, dq_rate=dq_rate))

    planning_base = base + timedelta(days=28)
    for d_idx, day_label in enumerate(["Day0", "Day1"]):
        ship_date = planning_base + timedelta(days=d_idx)
        for corridor in CORRIDORS:
            for _ in range(rng.randint(10, 14)):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                row = _row(ship_date.isoformat(), day_label, 1, corridor,
                           item_id, name, uid, loc)
                rows.append(_maybe_corrupt(rng, row, dq_rate=0.35))

    _inject_duplicate(rows, rng, count=5)
    return rows


def gen_tier1_surge(seed: int = 21) -> List[dict]:
    """
    5-day Tier 1 surge with peak in middle (oncology outbreak narrative).
    Day0 lands on the peak: 80% Tier 1. Forces explicit prioritization.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    base = date(2026, 2, 1)

    # Normal baseline for first 18 days
    for d_offset in range(18):
        ship_date = base + timedelta(days=d_offset)
        for corridor in CORRIDORS:
            for _ in range(rng.randint(3, 5)):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                rows.append(_row(ship_date.isoformat(), "History", 0, corridor,
                                 item_id, name, uid, loc))

    # 5-day surge ramp: Tier 1 share goes 30% → 60% → 80% → 70% → 50%
    surge_t1_share = [0.30, 0.60, 0.80, 0.70, 0.50]
    for d_offset, t1_share in enumerate(surge_t1_share, start=18):
        ship_date = base + timedelta(days=d_offset)
        for corridor in CORRIDORS:
            n = rng.randint(6, 8)
            for _ in range(n):
                if rng.random() < t1_share:
                    item_id, name = _pick_item(rng, TIER1_ITEMS)
                else:
                    item_id, name = _pick_item(rng, TIER2_ITEMS)
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                rows.append(_row(ship_date.isoformat(), "History", 0, corridor,
                                 item_id, name, uid, loc))

    # Day0 = surge peak. Day1 = tapering.
    planning_base = base + timedelta(days=23)
    for corridor in CORRIDORS:
        for _ in range(8):
            item_id, name = _pick_item(rng, TIER1_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(planning_base.isoformat(), "Day0", 1, corridor,
                             item_id, name, uid, loc))
        for _ in range(2):
            item_id, name = _pick_item(rng, TIER2_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(planning_base.isoformat(), "Day0", 1, corridor,
                             item_id, name, uid, loc))

    day1 = planning_base + timedelta(days=1)
    for corridor in CORRIDORS:
        for _ in range(5):
            item_id, name = _pick_item(rng, TIER1_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(day1.isoformat(), "Day1", 1, corridor,
                             item_id, name, uid, loc))
        for _ in range(4):
            item_id, name = _pick_item(rng, TIER2_ITEMS)
            loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
            uid = _make_uid(item_id, corridor, len(rows) + 1)
            rows.append(_row(day1.isoformat(), "Day1", 1, corridor,
                             item_id, name, uid, loc))

    return rows


def gen_growth_trend(seed: int = 99) -> List[dict]:
    """
    60-day history with a +50% upward trend on C2 (Philadelphia outgrowing
    capacity). C1 is flat. Planning window reflects sustained elevated C2.
    Allows 'is C2 outgrowing capacity?' analysis with strong signal.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    base = date(2026, 1, 1)

    for d_offset in range(60):
        ship_date = base + timedelta(days=d_offset)
        dow_mult = _dow_multiplier(ship_date)

        # C1 stays flat with day-of-week pattern
        c1_base = 5
        c1_n = max(1, int(round(c1_base * dow_mult * rng.uniform(0.85, 1.15))))
        for _ in range(c1_n):
            loc = rng.choice(CORRIDOR_LOCATIONS["C1_I95_NJ_BOS"])
            item_id, name = _pick_facility_item(rng, loc)
            uid = _make_uid(item_id, "C1_I95_NJ_BOS", len(rows) + 1)
            rows.append(_row(ship_date.isoformat(), "History", 0, "C1_I95_NJ_BOS",
                             item_id, name, uid, loc))

        # C2: linear upward trend from 3 → 7.5 across 60 days
        c2_base = 3 + (4.5 * d_offset / 60)
        c2_n = max(1, int(round(c2_base * dow_mult * rng.uniform(0.85, 1.15))))
        for _ in range(c2_n):
            loc = rng.choice(CORRIDOR_LOCATIONS["C2_NJ_PHL"])
            item_id, name = _pick_facility_item(rng, loc)
            uid = _make_uid(item_id, "C2_NJ_PHL", len(rows) + 1)
            rows.append(_row(ship_date.isoformat(), "History", 0, "C2_NJ_PHL",
                             item_id, name, uid, loc))

    planning_base = base + timedelta(days=60)
    for d_idx, day_label in enumerate(["Day0", "Day1"]):
        ship_date = planning_base + timedelta(days=d_idx)
        for _ in range(rng.randint(5, 7)):
            loc = rng.choice(CORRIDOR_LOCATIONS["C1_I95_NJ_BOS"])
            item_id, name = _pick_facility_item(rng, loc)
            uid = _make_uid(item_id, "C1_I95_NJ_BOS", len(rows) + 1)
            rows.append(_row(ship_date.isoformat(), day_label, 1, "C1_I95_NJ_BOS",
                             item_id, name, uid, loc))
        for _ in range(rng.randint(8, 11)):
            loc = rng.choice(CORRIDOR_LOCATIONS["C2_NJ_PHL"])
            item_id, name = _pick_facility_item(rng, loc)
            uid = _make_uid(item_id, "C2_NJ_PHL", len(rows) + 1)
            rows.append(_row(ship_date.isoformat(), day_label, 1, "C2_NJ_PHL",
                             item_id, name, uid, loc))

    return rows


def gen_rich_60d(seed: int = 2026) -> List[dict]:
    """
    Master rich dataset: 60 days history + Day0/Day1, with multiple embedded
    patterns to support 40-50 different analytical questions:
      - Strong M-F vs weekend day-of-week effect
      - Mild upward trend on both corridors (~25% over 60 days)
      - Two embedded volume bursts (around days 12 and 38)
      - Two DQ glitch days (days 17 and 44, ~50% DQ rate)
      - Facility specialty mixes drive realistic per-hospital drug profiles
      - Antiviral demand higher in second half (mild seasonal pattern)
      - Pembrolizumab demand concentrated at DanaFarber + UPenn
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    base = date(2026, 1, 1)
    burst_days = {12, 13, 38, 39}
    glitch_days = {17, 44}

    for d_offset in range(60):
        ship_date = base + timedelta(days=d_offset)
        dow_mult = _dow_multiplier(ship_date)

        # Mild upward trend across the period
        trend_mult = 1.0 + (0.25 * d_offset / 60)
        # Antiviral seasonality
        antiviral_share = 0.08 + (0.20 * d_offset / 60)
        # Volume burst days
        burst_mult = 1.6 if d_offset in burst_days else 1.0
        # Glitch day DQ rate
        dq_rate = 0.50 if d_offset in glitch_days else 0.06

        for corridor in CORRIDORS:
            base_n = 5 if corridor == "C1_I95_NJ_BOS" else 4
            n = max(1, int(round(base_n * dow_mult * trend_mult * burst_mult * rng.uniform(0.85, 1.15))))
            for _ in range(n):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                # Antiviral seasonal injection
                if rng.random() < antiviral_share:
                    item_id, name = _pick_item(rng, ANTIVIRAL_ITEMS)
                else:
                    item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                row = _row(ship_date.isoformat(), "History", 0, corridor,
                           item_id, name, uid, loc)
                rows.append(_maybe_corrupt(rng, row, dq_rate=dq_rate))

    # Planning window: continues the trend with realistic mix
    planning_base = base + timedelta(days=60)
    for d_idx, day_label in enumerate(["Day0", "Day1"]):
        ship_date = planning_base + timedelta(days=d_idx)
        for corridor in CORRIDORS:
            n = rng.randint(9, 12) if corridor == "C1_I95_NJ_BOS" else rng.randint(8, 11)
            for _ in range(n):
                loc = rng.choice(CORRIDOR_LOCATIONS[corridor])
                if rng.random() < 0.30:
                    item_id, name = _pick_item(rng, ANTIVIRAL_ITEMS)
                else:
                    item_id, name = _pick_facility_item(rng, loc)
                uid = _make_uid(item_id, corridor, len(rows) + 1)
                row = _row(ship_date.isoformat(), day_label, 1, corridor,
                           item_id, name, uid, loc)
                rows.append(_maybe_corrupt(rng, row, dq_rate=0.08))

    _inject_duplicate(rows, rng, count=3)
    return rows


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "shipment_date", "planning_day", "is_planning_window", "corridor_id",
    "item_id", "item_name", "unique_item_id", "dispatch_location",
]


def write_csv(filename: str, rows: List[dict]):
    path = OUT_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  {filename:42s} {len(rows):5d} rows  →  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PROFILES = [
    ("synthetic_baseline.csv",       gen_baseline,
     "30d history + Day0/Day1, day-of-week + facility specialty patterns"),
    ("synthetic_volume_spike.csv",   gen_volume_spike,
     "3-day cold-chain surge — Day0 needs ≥4 cold-chain trucks (only 2 avail)"),
    ("synthetic_dq_heavy.csv",       gen_dq_heavy,
     "DQ crisis: 35% planning DQ rate + glitch day in history"),
    ("synthetic_tier1_surge.csv",    gen_tier1_surge,
     "5-day Tier 1 oncology surge, Day0 lands on peak (80% Tier 1)"),
    ("synthetic_growth_trend.csv",   gen_growth_trend,
     "60d history with C2 +50% trend; C1 flat"),
    ("synthetic_rich_60d.csv",       gen_rich_60d,
     "Master 60d dataset: trend + DOW + bursts + glitches + seasonality"),
]


if __name__ == "__main__":
    print(f"Generating synthetic CSVs into {OUT_DIR}/\n")
    total_rows = 0
    for filename, gen_fn, desc in PROFILES:
        print(f"  ({desc})")
        rows = gen_fn()
        write_csv(filename, rows)
        total_rows += len(rows)
        print()
    print(f"Total: {len(PROFILES)} CSVs, {total_rows} rows.")
