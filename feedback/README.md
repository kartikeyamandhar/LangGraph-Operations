# feedback/ — Validation & Realism Layer

This directory contains the four feedback streams that close the loop between
the agentic dispatch system and the real world. They are read at the start of
every run and written to over time as managers and drivers interact with the
plan.

The system treats these files as authoritative inputs alongside the playbook
PDF and shipment CSV — modifying them changes downstream agent behaviour.

| File | Written by | Read by | What it does |
|---|---|---|---|
| `driver_state.csv` | Workforce ops (manual today; WMS API in production) | `node_load_workforce_state` → reduces `effective_resource_pool` | Encodes which drivers are eligible *today* (cert, hours, fatigue, leave) |
| `driver_post_shift_feedback.csv` | Drivers via mobile form after each shift | Aggregated for the report's "Workforce notes" + planner few-shot | "Tight schedule", "smooth run", "would prefer 1 more truck" — qualitative ground-truth |
| `manager_ratings.csv` | Manager via Streamlit "Rate this plan" card | Loaded into PLANNER_PROMPT as recent-feedback context; powers the Feedback History tab | Manager's verdict on each run (1-5 stars, tags, free text) |
| `outcome_log.csv` | Manager next morning ("Yesterday's outcomes" form) | Calibration tab + injected into OPS_ANALYSIS_PROMPT | Predicted vs actual: did the system call it right? |

## Schemas

### driver_state.csv
```
driver_id, name, certifications (semicolon-separated), hours_last_24h, hours_last_7d,
consecutive_days, fatigue_flag, preferred_corridors (semicolon-separated), active, notes
```
Eligibility rules applied by `workforce_tools.py`:
- `active == false` → ineligible (medical leave, etc.)
- `hours_last_7d >= 40` → ineligible (DOT-style rest enforcement)
- `consecutive_days >= 5` → ineligible (mandatory day off)
- `fatigue_flag == true` → eligible but flagged for warning
- A driver is "cold-chain certified" iff `cold_chain` is in their certifications list.

### driver_post_shift_feedback.csv
```
timestamp, driver_id, run_id, corridor, shift_quality (1-5),
tag (Right-sized | Aggressive | Risky | Wasteful), comment
```

### manager_ratings.csv
```
timestamp, run_id, manager_id, scenario, star_rating (1-5),
tags (semicolon-separated), comment
```

### outcome_log.csv
```
timestamp, run_id, scenario, predicted_penalty, actual_penalty,
predicted_deferred, actual_deferred, actual_tier1_late, actual_tier2_late,
actual_cold_chain_breaches, incident_notes
```

## Why this exists

The playbook penalty model (Tier-1 = 100 pts, Tier-2 = 40 pts, cold-chain breach = +80 pts)
is a *proxy* for real-world fitness. A plan can minimise this proxy and still be
operationally unsafe — e.g. assigning the same driver to back-to-back 12-hour
shifts. These feedback streams give the system ground-truth signals that the
proxy alone cannot capture:

- **Workforce constraints** ensure assignments respect human limits (Loop 2 in
  the architecture).
- **Manager ratings** provide validation of the plan's *fitness for use* over
  time (Loop 1).
- **Outcome telemetry** validates the system's *predictive accuracy* and
  surfaces miscalibration (Loop 3).

Together they convert the system from a one-shot LLM into a **learning agentic
operator** that improves with each run.
