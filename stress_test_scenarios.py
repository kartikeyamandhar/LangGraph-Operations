"""
Scenario stress tester — runs the full graph with different what-if inputs.
Usage:  python stress_test_scenarios.py
        python stress_test_scenarios.py --scenario 2   (run just scenario #2)
"""
from __future__ import annotations
import argparse
import json
import time
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, "src")

from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph

# ---------------------------------------------------------------------------
# Scenarios to test — add/edit freely
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "Baseline (no scenario)",
        "scenario": None,
    },
    {
        "name": "Demand spike C2",
        "scenario": "20% demand spike on the NJ→Philadelphia corridor (C2). "
                    "All existing orders must still be fulfilled; the spike adds "
                    "additional units on top.",
    },
    {
        "name": "Driver shortage",
        "scenario": "Driver shortage — only 3 drivers are available today instead "
                    "of the normal 6. Adjust allocation accordingly and flag any "
                    "SLA risk this creates.",
    },
    {
        "name": "Cold-chain truck breakdown",
        "scenario": "One temperature-controlled truck has broken down. Only 1 "
                    "cold-chain truck is available per day (instead of 2). "
                    "Identify which cold-chain medicines must be deferred and "
                    "calculate the penalty impact.",
    },
    {
        "name": "I-95 partial closure (C1 delay)",
        "scenario": "I-95 is partially closed due to an accident north of New "
                    "Haven. Estimated transit time for C1 (NJ→Boston) is extended "
                    "by 4 hours. Assess Tier 1 SLA compliance and adjust buffers.",
    },
    {
        "name": "Combined: demand spike + driver shortage",
        "scenario": "Dual disruption: 30% demand spike across both corridors AND "
                    "driver shortage — only 2 drivers available. Show worst-case "
                    "penalty score and which shipments get deferred first.",
    },
]


def run_scenario(idx: int, entry: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"SCENARIO {idx}: {entry['name']}")
    print(f"Input: {entry['scenario'] or '(none)'}")
    print(f"{'='*60}")

    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"stress-test-{idx}-{int(time.time())}"}}
    state = {
        "pdf_path": "data/SeeWeeS Specialty distribution.pdf",
        "csv_path": "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
        "scenario": entry["scenario"],
    }

    t0 = time.time()
    interrupted = False

    for event in app.stream(state, config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                print("  [INTERRUPT] Risk score 3 — auto-approving for stress test")
                from langgraph.types import Command
                for resume_event in app.stream(Command(resume="YES"), config, stream_mode="updates"):
                    for rn, _ in resume_event.items():
                        print(f"  [{rn}] resumed")
                interrupted = True
                break
            else:
                keys = list(node_output.keys()) if isinstance(node_output, dict) else []
                print(f"  [{node_name}] {keys}")
        if interrupted:
            break

    elapsed = time.time() - t0
    snap = app.get_state(config)
    final = snap.values

    result = {
        "scenario_name": entry["name"],
        "elapsed_sec": round(elapsed, 1),
        "audit_verdict": final.get("audit_verdict"),
        "audit_attempts": final.get("audit_attempts"),
        "human_approved": final.get("human_approved"),
        "allocation_penalty": (
            final.get("allocation_plan", {}).get("total_penalty_score", "N/A")
            if isinstance(final.get("allocation_plan"), dict) else "N/A"
        ),
        "allocation_deferred": (
            final.get("allocation_plan", {}).get("deferred_units", "N/A")
            if isinstance(final.get("allocation_plan"), dict) else "N/A"
        ),
        "plan_snippet": final.get("dispatch_plan", "")[:300],
        "report_html_len": len(final.get("report_html", "")),
    }

    print(f"\n  RESULT:")
    print(f"    Audit:           {result['audit_verdict']} (attempt {result['audit_attempts']})")
    print(f"    Penalty score:   {result['allocation_penalty']}")
    print(f"    Deferred units:  {result['allocation_deferred']}")
    print(f"    Human approved:  {result['human_approved']}")
    print(f"    Report length:   {result['report_html_len']} chars")
    print(f"    Time:            {result['elapsed_sec']}s")

    # Save the HTML report for this scenario
    html = final.get("report_html", "")
    if html:
        fname = f"report_scenario_{idx}.html"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"    Report saved:    {fname}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=None,
                        help="Run a single scenario by index (0-based). Omit to run all.")
    args = parser.parse_args()

    if args.scenario is not None:
        targets = [(args.scenario, SCENARIOS[args.scenario])]
    else:
        targets = list(enumerate(SCENARIOS))

    results = []
    for i, entry in targets:
        result = run_scenario(i, entry)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  [{r['scenario_name']}]")
        print(f"    Penalty={r['allocation_penalty']}  Deferred={r['allocation_deferred']}  "
              f"Audit={r['audit_verdict']}  Time={r['elapsed_sec']}s")
