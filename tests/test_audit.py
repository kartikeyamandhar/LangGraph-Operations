"""
Audit-loop tests.

The audit node combines deterministic Python checks against the playbook
(buffer policy, escalation flag) with an LLM "soft" check. The Python
checks alone must catch the most consequential failure modes — buffer
under-application and missed escalations — before any LLM is consulted.
"""
from __future__ import annotations

from typing import Any, Dict

import graph as graph_mod
from graph import _route_after_audit, node_audit


# ---------------------------------------------------------------------------
# Routing — one-line conditional edge that drives the cycle
# ---------------------------------------------------------------------------
def test_route_after_audit_pass_goes_to_allocator():
    assert _route_after_audit({"audit_verdict": "PASS"}) == "allocator"


def test_route_after_audit_fail_loops_back_to_planner():
    assert _route_after_audit({"audit_verdict": "FAIL"}) == "planner"


def test_route_after_audit_missing_verdict_treated_as_pass():
    """No verdict key means the audit didn't run — treat as 'don't loop'."""
    assert _route_after_audit({}) == "allocator"


# ---------------------------------------------------------------------------
# Hard rule checks — buffer policy
# ---------------------------------------------------------------------------
def test_audit_fails_when_buffer_below_policy(fake_weather_risk_critical, planner_violates_buffer):
    _, structured = planner_violates_buffer
    state: Dict[str, Any] = {
        "plan_structured": structured,
        "weather_risk": fake_weather_risk_critical,
        "dispatch_plan": "ignored on FAIL",
        "business_context": "ignored on FAIL",
        "audit_attempts": 1,
    }
    out = node_audit(state)

    assert out["audit_verdict"] == "FAIL"
    assert "buffer" in out["audit_feedback"].lower()
    assert "C1_I95_NJ_BOS" in out["audit_feedback"]


def test_audit_fails_when_escalation_missing_at_score_three(
    fake_weather_risk_critical, planner_missing_escalation, monkeypatch
):
    _, structured = planner_missing_escalation
    # Patch out the GPT soft-check so we know FAIL came from the Python rule
    monkeypatch.setattr(graph_mod, "run_audit_agent",
                        lambda **kw: ("PASS", ""))

    state: Dict[str, Any] = {
        "plan_structured": structured,
        "weather_risk": fake_weather_risk_critical,
        "dispatch_plan": "plan",
        "business_context": "rules",
        "audit_attempts": 1,
    }
    out = node_audit(state)

    assert out["audit_verdict"] == "FAIL"
    assert "escalation" in out["audit_feedback"].lower()


# ---------------------------------------------------------------------------
# Pass path — Python checks clear, LLM soft-check stub returns PASS
# ---------------------------------------------------------------------------
def test_audit_passes_when_plan_compliant(
    fake_weather_risk_critical, planner_pass_first_try, monkeypatch
):
    _, structured = planner_pass_first_try
    monkeypatch.setattr(graph_mod, "run_audit_agent",
                        lambda **kw: ("PASS", ""))

    state: Dict[str, Any] = {
        "plan_structured": structured,
        "weather_risk": fake_weather_risk_critical,
        "dispatch_plan": "good plan",
        "business_context": "rules",
        "audit_attempts": 1,
    }
    out = node_audit(state)

    assert out["audit_verdict"] == "PASS"
    assert out["audit_feedback"] == ""


# ---------------------------------------------------------------------------
# Soft check — Python checks pass but LLM disagrees
# ---------------------------------------------------------------------------
def test_audit_fails_when_llm_soft_check_objects(
    fake_weather_risk_critical, planner_pass_first_try, monkeypatch
):
    _, structured = planner_pass_first_try
    monkeypatch.setattr(
        graph_mod, "run_audit_agent",
        lambda **kw: ("FAIL", "Plan does not explicitly prioritise Tier 1."),
    )

    state: Dict[str, Any] = {
        "plan_structured": structured,
        "weather_risk": fake_weather_risk_critical,
        "dispatch_plan": "vague plan",
        "business_context": "rules",
        "audit_attempts": 1,
    }
    out = node_audit(state)

    assert out["audit_verdict"] == "FAIL"
    assert "Tier 1" in out["audit_feedback"]


# ---------------------------------------------------------------------------
# Retry budget — never loops forever
# ---------------------------------------------------------------------------
def test_audit_force_passes_after_max_attempts(
    fake_weather_risk_critical, planner_violates_buffer, monkeypatch
):
    _, structured = planner_violates_buffer
    monkeypatch.setattr(graph_mod, "run_audit_agent",
                        lambda **kw: ("PASS", ""))

    state: Dict[str, Any] = {
        "plan_structured": structured,
        "weather_risk": fake_weather_risk_critical,
        "dispatch_plan": "still wrong",
        "business_context": "rules",
        "audit_attempts": graph_mod.MAX_AUDIT_ATTEMPTS,  # at the cap
    }
    out = node_audit(state)

    # Force-pass so the graph escapes the loop, but report carries the flag
    assert out["audit_verdict"] == "PASS"
    assert "UNRESOLVED VIOLATIONS" in out["audit_feedback"]


# ---------------------------------------------------------------------------
# No structured plan: the audit should fail safely (not crash)
# ---------------------------------------------------------------------------
def test_audit_handles_missing_structured_plan(
    fake_weather_risk_safe, monkeypatch,
):
    monkeypatch.setattr(graph_mod, "run_audit_agent",
                        lambda **kw: ("PASS", ""))
    state: Dict[str, Any] = {
        "plan_structured": {},
        "weather_risk": fake_weather_risk_safe,
        "dispatch_plan": "narrative only",
        "business_context": "rules",
        "audit_attempts": 1,
    }
    # Should not raise, even with empty structured plan
    out = node_audit(state)
    assert out["audit_verdict"] in ("PASS", "FAIL")
