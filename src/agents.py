from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from prompts import (
    PDF_CONTEXT_PROMPT, OPS_ANALYSIS_PROMPT,
    PLANNER_PROMPT, AUDIT_PROMPT,
    ALLOCATOR_PROMPT, REPORT_PROMPT,
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    tags=["msba-demo", "multi-agent"],
    metadata={"repo": "MSBA_AI_Agents_Demo"},
)

# ---------------------------------------------------------------------------
# Agent 1 — Context (PDF rules)
# ---------------------------------------------------------------------------
def run_context_agent(snippets: str) -> str:
    return llm.invoke(PDF_CONTEXT_PROMPT.format_messages(snippets=snippets)).content


# ---------------------------------------------------------------------------
# Agent 2 — Ops data interpreter
# ---------------------------------------------------------------------------
def run_ops_agent(summary: Dict[str, Any], kpis: Dict[str, Any], anomalies_md: str) -> str:
    return llm.invoke(OPS_ANALYSIS_PROMPT.format_messages(
        summary=summary, kpis=kpis, anomalies_md=anomalies_md
    )).content


# ---------------------------------------------------------------------------
# Agent 3 — Planner (returns prose + structured JSON for audit)
# ---------------------------------------------------------------------------
def run_planner_agent(
    business_context: str,
    ops_insights: str,
    weather_risk: Dict[str, Any],
    corridor_kpis: List[Dict[str, Any]],
    resource_pool: Dict[str, Any],
    audit_feedback: str = "",
    scenario: str = "",
) -> Tuple[str, Dict[str, Any]]:

    raw = llm.invoke(PLANNER_PROMPT.format_messages(
        business_context=business_context,
        ops_insights=ops_insights,
        weather_risk=json.dumps(weather_risk, indent=2),
        corridor_kpis=json.dumps(corridor_kpis, indent=2),
        resource_pool=json.dumps(resource_pool, indent=2),
        audit_feedback=audit_feedback or "None — first attempt.",
        scenario=scenario or "None — run standard analysis.",
    )).content

    # Extract the JSON block the prompt asks for, fall back to safe defaults
    plan_structured: Dict[str, Any] = {}
    try:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        plan_structured = json.loads(raw[start:end].strip())
    except (ValueError, json.JSONDecodeError):
        # If GPT didn't wrap JSON properly, try to parse the whole response
        try:
            plan_structured = json.loads(raw.strip())
        except json.JSONDecodeError:
            plan_structured = {}

    return raw, plan_structured


# ---------------------------------------------------------------------------
# Agent 4 — Audit (GPT soft check — only runs after Python hard checks pass)
# ---------------------------------------------------------------------------
def run_audit_agent(
    dispatch_plan: str,
    business_context: str,
    weather_risk: Dict[str, Any],
) -> Tuple[str, str]:

    raw = llm.invoke(AUDIT_PROMPT.format_messages(
        dispatch_plan=dispatch_plan,
        business_context=business_context,
        weather_risk=json.dumps(weather_risk, indent=2),
    )).content

    upper = raw.strip().upper()
    if upper.startswith("PASS"):
        return "PASS", ""
    return "FAIL", raw.strip()


# ---------------------------------------------------------------------------
# Agent 5 — Resource allocator
# ---------------------------------------------------------------------------
def run_allocator_agent(
    corridor_kpis: List[Dict[str, Any]],
    resource_pool: Dict[str, Any],
    weather_risk: Dict[str, Any],
    business_context: str,
) -> Dict[str, Any]:

    raw = llm.invoke(ALLOCATOR_PROMPT.format_messages(
        corridor_kpis=json.dumps(corridor_kpis, indent=2),
        resource_pool=json.dumps(resource_pool, indent=2),
        weather_risk=json.dumps(weather_risk, indent=2),
        business_context=business_context,
    )).content

    try:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        return json.loads(raw[start:end].strip())
    except (ValueError, json.JSONDecodeError):
        return {"narrative": raw, "raw": True}


# ---------------------------------------------------------------------------
# Agent 6 — Report
# ---------------------------------------------------------------------------
def run_report_agent(
    business_context: str,
    kpis: Dict[str, Any],
    corridor_kpis: List[Dict[str, Any]],
    trend_summary: Dict[str, Any],
    anomaly_highlights: str,
    weather_risk: Dict[str, Any],
    dispatch_plan: str,
    allocation_plan: Dict[str, Any],
    audit_feedback: str = "",
    scenario: str = "",
    human_approved: bool = True,
) -> str:
    return llm.invoke(REPORT_PROMPT.format_messages(
        business_context=business_context,
        kpis=json.dumps(kpis, indent=2),
        corridor_kpis=json.dumps(corridor_kpis, indent=2),
        trend_summary=json.dumps(trend_summary, indent=2),
        anomaly_highlights=anomaly_highlights,
        weather_risk=json.dumps(weather_risk, indent=2),
        dispatch_plan=dispatch_plan,
        allocation_plan=json.dumps(allocation_plan, indent=2),
        audit_feedback=audit_feedback or "None — plan passed audit.",
        scenario=scenario or "Standard analysis — no scenario override.",
        human_approved=str(human_approved),
    )).content
