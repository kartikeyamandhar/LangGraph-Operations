"""
Graph-assembly tests.

We don't run the graph end-to-end here (that needs live OpenAI + Open-Meteo).
We instead verify that build_graph() produces a compiled object whose nodes,
edges, conditional edges, and entrypoint are wired exactly as the architecture
demands. A typo in graph.py that breaks the loop or skips a node should be
caught by these tests.
"""
from __future__ import annotations

from langgraph.graph import START, END

import graph as graph_mod
from graph import build_graph


EXPECTED_NODES = {
    "pdf_context",
    "csv_analysis",
    "weather",
    "planner",
    "audit",
    "allocator",
    "human_checkpoint",
    "report",
    "email",
}


def test_build_graph_returns_compiled():
    app = build_graph()
    # Compiled graphs expose .invoke / .stream
    assert hasattr(app, "invoke")
    assert hasattr(app, "stream")


def test_all_expected_nodes_registered():
    app = build_graph()
    nodes = set(app.get_graph().nodes.keys())
    # Compiled graph also includes synthetic START / __end__ entries
    assert EXPECTED_NODES <= nodes, f"Missing nodes: {EXPECTED_NODES - nodes}"


def test_parallel_fanout_from_start():
    """All three data-gathering nodes must be reachable from START."""
    app = build_graph()
    edges = app.get_graph().edges
    sources_of_data_nodes = {
        e.source for e in edges
        if e.target in {"pdf_context", "csv_analysis", "weather"}
    }
    assert "__start__" in sources_of_data_nodes


def test_data_nodes_converge_on_planner():
    app = build_graph()
    edges = app.get_graph().edges
    data_targets = {
        e.target for e in edges
        if e.source in {"pdf_context", "csv_analysis", "weather"}
    }
    assert data_targets == {"planner"}


def test_audit_has_conditional_edges_to_planner_and_allocator():
    app = build_graph()
    edges = app.get_graph().edges
    audit_targets = {e.target for e in edges if e.source == "audit"}
    assert {"planner", "allocator"} <= audit_targets


def test_linear_tail_after_audit():
    """allocator → human_checkpoint → report → email → END."""
    app = build_graph()
    edges = app.get_graph().edges
    pairs = {(e.source, e.target) for e in edges}

    assert ("allocator",        "human_checkpoint") in pairs
    assert ("human_checkpoint", "report") in pairs
    assert ("report",           "email") in pairs
    # email → END is the terminal edge
    assert any(s == "email" and t in ("__end__", END) for s, t in pairs)


def test_graph_is_acyclic_except_for_audit_loop():
    """Only edge that goes 'backwards' is audit → planner."""
    app = build_graph()
    edges = app.get_graph().edges
    pairs = {(e.source, e.target) for e in edges}

    # The cyclic edge we expect
    assert ("audit", "planner") in pairs

    # No other edge goes back to a data-gathering node
    backward_targets = {"pdf_context", "csv_analysis", "weather"}
    illegal = {(s, t) for s, t in pairs if t in backward_targets and s != "__start__"}
    assert not illegal, f"Unexpected backward edges: {illegal}"


def test_build_graph_accepts_optional_checkpointer():
    """The Streamlit UI passes a MemorySaver — build_graph must accept one."""
    from langgraph.checkpoint.memory import MemorySaver
    app = build_graph(checkpointer=MemorySaver())
    assert hasattr(app, "invoke")
