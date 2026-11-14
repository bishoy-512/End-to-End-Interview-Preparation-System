

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import AgentSet, build_agents
from .nodes import make_nodes
from .state import InterviewState


def build_checkpointer(db_path: str | None = None):
    """SQLite checkpointer (survives restarts). ORCH_DB=:memory: -> in-memory."""
    db_path = db_path or os.getenv("ORCH_DB", "orchestrator_sessions.sqlite")
    if db_path == ":memory:":
        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    except ImportError:  # langgraph-checkpoint-sqlite not installed
        return MemorySaver()


def build_graph(agents: AgentSet | None = None, checkpointer=None):
    agents = agents or build_agents()
    n = make_nodes(agents)

    g = StateGraph(InterviewState)
    for name, fn in n.items():
        g.add_node(name, fn)

    g.add_edge(START, "analyze")
    g.add_conditional_edges(
        "analyze",
        lambda s: "stop" if s.get("status") == "failed" else "go",
        {"stop": END, "go": "ask_question"},
    )
    g.add_conditional_edges(
        "ask_question",
        lambda s: "done" if s.get("interview_done") else "answer",
        {"done": "aggregate", "answer": "await_answer"},
    )
    g.add_conditional_edges(
        "await_answer",
        lambda s: "done" if s.get("interview_done") else "evaluate",
        {"done": "aggregate", "evaluate": "evaluate"},
    )
    g.add_edge("evaluate", "ask_question")
    g.add_edge("aggregate", "report")
    g.add_edge("report", "learning")
    g.add_edge("learning", END)

    return g.compile(checkpointer=checkpointer or build_checkpointer())
