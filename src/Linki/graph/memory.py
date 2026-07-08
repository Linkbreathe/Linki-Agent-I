"""Layered memory assembly for the Linki graph runtime.

Assembles a rules / working-memory / history-summary snapshot from graph
state plus the workspace's NOTEPAD.md and HISTORY_SUMMARY.md files. Agents
never write memory directly; the runtime assembles it and hands it to them
as read-only context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypedDict

from Linki.core.context import PROJECT_RULES_LIMIT, read_project_rules
from Linki.core.paths import resolve_workspace_path
from Linki.core.state import RuntimeState

NOTEPAD_FILENAME = "NOTEPAD.md"
HISTORY_SUMMARY_FILENAME = "HISTORY_SUMMARY.md"

RULES_LAYER: dict[str, Any] = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}


class CompressionEvent(TypedDict, total=False):
    node: str
    reason: str
    token_count: int
    token_limit: int
    summary: str


class WorkingMemory(TypedDict):
    node: str
    task: str
    session_id: str
    session_turn: int
    session_context: str
    plan_summary: str
    todos: list[Any]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    research_notes: str
    sources: list[dict[str, str]]
    agent_handoffs: list[Any]
    code_agent_summary: str
    verifier_summary: str
    last_error: str
    attempts: int
    max_attempts: int


class HistorySummaryStore(TypedDict):
    history_path: str
    history_exists: bool
    history_summary: str
    notepad_path: str
    notepad_exists: bool
    notepad: str
    context_summary: str
    compression_events: list[CompressionEvent]


class LayeredMemory(TypedDict):
    rules: dict[str, Any]
    working_memory: WorkingMemory
    history_summary_store: HistorySummaryStore


def _read_workspace_file(runtime: RuntimeState, filename: str) -> dict[str, Any]:
    path = resolve_workspace_path(runtime, filename)
    if not path.is_file():
        return {"exists": False, "content": ""}
    return {"exists": True, "content": path.read_text(encoding="utf-8")}


def read_notepad(runtime: RuntimeState) -> dict[str, Any]:
    return _read_workspace_file(runtime, NOTEPAD_FILENAME)


def read_history_summary(runtime: RuntimeState) -> dict[str, Any]:
    return _read_workspace_file(runtime, HISTORY_SUMMARY_FILENAME)


def _short_text(text: str, limit: int) -> str:
    """Truncate text when it exceeds the specified limit.

    Append "..." to truncated text.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _trim_handoffs(handoffs: list[Any]) -> list[Any]:
    """Retain only the six most recent agent handoff records."""
    if not handoffs:
        return []
    return list(handoffs[-6:])


def _build_rules_layer(runtime: RuntimeState) -> dict[str, Any]:
    """Return the base rules layer, merging in workspace LINKI.md when present."""

    rules = dict(RULES_LAYER)
    project_rules = read_project_rules(runtime)
    if project_rules:
        rules["project_rules"] = _short_text(project_rules, PROJECT_RULES_LIMIT)
    return rules


def build_layered_memory(state: Mapping[str, Any], *, node: str = "graph") -> LayeredMemory:
    runtime = state["runtime"]

    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)

    working_memory: WorkingMemory = {
        "node": node,
        "task": state.get("task", ""),
        "session_id": state.get("session_id", ""),
        "session_turn": state.get("session_turn", 0),
        "session_context": _short_text(
            state.get("session_context", ""),
            7000,
        ),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": _short_text(
            state.get("research_notes", ""),
            1600,
        ),
        "sources": [
            {
                "title": source.get("title", ""),
                "url": source.get("url", ""),
            }
            for source in state.get("sources", [])
        ],
        "agent_handoffs": _trim_handoffs(
            state.get("agent_handoffs", [])
        ),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""),
            1000,
        ),
        "verifier_summary": _short_text(
            state.get("verifier_summary", ""),
            1000,
        ),
        "last_error": _short_text(
            state.get("last_error", ""),
            1400,
        ),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

    history_summary = history.get("content", "")

    history_summary_store: HistorySummaryStore = {
        "history_path": HISTORY_SUMMARY_FILENAME,
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(
            history_summary,
            2200,
        ),
        "notepad_path": NOTEPAD_FILENAME,
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(
            notepad.get("content", ""),
            1800,
        ),
        "context_summary": _short_text(
            state.get("context_summary", ""),
            1600,
        ),
        "compression_events": state.get(
            "compression_events",
            [],
        )[-3:],
    }

    return {
        "rules": _build_rules_layer(runtime),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def format_layered_memory_for_prompt(memory: LayeredMemory) -> str:
    """Format the layered-memory object using json.dumps()."""
    return json.dumps(memory, ensure_ascii=False, indent=2)


def memory_event(memory: LayeredMemory, *, node: str) -> dict[str, Any]:
    """Build the stream event payload for a layered-memory snapshot handoff."""
    return {"type": "memory_snapshot", "node": node, "memory": memory}
