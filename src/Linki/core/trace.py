"""Execution tracing for Linki graph runs.

Records a structured, replayable trace of a run under
``<workspace>/.linki/traces/{trace_id}``: every emitted event plus derived
statistics (tool calls, handoffs, checkpoints, node visits) and a
human-readable timeline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from Linki.core.state import RuntimeState

VALID_TRACE_MODES = {"on", "off"}

TIMELINE_HEAD_LIMIT = 20
TIMELINE_TAIL_LIMIT = 80


def normalize_trace_mode(mode: str | None) -> str:
    """Normalize the trace mode.

    Missing or invalid values fall back to "on".
    """

    if mode in VALID_TRACE_MODES:
        return mode
    return "on"


def generate_trace_id() -> str:
    """Generate a chronologically sortable, unique trace id."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _final_state_summary(final_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "passed": final_state.get("passed"),
        "attempts": final_state.get("attempts", 0),
        "plan_summary": final_state.get("plan_summary", ""),
        "last_error": final_state.get("last_error", ""),
    }


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type") or "event")


def _tool_result_payloads(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = [event]
    result = event.get("result")
    if isinstance(result, Mapping):
        payloads.append(result)
        output = result.get("output")
        if isinstance(output, Mapping):
            payloads.append(output)
        elif isinstance(output, str):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                payloads.append(parsed)
    return payloads


def _updated_nodes(event: Mapping[str, Any]) -> list[str]:
    node = event.get("node")
    if node:
        return [str(node)]

    nodes: list[str] = []
    for key, value in event.items():
        if isinstance(value, Mapping):
            nodes.append(str(key))
    return nodes


def _timeline_entry_line(entry: Mapping[str, Any]) -> str:
    at = entry.get("at", "")
    event = entry.get("event") or {}
    event_type = _event_type(event)
    node = event.get("node")
    if event_type == "event" and not node:
        nodes = _updated_nodes(event)
        if nodes:
            event_type = "graph_update"
            node = ", ".join(nodes)
    label = event_type + (f" @ {node}" if node else "")

    detail = ""
    if event_type in {"ai_message", "node_update"}:
        detail = str(event.get("content", ""))[:160]
    elif event_type == "graph_update":
        detail = ", ".join(_updated_nodes(event))
    elif event_type in {"tool_call", "tool_result"}:
        detail = str(event.get("name", ""))
    elif event_type == "handoff":
        detail = f"{event.get('from')} -> {event.get('to')}"
    elif event_type == "checkpoint_saved":
        detail = f"status={event.get('status')} node={event.get('latest_node')}"
    elif event_type == "run_start":
        detail = f"resumed={event.get('resumed')}"
    elif event_type == "run_end":
        detail = f"status={event.get('status')}"

    suffix = f" — {detail}" if detail else ""
    return f"- `{at}` **{label}**{suffix}"


def build_timeline_markdown(payload: Mapping[str, Any]) -> str:
    """Generate the contents of timeline.md from a trace payload."""

    lines = [
        f"# Linki Trace: {payload.get('trace_id', '')}",
        "",
        f"- **Task**: {payload.get('task', '')}",
        f"- **Status**: {payload.get('status', '')}",
        f"- **Latest node**: {payload.get('latest_node') or '—'}",
        f"- **Started at**: {payload.get('started_at', '')}",
        f"- **Ended at**: {payload.get('ended_at', '')}",
        f"- **Duration**: {payload.get('duration_ms', 0)} ms",
        "",
        "## Statistics",
        "",
        f"- Node visits: {payload.get('node_visits', {})}",
        f"- Tool calls: {payload.get('tool_calls', 0)} (failed: {payload.get('failed_tool_calls', 0)})",
        f"- Approvals: {payload.get('approval_count', 0)}",
        f"- Checkpoints: {payload.get('checkpoint_count', 0)}",
        f"- Handoffs: {payload.get('handoff_count', 0)}",
        "",
        "## Timeline",
        "",
    ]

    head = payload.get("timeline_head") or []
    tail = payload.get("timeline_tail") or []
    omitted = int(payload.get("timeline_omitted", 0))

    for entry in head:
        lines.append(_timeline_entry_line(entry))

    if omitted:
        lines.append("")
        lines.append(f"_... {omitted} events omitted ..._")
        lines.append("")

    for entry in tail:
        lines.append(_timeline_entry_line(entry))

    lines.append("")
    return "\n".join(lines)


class TraceRecorder:
    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.workspace = runtime.workspace
        self.mode = normalize_trace_mode(runtime.trace_mode)
        self.task = task

        self.trace_id = runtime.trace_id or generate_trace_id()

        self.root = self.workspace / ".linki" / "traces" / self.trace_id

        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0

        self._timeline: list[dict[str, Any]] = []
        self._started_at: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def start(
        self,
        inputs: Mapping[str, Any],
        *,
        resumed: bool = False,
        resume_event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Record a run_start event. Returns None when tracing is disabled."""

        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        self._started_at = _now_iso()

        event: dict[str, Any] = {
            "type": "run_start",
            "trace_id": self.trace_id,
            "task": self.task,
            "workspace": str(self.workspace),
            "started_at": self._started_at,
            "resumed": resumed,
        }
        if resume_event is not None:
            event["resume_event"] = dict(resume_event)

        self._append_event(event)
        return event

    def record_custom_event(self, event: Mapping[str, Any]) -> None:
        """Record a custom runtime event and update trace statistics."""

        if not self.enabled:
            return

        event_type = _event_type(event)

        if event_type == "tool_call":
            self.tool_calls += 1
        elif event_type == "tool_result":
            results = _tool_result_payloads(event)
            if any(result.get("ok") is False for result in results):
                self.failed_tool_calls += 1
            if any(result.get("requires_approval") for result in results):
                self.approval_count += 1
        elif event_type == "handoff":
            self.handoff_count += 1
        elif event_type == "checkpoint_saved":
            self.checkpoint_count += 1

        self._append_event(dict(event))

    def record_graph_update(self, event: Mapping[str, Any]) -> None:
        """Record a graph update event and track node visits."""

        if not self.enabled:
            return

        for node in _updated_nodes(event):
            self.node_visits[node] = self.node_visits.get(node, 0) + 1

        self._append_event(dict(event))

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {"at": _now_iso(), "event": dict(event)}
        self._timeline.append(entry)

        events_path = self.root / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def end(
        self,
        *,
        status: str,
        latest_node: str | None,
        final_state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Finish the trace and generate trace.json and timeline.md."""

        if not self.enabled:
            return None

        self._append_event(
            {
                "type": "run_end",
                "status": status,
                "latest_node": latest_node,
                "final_state_summary": _final_state_summary(final_state),
            }
        )

        ended_at = _now_iso()
        started_at = self._started_at or ended_at
        duration_ms = int(
            (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds() * 1000
        )

        timeline = self._timeline
        head = timeline[:TIMELINE_HEAD_LIMIT]
        tail_start = max(len(timeline) - TIMELINE_TAIL_LIMIT, len(head))
        tail = timeline[tail_start:]
        omitted = max(len(timeline) - len(head) - len(tail), 0)

        payload = {
            "trace_id": self.trace_id,
            "task": self.task,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "latest_node": latest_node,
            "node_visits": self.node_visits,
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "timeline_head": head,
            "timeline_tail": tail,
            "timeline_omitted": omitted,
        }

        _write_json(self.root / "trace.json", payload)
        (self.root / "timeline.md").write_text(build_timeline_markdown(payload), encoding="utf-8")

        return {**payload, "path": str(self.root)}
