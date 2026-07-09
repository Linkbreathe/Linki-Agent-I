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
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _final_state_summary(final_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "passed": final_state.get("passed"),
        "attempts": final_state.get("attempts", 0),
        "max_attempts": final_state.get("max_attempts", 0),
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


def _verification_summary(final_state: Mapping[str, Any]) -> dict[str, Any]:
    checks = final_state.get("verification_checks") or []
    if not isinstance(checks, list):
        checks = []
    failed = [
        check
        for check in checks
        if isinstance(check, Mapping) and check.get("passed") is False
    ]
    return {
        "verified": bool(final_state.get("passed")),
        "attempts": int(final_state.get("attempts") or 0),
        "max_attempts": int(final_state.get("max_attempts") or 0),
        "verification_checks": len(checks),
        "failed_verification_checks": len(failed),
    }


def _usage_from_event(event: Mapping[str, Any]) -> dict[str, int]:
    raw = event.get("usage_metadata")
    if not isinstance(raw, Mapping):
        response_metadata = event.get("response_metadata")
        if isinstance(response_metadata, Mapping):
            raw = response_metadata.get("token_usage")
    if not isinstance(raw, Mapping):
        return {}

    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens", 0))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens", 0))
    total_tokens = raw.get("total_tokens", 0)
    try:
        input_value = int(input_tokens or 0)
        output_value = int(output_tokens or 0)
        total_value = int(total_tokens or input_value + output_value)
    except (TypeError, ValueError):
        return {}
    return {
        "input_tokens": input_value,
        "output_tokens": output_value,
        "total_tokens": total_value,
    }


def _merge_usage(target: dict[str, int], usage: Mapping[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        target[key] = int(target.get(key, 0)) + int(usage.get(key, 0))


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
    agent = event.get("agent")
    label = event_type + (f" @ {node}" if node else "")
    if agent:
        label = f"{label} [{agent}]"

    detail = ""
    if event_type in {"ai_message", "node_update"}:
        detail = str(event.get("content", ""))[:160]
    elif event_type == "subagent_start":
        detail = str(event.get("description") or "")
    elif event_type == "subagent_result":
        detail = str(event.get("summary") or "")[:160]
    elif event_type == "graph_update":
        detail = ", ".join(_updated_nodes(event))
    elif event_type in {"tool_call", "tool_result"}:
        detail = str(event.get("name", ""))
    elif event_type == "approval_requested":
        detail = f"{event.get('tool')} {event.get('reason', '')}".strip()
    elif event_type == "approval_decision":
        detail = f"{event.get('tool')} approved={event.get('approved')}"
    elif event_type == "handoff":
        detail = f"{event.get('from')} -> {event.get('to')}"
    elif event_type == "checkpoint_saved":
        detail = f"status={event.get('status')} node={event.get('latest_node')}"
    elif event_type == "run_start":
        detail = f"resumed={event.get('resumed')}"
    elif event_type == "run_end":
        detail = (
            f"status={event.get('status')} verified={event.get('verified')} "
            f"attempts={event.get('attempts')}/{event.get('max_attempts')}"
        )

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
        self.approval_requested_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0
        self.token_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        self._timeline: list[dict[str, Any]] = []
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._status = "running"
        self._latest_node: str | None = None
        self._final_state_summary: dict[str, Any] = {}
        self._verification: dict[str, Any] = {
            "verified": False,
            "attempts": 0,
            "max_attempts": 0,
            "verification_checks": 0,
            "failed_verification_checks": 0,
        }

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
        self._status = "running"
        self._latest_node = "start"

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
            if event.get("latest_node"):
                self._latest_node = str(event.get("latest_node"))
        elif event_type == "approval_requested":
            self.approval_requested_count += 1
        elif event_type == "ai_message":
            _merge_usage(self.token_usage, _usage_from_event(event))

        self._append_event(dict(event))

    def record_graph_update(self, event: Mapping[str, Any]) -> None:
        """Record a graph update event and track node visits."""

        if not self.enabled:
            return

        for node in _updated_nodes(event):
            self.node_visits[node] = self.node_visits.get(node, 0) + 1
            self._latest_node = node

        self._append_event(dict(event))

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {"at": _now_iso(), "event": dict(event)}
        self._timeline.append(entry)

        events_path = self.root / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._write_snapshot()

    def _duration_ms(self, ended_at: str | None = None) -> int:
        end = ended_at or self._ended_at or _now_iso()
        start = self._started_at or end
        return int(
            (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000
        )

    def _payload(self) -> dict[str, Any]:
        ended_at = self._ended_at or ""
        timeline = self._timeline
        head = timeline[:TIMELINE_HEAD_LIMIT]
        tail_start = max(len(timeline) - TIMELINE_TAIL_LIMIT, len(head))
        tail = timeline[tail_start:]
        omitted = max(len(timeline) - len(head) - len(tail), 0)

        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "status": self._status,
            "started_at": self._started_at or "",
            "ended_at": ended_at,
            "duration_ms": self._duration_ms(self._ended_at),
            "latest_node": self._latest_node,
            "verified": self._verification.get("verified", False),
            "attempts": self._verification.get("attempts", 0),
            "max_attempts": self._verification.get("max_attempts", 0),
            "verification_checks": self._verification.get("verification_checks", 0),
            "failed_verification_checks": self._verification.get("failed_verification_checks", 0),
            "final_state_summary": self._final_state_summary,
            "node_visits": self.node_visits,
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "approval_requested_count": self.approval_requested_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "token_usage": dict(self.token_usage),
            "timeline_head": head,
            "timeline_tail": tail,
            "timeline_omitted": omitted,
        }

    def _write_snapshot(self) -> None:
        payload = self._payload()
        _write_json(self.root / "trace.json", payload)
        _write_text_atomic(self.root / "timeline.md", build_timeline_markdown(payload))

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

        ended_at = _now_iso()
        self._ended_at = ended_at
        self._status = status
        self._latest_node = latest_node
        self._final_state_summary = _final_state_summary(final_state)
        self._verification = _verification_summary(final_state)
        duration_ms = self._duration_ms(ended_at)

        self._append_event(
            {
                "type": "run_end",
                "status": status,
                "latest_node": latest_node,
                "duration_ms": duration_ms,
                **self._verification,
                "tool_calls": self.tool_calls,
                "failed_tool_calls": self.failed_tool_calls,
                "approval_count": self.approval_count,
                "approval_requested_count": self.approval_requested_count,
                "checkpoint_count": self.checkpoint_count,
                "handoff_count": self.handoff_count,
                "token_usage": dict(self.token_usage),
                "final_state_summary": self._final_state_summary,
            }
        )

        payload = self._payload()
        self._write_snapshot()

        return {**payload, "path": str(self.root)}


def recover_trace_artifacts(trace_dir: str | Path) -> dict[str, Any] | None:
    """Rebuild trace.json and timeline.md from events.jsonl.

    This is intentionally best-effort: it exists for crashed or killed runs
    where per-event JSONL survived but the final aggregate files did not.
    """

    root = Path(trace_dir)
    events_path = root / "events.jsonl"
    if not events_path.is_file():
        return None

    timeline: list[dict[str, Any]] = []
    node_visits: dict[str, int] = {}
    tool_calls = 0
    failed_tool_calls = 0
    approval_count = 0
    approval_requested_count = 0
    checkpoint_count = 0
    handoff_count = 0
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    trace_id = root.name
    task = ""
    started_at = ""
    ended_at = ""
    status = "incomplete"
    latest_node: str | None = None
    final_state_summary: dict[str, Any] = {}
    verification: dict[str, Any] = {
        "verified": False,
        "attempts": 0,
        "max_attempts": 0,
        "verification_checks": 0,
        "failed_verification_checks": 0,
    }

    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, Mapping):
            continue
        event = entry.get("event")
        if not isinstance(event, Mapping):
            continue
        timeline.append({"at": str(entry.get("at") or ""), "event": dict(event)})
        event_type = _event_type(event)

        if event_type == "run_start":
            trace_id = str(event.get("trace_id") or trace_id)
            task = str(event.get("task") or task)
            started_at = str(event.get("started_at") or entry.get("at") or started_at)
            status = "running"
            latest_node = "start"
        elif event_type == "tool_call":
            tool_calls += 1
        elif event_type == "tool_result":
            results = _tool_result_payloads(event)
            if any(result.get("ok") is False for result in results):
                failed_tool_calls += 1
            if any(result.get("requires_approval") for result in results):
                approval_count += 1
        elif event_type == "approval_requested":
            approval_requested_count += 1
        elif event_type == "checkpoint_saved":
            checkpoint_count += 1
            if event.get("latest_node"):
                latest_node = str(event.get("latest_node"))
        elif event_type == "handoff":
            handoff_count += 1
        elif event_type == "ai_message":
            _merge_usage(token_usage, _usage_from_event(event))
        elif event_type == "run_end":
            status = str(event.get("status") or status)
            latest_node = str(event.get("latest_node") or latest_node or "")
            ended_at = str(entry.get("at") or "")
            summary = event.get("final_state_summary")
            if isinstance(summary, Mapping):
                final_state_summary = dict(summary)
            for key in verification:
                if key in event:
                    verification[key] = event[key]
        elif event_type == "error":
            status = "error"

        if event.get("node"):
            node = str(event["node"])
            node_visits[node] = node_visits.get(node, 0) + 1
            latest_node = node
        elif event_type == "event":
            for node in _updated_nodes(event):
                node_visits[node] = node_visits.get(node, 0) + 1
                latest_node = node

    if timeline and not started_at:
        started_at = str(timeline[0].get("at") or "")
    if timeline and not ended_at and status != "running":
        ended_at = str(timeline[-1].get("at") or "")

    duration_ms = 0
    if started_at:
        end_for_duration = ended_at or str(timeline[-1].get("at") or started_at)
        try:
            duration_ms = int(
                (datetime.fromisoformat(end_for_duration) - datetime.fromisoformat(started_at)).total_seconds()
                * 1000
            )
        except ValueError:
            duration_ms = 0

    head = timeline[:TIMELINE_HEAD_LIMIT]
    tail_start = max(len(timeline) - TIMELINE_TAIL_LIMIT, len(head))
    tail = timeline[tail_start:]
    omitted = max(len(timeline) - len(head) - len(tail), 0)
    payload = {
        "trace_id": trace_id,
        "task": task,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "latest_node": latest_node,
        **verification,
        "final_state_summary": final_state_summary,
        "node_visits": node_visits,
        "tool_calls": tool_calls,
        "failed_tool_calls": failed_tool_calls,
        "approval_count": approval_count,
        "approval_requested_count": approval_requested_count,
        "checkpoint_count": checkpoint_count,
        "handoff_count": handoff_count,
        "token_usage": token_usage,
        "timeline_head": head,
        "timeline_tail": tail,
        "timeline_omitted": omitted,
    }

    _write_json(root / "trace.json", payload)
    _write_text_atomic(root / "timeline.md", build_timeline_markdown(payload))
    return {**payload, "path": str(root)}
