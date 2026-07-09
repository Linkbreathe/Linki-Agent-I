from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from Linki.core.agent import stream_session_events
from Linki.core.state import create_runtime
from Linki.core.trace import TraceRecorder, recover_trace_artifacts


class QueueModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


def test_trace_snapshot_exists_before_end_and_run_end_has_metrics(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path, trace_id="trace-metrics")
    trace = TraceRecorder(runtime, task="do work")

    trace.start({"task": "do work"})
    trace_path = tmp_path / ".linki" / "traces" / "trace-metrics" / "trace.json"
    assert trace_path.is_file()
    assert json.loads(trace_path.read_text(encoding="utf-8"))["status"] == "running"

    trace.record_custom_event(
        {
            "type": "ai_message",
            "node": "planner",
            "content": "hello",
            "usage_metadata": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        }
    )
    trace.record_custom_event({"type": "tool_call", "node": "planner", "name": "GrepTool"})
    trace.record_custom_event(
        {"type": "tool_result", "node": "planner", "name": "GrepTool", "result": {"ok": False}}
    )
    summary = trace.end(
        status="finished",
        latest_node="final",
        final_state={
            "passed": True,
            "attempts": 2,
            "max_attempts": 3,
            "verification_checks": [{"passed": True}],
        },
    )

    assert summary is not None
    assert summary["verified"] is True
    assert summary["attempts"] == 2
    assert summary["max_attempts"] == 3
    assert summary["failed_tool_calls"] == 1
    assert summary["token_usage"]["total_tokens"] == 7

    run_end = [
        json.loads(line)["event"]
        for line in (trace_path.parent / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"].get("type") == "run_end"
    ][0]
    assert run_end["duration_ms"] >= 0
    assert run_end["verified"] is True
    assert run_end["attempts"] == 2


def test_recover_trace_artifacts_from_events_only(tmp_path: Path) -> None:
    trace_dir = tmp_path / ".linki" / "traces" / "trace-crashed"
    trace_dir.mkdir(parents=True)
    events = [
        {
            "at": "2026-07-09T00:00:00+00:00",
            "event": {
                "type": "run_start",
                "trace_id": "trace-crashed",
                "task": "x",
                "started_at": "2026-07-09T00:00:00+00:00",
            },
        },
        {
            "at": "2026-07-09T00:00:01+00:00",
            "event": {"type": "ai_message", "node": "codeAgent", "content": "x"},
        },
        {
            "at": "2026-07-09T00:00:02+00:00",
            "event": {"type": "error", "error_type": "RuntimeError", "error": "boom"},
        },
    ]
    (trace_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    payload = recover_trace_artifacts(trace_dir)

    assert payload is not None
    assert payload["status"] == "error"
    assert payload["latest_node"] == "codeAgent"
    assert payload["duration_ms"] == 2000
    assert (trace_dir / "trace.json").is_file()
    assert (trace_dir / "timeline.md").is_file()


def test_chat_session_intent_router_is_traced(tmp_path: Path) -> None:
    model = QueueModel(
        [
            AIMessage(content='{"route": "chat", "confidence": 0.99, "reason": "small talk"}'),
            AIMessage(content="hello"),
        ]
    )

    events = list(stream_session_events("hi", session_workspace=tmp_path, model=model))
    trace_event = [event for event in events if event.get("type") == "trace_finished"][-1]
    trace = trace_event["trace"]

    assert trace["node_visits"]["intent_router"] == 1
    assert trace["node_visits"]["chat_responder"] == 1
    trace_path = Path(trace["path"]) / "trace.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["node_visits"]["intent_router"] == 1
