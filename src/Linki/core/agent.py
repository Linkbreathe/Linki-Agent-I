import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from Linki.core.paths import ensure_workspace
from Linki.core.state import RuntimeState
from Linki.graph.workflow import build_workflow
from Linki.providers.openai_provider import create_model


def _message_content(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        data = {
            "type": getattr(value, "type", type(value).__name__),
            "content": _message_content(value),
        }
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = tool_calls
        return data

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _content_for_node(node: str, update: Mapping[str, Any]) -> str:
    if node == "planner":
        return str(update.get("plan_summary", ""))

    if node == "actor":
        return str(update.get("last_actor_summary", ""))

    if node == "verifier":
        passed = bool(update.get("passed"))
        reason = "passed" if passed else "failed"
        checks = update.get("verification_checks") or []
        return f"Verification {reason}. Checks: {len(checks)}"

    if node == "final":
        return str(update.get("final_answer", ""))

    return ""


def _node_update_event(node: str, update: Mapping[str, Any]) -> dict:
    safe_update = _json_safe(update)
    return {
        "type": "node_update",
        "node": node,
        "content": _content_for_node(node, update),
        "data": safe_update,
    }


def _parse_graph_event(raw_event: Any) -> Iterator[dict]:
    if isinstance(raw_event, tuple) and len(raw_event) == 2:
        mode, payload = raw_event
    else:
        mode, payload = "updates", raw_event

    if mode == "updates" and isinstance(payload, Mapping):
        for node, update in payload.items():
            if isinstance(update, Mapping):
                yield _node_update_event(str(node), update)
            else:
                yield {
                    "type": "node_update",
                    "node": str(node),
                    "content": "",
                    "data": _json_safe(update),
                }
        return

    if mode == "custom":
        if isinstance(payload, Mapping):
            event = dict(payload)
            event.setdefault("type", "custom")
            event["data"] = _json_safe(event.get("data", payload))
            yield _json_safe(event)
        else:
            yield {"type": "custom", "data": _json_safe(payload)}
        return

    yield {
        "type": "graph_event",
        "mode": str(mode),
        "data": _json_safe(payload),
    }


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_attempts: int = 3,
    provider: str = "openai",
    model_name: str | None = None,
    model: Any | None = None,
) -> Iterator[dict]:
    """Stream normalized events from Linki's LangGraph workflow."""

    runtime = RuntimeState(workspace=Path(workspace))
    ensure_workspace(runtime, create=True)

    inputs = {
        "task": task,
        "runtime": runtime,
        "attempts": 0,
        "max_attempts": max_attempts,
        "provider": provider,
        "model_name": model_name,
        "model": model or create_model(provider=provider, model=model_name),
    }

    workflow = build_workflow()
    for raw_event in workflow.stream(inputs, stream_mode=["updates", "custom"]):
        yield from _parse_graph_event(raw_event)
