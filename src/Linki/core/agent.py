import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from Linki.core.checkpoint import CheckpointManager, resume_command
from Linki.core.paths import ensure_workspace
from Linki.core.state import create_runtime
from Linki.core.trace import TraceRecorder
from Linki.graph.workflow import build_complex_workflow
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
        parts = [str(update.get("plan_summary", ""))]
        code_agent_summary = update.get("code_agent_summary")
        if code_agent_summary:
            parts.append(str(code_agent_summary))
        return "\n".join(part for part in parts if part)

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


def _ensure_event_mapping(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    return {"type": "event", "data": _json_safe(event)}


def _extract_latest_node(event: Mapping[str, Any], *, default: str | None = None) -> str | None:
    node = event.get("node")
    if node:
        return str(node)

    latest = default
    for key in event:
        latest = str(key)
    return latest


def _merge_graph_update(current_state: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(current_state)
    for node, update in event.items():
        if isinstance(update, Mapping):
            merged.update(update)
        else:
            merged[str(node)] = update
    return merged


def _candidate_result_payloads(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = [event]
    result = event.get("result")
    if isinstance(result, Mapping):
        candidates.append(result)
        output = result.get("output")
        if isinstance(output, Mapping):
            candidates.append(output)
        elif isinstance(output, str):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                candidates.append(parsed)
    return candidates


def _tool_result_failed(event: Mapping[str, Any]) -> bool:
    return any(payload.get("ok") is False for payload in _candidate_result_payloads(event))


def _tool_result_requires_approval(event: Mapping[str, Any]) -> bool:
    return any(bool(payload.get("requires_approval")) for payload in _candidate_result_payloads(event))


def _custom_event_needs_checkpoint(event: Mapping[str, Any]) -> bool:
    event_type = event.get("type")
    if event_type in {"handoff", "checkpoint_resumed"}:
        return True
    if event_type == "tool_result":
        return _tool_result_failed(event) or _tool_result_requires_approval(event)
    return False


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: Callable[[Any], Any] | None = None,
    checkpoint_mode: str = "light",
    resume_workspace: str | Path | None = None,
    trace_mode: str = "on",
    provider: str = "openai",
    model_name: str | None = None,
    model: Any | None = None,
) -> Iterator[dict]:
    """Stream graph/custom events while recording checkpoints and traces."""

    runtime = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        resume_from=resume_workspace,
        trace_mode=trace_mode,
    )
    ensure_workspace(runtime, create=True)

    manager = CheckpointManager(runtime, task=task)
    trace = TraceRecorder(runtime, task=task)

    resume_event: dict[str, Any] | None = None
    current_state: dict[str, Any] = {
        "task": task,
        "runtime": runtime,
        "attempts": 0,
        "max_attempts": max_attempts,
        "provider": provider,
        "model_name": model_name,
    }
    latest_node: str | None = "start"
    trace_started = False

    try:
        if resume_workspace is not None:
            inputs, resume_event = CheckpointManager.load_resume_inputs(
                runtime,
                task=task or None,
                max_attempts=max_attempts,
            )
            restored_task = str(inputs.get("task") or task)
            manager.task = restored_task
            trace.task = restored_task
            inputs["runtime"] = runtime
            inputs["provider"] = provider
            inputs["model_name"] = model_name
            inputs["model"] = model or create_model(provider=provider, model=model_name)
        else:
            inputs = {
                "task": task,
                "runtime": runtime,
                "attempts": 0,
                "max_attempts": max_attempts,
                "provider": provider,
                "model_name": model_name,
                "model": model or create_model(provider=provider, model=model_name),
            }

        current_state = dict(inputs)
        workflow = build_complex_workflow()

        trace.start(
            inputs,
            resumed=resume_workspace is not None,
            resume_event=resume_event,
        )
        trace_started = True

        checkpoint_event = manager.save(
            current_state,
            status="started",
            latest_node="start",
        )
        if checkpoint_event is not None:
            trace.record_custom_event(checkpoint_event)

        for raw_event in workflow.stream(inputs, stream_mode=["updates", "custom"]):
            if isinstance(raw_event, tuple) and len(raw_event) == 2:
                mode, event = raw_event
            else:
                mode, event = "updates", raw_event

            if mode == "custom":
                custom_event = _ensure_event_mapping(event)
                trace.record_custom_event(custom_event)

                if manager.mode == "strict" or _custom_event_needs_checkpoint(custom_event):
                    checkpoint_event = manager.save(
                        current_state,
                        status="running",
                        latest_node=latest_node,
                        event=custom_event,
                    )
                    if checkpoint_event is not None:
                        trace.record_custom_event(checkpoint_event)

                yield {
                    "type": "custom_event",
                    "event": custom_event,
                }
                continue

            graph_event = _ensure_event_mapping(event)
            trace.record_graph_update(graph_event)
            current_state = _merge_graph_update(current_state, graph_event)
            latest_node = _extract_latest_node(graph_event, default=latest_node)

            checkpoint_event = manager.save(
                current_state,
                status="running",
                latest_node=latest_node,
                event=graph_event,
            )
            if checkpoint_event is not None:
                trace.record_custom_event(checkpoint_event)

            yield {
                "type": "graph_event",
                "event": graph_event,
            }

        checkpoint_event = manager.save(
            current_state,
            status="finished",
            latest_node=latest_node,
        )
        if checkpoint_event is not None:
            trace.record_custom_event(checkpoint_event)

        trace_summary = trace.end(
            status="finished",
            latest_node=latest_node,
            final_state=current_state,
        )

        yield {
            "type": "trace_finished",
            "trace": trace_summary,
        }
    except KeyboardInterrupt:
        if not trace_started:
            trace.start(
                current_state,
                resumed=resume_workspace is not None,
                resume_event=resume_event,
            )

        checkpoint_event = manager.save(
            current_state,
            status="interrupted",
            latest_node=latest_node,
        )
        if checkpoint_event is not None:
            trace.record_custom_event(checkpoint_event)

        trace.end(
            status="interrupted",
            latest_node=latest_node,
            final_state=current_state,
        )

        yield {
            "type": "interrupted",
            "workspace": str(runtime.workspace),
            "resume_command": resume_command(runtime.workspace),
        }
        raise
    except Exception as exc:
        error_event = {
            "type": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not trace_started:
            trace.start(
                current_state,
                resumed=resume_workspace is not None,
                resume_event=resume_event,
            )
        trace.record_custom_event(error_event)

        checkpoint_event = manager.save(
            current_state,
            status="error",
            latest_node=latest_node,
            event=error_event,
        )
        if checkpoint_event is not None:
            trace.record_custom_event(checkpoint_event)

        trace.end(
            status="error",
            latest_node=latest_node,
            final_state=current_state,
        )
        raise
