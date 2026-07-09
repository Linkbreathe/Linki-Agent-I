import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from Linki.core.checkpoint import CheckpointManager, resume_command
from Linki.core.context import assemble_project_context
from Linki.core.hooks import load_hooks_config
from Linki.core.memory_store import extract_run_memories
from Linki.core.paths import ensure_scratch_dir, ensure_workspace
from Linki.core.session import (
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    resolve_session_workspace,
    save_session,
)
from Linki.core.state import create_runtime
from Linki.core.trace import TraceRecorder
from Linki.skills.registry import load_skills_into_runtime
from Linki.graph.workflow import build_complex_workflow, build_entry_workflow
from Linki.providers.openai_provider import create_model, validate_provider_config


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
    if node == "intent_router":
        route = str(update.get("intent_route", "workflow"))
        confidence = update.get("intent_confidence", 0.0)
        reason = str(update.get("intent_reason", ""))
        return f"Route: {route} (confidence={confidence}). {reason}".strip()

    if node == "chat_responder":
        return str(update.get("chat_response") or update.get("final_answer") or "")

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
    session_id: str = "",
    session_turn: int = 0,
    session_context: str = "",
    plan_mode: bool = False,
    preface_events: list[Mapping[str, Any]] | None = None,
) -> Iterator[dict]:
    """Stream graph/custom events while recording checkpoints and traces."""

    if model is None:
        validate_provider_config(provider, model_name)

    runtime = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        resume_from=resume_workspace,
        trace_mode=trace_mode,
    )
    ensure_workspace(runtime, create=True)
    ensure_scratch_dir(runtime)
    load_hooks_config(runtime)
    # Populate the run's skill catalog and clear any prior run's disclosures.
    load_skills_into_runtime(runtime)

    # Assemble the workspace's project context once per run; planner/codeAgent
    # prompt builders inject it from graph state.
    project_context = assemble_project_context(runtime)

    manager = CheckpointManager(runtime, task=task)
    trace = TraceRecorder(runtime, task=task)

    resume_event: dict[str, Any] | None = None
    current_state: dict[str, Any] = {
        "task": task,
        "runtime": runtime,
        "project_context": project_context,
        "attempts": 0,
        "max_attempts": max_attempts,
        "provider": provider,
        "model_name": model_name,
        "session_id": session_id,
        "session_turn": session_turn,
        "session_context": session_context,
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
            inputs["project_context"] = project_context
            inputs.setdefault("ask_budget", 2)
            if plan_mode:
                inputs["plan_mode"] = True
                inputs.setdefault("pre_plan_approval_mode", approval_mode)
            inputs["provider"] = provider
            inputs["model_name"] = model_name
            inputs["model"] = model or create_model(provider=provider, model=model_name)
            inputs["session_id"] = session_id
            inputs["session_turn"] = session_turn
            inputs["session_context"] = session_context
        else:
            inputs = {
                "task": task,
                "runtime": runtime,
                "project_context": project_context,
                "ask_budget": 2,
                "plan_mode": plan_mode,
                "pre_plan_approval_mode": approval_mode if plan_mode else None,
                "attempts": 0,
                "max_attempts": max_attempts,
                "provider": provider,
                "model_name": model_name,
                "model": model or create_model(provider=provider, model=model_name),
                "session_id": session_id,
                "session_turn": session_turn,
                "session_context": session_context,
            }

        current_state = dict(inputs)
        workflow = build_complex_workflow()

        trace.start(
            inputs,
            resumed=resume_workspace is not None,
            resume_event=resume_event,
        )
        trace_started = True

        for preface_event in preface_events or []:
            trace.record_graph_update(preface_event)
            current_state = _merge_graph_update(current_state, preface_event)
            latest_node = _extract_latest_node(preface_event, default=latest_node)

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

        memory_stats = extract_run_memories(current_state)
        if memory_stats.get("added") or memory_stats.get("replaced"):
            memory_event = {
                "type": "memory_extract",
                "added": memory_stats.get("added", 0),
                "replaced": memory_stats.get("replaced", 0),
                "total": memory_stats.get("total", 0),
            }
            trace.record_custom_event(memory_event)
            yield {"type": "custom_event", "event": memory_event}

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


def stream_session_events(
    task: str,
    *,
    session_workspace: str | Path | None = None,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: Callable[[Any], Any] | None = None,
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
    plan_mode: bool = False,
    **kwargs: Any,
) -> Iterator[dict]:
    """
    Stream events for a multi-turn conversation session.

    The session entry graph routes each turn to either lightweight chat or the
    full task workflow. The user's turn is saved before model work starts so
    interruptions do not lose the latest input.
    """

    workspace_arg = session_workspace if session_workspace is not None else kwargs.pop("workspace", None)
    workspace = resolve_session_workspace(workspace_arg)
    provider = str(kwargs.pop("provider", "openai"))
    model_name = kwargs.pop("model_name", None)
    model = kwargs.pop("model", None)
    if model is None:
        validate_provider_config(provider, model_name)

    runtime = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
    )
    ensure_workspace(runtime, create=True)
    load_hooks_config(runtime)

    session = load_or_create_session(workspace)
    turn = append_user_turn(session, task)
    saved_event = save_session(workspace, session)
    yield saved_event

    session_context = build_session_context(workspace, session)
    entry_model = model or create_model(provider=provider, model=model_name)
    entry_inputs: dict[str, Any] = {
        "task": task,
        "runtime": runtime,
        "session_id": session["session_id"],
        "session_turn": turn,
        "session_context": session_context,
        "provider": provider,
        "model_name": model_name,
        "model": entry_model,
    }

    route = "workflow"
    intent_reason = ""
    intent_confidence = 0.0
    assistant_recorded = False
    entry_trace_events: list[Mapping[str, Any]] = []

    try:
        entry_state = dict(entry_inputs)
        entry_workflow = build_entry_workflow()

        for raw_event in entry_workflow.stream(entry_inputs, stream_mode=["updates", "custom"]):
            if isinstance(raw_event, tuple) and len(raw_event) == 2:
                mode, event = raw_event
            else:
                mode, event = "updates", raw_event

            if mode == "custom":
                custom_event = _ensure_event_mapping(event)
                yield {
                    "type": "custom_event",
                    "event": custom_event,
                }
                continue

            graph_event = _ensure_event_mapping(event)
            entry_state = _merge_graph_update(entry_state, graph_event)
            entry_trace_events.append(dict(graph_event))
            yield {
                "type": "graph_event",
                "event": graph_event,
            }

        route = str(entry_state.get("intent_route") or "workflow")
        if route not in {"chat", "workflow"}:
            route = "workflow"
        intent_reason = str(entry_state.get("intent_reason") or "")
        try:
            intent_confidence = float(entry_state.get("intent_confidence") or 0.0)
        except (TypeError, ValueError):
            intent_confidence = 0.0

        yield {
            "type": "intent_route",
            "route": route,
            "reason": intent_reason,
            "confidence": intent_confidence,
            "session_id": session["session_id"],
            "turn": turn,
        }

        if route == "chat":
            chat_response = str(entry_state.get("chat_response") or entry_state.get("final_answer") or "")
            append_assistant_turn(
                session,
                turn=turn,
                route="chat",
                content=chat_response,
                summary=chat_response,
            )
            assistant_recorded = True
            yield save_session(workspace, session)
            yield {
                "type": "final_answer",
                "route": "chat",
                "content": chat_response,
                "session_id": session["session_id"],
                "turn": turn,
            }
            trace = TraceRecorder(runtime, task=task)
            latest = "start"
            trace.start(entry_inputs, resumed=False)
            for graph_event in entry_trace_events:
                trace.record_graph_update(graph_event)
                latest = _extract_latest_node(graph_event, default=latest) or latest
            trace_summary = trace.end(
                status="finished",
                latest_node=latest,
                final_state=entry_state,
            )
            yield {"type": "trace_finished", "trace": trace_summary}
            return

        final_answer = ""
        for event in stream_agent_events(
            task,
            workspace=workspace,
            max_attempts=max_attempts,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
            provider=provider,
            model_name=model_name,
            model=entry_model,
            session_id=session["session_id"],
            session_turn=turn,
            session_context=session_context,
            plan_mode=plan_mode,
            preface_events=entry_trace_events,
        ):
            if event.get("type") == "graph_event":
                inner = event.get("event")
                if isinstance(inner, Mapping):
                    final_update = inner.get("final")
                    if isinstance(final_update, Mapping):
                        final_answer = str(final_update.get("final_answer") or final_answer)
            yield event

        if not final_answer:
            final_answer = "Workflow completed."

        append_assistant_turn(
            session,
            turn=turn,
            route="workflow",
            content=final_answer,
            summary=final_answer,
        )
        assistant_recorded = True
        yield save_session(workspace, session)
        yield {
            "type": "final_answer",
            "route": "workflow",
            "content": final_answer,
            "session_id": session["session_id"],
            "turn": turn,
        }
    except KeyboardInterrupt:
        save_session(workspace, session)
        raise
    except Exception:
        if not assistant_recorded:
            save_session(workspace, session)
        raise
