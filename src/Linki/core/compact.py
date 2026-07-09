"""Structured context compaction for Linki stage four."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from Linki.core.paths import resolve_workspace_path
from Linki.core.state import RuntimeState
from Linki.graph.memory import (
    HISTORY_SUMMARY_FILENAME,
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
)
from Linki.prompts.compact import COMPACT_PROMPT

DEFAULT_CONTEXT_TOKEN_LIMIT = 400_000


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _message_token_estimate(message: BaseMessage) -> int:
    text = _message_content(message)
    return max(len(text) // 4, 1) if text.strip() else 0


def _has_text_body(message: BaseMessage) -> bool:
    return bool(_message_content(message).strip())


def _tool_call_ids(message: BaseMessage) -> set[str]:
    ids: set[str] = set()
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, Mapping) and call.get("id"):
            ids.add(str(call["id"]))
    return ids


def _align_tool_boundary(messages: list[BaseMessage], cut: int) -> int:
    if cut <= 0 or cut >= len(messages):
        return cut

    first_tail = messages[cut]
    if not isinstance(first_tail, ToolMessage):
        return cut

    tail_tool_ids: set[str] = set()
    index = cut
    while index < len(messages) and isinstance(messages[index], ToolMessage):
        tool_id = getattr(messages[index], "tool_call_id", None)
        if tool_id:
            tail_tool_ids.add(str(tool_id))
        index += 1

    scan = cut - 1
    while scan >= 0:
        ids = _tool_call_ids(messages[scan])
        if ids and (not tail_tool_ids or ids.intersection(tail_tool_ids)):
            return scan
        if ids:
            return scan
        scan -= 1
    return cut


def split_messages(
    messages: list[BaseMessage],
    min_tail_messages: int = 5,
    min_tail_tokens: int = 2000,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split messages into compactable head and retained tail.

    The tail first retains at least ``min_tail_messages`` messages with textual
    content, then expands until the estimated token budget is met. The cut point
    is shifted backward if it would separate an AI tool call from its ToolMessage
    result.
    """

    if not messages:
        return [], []

    text_count = 0
    token_count = 0
    cut = len(messages)

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if _has_text_body(message):
            text_count += 1
        token_count += _message_token_estimate(message)
        cut = index
        if text_count >= min_tail_messages and token_count >= min_tail_tokens:
            break

    cut = _align_tool_boundary(messages, cut)
    if cut <= 0:
        return [], messages
    return messages[:cut], messages[cut:]


def _default_model() -> Any:
    from Linki.providers.openai_provider import create_model

    return create_model()


def summarize_structured(head: list[BaseMessage], focus: str | None = None, *, model: Any | None = None) -> str:
    focus_clause = f"\n\nFocus for this compaction:\n{focus.strip()}" if focus and focus.strip() else ""
    transcript = "\n\n".join(
        f"{type(message).__name__}: {_message_content(message)}" for message in head
    )
    response = (model or _default_model()).invoke(
        [
            HumanMessage(content=f"{COMPACT_PROMPT}{focus_clause}\n\nTranscript head:\n{transcript}"),
        ]
    )
    return _message_content(response).strip()


def _emit(runtime: RuntimeState | None, event: dict[str, Any]) -> None:
    if runtime is not None and runtime.event_handler is not None:
        runtime.event_handler(dict(event))


def _ensure_acceptance_section(summary: str) -> str:
    if "# Acceptance Criteria" in summary:
        return summary
    return f"{summary.rstrip()}\n\n# Acceptance Criteria\n"


def enforce_acceptance_criteria(
    summary: str,
    criteria: list[str],
    *,
    runtime: RuntimeState | None = None,
) -> str:
    missing = [criterion for criterion in criteria if criterion and criterion not in summary]
    if not missing:
        return summary

    updated = _ensure_acceptance_section(summary).rstrip()
    additions = "\n".join(f"- {criterion}" for criterion in missing)
    updated = f"{updated}\n{additions}\n"

    for criterion in missing:
        _emit(
            runtime,
            {
                "type": "trace.warn",
                "event": "context_compaction",
                "reason": "acceptance criterion restored",
                "criterion": criterion,
            },
        )
    return updated


def recent_files_note(runtime: RuntimeState) -> str:
    files = list(getattr(runtime, "recent_files", {}).keys())
    if not files:
        return ""
    return (
        "Recently touched files (summaries may be stale - ALWAYS use "
        f"FileReadTool before editing): {', '.join(files)}"
    )


def _format_compact_summary(summary: str, runtime: RuntimeState) -> str:
    note = recent_files_note(runtime)
    parts = ["<compact_summary>", summary.strip()]
    if note:
        parts.extend(["", note])
    parts.append("</compact_summary>")
    return "\n".join(parts)


def _estimate_token_count(model: Any, messages: list[BaseMessage], memory_payload: str = "") -> int:
    payload = [*messages]
    if memory_payload:
        payload.append(HumanMessage(content=memory_payload))
    try:
        return int(model.get_num_tokens_from_messages(payload))
    except Exception:
        text = "\n".join(_message_content(message) for message in payload)
        return len(text) // 4


def _compact_once(
    state: Mapping[str, Any],
    *,
    focus: str | None,
    trigger: str,
    min_tail_tokens: int,
) -> dict[str, Any]:
    runtime = state["runtime"]
    model = state.get("model")
    messages = list(state.get("messages", []))
    head, tail = split_messages(messages, min_tail_tokens=min_tail_tokens)
    if not head:
        return {
            "changed": False,
            "head": head,
            "tail": tail,
            "updates": {"context_should_compress": False},
        }

    before_tokens = int(state.get("context_token_count") or _estimate_token_count(model, messages))
    summary = summarize_structured(head, focus, model=model)
    summary = enforce_acceptance_criteria(
        summary,
        [str(item) for item in state.get("acceptance_criteria", [])],
        runtime=runtime,
    )
    compact_content = _format_compact_summary(summary, runtime)

    resolve_workspace_path(runtime, HISTORY_SUMMARY_FILENAME).write_text(compact_content, encoding="utf-8")
    compact_messages: list[BaseMessage] = [AIMessage(content=compact_content), *tail]
    memory_payload = format_layered_memory_for_prompt(build_layered_memory(state, node="context_compressor"))
    after_tokens = _estimate_token_count(model, compact_messages, memory_payload)

    token_limit = int(state.get("context_token_limit") or DEFAULT_CONTEXT_TOKEN_LIMIT)
    compression_event = {
        "node": "context_compressor",
        "reason": f"{trigger} context compaction",
        "token_count": before_tokens,
        "token_limit": token_limit,
        "summary": _short_text(summary, 400),
        "tail_messages": len(tail),
        "after_token_count": after_tokens,
    }

    updates = {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *compact_messages],
        "context_summary": compact_content,
        "context_token_count": after_tokens,
        "context_should_compress": after_tokens > token_limit,
        "research_notes": _short_text(state.get("research_notes", ""), 1600),
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),
        "code_agent_summary": _short_text(state.get("code_agent_summary", ""), 1000),
        "last_actor_summary": _short_text(state.get("last_actor_summary", ""), 1000),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "history_summary": compact_content,
        "compression_events": [*state.get("compression_events", []), compression_event],
    }
    return {"changed": True, "head": head, "tail": tail, "updates": updates}


def compact_pipeline(state: Mapping[str, Any], focus: str | None = None, trigger: str = "auto") -> dict[str, Any]:
    """Run structured compaction and return graph-state updates."""

    runtime = state["runtime"]
    first = _compact_once(state, focus=focus, trigger=trigger, min_tail_tokens=2000)
    updates = first["updates"]

    if not first["changed"]:
        _emit(
            runtime,
            {
                "type": "compact_skipped",
                "trigger": trigger,
                "reason": "message history is within retained tail",
                "tail_messages": len(first["tail"]),
            },
        )
        return updates

    event = updates["compression_events"][-1]
    _emit(
        runtime,
        {
            "type": "compact_result",
            "trigger": trigger,
            "before_tokens": event["token_count"],
            "after_tokens": event["after_token_count"],
            "tail_messages": event["tail_messages"],
        },
    )

    token_limit = int(state.get("context_token_limit") or DEFAULT_CONTEXT_TOKEN_LIMIT)
    if int(updates.get("context_token_count") or 0) <= token_limit:
        updates["context_should_compress"] = False
        return updates

    second_state = {**state, **updates}
    second = _compact_once(second_state, focus=focus, trigger=f"{trigger}:fallback", min_tail_tokens=0)
    updates = second["updates"]
    if second["changed"]:
        event = updates["compression_events"][-1]
        _emit(
            runtime,
            {
                "type": "compact_result",
                "trigger": f"{trigger}:fallback",
                "before_tokens": event["token_count"],
                "after_tokens": event["after_token_count"],
                "tail_messages": event["tail_messages"],
            },
        )

    if int(updates.get("context_token_count") or 0) > token_limit:
        _emit(
            runtime,
            {
                "type": "compact_warning",
                "trigger": trigger,
                "message": "Context is still too large; consider /clear to start a fresh session.",
                "after_tokens": updates.get("context_token_count"),
                "token_limit": token_limit,
            },
        )
    else:
        updates["context_should_compress"] = False

    return updates
