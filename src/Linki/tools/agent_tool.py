"""AgentTool: controlled subagent dispatch.

The planner and codeAgent dispatch specialist subagents through a single tool.
Each subagent runs an isolated ReAct loop with a restricted tool pool, an
independent message history, and a nested trace span. Every tool call still flows
through :func:`execute_tool`, so hooks, risk classification, and approval remain
active inside subagents, and their events bubble up to the parent stream tagged
with the subagent's ``agent`` name.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from Linki.agents.registry import AgentSpec, load_agent_registry
from Linki.core.approval import ApprovalDecision
from Linki.core.state import RuntimeState
from Linki.providers.openai_provider import create_model
from Linki.tools.registry import AGENT_TOOL_NAME, build_subagent_tools

MAX_SUBAGENT_LOOPS = 6


class AgentToolInput(BaseModel):
    subagent_type: str = Field(description="Registered agent type to run.")
    description: str = Field(description="Short 3-5 word label for traces and the TUI.")
    prompt: str = Field(
        description="Complete, self-contained task for the subagent. The subagent "
        "cannot see the parent conversation."
    )


def _model(state: Any) -> Any:
    values = state if isinstance(state, Mapping) else {}
    if values.get("model") is not None:
        return values["model"]
    return create_model(provider=values.get("provider", "openai"), model=values.get("model_name"))


def _runtime(state: Any) -> RuntimeState | None:
    if isinstance(state, RuntimeState):
        return state
    if isinstance(state, Mapping):
        return state.get("runtime")
    return None


def _parent_label(state: Any) -> str:
    values = state if isinstance(state, Mapping) else {}
    return str(values.get("current_node") or values.get("parent_agent") or "planner")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _resolve_sink(runtime: RuntimeState | None):
    """Resolve where events should be written: the runtime handler, or the
    LangGraph stream writer when running inside a graph node."""

    if runtime is not None and runtime.event_handler is not None:
        return runtime.event_handler
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (ImportError, RuntimeError, KeyError):
        return None


def allowed_subagent_tools(runtime: RuntimeState, spec: AgentSpec) -> list[StructuredTool]:
    """Filter the full subagent tool pool down to the spec's allowlist.

    ``AgentTool`` is always removed so a subagent cannot dispatch further
    subagents, even if a definition erroneously lists it.
    """

    allow = set(spec.tools)
    return [
        tool
        for tool in build_subagent_tools(runtime)
        if tool.name in allow and tool.name != AGENT_TOOL_NAME
    ]


def run_subagent(state: Any, spec: AgentSpec, prompt: str, *, description: str = "") -> str:
    """Run a subagent's isolated ReAct loop and return its final text.

    Tool calls run through the canonical pipeline; subagent messages are kept in
    an independent history and never appended to the parent graph messages.
    """

    runtime = _runtime(state)
    agent_name = spec.name
    parent = _parent_label(state)
    sink = _resolve_sink(runtime)

    def emit(event: dict[str, Any]) -> None:
        if sink is None:
            return
        payload = dict(event)
        payload.setdefault("agent", agent_name)
        sink(payload)

    # Wrap the runtime so tool/hook/approval events emitted deep inside
    # execute_tool bubble up tagged with this subagent's name.
    if runtime is not None:
        def tagging_handler(event: dict[str, Any]) -> None:
            if sink is None:
                return
            payload = dict(event)
            payload.setdefault("agent", agent_name)
            sink(payload)

        real_approval = runtime.approval_handler

        def approval_handler(request: Any) -> ApprovalDecision:
            emit(
                {
                    "type": "approval_requested",
                    "tool": getattr(request, "tool_name", ""),
                    "reason": getattr(request, "risk_reason", ""),
                    "command": getattr(request, "command", ""),
                }
            )
            if real_approval is None:
                return ApprovalDecision(approved=False, reason="no approval handler")
            return real_approval(request)

        scoped_runtime = replace(runtime, event_handler=tagging_handler, approval_handler=approval_handler)
    else:
        scoped_runtime = runtime

    tools = allowed_subagent_tools(scoped_runtime, spec) if scoped_runtime is not None else []
    tools_by_name = {tool.name: tool for tool in tools}
    agent = _model(state).bind_tools(tools) if tools else _model(state)

    emit(
        {
            "type": "subagent_start",
            "description": description,
            "parent": parent,
            "tools": [tool.name for tool in tools],
        }
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=spec.system_prompt),
        HumanMessage(content=prompt),
    ]

    summary = ""
    for _ in range(MAX_SUBAGENT_LOOPS):
        response = agent.invoke(messages)
        messages.append(response)
        summary = _message_text(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {})
            emit({"type": "tool_call", "name": name, "args": args})

            tool = tools_by_name.get(name)
            if tool is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "name": name,
                    "error": f"tool '{name}' is not available to subagent '{agent_name}'",
                }
            else:
                result = tool.invoke(args)

            emit({"type": "tool_result", "name": name, "result": result})
            if name == "WebSearchTool":
                emit({"type": "search_results", "name": name, "query": args.get("query", ""), "result": result})

            messages.append(
                ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str), tool_call_id=call["id"])
            )
    else:
        summary = summary or f"subagent '{agent_name}' reached the {MAX_SUBAGENT_LOOPS}-step limit"

    emit({"type": "subagent_result", "description": description, "parent": parent, "summary": summary})
    return summary


def make_agent_tool(state: Any) -> StructuredTool:
    """Build the AgentTool that dispatches registered subagents."""

    runtime = _runtime(state)
    registry = load_agent_registry(runtime) if runtime is not None else {}

    def agent_tool(subagent_type: str, description: str, prompt: str) -> dict[str, Any]:
        spec = registry.get(subagent_type)
        if spec is None:
            available = ", ".join(sorted(registry)) or "(none)"
            return {
                "ok": False,
                "name": AGENT_TOOL_NAME,
                "error": f"unknown subagent type: {subagent_type}\navailable types: {available}",
            }

        result = run_subagent(state, spec, prompt, description=description)
        return {
            "ok": True,
            "name": AGENT_TOOL_NAME,
            "subagent_type": subagent_type,
            "description": description,
            "output": result,
        }

    return StructuredTool.from_function(
        func=agent_tool,
        name=AGENT_TOOL_NAME,
        description=(
            "Dispatch a specialist subagent by type with a self-contained prompt. "
            "Use for research, documentation, or review. The subagent cannot see this "
            "conversation, so the prompt must be complete."
        ),
        args_schema=AgentToolInput,
    )
