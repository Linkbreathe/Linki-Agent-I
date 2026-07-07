import json
import re
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from Linki.agents.code_agent import run_code_agent
from Linki.agents.search_agent import run_search_agent
from Linki.core.paths import ensure_workspace, resolve_workspace_path
from Linki.core.state import RuntimeState
from Linki.graph.memory import (
    HISTORY_SUMMARY_FILENAME,
    CompressionEvent,
    LayeredMemory,
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from Linki.graph.state import AgentHandoff, LinkiGraphState, SourceItem, TodoItem, VerificationCheck, VerificationResult
from Linki.providers.openai_provider import create_model
from Linki.prompts.stage3 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from Linki.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from Linki.tools.bash_tool import _decode_timeout_output, _validate_workspace_command
from Linki.tools.registry import build_read_only_tools, build_tools


TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}
CONTEXT_TOKEN_LIMIT_DEFAULT = 400_000


class TodoItemSchema(BaseModel):
    id: str = Field(description="Stable todo identifier.")
    content: str = Field(description="Concrete work item.")
    status: str = Field(description="pending, in_progress, completed, or blocked.")
    note: str = Field(default="", description="Short context or blocker note.")


def _state_mapping(state: LinkiGraphState) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], state)


def _runtime(state: LinkiGraphState) -> RuntimeState:
    runtime = state.get("runtime")
    if runtime is None:
        raise ValueError("LinkiGraphState.runtime is required")
    return runtime


def _model(state: LinkiGraphState) -> Any:
    values = _state_mapping(state)
    if values.get("model") is not None:
        return values["model"]
    return create_model(
        provider=values.get("provider", "openai"),
        model=values.get("model_name"),
    )


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _emit_custom_event(event: Mapping[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):
        return
    writer(dict(event))


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}

    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return value if isinstance(value, dict) else {}


def _session_context_block(state: Mapping[str, Any]) -> str:
    session_context = str(state.get("session_context") or "").strip()
    if not session_context:
        return "Session context: none"
    return f"Session context:\n{session_context}"


def _latest_user_input_block(state: Mapping[str, Any]) -> str:
    return f"Latest user input:\n{state.get('task', '')}"


def intent_router_node(state: LinkiGraphState) -> dict:
    """Classify whether the latest input should be answered as chat or workflow."""

    messages: list[BaseMessage] = [
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        HumanMessage(
            content="\n\n".join(
                [
                    _latest_user_input_block(_state_mapping(state)),
                    _session_context_block(_state_mapping(state)),
                ]
            )
        ),
    ]

    try:
        response = _model(state).invoke(messages)
        payload = _json_from_text(_message_content(response))
    except Exception as exc:
        return {
            "intent_route": "workflow",
            "intent_reason": f"intent router failed: {type(exc).__name__}: {exc}",
            "intent_confidence": 0.0,
            "context_next_node": "planner",
        }

    route = str(payload.get("route") or "").strip().lower()
    reason = str(payload.get("reason") or "")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if route not in {"chat", "workflow"} or confidence < 0.55:
        if not reason:
            reason = "Invalid or low-confidence intent classification."
        route = "workflow"

    return {
        "intent_route": route,
        "intent_reason": reason,
        "intent_confidence": confidence,
        "context_next_node": route,
    }


def chat_responder_node(state: LinkiGraphState) -> dict:
    """Answer lightweight conversational turns without workspace tools."""

    messages: list[BaseMessage] = [
        SystemMessage(content=CHAT_RESPONDER_PROMPT),
        HumanMessage(
            content="\n\n".join(
                [
                    _latest_user_input_block(_state_mapping(state)),
                    _session_context_block(_state_mapping(state)),
                ]
            )
        ),
    ]

    response = _model(state).invoke(messages)
    chat_response = _message_content(response).strip()
    return {
        "chat_response": chat_response,
        "final_answer": chat_response,
    }


def intent_route_fn(state: LinkiGraphState) -> str:
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"


def _todo_dict(value: Any, index: int) -> TodoItem:
    if isinstance(value, BaseModel):
        raw = value.model_dump()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {}

    status = str(raw.get("status") or "pending")
    if status not in TODO_STATUSES:
        status = "pending"

    return {
        "id": str(raw.get("id") or f"todo-{index + 1}"),
        "content": str(raw.get("content") or ""),
        "status": status,
        "note": str(raw.get("note") or ""),
    }


def _normalize_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    todos = [_todo_dict(todo, index) for index, todo in enumerate(payload.get("todos") or [])]
    acceptance_criteria = [str(item) for item in payload.get("acceptance_criteria") or []]
    verification_commands = [str(item) for item in payload.get("verification_commands") or []]

    return {
        "plan_summary": str(payload.get("plan_summary") or ""),
        "todos": todos,
        "acceptance_criteria": acceptance_criteria,
        "verification_commands": verification_commands,
    }


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _plan_context(state: LinkiGraphState) -> str:
    return _format_json(
        {
            "plan_summary": state.get("plan_summary", ""),
            "todos": state.get("todos", []),
            "acceptance_criteria": state.get("acceptance_criteria", []),
            "verification_commands": state.get("verification_commands", []),
        }
    )


def _tool_result(name: str, ok: bool, output: Any = None, error: BaseException | None = None) -> dict:
    result = {"ok": ok, "name": name}
    if error is not None:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    else:
        result["output"] = output
    return result


def _execute_call(call: dict, tools_by_name: Mapping[str, StructuredTool]) -> dict:
    name = call["name"]
    args = call.get("args", {})

    tool = tools_by_name.get(name)
    if tool is None:
        return _tool_result(name, False, error=ValueError(f"Unknown tool: {name}"))

    try:
        return _tool_result(name, True, output=tool.invoke(args))
    except Exception as exc:
        return _tool_result(name, False, error=exc)


def _react_events(
    agent: Any,
    messages: list[BaseMessage],
    tools_by_name: Mapping[str, StructuredTool],
    *,
    node: str,
    max_loops: int = 10,
) -> Iterator[dict[str, Any]]:
    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        event = {"type": "ai_message", "node": node, "content": _message_content(response)}
        _emit_custom_event(event)
        yield {**event, "message": response}

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return

        for call in tool_calls:
            event = {"type": "tool_call", "node": node, "name": call["name"], "args": call.get("args", {})}
            _emit_custom_event(event)
            yield event
            result = _execute_call(call, tools_by_name)
            tool_message = ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=call["id"],
            )
            messages.append(tool_message)
            event = {
                "type": "tool_result",
                "node": node,
                "name": call["name"],
                "result": result,
            }
            _emit_custom_event(event)
            yield {**event, "message": tool_message}


def _append_research_notes(existing: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing}\n\n{addition}"


def _merge_sources(existing: list[SourceItem], new_sources: list[Any]) -> list[SourceItem]:
    merged: dict[str, SourceItem] = {}
    for item in list(existing) + list(new_sources or []):
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "")
        if not url:
            continue
        merged[url] = {
            "title": str(item.get("title") or ""),
            "url": url,
            "content": str(item.get("content") or ""),
            "score": float(item.get("score") or 0.0),
        }
    return list(merged.values())


def _call_search_agent_tool(state: dict[str, Any], writer: Any, instruction: str) -> dict:
    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "searchAgent",
            "instruction": instruction,
        }
    )

    result = run_search_agent(
        state,
        instruction,
        writer=writer,
    )

    state["research_notes"] = _append_research_notes(state.get("research_notes", ""), str(result.get("summary", "")))
    state["sources"] = _merge_sources(state.get("sources", []), result.get("sources", []))
    state.setdefault("agent_handoffs", []).append(
        {
            "from_agent": "planner",
            "to_agent": "searchAgent",
            "instruction": instruction,
            "result": str(result.get("summary", "")),
        }
    )

    return result


def _call_code_agent_tool(state: dict[str, Any], writer: Any, instruction: str) -> dict:
    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "codeAgent",
            "instruction": instruction,
        }
    )

    result = run_code_agent(
        state,
        instruction,
        writer=writer,
    )

    state["todos"] = result.get("todos", state.get("todos", []))
    state["code_agent_summary"] = str(result.get("summary", ""))
    state.setdefault("agent_handoffs", []).append(
        {
            "from_agent": "planner",
            "to_agent": "codeAgent",
            "instruction": instruction,
            "result": str(result.get("summary", "")),
        }
    )
    state["messages"] = result.get("messages", state.get("messages", []))

    return result


def _build_planner_tools(working: dict[str, Any]) -> list[StructuredTool]:
    def todo_write_tool(
        plan_summary: str,
        todos: list[TodoItemSchema],
        acceptance_criteria: list[str],
        verification_commands: list[str],
    ) -> dict[str, Any]:
        plan = _normalize_plan(
            {
                "plan_summary": plan_summary,
                "todos": todos,
                "acceptance_criteria": acceptance_criteria,
                "verification_commands": verification_commands,
            }
        )
        working.update(plan)
        return {"ok": True, "plan": plan}

    def call_search_agent_tool(instruction: str) -> dict[str, Any]:
        result = _call_search_agent_tool(working, _emit_custom_event, instruction)
        return {
            "ok": bool(result.get("ok", True)),
            "summary": result.get("summary", ""),
            "queries": result.get("queries", []),
            "sources": result.get("sources", []),
        }

    def call_code_agent_tool(instruction: str) -> dict[str, Any]:
        result = _call_code_agent_tool(working, _emit_custom_event, instruction)
        return {
            "ok": bool(result.get("ok", True)),
            "summary": result.get("summary", ""),
            "todos": result.get("todos", []),
        }

    return [
        StructuredTool.from_function(
            func=todo_write_tool,
            name="TodoWriteTool",
            description="Publish or revise the plan, todos, acceptance criteria, and verification commands.",
        ),
        StructuredTool.from_function(
            func=call_search_agent_tool,
            name="CallSearchAgentTool",
            description="Delegate a research task to searchAgent.",
        ),
        StructuredTool.from_function(
            func=call_code_agent_tool,
            name="CallCodeAgentTool",
            description="Delegate an implementation task to codeAgent.",
        ),
    ]


def _planner_input(working_state: Mapping[str, Any], memory: LayeredMemory) -> str:
    failed_previous_verification = working_state.get("passed") is False or bool(working_state.get("last_error"))

    if failed_previous_verification:
        instruction = "\n".join(
            [
                "Revise the existing plan based on the verifier failure, then delegate only the missing fix.",
                f"Task:\n{working_state.get('task', '')}",
                f"Last error:\n{working_state.get('last_error', '')}",
                f"Current plan:\n{_plan_context(working_state)}",
            ]
        )
    else:
        instruction = "\n".join(
            [
                "Plan this task and delegate the needed work to the specialist agents.",
                f"Task:\n{working_state.get('task', '')}",
            ]
        )

    return "\n\n".join([instruction, format_layered_memory_for_prompt(memory)])


def planner_node(state: LinkiGraphState) -> dict:
    """Run the planner/supervisor node: publish the plan and delegate to searchAgent/codeAgent."""

    runtime = _runtime(state)

    working: dict[str, Any] = {
        "task": state.get("task", ""),
        "runtime": runtime,
        "provider": state.get("provider", "openai"),
        "model_name": state.get("model_name"),
        "model": state.get("model"),
        "todos": [_todo_dict(todo, index) for index, todo in enumerate(state.get("todos", []))],
        "plan_summary": state.get("plan_summary", ""),
        "acceptance_criteria": list(state.get("acceptance_criteria", [])),
        "verification_commands": list(state.get("verification_commands", [])),
        "research_notes": state.get("research_notes", ""),
        "sources": list(state.get("sources", [])),
        "agent_handoffs": list(state.get("agent_handoffs", [])),
        "code_agent_summary": state.get("code_agent_summary", ""),
        "messages": list(state.get("messages", [])),
        "passed": state.get("passed"),
        "last_error": state.get("last_error", ""),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
        "context_summary": state.get("context_summary", ""),
        "compression_events": list(state.get("compression_events", [])),
    }

    tools = _build_planner_tools(working)
    tools_by_name = {tool.name: tool for tool in tools}
    agent = _model(state).bind_tools(tools)

    memory = build_layered_memory(working, node="planner")
    _emit_custom_event(memory_event(memory, node="planner"))

    messages: list[BaseMessage] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working, memory)),
    ]

    supervisor_summary = ""
    for event in _react_events(agent, messages, tools_by_name, node="planner", max_loops=10):
        if event["type"] == "ai_message":
            supervisor_summary = str(event["content"])

    return {
        "plan_summary": working["plan_summary"],
        "todos": working["todos"],
        "acceptance_criteria": working["acceptance_criteria"],
        "verification_commands": working["verification_commands"],
        "research_notes": working["research_notes"],
        "sources": working["sources"],
        "agent_handoffs": working["agent_handoffs"],
        "code_agent_summary": working["code_agent_summary"],
        "messages": working["messages"],
        "last_actor_summary": working["code_agent_summary"] or supervisor_summary,
        "context_next_node": "verifier",
    }


def _run_verification_command(runtime: RuntimeState, command: str, timeout_seconds: int = 60) -> VerificationResult:
    try:
        _validate_workspace_command(command)
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=ensure_workspace(runtime),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "ok": False,
            "exit_code": None,
            "stdout": _decode_timeout_output(exc.stdout),
            "stderr": f"Command timed out after {timeout_seconds}s\n{_decode_timeout_output(exc.stderr)}".strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }

    return {
        "command": command,
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _normalize_checks(items: Iterable[Any]) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    for index, item in enumerate(items):
        raw = dict(item) if isinstance(item, Mapping) else {}
        checks.append(
            {
                "name": str(raw.get("name") or f"Check {index + 1}"),
                "passed": bool(raw.get("passed")),
                "detail": str(raw.get("detail") or ""),
            }
        )
    return checks


def _command_checks(results: list[VerificationResult]) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    for result in results:
        detail_parts = []
        if result["stdout"]:
            detail_parts.append(f"stdout:\n{result['stdout'].strip()}")
        if result["stderr"]:
            detail_parts.append(f"stderr:\n{result['stderr'].strip()}")
        detail_parts.append(f"exit_code={result['exit_code']}")
        checks.append(
            {
                "name": f"Command: {result['command']}",
                "passed": result["ok"],
                "detail": "\n".join(detail_parts),
            }
        )
    return checks


def _verification_error(reason: str, checks: list[VerificationCheck], recommended_next_instruction: str) -> str:
    failed = [check for check in checks if not check["passed"]]
    failed_details = "\n".join(f"- {check['name']}: {check['detail']}" for check in failed)
    parts = [part for part in [reason, failed_details, recommended_next_instruction] if part]
    return "\n".join(parts)


def _verified_todos(todos: list[TodoItem], passed: bool, last_error: str) -> list[TodoItem]:
    if passed:
        return [{**todo, "status": "completed", "note": todo.get("note", "")} for todo in todos]

    updated: list[TodoItem] = []
    marked_blocked = False
    for todo in todos:
        next_todo = dict(todo)
        if not marked_blocked and next_todo.get("status") != "completed":
            next_todo["status"] = "blocked"
            next_todo["note"] = last_error
            marked_blocked = True
        elif next_todo.get("status") == "in_progress":
            next_todo["status"] = "pending"
        updated.append(cast(TodoItem, next_todo))
    return updated


def _verifier_input(working_state: Mapping[str, Any], memory: LayeredMemory) -> str:
    instruction = "\n".join(
        [
            f"Task:\n{working_state.get('task', '')}",
            f"Plan:\n{_plan_context(working_state)}",
            f"Acceptance criteria:\n{_format_json(working_state.get('acceptance_criteria', []))}",
            f"Verification commands:\n{_format_json(working_state.get('verification_commands', []))}",
            f"Verification command results:\n{_format_json(working_state.get('verification_results', []))}",
            f"Latest actor output:\n{working_state.get('last_actor_summary', '')}",
        ]
    )
    return "\n\n".join([instruction, format_layered_memory_for_prompt(memory)])


def verifier_node(state: LinkiGraphState) -> dict:
    """Verify actor output, run verification commands, and update graph status."""

    runtime = _runtime(state)
    verification_results = [
        _run_verification_command(runtime, command)
        for command in state.get("verification_commands", [])
    ]

    tools = build_read_only_tools(runtime)
    tools_by_name = {tool.name: tool for tool in tools}
    agent = _model(state).bind_tools(tools)

    working_state: dict[str, Any] = {**state, "verification_results": verification_results}
    memory = build_layered_memory(working_state, node="verifier")
    _emit_custom_event(memory_event(memory, node="verifier"))

    messages: list[BaseMessage] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=_verifier_input(working_state, memory)),
    ]

    final_content = ""
    for event in _react_events(agent, messages, tools_by_name, node="verifier", max_loops=5):
        if event["type"] == "ai_message":
            final_content = str(event["content"])

    payload = _json_from_text(final_content)
    reason = str(payload.get("reason") or "")
    recommended_next_instruction = str(payload.get("recommended_next_instruction") or "")
    verification_checks = _normalize_checks(payload.get("checks") or []) + _command_checks(verification_results)
    passed = bool(payload.get("passed")) and all(check["passed"] for check in verification_checks)
    last_error = "" if passed else _verification_error(reason, verification_checks, recommended_next_instruction)
    todos = _verified_todos(
        [_todo_dict(todo, index) for index, todo in enumerate(state.get("todos", []))],
        passed,
        last_error,
    )

    updates = {
        "passed": passed,
        "attempts": int(state.get("attempts", 0)) + 1,
        "verification_results": verification_results,
        "verification_checks": verification_checks,
        "todos": todos,
    }
    if not passed:
        updates["last_error"] = last_error
        updates["context_next_node"] = "planner"
    return updates


def verifier_route(state: LinkiGraphState) -> str:
    if state.get("passed"):
        return "final"

    if state.get("attempts", 0) >= state.get("max_attempts", 0):
        return "final"

    return "planner"


def _messages_text(messages: Iterable[BaseMessage]) -> str:
    return "\n".join(_message_content(message) for message in messages)


def _estimate_token_count(model: Any, messages: list[BaseMessage], memory_payload: str) -> int:
    payload_message = HumanMessage(content=memory_payload)
    try:
        return model.get_num_tokens_from_messages(messages + [payload_message])
    except Exception:
        text = _messages_text(messages) + memory_payload
        return len(text) // 4


def context_monitor_node(state: LinkiGraphState) -> dict:
    """Estimate context token usage and flag whether compression is required."""

    model = _model(state)
    messages = list(state.get("messages", []))
    memory_payload = format_layered_memory_for_prompt(build_layered_memory(state, node="context_monitor"))

    token_count = _estimate_token_count(model, messages, memory_payload)
    token_limit = int(state.get("context_token_limit") or CONTEXT_TOKEN_LIMIT_DEFAULT)
    should_compress = token_count > token_limit

    return {
        "context_token_count": token_count,
        "context_should_compress": should_compress,
        "context_next_node": state.get("context_next_node", "verifier"),
    }


def context_monitor_route(state: LinkiGraphState) -> str:
    if state.get("passed"):
        return "final"

    if state.get("context_should_compress"):
        return "context_compressor"

    return state.get("context_next_node", "verifier")


def context_compressor_node(state: LinkiGraphState) -> dict:
    """Compress the message history and durable context into one summary."""

    runtime = _runtime(state)
    model = _model(state)
    messages = list(state.get("messages", []))
    memory_payload = format_layered_memory_for_prompt(
        build_layered_memory(state, node="context_compressor")
    )

    compression_input = "\n\n".join(
        [
            f"Current messages:\n{_format_json([_message_content(message) for message in messages])}",
            f"Layered memory snapshot:\n{memory_payload}",
        ]
    )

    response = model.invoke(
        [
            SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
            HumanMessage(content=compression_input),
        ]
    )
    response_text = _message_content(response)
    payload = _json_from_text(response_text)
    summary = _format_json(payload) if payload else response_text

    resolve_workspace_path(runtime, HISTORY_SUMMARY_FILENAME).write_text(summary, encoding="utf-8")

    new_token_count = _estimate_token_count(model, [AIMessage(content=summary)], "")

    compression_event: CompressionEvent = {
        "node": "context_compressor",
        "reason": "context_token_count exceeded context_token_limit",
        "token_count": int(state.get("context_token_count", 0)),
        "token_limit": int(state.get("context_token_limit") or CONTEXT_TOKEN_LIMIT_DEFAULT),
        "summary": _short_text(summary, 400),
    }

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage(content=summary),
        ],
        "context_summary": summary,
        "context_token_count": new_token_count,
        "context_should_compress": False,
        "research_notes": _short_text(state.get("research_notes", ""), 1600),
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),
        "code_agent_summary": _short_text(state.get("code_agent_summary", ""), 1000),
        "last_actor_summary": _short_text(state.get("last_actor_summary", ""), 1000),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "history_summary": summary,
        "compression_events": [*state.get("compression_events", []), compression_event],
    }


def context_compressor_route(state: LinkiGraphState) -> str:
    """Route to the node selected before compression."""

    return state.get("context_next_node", "verifier")


def final_node(state: LinkiGraphState) -> dict:
    """Format the final graph outcome."""

    status = "passed" if state.get("passed") else "failed"
    attempts = int(state.get("attempts", 0))
    plan_summary = state.get("plan_summary", "")
    last_error = state.get("last_error", "")

    parts = [
        f"Verification {status}.",
        f"Attempts: {attempts}",
    ]
    if plan_summary:
        parts.append(f"Plan: {plan_summary}")
    if last_error and not state.get("passed"):
        parts.append(f"Reason: {last_error}")

    return {"final_answer": "\n".join(parts)}
