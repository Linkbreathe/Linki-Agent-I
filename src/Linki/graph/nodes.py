import json
import re
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from Linki.core.paths import ensure_workspace
from Linki.core.state import RuntimeState
from Linki.graph.state import LinkiGraphState, TodoItem, VerificationCheck, VerificationResult
from Linki.providers.openai_provider import create_model
from Linki.prompts.stage2 import ACTOR_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from Linki.tools.bash_tool import _decode_timeout_output, _validate_workspace_command
from Linki.tools.registry import build_read_only_tools, build_tools


TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}


class TodoItemSchema(BaseModel):
    id: str = Field(description="Stable todo identifier.")
    content: str = Field(description="Concrete work item.")
    status: str = Field(description="pending, in_progress, completed, or blocked.")
    note: str = Field(default="", description="Short context or blocker note.")


class TodoWriteTool(BaseModel):
    """Write the complete plan for the current task."""

    plan_summary: str
    todos: list[TodoItemSchema]
    acceptance_criteria: list[str]
    verification_commands: list[str]


class TodoUpdateTool(BaseModel):
    """Update the status and note for one todo item."""

    id: str
    status: str = Field(description="pending, in_progress, completed, or blocked.")
    note: str = ""


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
    except RuntimeError:
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


def _structured_payload(message: Any, preferred_tool: str | None = None) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        if preferred_tool is None or call.get("name") == preferred_tool:
            args = call.get("args", {})
            return args if isinstance(args, dict) else {}

    return _json_from_text(_message_content(message))


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


def planner_node(state: LinkiGraphState) -> dict:
    """Create or revise the graph plan."""

    has_todos = bool(state.get("todos"))
    failed_previous_verification = state.get("passed") is False or bool(state.get("last_error"))
    if has_todos and not failed_previous_verification:
        return {
            "plan_summary": state.get("plan_summary", ""),
            "todos": state.get("todos", []),
            "acceptance_criteria": state.get("acceptance_criteria", []),
            "verification_commands": state.get("verification_commands", []),
        }

    if has_todos:
        user_prompt = "\n".join(
            [
                "Revise the existing plan based on the verifier failure.",
                f"Task:\n{state.get('task', '')}",
                f"Last error:\n{state.get('last_error', '')}",
                f"Current plan:\n{_plan_context(state)}",
            ]
        )
    else:
        user_prompt = "\n".join(
            [
                "Create an implementation plan for this task.",
                f"Task:\n{state.get('task', '')}",
            ]
        )

    agent = _model(state).bind_tools([TodoWriteTool])
    response = agent.invoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return _normalize_plan(_structured_payload(response, "TodoWriteTool"))


def _tool_result(name: str, ok: bool, output: Any = None, error: BaseException | None = None) -> dict:
    result = {"ok": ok, "name": name}
    if error is not None:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    else:
        result["output"] = output
    return result


def _update_todo(todos: list[TodoItem], todo_id: str, status: str, note: str) -> dict:
    if status not in TODO_STATUSES:
        raise ValueError(f"Unsupported todo status: {status}")

    for todo in todos:
        if todo["id"] == todo_id:
            todo["status"] = status
            todo["note"] = note
            return {"updated": todo}

    raise ValueError(f"Unknown todo id: {todo_id}")


def _execute_call(
    call: dict,
    tools_by_name: Mapping[str, StructuredTool],
    todos: list[TodoItem] | None = None,
) -> dict:
    name = call["name"]
    args = call.get("args", {})

    if name == "TodoUpdateTool" and todos is not None:
        try:
            output = _update_todo(
                todos,
                todo_id=str(args.get("id", "")),
                status=str(args.get("status", "")),
                note=str(args.get("note", "")),
            )
        except Exception as exc:
            return _tool_result(name, False, error=exc)
        return _tool_result(name, True, output=output)

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
    todos: list[TodoItem] | None = None,
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
            result = _execute_call(call, tools_by_name, todos)
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


def actor_node(state: LinkiGraphState) -> dict:
    """Run the actor ReAct loop against the current plan."""

    runtime = _runtime(state)
    tools = build_tools(runtime)
    tools_by_name = {tool.name: tool for tool in tools}
    todos = [_todo_dict(todo, index) for index, todo in enumerate(state.get("todos", []))]

    agent = _model(state).bind_tools(tools + [TodoUpdateTool])
    actor_input = "\n".join(
        [
            f"Current plan:\n{_plan_context(state)}",
            f"Task:\n{state.get('task', '')}",
        ]
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=actor_input),
    ]

    last_actor_summary = ""
    for event in _react_events(agent, messages, tools_by_name, node="actor", todos=todos, max_loops=10):
        if event["type"] == "ai_message":
            last_actor_summary = str(event["content"])

    return {
        "messages": messages,
        "last_actor_summary": last_actor_summary,
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
    verifier_input = "\n".join(
        [
            f"Task:\n{state.get('task', '')}",
            f"Plan:\n{_plan_context(state)}",
            f"Acceptance criteria:\n{_format_json(state.get('acceptance_criteria', []))}",
            f"Verification commands:\n{_format_json(state.get('verification_commands', []))}",
            f"Verification command results:\n{_format_json(verification_results)}",
            f"Latest actor output:\n{state.get('last_actor_summary', '')}",
        ]
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=verifier_input),
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
    return updates


def verifier_route(state: LinkiGraphState) -> str:
    if state.get("passed"):
        return "final"

    if state.get("attempts", 0) >= state.get("max_attempts", 0):
        return "final"

    return "planner"


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
