import json
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from Linki.core.state import RuntimeState
from Linki.graph.memory import LayeredMemory, build_layered_memory, format_layered_memory_for_prompt, memory_event
from Linki.graph.state import TodoItem
from Linki.providers.openai_provider import create_model
from Linki.tools.registry import build_tools

CODE_AGENT_PROMPT = """You are codeAgent, a focused implementation specialist.

You implement the planner's instruction inside the workspace using file and
shell tools.

Rules:
- You must update todo progress explicitly.
- Before starting a todo, call TodoUpdateTool with status "in_progress".
- After finishing that todo, call TodoUpdateTool with status "completed".
- If a todo is impossible, call TodoUpdateTool with status "blocked" and explain.
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool for non-interactive checks.
- Use NotepadAppendTool to record durable findings, decisions, important files,
  blockers, and next-step context that should survive compression.
- Use NotepadReadTool when you need to recover prior notes.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- Incorporate research notes and source URLs when the task asks for researched content.
- End with a concise summary of files changed and checks run.
"""

TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}


class TodoUpdateTool(BaseModel):
    """Update the status and note for one todo item."""

    id: str
    status: str = Field(description="pending, in_progress, completed, or blocked.")
    note: str = ""


def _model(state: Any) -> Any:
    values = state if isinstance(state, Mapping) else {}
    if values.get("model") is not None:
        return values["model"]
    return create_model(
        provider=values.get("provider", "openai"),
        model=values.get("model_name"),
    )


def _runtime(state: Any) -> RuntimeState:
    values = state if isinstance(state, Mapping) else {}
    runtime = values.get("runtime")
    if runtime is None:
        raise ValueError("state['runtime'] is required")
    return runtime


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


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


def _update_todo(todos: list[TodoItem], todo_id: str, status: str, note: str) -> dict:
    if status not in TODO_STATUSES:
        raise ValueError(f"Unsupported todo status: {status}")

    for todo in todos:
        if todo["id"] == todo_id:
            todo["status"] = status
            todo["note"] = note
            return {"updated": todo}

    raise ValueError(f"Unknown todo id: {todo_id}")


def _session_context(state: Any) -> str:
    values = state if isinstance(state, Mapping) else {}
    context: dict[str, Any] = {
        "plan_summary": values.get("plan_summary", ""),
        "todos": values.get("todos", []),
        "acceptance_criteria": values.get("acceptance_criteria", []),
        "verification_commands": values.get("verification_commands", []),
    }
    if values.get("last_error"):
        context["last_error"] = values["last_error"]

    research_notes = values.get("research_notes")
    if research_notes:
        context["research_notes"] = research_notes

    sources = values.get("sources")
    if sources:
        context["sources"] = sources

    return _format_json(context)


def _code_agent_input(state: Any, instruction: str, memory: LayeredMemory) -> str:
    values = state if isinstance(state, Mapping) else {}
    parts: list[str] = []

    project_context = str(values.get("project_context") or "").strip()
    if project_context:
        parts.append(project_context)

    parts += [
        f"Task:\n{values.get('task', '')}",
        f"Instruction:\n{instruction}",
        f"Session context:\n{_session_context(state)}",
        format_layered_memory_for_prompt(memory),
    ]

    return "\n\n".join(parts)


def _tool_result(name: str, ok: bool, output: Any = None, error: BaseException | None = None) -> dict:
    result: dict[str, Any] = {"ok": ok, "name": name}
    if error is not None:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    else:
        result["output"] = output
    return result


def _execute_call(call: dict, tools_by_name: Mapping[str, Any], todos: list[TodoItem]) -> dict:
    name = call["name"]
    args = call.get("args", {})

    if name == "TodoUpdateTool":
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


def run_code_agent(
    state: Any,
    instruction: str,
    *,
    writer: Callable[[Mapping[str, Any]], None] | None = None,
    max_loops: int = 10,
) -> dict[str, Any]:
    """Run codeAgent's ReAct loop against the workspace tools + TodoUpdateTool.

    Returns a dict with the final summary, the updated todo list, the full
    message trace, and the raw tool events emitted along the way.
    """

    runtime = _runtime(state)
    values = state if isinstance(state, Mapping) else {}
    # Defense-in-depth: even if reached in plan mode, keep the code agent read-only.
    tools = build_tools(
        runtime,
        plan_mode=bool(values.get("plan_mode")),
        ask_budget_left=values.get("ask_budget"),
    )
    tools_by_name = {tool.name: tool for tool in tools}

    todos = [_todo_dict(todo, index) for index, todo in enumerate(values.get("todos") or [])]

    agent = _model(state).bind_tools(tools + [TodoUpdateTool])

    memory = build_layered_memory(state, node="codeAgent")
    if writer is not None:
        writer(memory_event(memory, node="codeAgent"))

    messages: list[BaseMessage] = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=_code_agent_input(state, instruction, memory)),
    ]

    tool_events: list[dict[str, Any]] = []
    summary = ""

    def _emit(event: dict[str, Any]) -> None:
        tool_events.append(event)
        if writer is not None:
            writer(dict(event))

    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        summary = _message_content(response)
        _emit({"type": "ai_message", "content": summary})

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            _emit({"type": "tool_call", "name": call["name"], "args": call.get("args", {})})

            result = _execute_call(call, tools_by_name, todos)
            _emit({"type": "tool_result", "name": call["name"], "result": result})

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call["id"],
                )
            )

    return {
        "ok": True,
        "summary": summary,
        "todos": todos,
        "messages": messages,
        "tool_events": tool_events,
    }
