import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from Linki.core.paths import ensure_workspace
from Linki.core.state import RuntimeState
from Linki.providers.openai_provider import create_model
from Linki.tools.registry import build_tools

ACTOR_PROMPT = """You are the actor node in Linki's ReAct workflow.

You implement the user's task using tools. Work inside the workspace only.

Rules:
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool to run commands and test results.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- End with a concise summary of files changed and commands run.
"""


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _execute_tool(call: dict, tools_by_name: dict[str, StructuredTool]) -> dict:
    name = call["name"]
    tool = tools_by_name.get(name)
    if tool is None:
        return {
            "ok": False,
            "name": name,
            "error_type": "UnknownTool",
            "error": f"Unknown tool: {name}",
        }

    try:
        output = tool.invoke(call.get("args", {}))
    except Exception as exc:
        return {
            "ok": False,
            "name": name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "ok": True,
        "name": name,
        "output": output,
    }


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_loops: int = 10,
    provider: str = "openai",
    model_name: str | None = None,
    model: Any | None = None,
) -> Iterator[dict]:
    """Stream events from Linki's ReAct tool-calling loop."""

    state = RuntimeState(workspace=Path(workspace))
    ensure_workspace(state, create=True)

    tools = build_tools(state)
    tools_by_name = {tool.name: tool for tool in tools}
    chat_model = model or create_model(provider=provider, model=model_name)
    agent = chat_model.bind_tools(tools)
    messages = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content=task),
    ]

    last_ai_content = ""
    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        last_ai_content = _message_content(response)

        yield {
            "type": "ai_message",
            "content": last_ai_content,
        }

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {})
            yield {
                "type": "tool_call",
                "name": name,
                "args": args,
            }

            result = _execute_tool(call, tools_by_name)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call["id"],
                )
            )

            yield {
                "type": "tool_result",
                "name": name,
                "result": result,
            }

    yield {
        "type": "final_answer",
        "content": last_ai_content,
    }
