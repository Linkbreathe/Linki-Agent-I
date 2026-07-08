import json
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from Linki.providers.openai_provider import create_model
from Linki.tools.executor import execute_tool, is_tool_result
from Linki.tools.web_search_tool import WebSearchTool

SEARCH_AGENT_PROMPT = """You are searchAgent, a focused research specialist.

Your only external capability is WebSearchTool. Search for reliable information
needed by the planner and codeAgent.

Rules:
- Use WebSearchTool for factual research.
- Prefer official or encyclopedia-style sources when available.
- Return a concise research summary and list the useful source URLs.
- Do not write files or produce application code.
"""


def _model(state: Any) -> Any:
    values = state if isinstance(state, Mapping) else {}
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


def _research_notes_text(state: Any) -> str:
    values = state if isinstance(state, Mapping) else {}
    notes = values.get("research_notes")
    if not notes:
        return ""
    if isinstance(notes, list):
        return "\n".join(str(note) for note in notes)
    return str(notes)


def _build_human_message(state: Any, instruction: str) -> str:
    values = state if isinstance(state, Mapping) else {}
    parts = [f"Task:\n{values.get('task', '')}", f"Instruction:\n{instruction}"]

    notes = _research_notes_text(state)
    if notes:
        parts.append(f"Existing research notes:\n{notes}")

    return "\n\n".join(parts)


def _runtime(state: Any) -> Any:
    values = state if isinstance(state, Mapping) else {}
    return values.get("runtime")


def _build_web_search_tool(state: Any, web_search: WebSearchTool) -> StructuredTool:
    runtime = _runtime(state)

    def web_search_tool(query: str) -> dict:
        if runtime is None:
            return {"ok": True, "name": "WebSearchTool", "output": web_search(query)}
        return execute_tool(runtime, "WebSearchTool", {"query": query}, web_search)

    return StructuredTool.from_function(
        func=web_search_tool,
        name="WebSearchTool",
        description="Search the web for factual information and return sources.",
    )


def _effective_tool_output(result: Any, tool_name: str) -> dict[str, Any]:
    if is_tool_result(result, tool_name):
        output = result.get("output")
        return dict(output) if isinstance(output, Mapping) else {"ok": result.get("ok"), "output": output}
    return dict(result) if isinstance(result, Mapping) else {"ok": False, "error": str(result)}


def run_search_agent(
    state: Any,
    instruction: str,
    *,
    writer: Callable[[Mapping[str, Any]], None] | None = None,
    max_loops: int = 4,
) -> dict[str, Any]:
    """Run searchAgent's ReAct loop against WebSearchTool.

    Returns a dict with the research summary, issued queries, deduplicated
    sources, the full message trace, and the raw tool events emitted along
    the way.
    """

    web_search = WebSearchTool()
    tool = _build_web_search_tool(state, web_search)
    agent = _model(state).bind_tools([tool])

    messages: list[BaseMessage] = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(content=_build_human_message(state, instruction)),
    ]

    queries: list[str] = []
    sources: dict[str, dict[str, Any]] = {}
    tool_events: list[dict[str, Any]] = []
    summary = ""

    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        summary = _message_content(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            query = str(call.get("args", {}).get("query", ""))
            queries.append(query)

            tool_call_event = {"type": "tool_call", "name": call["name"], "args": call.get("args", {})}
            tool_events.append(tool_call_event)
            if writer is not None:
                writer(dict(tool_call_event))

            result = tool.invoke(call.get("args", {}))
            effective_result = _effective_tool_output(result, call["name"])
            if effective_result.get("ok"):
                for item in effective_result.get("results", []):
                    url = item.get("url", "")
                    if url:
                        sources.setdefault(url, item)

            search_results_event = {
                "type": "search_results",
                "name": call["name"],
                "query": query,
                "result": result,
            }
            tool_events.append(search_results_event)
            if writer is not None:
                writer(dict(search_results_event))

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call["id"],
                )
            )

    return {
        "ok": True,
        "summary": summary,
        "queries": queries,
        "sources": list(sources.values()),
        "messages": messages,
        "tool_events": tool_events,
    }
