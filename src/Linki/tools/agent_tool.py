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
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from Linki.agents.registry import AgentSpec, load_agent_registry
from Linki.core.approval import ApprovalDecision
from Linki.core.state import RuntimeState
from Linki.providers.openai_provider import create_model
from Linki.tools.registry import AGENT_DISPATCH_TOOL_NAME, AGENT_TOOL_NAME, build_subagent_tools

MAX_SUBAGENT_LOOPS = 6
MAX_PARALLEL_JOBS = 3


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


def run_subagent(
    state: Any,
    spec: AgentSpec,
    prompt: str,
    *,
    description: str = "",
    job_id: str | None = None,
    approval_lock: threading.Lock | None = None,
    extra_tools: list[StructuredTool] | None = None,
) -> str:
    """Run a subagent's isolated ReAct loop and return its final text.

    Tool calls run through the canonical pipeline; subagent messages are kept in
    an independent history and never appended to the parent graph messages.

    When dispatched as one of several parallel jobs, ``job_id`` (e.g. "job-2")
    is stamped onto every emitted event, and ``approval_lock`` serializes this
    job's approval prompts against its siblings so only one popup is presented at
    a time. The approval request is labelled ``[job-i · agent]`` for the UI.

    ``extra_tools`` are appended to the spec's allowlisted pool — used by the
    swarm scheduler to grant board/mailbox tools on top of the agent's own tools.
    """

    runtime = _runtime(state)
    agent_name = spec.name
    parent = _parent_label(state)
    sink = _resolve_sink(runtime)
    job_label = f"[{job_id} · {agent_name}]" if job_id else ""

    def _stamp(event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        payload.setdefault("agent", agent_name)
        if job_id:
            payload.setdefault("job_id", job_id)
        return payload

    def emit(event: dict[str, Any]) -> None:
        if sink is None:
            return
        sink(_stamp(event))

    # Wrap the runtime so tool/hook/approval events emitted deep inside
    # execute_tool bubble up tagged with this subagent's name (and job id).
    if runtime is not None:
        def tagging_handler(event: dict[str, Any]) -> None:
            if sink is None:
                return
            sink(_stamp(event))

        real_approval = runtime.approval_handler

        def approval_handler(request: Any) -> ApprovalDecision:
            emit(
                {
                    "type": "approval_requested",
                    "tool": getattr(request, "tool_name", ""),
                    "reason": getattr(request, "risk_reason", ""),
                    "command": getattr(request, "command", ""),
                    "label": job_label,
                }
            )
            if real_approval is None:
                return ApprovalDecision(approved=False, reason="no approval handler")
            labeled = replace(request, label=job_label) if job_label else request
            # Serialize concurrent approvals so parallel jobs present one popup
            # at a time rather than racing for the terminal.
            if approval_lock is not None:
                with approval_lock:
                    return real_approval(labeled)
            return real_approval(labeled)

        scoped_runtime = replace(runtime, event_handler=tagging_handler, approval_handler=approval_handler)
    else:
        scoped_runtime = runtime

    tools = allowed_subagent_tools(scoped_runtime, spec) if scoped_runtime is not None else []
    if extra_tools:
        tools = tools + list(extra_tools)
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


# --------------------------------------------------------------------------- #
# Parallel dispatch (Coordinator physical layer)
# --------------------------------------------------------------------------- #


class JobSpec(BaseModel):
    subagent_type: str = Field(description="Registered agent type to run for this job.")
    description: str = Field(description="Short 3-5 word label for traces and the TUI.")
    prompt: str = Field(
        description="Complete, self-contained task for this job. The subagent cannot "
        "see the parent conversation."
    )


class AgentDispatchToolInput(BaseModel):
    jobs: list[JobSpec] = Field(
        description=f"Up to {MAX_PARALLEL_JOBS} independent subagent jobs to run in parallel."
    )


def _job_dict(job: Any) -> dict[str, Any]:
    if isinstance(job, BaseModel):
        return job.model_dump()
    if isinstance(job, Mapping):
        return dict(job)
    return {}


def _run_one_job(
    state: Any,
    registry: Mapping[str, AgentSpec],
    job: Mapping[str, Any],
    job_id: str,
    approval_lock: threading.Lock,
) -> str:
    """Run a single dispatch job, returning a labelled line (never raising)."""

    subagent_type = str(job.get("subagent_type", ""))
    label = f"[{job_id} · {subagent_type or '?'}]"
    spec = registry.get(subagent_type)
    if spec is None:
        available = ", ".join(sorted(registry)) or "(none)"
        return f"{label} FAILED: unknown subagent type: {subagent_type} (available: {available})"

    try:
        summary = run_subagent(
            state,
            spec,
            str(job.get("prompt", "")),
            description=str(job.get("description", "")),
            job_id=job_id,
            approval_lock=approval_lock,
        )
    except Exception as exc:  # a single job's failure must not sink its siblings
        return f"{label} FAILED: {type(exc).__name__}: {exc}"
    return f"{label} {summary}"


def dispatch_parallel(state: Any, jobs: list[Any]) -> str:
    """Run up to ``MAX_PARALLEL_JOBS`` subagent jobs concurrently.

    Extra jobs beyond the cap are dropped and reported in a trailing warning
    line. Results are joined back in submission order; a job that raises is
    replaced by a ``[job-i] FAILED: …`` placeholder so the batch always returns.
    """

    runtime = _runtime(state)
    registry = load_agent_registry(runtime) if runtime is not None else {}

    normalized = [_job_dict(job) for job in jobs]
    accepted = normalized[:MAX_PARALLEL_JOBS]
    dropped = len(normalized) - len(accepted)

    approval_lock = threading.Lock()
    results: list[str] = [""] * len(accepted)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        futures = {
            executor.submit(
                _run_one_job, state, registry, job, f"job-{index + 1}", approval_lock
            ): index
            for index, job in enumerate(accepted)
        }
        for future, index in futures.items():
            results[index] = future.result()

    combined = "\n\n".join(results)
    if dropped > 0:
        combined += (
            f"\n\n[warning] AgentDispatchTool accepts at most {MAX_PARALLEL_JOBS} jobs "
            f"per call; {dropped} extra job(s) were dropped."
        )
    return combined


def make_agent_dispatch_tool(state: Any) -> StructuredTool:
    """Build the planner-only AgentDispatchTool for parallel subagent dispatch."""

    def agent_dispatch_tool(jobs: list[Any]) -> dict[str, Any]:
        output = dispatch_parallel(state, jobs)
        return {"ok": True, "name": AGENT_DISPATCH_TOOL_NAME, "output": output}

    return StructuredTool.from_function(
        func=agent_dispatch_tool,
        name=AGENT_DISPATCH_TOOL_NAME,
        description=(
            "Dispatch up to three INDEPENDENT subagent jobs in parallel, each with a "
            "self-contained prompt. Use only for genuinely independent research/review "
            "work; the main implementation trunk stays serial. Extra jobs are dropped."
        ),
        args_schema=AgentDispatchToolInput,
    )
