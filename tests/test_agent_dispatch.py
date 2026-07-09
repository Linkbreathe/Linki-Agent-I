"""Tests for AgentDispatchTool: parallel subagent dispatch (Coordinator layer)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage

from Linki.agents.registry import load_agent_registry
from Linki.core.approval import ApprovalDecision
from Linki.core.state import create_runtime
from Linki.tools.agent_tool import make_agent_dispatch_tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _seed_workspace_agents(workspace: Path) -> None:
    src = PROJECT_ROOT / "src" / "Linki" / "agents" / "builtin"
    dst = workspace / ".linki" / "agents"
    dst.mkdir(parents=True, exist_ok=True)
    for md in src.glob("*.md"):
        dst.joinpath(md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")


class SleepModel:
    """Returns a final answer after sleeping — used to measure parallelism."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        time.sleep(self.delay)
        return AIMessage(content="done")


class ConditionalModel:
    """Raises when a job's prompt contains BOOM; otherwise returns a summary."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        text = "".join(str(getattr(m, "content", "")) for m in messages)
        if "BOOM" in text:
            raise RuntimeError("kaboom")
        return AIMessage(content="ok-summary")


class BashThenDoneModel:
    """Per-thread: first turn issues a risky BashTool call, second turn finishes."""

    def __init__(self) -> None:
        self._local = threading.local()

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        n = getattr(self._local, "n", 0)
        self._local.n = n + 1
        if n == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "BashTool",
                        "args": {"command": "pip install requests", "timeout_seconds": 30},
                        "id": f"c-{threading.get_ident()}",
                    }
                ],
            )
        return AIMessage(content="done")


def _job(agent: str, prompt: str, description: str = "x") -> dict:
    return {"subagent_type": agent, "description": description, "prompt": prompt}


def test_parallel_dispatch_beats_serial(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    delay = 0.3
    runtime = create_runtime(tmp_path)
    state = {"runtime": runtime, "model": SleepModel(delay)}
    tool = make_agent_dispatch_tool(state)

    jobs = [_job("search-agent", f"task {i}") for i in range(3)]
    started = time.monotonic()
    result = tool.invoke({"jobs": jobs})
    elapsed = time.monotonic() - started

    assert result["ok"] is True
    # Three parallel 0.3s jobs must finish well under the 0.9s serial sum.
    assert elapsed < 2 * delay, f"elapsed {elapsed:.3f}s suggests serial execution"


def test_one_job_fails_others_succeed(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    runtime = create_runtime(tmp_path)
    state = {"runtime": runtime, "model": ConditionalModel()}
    tool = make_agent_dispatch_tool(state)

    jobs = [
        _job("search-agent", "first job"),
        _job("search-agent", "second job BOOM"),
        _job("search-agent", "third job"),
    ]
    output = tool.invoke({"jobs": jobs})["output"]

    assert "[job-2" in output
    assert "FAILED" in output
    # The two healthy jobs still produced their summaries.
    assert output.count("ok-summary") == 2


def test_failed_parallel_job_emits_terminal_subagent_result(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    events: list[dict] = []
    runtime = create_runtime(tmp_path, event_handler=events.append)
    state = {"runtime": runtime, "model": ConditionalModel()}
    tool = make_agent_dispatch_tool(state)

    tool.invoke({"jobs": [_job("search-agent", "BOOM")]})

    failures = [
        event
        for event in events
        if event.get("type") == "subagent_result" and event.get("ok") is False
    ]
    assert failures
    assert failures[0]["job_id"] == "job-1"
    assert failures[0]["agent"] == "search-agent"


def test_fourth_job_truncated_with_warning(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    runtime = create_runtime(tmp_path)
    state = {"runtime": runtime, "model": ConditionalModel()}
    tool = make_agent_dispatch_tool(state)

    jobs = [_job("search-agent", f"job {i}") for i in range(4)]
    output = tool.invoke({"jobs": jobs})["output"]

    # Only three jobs ran; the fourth was dropped with a visible warning.
    assert output.count("ok-summary") == 3
    assert "3" in output and "warning" in output.lower()


def test_concurrent_approvals_are_serialized_and_tagged(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)

    order: list[str] = []
    order_lock = threading.Lock()

    def handler(request):
        with order_lock:
            order.append("start")
        time.sleep(0.1)
        with order_lock:
            order.append("end")
        return ApprovalDecision(approved=True, reason="ok")

    events: list[dict] = []
    runtime = create_runtime(
        tmp_path,
        approval_mode="inline",
        approval_handler=handler,
        event_handler=events.append,
    )
    state = {"runtime": runtime, "model": BashThenDoneModel()}
    tool = make_agent_dispatch_tool(state)

    jobs = [_job("reviewer", "review one"), _job("reviewer", "review two")]
    tool.invoke({"jobs": jobs})

    # Two approvals happened, and the serialization lock kept them from overlapping:
    # a well-formed sequence never has two "start" in a row.
    assert order.count("start") == 2
    for i in range(1, len(order)):
        assert not (order[i] == "start" and order[i - 1] == "start"), order

    approval_events = [e for e in events if e.get("type") == "approval_requested"]
    assert len(approval_events) == 2
    assert {e.get("job_id") for e in approval_events} == {"job-1", "job-2"}
    assert all("reviewer" in str(e.get("label", "")) for e in approval_events)


def test_dispatch_tool_registered_for_planner_only(tmp_path: Path) -> None:
    """AgentDispatchTool must be a planner tool, never a subagent tool."""
    from Linki.tools.registry import build_subagent_tools

    runtime = create_runtime(tmp_path)
    sub_names = {t.name for t in build_subagent_tools(runtime)}
    assert "AgentDispatchTool" not in sub_names

    # But it IS in the planner's tool pool.
    from Linki.graph.nodes import _build_planner_tools

    working = {"task": "x", "runtime": runtime, "todos": [], "acceptance_criteria": [], "verification_commands": []}
    planner_names = {t.name for t in _build_planner_tools(working)}
    assert "AgentDispatchTool" in planner_names

    # Also never granted to a subagent even if a definition tried to name it.
    _seed_workspace_agents(tmp_path)
    from Linki.tools.agent_tool import allowed_subagent_tools

    registry = load_agent_registry(runtime)
    for spec in registry.values():
        names = {t.name for t in allowed_subagent_tools(runtime, spec)}
        assert "AgentDispatchTool" not in names
