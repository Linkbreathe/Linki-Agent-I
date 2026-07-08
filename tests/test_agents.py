"""Tests for the declarative agent registry and controlled subagent runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from Linki.agents.registry import load_agent_registry, parse_frontmatter_markdown
from Linki.core.state import create_runtime
from Linki.tools.agent_tool import allowed_subagent_tools, make_agent_tool, run_subagent

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """A scripted chat model: returns queued responses and records bound tools."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


def _seed_workspace_agents(workspace: Path) -> None:
    src = PROJECT_ROOT / ".linki" / "agents"
    dst = workspace / ".linki" / "agents"
    dst.mkdir(parents=True, exist_ok=True)
    for md in src.glob("*.md"):
        dst.joinpath(md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Step 1: registry
# --------------------------------------------------------------------------- #


def test_builtin_registry_loads_and_parses(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    registry = load_agent_registry(runtime)

    assert "search-agent" in registry
    spec = registry["search-agent"]
    assert spec.description == "Search the web and organize findings"
    assert spec.tools == ["WebSearchTool", "NotepadAppendTool"]
    assert spec.system_prompt.startswith("You are searchAgent")
    # registry is keyed by agent name
    assert registry["search-agent"].name == "search-agent"


def test_unknown_tool_aborts_loading(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".linki" / "agents"
    agents_dir.mkdir(parents=True)
    bad = agents_dir / "bad-agent.md"
    bad.write_text(
        "---\nname: bad-agent\ntools: [FileReadTool, TotallyFakeTool]\n---\n\nbody\n",
        encoding="utf-8",
    )
    runtime = create_runtime(tmp_path)

    with pytest.raises(ValueError) as exc:
        load_agent_registry(runtime)

    message = str(exc.value)
    assert "bad-agent.md" in message
    assert "TotallyFakeTool" in message


def test_missing_name_raises_file_specific_error(tmp_path: Path) -> None:
    path = tmp_path / "no-name.md"
    path.write_text("---\ntools: [FileReadTool]\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        parse_frontmatter_markdown(path)
    assert "no-name.md" in str(exc.value)


def test_malformed_yaml_raises_file_specific_error(tmp_path: Path) -> None:
    path = tmp_path / "bad-yaml.md"
    path.write_text("---\nname: [unterminated\ntools: [FileReadTool]\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        parse_frontmatter_markdown(path)
    message = str(exc.value)
    assert "bad-yaml.md" in message
    assert "invalid YAML" in message


def test_workspace_definition_overrides_builtin(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".linki" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "search-agent.md").write_text(
        "---\nname: search-agent\ndescription: Overridden searcher\n"
        "tools: [WebSearchTool]\n---\n\nOverridden prompt body.\n",
        encoding="utf-8",
    )
    runtime = create_runtime(tmp_path)
    registry = load_agent_registry(runtime)

    spec = registry["search-agent"]
    assert spec.description == "Overridden searcher"
    assert spec.tools == ["WebSearchTool"]
    assert spec.system_prompt == "Overridden prompt body."


# --------------------------------------------------------------------------- #
# Step 2/3: subagent runtime
# --------------------------------------------------------------------------- #


def test_reviewer_tool_pool_is_trimmed(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    runtime = create_runtime(tmp_path)
    registry = load_agent_registry(runtime)

    names = {t.name for t in allowed_subagent_tools(runtime, registry["reviewer"])}
    assert "FileReadTool" in names
    assert "GrepTool" in names
    assert "BashTool" in names
    assert "FileWriteTool" not in names
    assert "FileEditTool" not in names


def test_agent_tool_never_available_to_subagents(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    # An agent that erroneously lists AgentTool must still not receive it.
    (tmp_path / ".linki" / "agents" / "greedy.md").write_text(
        "---\nname: greedy\ntools: [FileReadTool, AgentTool]\n---\n\nbody\n",
        encoding="utf-8",
    )
    runtime = create_runtime(tmp_path)
    registry = load_agent_registry(runtime)

    for spec in registry.values():
        names = {t.name for t in allowed_subagent_tools(runtime, spec)}
        assert "AgentTool" not in names


def test_unknown_subagent_type_returns_error_not_exception(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    runtime = create_runtime(tmp_path)
    state = {"runtime": runtime, "model": FakeModel([])}
    tool = make_agent_tool(state)

    result = tool.invoke(
        {"subagent_type": "security-reviewer", "description": "x", "prompt": "y"}
    )
    assert result["ok"] is False
    assert "unknown subagent type: security-reviewer" in result["error"]
    assert "available types:" in result["error"]
    assert "reviewer" in result["error"]


def test_context_isolation_and_result_return(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    events: list[dict] = []
    runtime = create_runtime(tmp_path, event_handler=events.append)
    registry = load_agent_registry(runtime)

    fake = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "NotepadAppendTool", "args": {"note": "finding"}, "id": "c1"}],
            ),
            AIMessage(content="research summary with sources"),
        ]
    )
    state = {"runtime": runtime, "model": fake}

    parent_messages = [SystemMessage(content="parent"), HumanMessage(content="parent task")]
    before = list(parent_messages)

    summary = run_subagent(state, registry["search-agent"], "self-contained prompt")

    # subagent history must not leak into the parent messages
    assert parent_messages == before
    assert summary == "research summary with sources"
    # subagent got only its allowlisted tools
    assert set(fake.bound_tool_names) == {"WebSearchTool", "NotepadAppendTool"}
    # nested events were recorded and tagged with the agent name
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert tool_results and all(e["agent"] == "search-agent" for e in tool_results)
    assert any(e.get("type") == "subagent_result" and e["agent"] == "search-agent" for e in events)
    # the notepad append actually happened through the canonical pipeline
    assert (tmp_path / "NOTEPAD.md").read_text(encoding="utf-8").find("finding") != -1


def test_approval_propagates_from_subagent_deny(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    events: list[dict] = []
    requests: list = []

    def handler(request):
        from Linki.core.approval import ApprovalDecision

        requests.append(request)
        return ApprovalDecision(approved=False, reason="denied in test")

    runtime = create_runtime(
        tmp_path, approval_mode="inline", approval_handler=handler, event_handler=events.append
    )
    registry = load_agent_registry(runtime)

    fake = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "BashTool", "args": {"command": "pip install requests", "timeout_seconds": 30}, "id": "c1"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    run_subagent({"runtime": runtime, "model": fake}, registry["reviewer"], "review and try install")

    # approval reached the parent stream, tagged with the subagent name
    approval_events = [e for e in events if e.get("type") == "approval_requested"]
    assert approval_events
    assert approval_events[0]["agent"] == "reviewer"
    assert requests  # the real handler was consulted
    # denial prevented execution: no sentinel file, tool_result marked failed
    tool_results = [e for e in events if e.get("type") == "tool_result" and e["name"] == "BashTool"]
    assert tool_results and tool_results[0]["result"]["ok"] is False


def test_approval_allows_execution_when_approved(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)

    def handler(request):
        from Linki.core.approval import ApprovalDecision

        return ApprovalDecision(approved=True, reason="approved in test")

    runtime = create_runtime(tmp_path, approval_mode="inline", approval_handler=handler)
    registry = load_agent_registry(runtime)

    fake = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    # matches the install risk pattern but harmlessly prints help
                    {"name": "BashTool", "args": {"command": "pip install --help", "timeout_seconds": 60}, "id": "c1"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    events: list[dict] = []
    runtime = create_runtime(
        tmp_path, approval_mode="inline", approval_handler=handler, event_handler=events.append
    )
    registry = load_agent_registry(runtime)
    run_subagent({"runtime": runtime, "model": fake}, registry["reviewer"], "run pip help")

    tool_results = [e for e in events if e.get("type") == "tool_result" and e["name"] == "BashTool"]
    assert tool_results and tool_results[0]["result"]["ok"] is True
