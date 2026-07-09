"""Focused rendering checks for hook and subagent CLI events."""

from __future__ import annotations

from rich.console import Console

from Linki.cli import app as cli_app


def test_agent_tool_result_table_surfaces_subagent_summary() -> None:
    table = cli_app._tool_result_table(
        {
            "ok": True,
            "name": "AgentTool",
            "output": {
                "ok": True,
                "subagent_type": "reviewer",
                "description": "review hooks",
                "output": "No significant issue found.",
            },
        },
        tool_name="AgentTool",
    )

    console = Console(record=True, width=100)
    console.print(table)
    rendered = console.export_text()
    assert "Subagent" in rendered
    assert "reviewer" in rendered
    assert "review hooks" in rendered
    assert "No significant issue found." in rendered


def test_cli_hook_decision_rendering_includes_agent(monkeypatch) -> None:
    console = Console(record=True, width=100)
    monkeypatch.setattr(cli_app, "console", console)

    cli_app._print_event(
        {
            "type": "hook_decision",
            "agent": "reviewer",
            "event": "PreToolUse",
            "tool": "BashTool",
            "decision": "ask",
            "reason": "package install",
            "hook": ".linki/hooks/bash_guard.py",
        }
    )

    rendered = console.export_text()
    assert "reviewer" in rendered
    assert "Hook ask" in rendered
    assert "BashTool" in rendered
    assert "package install" in rendered


def test_cli_skill_load_rendering(monkeypatch) -> None:
    console = Console(record=True, width=100)
    monkeypatch.setattr(cli_app, "console", console)

    cli_app._print_event(
        {"type": "skill_load", "name": "conventional-commit", "tokens": 128}
    )

    rendered = console.export_text()
    assert "📚" in rendered
    assert "conventional-commit" in rendered


def test_cli_subagent_start_rendering_lists_tools(monkeypatch) -> None:
    console = Console(record=True, width=100)
    monkeypatch.setattr(cli_app, "console", console)

    cli_app._print_event(
        {
            "type": "subagent_start",
            "agent": "search-agent",
            "description": "research langgraph",
            "tools": ["WebSearchTool", "NotepadAppendTool"],
        }
    )

    rendered = console.export_text()
    assert "search-agent" in rendered
    assert "research langgraph" in rendered
    assert "WebSearchTool" in rendered
    assert "NotepadAppendTool" in rendered
