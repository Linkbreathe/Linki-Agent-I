"""Tests for coordinator discipline rules and the scratchpad convention."""

from __future__ import annotations

from pathlib import Path

from Linki.core.state import create_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_coordinator_rules_injected_after_available_agents(tmp_path: Path) -> None:
    from Linki.graph.memory import build_layered_memory
    from Linki.graph.nodes import _planner_input
    from Linki.prompts.coordinator import COORDINATOR_RULES

    runtime = create_runtime(tmp_path)
    working = {"task": "do work", "runtime": runtime}
    memory = build_layered_memory(working, node="planner")
    prompt = _planner_input(working, memory)

    assert COORDINATOR_RULES in prompt
    # Injected after the <available_agents> block.
    assert prompt.index("<available_agents>") < prompt.index(COORDINATOR_RULES)


def test_ensure_scratch_dir_creates_it(tmp_path: Path) -> None:
    from Linki.core.paths import ensure_scratch_dir

    runtime = create_runtime(tmp_path)
    scratch = ensure_scratch_dir(runtime)

    assert scratch.is_dir()
    assert scratch == runtime.workspace / ".linki" / "scratch"


def test_search_agent_body_has_pass_by_path_rule() -> None:
    body = (PROJECT_ROOT / "src" / "Linki" / "agents" / "builtin" / "search-agent.md").read_text(
        encoding="utf-8"
    )
    assert ".linki/scratch/" in body
    assert "200 words" in body
