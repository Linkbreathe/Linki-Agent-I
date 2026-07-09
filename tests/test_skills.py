"""Tests for the skill registry and SkillTool progressive disclosure."""

from __future__ import annotations

from pathlib import Path

import pytest

from Linki.core.state import create_runtime


# --------------------------------------------------------------------------- #
# Step 1: skill registry
# --------------------------------------------------------------------------- #


def _write_skill(root: Path, name: str, description: str, body: str = "Full body.") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_builtin_skill_registry_loads_and_parses(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skill_registry

    runtime = create_runtime(tmp_path)
    registry = load_skill_registry(runtime)

    assert "conventional-commit" in registry
    spec = registry["conventional-commit"]
    assert spec.description
    assert len(spec.description) <= 120
    assert spec.body  # the full instructions are present in the spec
    assert spec.dir.name == "conventional-commit"


def test_description_over_limit_raises_file_specific_error(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skill_registry

    skills_dir = tmp_path / ".linki" / "skills"
    _write_skill(skills_dir, "too-long", "x" * 121)
    runtime = create_runtime(tmp_path)

    with pytest.raises(ValueError) as exc:
        load_skill_registry(runtime)

    message = str(exc.value)
    assert "too-long" in message
    assert "120" in message


def test_workspace_skill_overrides_builtin(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skill_registry

    skills_dir = tmp_path / ".linki" / "skills"
    _write_skill(
        skills_dir,
        "conventional-commit",
        "Overridden commit skill",
        body="Overridden instructions body.",
    )
    runtime = create_runtime(tmp_path)
    registry = load_skill_registry(runtime)

    spec = registry["conventional-commit"]
    assert spec.description == "Overridden commit skill"
    assert spec.body == "Overridden instructions body."


def test_missing_description_raises(tmp_path: Path) -> None:
    from Linki.core.frontmatter import parse_frontmatter_markdown

    skill_dir = tmp_path / "no-desc"
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    path.write_text("---\nname: no-desc\n---\n\nbody\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        parse_frontmatter_markdown(path, kind="skill")
    assert "SKILL.md" in str(exc.value)
    assert "description" in str(exc.value)


# --------------------------------------------------------------------------- #
# Step 2: SkillTool + progressive disclosure
# --------------------------------------------------------------------------- #

BODY_MARKER = "imperative mood"  # appears in the conventional-commit body, not its description


def test_available_skills_block_shows_description_not_body(tmp_path: Path) -> None:
    """Lazy disclosure: the startup prompt lists the description but not the body."""
    from Linki.skills.registry import load_skills_into_runtime, render_available_skills

    runtime = create_runtime(tmp_path)
    load_skills_into_runtime(runtime)

    block = render_available_skills(runtime)
    assert "<available_skills>" in block
    assert "· conventional-commit — " in block
    assert runtime.skills["conventional-commit"].description in block
    # The full instructions must NOT leak into the prompt before the tool is called.
    assert BODY_MARKER not in block


def test_skill_tool_loads_full_body_on_call(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skills_into_runtime
    from Linki.tools.skill_tool import make_skill_tool

    runtime = create_runtime(tmp_path)
    load_skills_into_runtime(runtime)
    tool = make_skill_tool({"runtime": runtime})

    result = tool.invoke({"name": "conventional-commit"})
    assert result["ok"] is True
    assert BODY_MARKER in result["output"]  # ToolMessage carries the full body
    assert "conventional-commit" in runtime.loaded_skills


def test_skill_tool_dedupes_within_a_run(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skills_into_runtime
    from Linki.tools.skill_tool import make_skill_tool

    runtime = create_runtime(tmp_path)
    load_skills_into_runtime(runtime)
    tool = make_skill_tool({"runtime": runtime})

    first = tool.invoke({"name": "conventional-commit"})
    second = tool.invoke({"name": "conventional-commit"})

    assert BODY_MARKER in first["output"]
    assert second["ok"] is True
    assert "already loaded" in second["output"].lower()
    assert BODY_MARKER not in second["output"]  # body not re-sent


def test_skill_tool_unknown_name_lists_available_without_raising(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skills_into_runtime
    from Linki.tools.skill_tool import make_skill_tool

    runtime = create_runtime(tmp_path)
    load_skills_into_runtime(runtime)
    tool = make_skill_tool({"runtime": runtime})

    result = tool.invoke({"name": "does-not-exist"})
    assert result["ok"] is True  # no exception
    assert "conventional-commit" in result["output"]  # available skills listed


def test_skill_tool_emits_skill_load_trace_event(tmp_path: Path) -> None:
    from Linki.skills.registry import load_skills_into_runtime
    from Linki.tools.skill_tool import make_skill_tool

    events: list[dict] = []
    runtime = create_runtime(tmp_path, event_handler=events.append)
    load_skills_into_runtime(runtime)
    tool = make_skill_tool({"runtime": runtime})

    tool.invoke({"name": "conventional-commit"})

    skill_loads = [e for e in events if e.get("type") == "skill_load"]
    assert len(skill_loads) == 1
    assert skill_loads[0]["name"] == "conventional-commit"
    assert isinstance(skill_loads[0]["tokens"], int) and skill_loads[0]["tokens"] > 0


def test_subagent_whitelist_grants_skill_tool(tmp_path: Path) -> None:
    """A subagent whose allowlist names SkillTool receives it (stage-nine mechanism)."""
    from Linki.agents.registry import load_agent_registry
    from Linki.tools.agent_tool import allowed_subagent_tools

    agents_dir = tmp_path / ".linki" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "committer.md").write_text(
        "---\nname: committer\ntools: [FileReadTool, SkillTool]\n---\n\nbody\n",
        encoding="utf-8",
    )
    runtime = create_runtime(tmp_path)
    registry = load_agent_registry(runtime)  # SkillTool must be a KNOWN_TOOL_NAME

    names = {t.name for t in allowed_subagent_tools(runtime, registry["committer"])}
    assert "SkillTool" in names
