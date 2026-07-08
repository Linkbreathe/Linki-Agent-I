"""Tests for project-context assembly and its merge into the rules layer."""

from __future__ import annotations

import subprocess
from pathlib import Path

from Linki.core.context import (
    PROJECT_RULES_LIMIT,
    assemble_project_context,
    read_project_rules,
)
from Linki.core.state import create_runtime
from Linki.graph.memory import build_layered_memory


def _init_git_repo(path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(path),
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("config", "commit.gpgsign", "false")
    (path / "seed.txt").write_text("seed", encoding="utf-8")
    git("add", "seed.txt")
    git("commit", "-m", "initial commit")


def test_returns_empty_for_bare_non_git_workspace(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    assert assemble_project_context(runtime) == ""


def test_includes_project_rules_when_linki_md_present(tmp_path: Path) -> None:
    (tmp_path / "LINKI.md").write_text("# House Rules\nBe careful.", encoding="utf-8")
    runtime = create_runtime(tmp_path)

    out = assemble_project_context(runtime)

    assert "<project_rules>" in out
    assert "Be careful." in out
    # No git repo, so no state section.
    assert "<project_state>" not in out


def test_project_rules_truncated_to_limit(tmp_path: Path) -> None:
    (tmp_path / "LINKI.md").write_text("x" * (PROJECT_RULES_LIMIT + 500), encoding="utf-8")
    runtime = create_runtime(tmp_path)

    out = assemble_project_context(runtime)

    assert "x" * PROJECT_RULES_LIMIT in out
    assert "x" * (PROJECT_RULES_LIMIT + 1) not in out
    assert "..." in out


def test_includes_git_state_for_repo(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    # An uncommitted change so modified count is non-zero.
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")
    runtime = create_runtime(tmp_path)

    out = assemble_project_context(runtime)

    assert "<project_state>" in out
    assert "branch:" in out
    assert "modified: 1 files" in out
    assert "initial commit" in out


def test_both_sections_present_together(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "LINKI.md").write_text("project rules body", encoding="utf-8")
    runtime = create_runtime(tmp_path)

    out = assemble_project_context(runtime)

    assert out.index("<project_rules>") < out.index("<project_state>")
    assert "project rules body" in out


def test_read_project_rules_missing_returns_none(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    assert read_project_rules(runtime) is None


def test_linki_md_merged_into_rules_layer(tmp_path: Path) -> None:
    (tmp_path / "LINKI.md").write_text("layered project rule", encoding="utf-8")
    runtime = create_runtime(tmp_path)

    memory = build_layered_memory({"runtime": runtime}, node="test")

    assert memory["rules"]["project_rules"] == "layered project rule"
    # Base rules must survive the merge.
    assert isinstance(memory["rules"]["rules"], list)
    assert memory["rules"]["rules"]


def test_rules_layer_has_no_project_rules_without_linki_md(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    memory = build_layered_memory({"runtime": runtime}, node="test")
    assert "project_rules" not in memory["rules"]
