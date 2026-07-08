"""Project-level context assembly for the Linki runtime.

Reads durable, workspace-scoped project signals that should frame every agent
turn: the workspace's ``LINKI.md`` house rules and, when the workspace is a git
repository, a compact snapshot of branch / dirty-file / recent-commit state.

The result is assembled once per run and threaded through graph state so the
planner and codeAgent prompt builders inject the same block. Everything here is
best-effort: a missing ``LINKI.md`` or a non-git (or broken git) workspace is
silently skipped rather than raised, so a run never fails on project context.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from Linki.core.state import RuntimeState

LINKI_FILENAME = "LINKI.md"
PROJECT_RULES_LIMIT = 4000
GIT_TIMEOUT_SECONDS = 5
RECENT_COMMITS = 5


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters, marking cuts with an ellipsis."""

    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def read_project_rules(state: RuntimeState) -> str | None:
    """Return the raw ``LINKI.md`` contents from the workspace root, or ``None``.

    Returns ``None`` when the file is absent or cannot be read; content is not
    truncated here so callers can apply their own limits.
    """

    path = Path(state.workspace) / LINKI_FILENAME
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _run_git(state: RuntimeState, args: list[str]) -> str | None:
    """Run a git command inside the workspace, returning stdout or ``None``.

    Any failure -- non-git workspace, missing git binary, timeout, or non-zero
    exit -- is swallowed and reported as ``None`` so callers can skip silently.
    """

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(state.workspace),
            text=True,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_status(porcelain_branch: str) -> tuple[str, int]:
    """Parse ``git status --porcelain -b`` output into (branch, modified_count)."""

    branch = "unknown"
    modified = 0
    for line in porcelain_branch.splitlines():
        if line.startswith("## "):
            header = line[3:].strip()
            # Formats: "main...origin/main [ahead 1]", "main", "HEAD (no branch)".
            branch = header.split("...", 1)[0].split(" ", 1)[0] or "unknown"
        elif line.strip():
            modified += 1
    return branch, modified


def _format_project_state(state: RuntimeState) -> str | None:
    """Build the ``project_state`` summary line, or ``None`` if not a git repo."""

    status = _run_git(state, ["status", "--porcelain", "-b"])
    if status is None:
        return None

    branch, modified = _parse_status(status)

    log = _run_git(state, ["log", "--oneline", f"-{RECENT_COMMITS}"])
    commits = [line.strip() for line in (log or "").splitlines() if line.strip()]
    recent = "; ".join(commits) if commits else "none"

    return f"branch: {branch} | modified: {modified} files | recent commits: {recent}"


def assemble_project_context(state: RuntimeState) -> str:
    """Assemble the workspace's project-context block for prompt injection.

    Combines ``LINKI.md`` house rules (truncated) and a git state summary into a
    ``<project_rules>``/``<project_state>`` block. Each section is included only
    when its source is available; when neither is, an empty string is returned.
    """

    sections: list[str] = []

    rules = read_project_rules(state)
    if rules is not None:
        sections.append(
            "<project_rules>\n"
            f"{_truncate(rules, PROJECT_RULES_LIMIT)}\n"
            "</project_rules>"
        )

    project_state = _format_project_state(state)
    if project_state is not None:
        sections.append(
            "<project_state>\n"
            f"{project_state}\n"
            "</project_state>"
        )

    return "\n".join(sections)
