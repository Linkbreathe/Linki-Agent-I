from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

SESSION_ROOT = ".linki/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"

DEFAULT_WORKSPACE_BASE = "workspace"
RUN_DIR_PREFIX = "run-"

MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000

_MAX_STORED_TURN_ENTRIES = 80
_MAX_CONTEXT_FILES = 30
_MAX_CONTEXT_TURNS = 10
_EXCLUDED_PATH_PARTS = {
    ".git",
    ".linki",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
    "dist",
    "build",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _session_root(workspace: Path) -> Path:
    return workspace / SESSION_ROOT


def _session_path(workspace: Path) -> Path:
    return _session_root(workspace) / SESSION_FILE


def _session_summary_path(workspace: Path) -> Path:
    return _session_root(workspace) / SESSION_SUMMARY_FILE


def resolve_session_workspace(session_workspace: str | Path | None = None) -> Path:
    """Resolve the workspace used for a multi-turn session."""

    workspace = Path(session_workspace or ".").expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not workspace.is_dir():
        raise NotADirectoryError(f"Session workspace is not a directory: {workspace}")
    return workspace


def create_run_workspace(base_dir: str | Path | None = None) -> Path:
    """Create and return a fresh run workspace under ``base_dir``.

    Each application start gets its own ``run-<timestamp>`` folder so a new run
    never overwrites the checkpoints of a previous one. ``base_dir`` is treated
    as a parent directory and defaults to ``workspace``.
    """

    base = Path(base_dir or DEFAULT_WORKSPACE_BASE).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base / f"{RUN_DIR_PREFIX}{stamp}"

    suffix = 2
    while candidate.exists():
        candidate = base / f"{RUN_DIR_PREFIX}{stamp}-{suffix}"
        suffix += 1

    candidate.mkdir(parents=True)
    return candidate


def _new_session() -> dict[str, Any]:
    now = _now_iso()
    return {
        "session_id": str(uuid4()),
        "turn_index": 0,
        "recent_turns": [],
        "created_at": now,
        "updated_at": now,
    }


def _valid_session(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("session_id"), str)
        and isinstance(value.get("turn_index"), int)
        and isinstance(value.get("recent_turns"), list)
        and isinstance(value.get("created_at"), str)
        and isinstance(value.get("updated_at"), str)
    )


def load_or_create_session(workspace: Path) -> dict:
    """
    Load session.json from the workspace, or create a new session when no
    valid session file exists.
    """

    path = _session_path(Path(workspace))
    if not path.is_file():
        return _new_session()

    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _new_session()

    if not _valid_session(session):
        return _new_session()

    return session


def append_user_turn(session: dict, content: str) -> int:
    """
    Record the user's input and return the new turn number.

    Truncate content to MAX_TURN_CONTENT when necessary.
    """

    turn = int(session.get("turn_index", 0)) + 1
    session["turn_index"] = turn
    session.setdefault("recent_turns", []).append(
        {
            "turn": turn,
            "role": "user",
            "content": _truncate(content, MAX_TURN_CONTENT),
            "timestamp": _now_iso(),
        }
    )
    return turn


def append_assistant_turn(
    session: dict,
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    """
    Record an assistant response.

    route must be either "chat" or "workflow".
    """

    if route not in {"chat", "workflow"}:
        raise ValueError("route must be either 'chat' or 'workflow'")

    session.setdefault("recent_turns", []).append(
        {
            "turn": turn,
            "role": "assistant",
            "route": route,
            "content": _truncate(content, MAX_TURN_CONTENT),
            "summary": _truncate(summary, MAX_TURN_CONTENT),
            "timestamp": _now_iso(),
        }
    )


def _bounded_recent_turns(session: dict) -> list[dict[str, Any]]:
    turns = session.get("recent_turns", [])
    if not isinstance(turns, list):
        return []
    return [turn for turn in turns[-_MAX_STORED_TURN_ENTRIES:] if isinstance(turn, dict)]


def _turn_summary(turn: dict[str, Any], *, limit: int = 500) -> str:
    if turn.get("role") == "assistant" and turn.get("summary"):
        text = str(turn.get("summary") or "")
    else:
        text = str(turn.get("content") or "")
    text = " ".join(text.split())
    return _truncate(text, limit)


def _build_session_summary(session: dict) -> str:
    lines = [
        "# Linki Session Summary",
        "",
        f"- Session: {session.get('session_id', '')}",
        f"- Turn index: {session.get('turn_index', 0)}",
        f"- Created at: {session.get('created_at', '')}",
        f"- Updated at: {session.get('updated_at', '')}",
        "",
        "## Recent Turns",
        "",
    ]

    recent_turns = _bounded_recent_turns(session)[-_MAX_CONTEXT_TURNS:]
    if not recent_turns:
        lines.append("_No turns recorded yet._")
    for turn in recent_turns:
        role = turn.get("role", "unknown")
        route = f" ({turn.get('route')})" if turn.get("route") else ""
        lines.append(f"- Turn {turn.get('turn')}: {role}{route} - {_turn_summary(turn)}")

    lines.append("")
    return "\n".join(lines)


def save_session(
    workspace: Path,
    session: dict,
) -> dict:
    """
    Save session.json and generate SESSION_SUMMARY.md.

    Return useful metadata about the saved session.
    """

    workspace = Path(workspace)
    root = _session_root(workspace)
    root.mkdir(parents=True, exist_ok=True)

    session["recent_turns"] = _bounded_recent_turns(session)
    session["updated_at"] = _now_iso()

    session_path = _session_path(workspace)
    summary_path = _session_summary_path(workspace)
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _build_session_summary(session)
    summary_path.write_text(summary, encoding="utf-8")

    return {
        "type": "session_saved",
        "session_id": session.get("session_id"),
        "turn_index": session.get("turn_index", 0),
        "recent_turns": len(session.get("recent_turns", [])),
        "path": str(session_path),
        "summary_path": str(summary_path),
    }


def _workspace_files(workspace: Path) -> list[str]:
    files: list[str] = []
    if not workspace.is_dir():
        return files

    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if any(part in _EXCLUDED_PATH_PARTS for part in relative.parts):
            continue
        files.append(relative.as_posix())
        if len(files) >= _MAX_CONTEXT_FILES:
            break
    return files


def build_session_context(
    workspace: Path,
    session: dict | None = None,
) -> str:
    """
    Build the session-context string supplied to the intent router and chat
    responder.
    """

    workspace = Path(workspace)
    session = session or load_or_create_session(workspace)
    files = _workspace_files(workspace)
    recent_turns = _bounded_recent_turns(session)[-_MAX_CONTEXT_TURNS:]

    lines = [
        f"session_id: {session.get('session_id', '')}",
        f"turn_index: {session.get('turn_index', 0)}",
        "",
        "workspace_files:",
    ]
    if files:
        lines.extend(f"- {file}" for file in files)
    else:
        lines.append("- none")

    lines.extend(["", "recent_turns:"])
    if recent_turns:
        for turn in recent_turns:
            role = str(turn.get("role") or "unknown")
            route = f"/{turn.get('route')}" if turn.get("route") else ""
            lines.append(f"- turn {turn.get('turn')} {role}{route}: {_turn_summary(turn, limit=700)}")
    else:
        lines.append("- none")

    return _truncate("\n".join(lines), MAX_SESSION_CONTEXT)
