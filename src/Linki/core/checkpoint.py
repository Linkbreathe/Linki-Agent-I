"""Checkpoint saving and recovery for Linki graph runs.

Persists enough state under ``<workspace>/.linki/checkpoints`` to recover a
run after an interruption: a Git snapshot of the workspace tree, a
human-readable recovery guide, and (in "strict" mode) the full graph state
and event log.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from Linki.core.state import RuntimeState

VALID_CHECKPOINT_MODES = {
    "light",
    "strict",
    "off",
}

GIT_SNAPSHOT_DIRNAME = "workspace.git"
EXCLUDED_TOP_LEVEL_ENTRIES = {".linki", ".git", "__pycache__", ".venv", "node_modules"}


def normalize_checkpoint_mode(mode: str | None) -> str:
    """Normalize the checkpoint mode.

    Missing or invalid values fall back to "light".
    """

    if mode in VALID_CHECKPOINT_MODES:
        return mode
    return "light"


def _message_content(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        data: dict[str, Any] = {
            "type": getattr(value, "type", type(value).__name__),
            "content": _message_content(value),
        }
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = tool_calls
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        return data

    if isinstance(value, RuntimeState):
        return str(value.workspace)

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_summary": state.get("plan_summary", ""),
        "todos": _json_safe(state.get("todos", [])),
        "acceptance_criteria": list(state.get("acceptance_criteria", [])),
        "verification_commands": list(state.get("verification_commands", [])),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 0),
        "passed": state.get("passed"),
        "last_error": state.get("last_error", ""),
        "last_actor_summary": state.get("last_actor_summary", ""),
    }


def workspace_manifest(workspace: Path) -> list[dict[str, Any]]:
    """List workspace files (excluding checkpoint/VCS/build directories)."""

    manifest: list[dict[str, Any]] = []
    if not workspace.is_dir():
        return manifest

    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL_ENTRIES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        manifest.append(
            {
                "path": relative.as_posix(),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return manifest


def _git_dir(root: Path) -> Path:
    return root / GIT_SNAPSHOT_DIRNAME


def _run_git(workspace: Path, git_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={workspace}", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )


def _ensure_git_snapshot_repo(workspace: Path, git_dir: Path) -> None:
    if git_dir.exists():
        return
    subprocess.run(
        ["git", "init", "--quiet", "--bare", str(git_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    exclude_path = git_dir / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_path.write_text(".linki/\n.git/\n", encoding="utf-8")
    _run_git(workspace, git_dir, "config", "user.email", "linki-checkpoint@localhost")
    _run_git(workspace, git_dir, "config", "user.name", "Linki Checkpoint")


def snapshot_workspace_git(workspace: Path, root: Path) -> str | None:
    """Create a Git snapshot commit of the workspace tree.

    Uses a shadow bare repo under ``root`` (separate from any real Git repo
    the workspace may already contain) so checkpoint snapshots never
    interfere with the user's own version control. Returns the resulting
    commit hash, or None when git is unavailable or the snapshot fails.
    """

    git_dir = _git_dir(root)
    try:
        _ensure_git_snapshot_repo(workspace, git_dir)
        _run_git(workspace, git_dir, "add", "-A")
        _run_git(workspace, git_dir, "commit", "--quiet", "--allow-empty", "-m", "Linki checkpoint")
        result = _run_git(workspace, git_dir, "rev-parse", "HEAD")
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def restore_workspace_git(workspace: Path, root: Path, commit: str) -> bool:
    """Restore workspace files from a prior Git snapshot commit.

    Returns True on success, False when the restore could not be completed
    (e.g. no snapshot repo exists yet).
    """

    git_dir = _git_dir(root)
    if not git_dir.exists():
        return False
    try:
        _run_git(workspace, git_dir, "checkout", commit, "--", ".")
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def resume_command(workspace: Path) -> str:
    """Generate the CLI command used to resume the workspace."""

    return f"Linki --resume {workspace}"


def build_recovery_markdown(payload: Mapping[str, Any]) -> str:
    """Generate the contents of RECOVERY.md from a checkpoint payload."""

    manifest = payload.get("workspace_manifest") or []
    if manifest:
        manifest_lines = "\n".join(f"- `{item['path']}` ({item['size']} bytes)" for item in manifest)
    else:
        manifest_lines = "_No tracked workspace files._"

    workspace = Path(str(payload.get("workspace", "")))

    lines = [
        "# Linki Recovery",
        "",
        f"- **Task**: {payload.get('task', '')}",
        f"- **Status**: {payload.get('status', '')}",
        f"- **Latest node**: {payload.get('latest_node') or '—'}",
        f"- **Checkpoint mode**: {payload.get('checkpoint_mode', '')}",
        f"- **Workspace**: {payload.get('workspace', '')}",
        f"- **Git snapshot commit**: {payload.get('git_commit') or '—'}",
        f"- **Saved at**: {payload.get('saved_at', '')}",
        "",
        "## Workspace files",
        "",
        manifest_lines,
        "",
        "## Resume",
        "",
        "```",
        resume_command(workspace),
        "```",
        "",
    ]
    return "\n".join(lines)


class CheckpointManager:
    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.workspace = runtime.workspace
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        self.task = task
        self.root = self.workspace / ".linki" / "checkpoints"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def save(
        self,
        state: Mapping[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Save a checkpoint. Returns None when checkpointing is disabled."""

        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)

        if self.mode == "strict":
            if event is not None:
                self._append_event(event)
            self._save_state(state)

        manifest = workspace_manifest(self.workspace)
        commit = snapshot_workspace_git(self.workspace, self.root)
        saved_at = datetime.now(timezone.utc).isoformat()

        payload = {
            "task": self.task,
            "status": status,
            "latest_node": latest_node,
            "checkpoint_mode": self.mode,
            "workspace": str(self.workspace),
            "workspace_manifest": manifest,
            "git_commit": commit,
            "saved_at": saved_at,
            "state_summary": _state_summary(state),
        }

        _write_json(self.root / "checkpoint.json", payload)
        (self.root / "RECOVERY.md").write_text(build_recovery_markdown(payload), encoding="utf-8")

        return {
            "type": "checkpoint_saved",
            "checkpoint_mode": self.mode,
            "status": status,
            "latest_node": latest_node,
            "git_commit": commit,
            "saved_at": saved_at,
            "path": str(self.root),
        }

    def _append_event(self, event: Mapping[str, Any]) -> None:
        events_path = self.root / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dict(event)), ensure_ascii=False) + "\n")

    def _save_state(self, state: Mapping[str, Any]) -> None:
        skip_keys = {"runtime", "model"}
        safe_state = {key: _json_safe(value) for key, value in state.items() if key not in skip_keys}
        _write_json(self.root / "state.json", safe_state)

    @classmethod
    def load_resume_inputs(
        cls,
        runtime: RuntimeState,
        task: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore workflow inputs from a checkpoint.

        Returns (inputs, resume_event).
        """

        manager = cls(runtime, task=task or "")
        checkpoint_path = manager.root / "checkpoint.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        git_commit = payload.get("git_commit")
        restored = False
        if git_commit:
            restored = restore_workspace_git(manager.workspace, manager.root, git_commit)

        state_path = manager.root / "state.json"
        persisted_state: dict[str, Any] = {}
        if state_path.is_file():
            persisted_state = json.loads(state_path.read_text(encoding="utf-8"))

        resolved_task = task or persisted_state.get("task") or payload.get("task", "")

        inputs: dict[str, Any] = {
            "task": resolved_task,
            "runtime": runtime,
            "messages": persisted_state.get("messages", []),
            "attempts": persisted_state.get("attempts", payload.get("state_summary", {}).get("attempts", 0)),
            "max_attempts": max_attempts,
        }

        carry_over_keys = {"task", "messages", "attempts", "max_attempts"}
        for key, value in persisted_state.items():
            if key in carry_over_keys:
                continue
            inputs[key] = value

        if not persisted_state:
            for key, value in (payload.get("state_summary") or {}).items():
                inputs.setdefault(key, value)

        resume_event = {
            "type": "checkpoint_resumed",
            "checkpoint_mode": payload.get("checkpoint_mode"),
            "status": payload.get("status"),
            "latest_node": payload.get("latest_node"),
            "git_commit": git_commit,
            "git_restored": restored,
            "saved_at": payload.get("saved_at"),
            "path": str(manager.root),
        }

        return inputs, resume_event
