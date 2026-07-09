from pathlib import Path

from Linki.core.state import RuntimeState


def ensure_workspace(state: RuntimeState, create: bool = True) -> Path:
    """Return the resolved workspace, creating it when requested."""

    if create:
        state.workspace.mkdir(parents=True, exist_ok=True)
    if not state.workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {state.workspace}")
    if not state.workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {state.workspace}")
    return state.workspace


def ensure_scratch_dir(state: RuntimeState) -> Path:
    """Return the workspace scratchpad (``.linki/scratch``), creating it.

    Subagents write large findings here and pass the path downstream instead of
    embedding the content in prompts (see the coordinator rules).
    """

    scratch = state.workspace / ".linki" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def resolve_workspace_path(state: RuntimeState, file_path: str | Path) -> Path:
    """Resolve a user path and reject paths outside the workspace."""

    ensure_workspace(state)
    raw_path = Path(file_path).expanduser()
    candidate = raw_path if raw_path.is_absolute() else state.workspace / raw_path
    resolved = candidate.resolve(strict=False)

    if resolved == state.workspace or state.workspace in resolved.parents:
        return resolved

    raise PermissionError(f"Path escapes workspace: {file_path}")
