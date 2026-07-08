from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeState:
    """Runtime configuration shared by Linki tools."""

    workspace: Path
    approval_mode: str = "inline"
    approval_handler: Callable[[Any], Any] | None = None
    checkpoint_mode: str | None = None
    trace_mode: str | None = None
    trace_id: str | None = None
    resume_from: Path | None = None
    event_handler: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).expanduser().resolve())
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from).expanduser().resolve())


def create_runtime(
    workspace: str | Path,
    *,
    approval_mode: str = "inline",
    approval_handler: Callable[[Any], Any] | None = None,
    checkpoint_mode: str = "light",
    resume_from: str | Path | None = None,
    trace_mode: str = "on",
    trace_id: str | None = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
) -> RuntimeState:
    """Create the normalized runtime object shared by graph nodes and tools."""

    return RuntimeState(
        workspace=Path(workspace),
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        trace_id=trace_id,
        resume_from=Path(resume_from) if resume_from is not None else None,
        event_handler=event_handler,
    )
