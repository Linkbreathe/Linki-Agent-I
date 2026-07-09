from collections.abc import Callable
from collections import OrderedDict
from dataclasses import dataclass, field
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
    recent_files: OrderedDict[str, None] = field(default_factory=OrderedDict)
    # Skills loaded for the current run. ``skills`` maps skill name -> SkillSpec
    # (populated at run start); ``loaded_skills`` tracks which have had their full
    # body disclosed via SkillTool so far this run. Both are cleared per run.
    # Typed as ``Any`` values to avoid importing the skills package into core.
    skills: dict[str, Any] = field(default_factory=dict)
    loaded_skills: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).expanduser().resolve())
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from).expanduser().resolve())

    def touch_file(self, path: str | Path) -> None:
        file_path = Path(path)
        try:
            label = file_path.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            label = file_path.as_posix()
        self.recent_files.pop(label, None)
        self.recent_files[label] = None
        while len(self.recent_files) > 8:
            self.recent_files.popitem(last=False)


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
