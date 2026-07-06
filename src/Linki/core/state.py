from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeState:
    """Runtime configuration shared by Linki tools."""

    workspace: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", self.workspace.expanduser().resolve())
