"""Durable-notes tools backed by the workspace ``NOTEPAD.md`` file.

The notepad is the project's durable memory surface (see ``graph/memory.py``).
These tools expose read/append access so agents — especially subagents whose
findings should survive context compression — can record and recover notes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from Linki.core.paths import resolve_workspace_path
from Linki.core.state import RuntimeState

NOTEPAD_FILENAME = "NOTEPAD.md"


def _notepad_path(state: RuntimeState):
    return resolve_workspace_path(state, NOTEPAD_FILENAME)


class NotepadReadTool:
    """Return the current contents of the workspace notepad."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self) -> str:
        path = _notepad_path(self.state)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")


class NotepadAppendTool:
    """Append a durable note to the workspace notepad, creating it if needed."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, note: str) -> str:
        note = (note or "").strip()
        if not note:
            return "empty note ignored"

        path = _notepad_path(self.state)
        path.parent.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = f"- [{stamp}] {note}\n"

        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            path.write_text(existing + entry, encoding="utf-8")
        else:
            path.write_text(f"# NOTEPAD\n\n{entry}", encoding="utf-8")

        return f"appended note to {NOTEPAD_FILENAME}"
