"""Per-agent mailbox for the swarm lab.

Messages are appended as JSON lines to ``inbox/<agent>.jsonl``. Each agent has a
cursor (``inbox/<agent>.cursor``) recording how many lines it has already read,
so ``read_new`` returns only messages that arrived since the last read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _inbox_dir(team_dir: str | Path) -> Path:
    inbox = Path(team_dir) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox


def send(team_dir: str | Path, to: str, frm: str, text: str) -> dict:
    """Append a message to ``to``'s inbox."""

    entry = {"to": to, "from": frm, "text": text, "ts": time.time()}
    path = _inbox_dir(team_dir) / f"{to}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_new(team_dir: str | Path, agent: str) -> list[dict]:
    """Return messages for ``agent`` newer than its cursor, then advance it."""

    inbox = _inbox_dir(team_dir)
    path = inbox / f"{agent}.jsonl"
    cursor_path = inbox / f"{agent}.cursor"
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    if cursor_path.exists():
        try:
            cursor = int(cursor_path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            cursor = 0

    new = [json.loads(line) for line in lines[cursor:] if line.strip()]
    cursor_path.write_text(str(len(lines)), encoding="utf-8")
    return new
