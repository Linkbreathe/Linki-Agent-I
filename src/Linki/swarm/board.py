"""Shared task board for the swarm lab.

The board is a single ``board.json`` file under the team directory. All mutations
happen under a filesystem lock (an atomically-created lock directory) and are
committed with ``os.replace`` so a reader never observes a half-written file and
two workers never lose each other's update — ``claim`` is a genuine
compare-and-set, so two threads racing for one task yield exactly one winner.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BOARD_FILE = "board.json"
LOCK_DIR = "board.lock"

# A claimed task whose owner has been silent for this many rounds is released.
REAP_ROUNDS = 2


def _board_path(team_dir: str | Path) -> Path:
    return Path(team_dir) / BOARD_FILE


class _BoardLock:
    """Cross-thread/process mutual exclusion via an atomic ``mkdir`` on a dir."""

    def __init__(self, team_dir: str | Path, timeout: float = 5.0) -> None:
        self._lock = Path(team_dir) / LOCK_DIR
        self._timeout = timeout

    def __enter__(self) -> "_BoardLock":
        start = time.monotonic()
        while True:
            try:
                self._lock.mkdir(parents=True, exist_ok=False)
                return self
            except FileExistsError:
                if time.monotonic() - start > self._timeout:
                    raise TimeoutError(f"board lock not acquired within {self._timeout}s")
                time.sleep(0.005)

    def __exit__(self, *_exc: Any) -> None:
        try:
            self._lock.rmdir()
        except FileNotFoundError:
            pass


def _read(team_dir: str | Path) -> dict[str, Any]:
    return json.loads(_board_path(team_dir).read_text(encoding="utf-8"))


def _atomic_write(team_dir: str | Path, board: Mapping[str, Any]) -> None:
    path = _board_path(team_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX and Windows


def _deps_done(board: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    tasks = board["tasks"]
    return all(tasks.get(dep, {}).get("status") == "done" for dep in task.get("depends_on", []))


def init_board(team_dir: str | Path, tasks: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Initialize board.json from a planner's task list (with ``depends_on``)."""

    Path(team_dir).mkdir(parents=True, exist_ok=True)
    board: dict[str, Any] = {"round": 0, "tasks": {}}
    for task in tasks:
        tid = str(task["id"])
        deps = [str(dep) for dep in task.get("depends_on", [])]
        board["tasks"][tid] = {
            "id": tid,
            "title": str(task.get("title", "")),
            "depends_on": deps,
            # A task with unmet dependencies starts blocked; it is unblocked once
            # all of its dependencies reach "done".
            "status": "blocked" if deps else "pending",
            "owner": None,
            "claimed_round": None,
            "result": "",
        }
    with _BoardLock(team_dir):
        _atomic_write(team_dir, board)
    return board


def claim(team_dir: str | Path, task_id: str, agent: str, rnd: int) -> bool:
    """Atomically claim a claimable (pending, deps-done) task. One winner only."""

    with _BoardLock(team_dir):
        board = _read(team_dir)
        task = board["tasks"].get(str(task_id))
        if task is None or task["status"] != "pending" or not _deps_done(board, task):
            return False
        task["status"] = "claimed"
        task["owner"] = agent
        task["claimed_round"] = rnd
        _atomic_write(team_dir, board)
        return True


def update(
    team_dir: str | Path,
    task_id: str,
    *,
    status: str | None = None,
    result: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    """Update a task's status/result under the board lock."""

    with _BoardLock(team_dir):
        board = _read(team_dir)
        task = board["tasks"].get(str(task_id))
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if status is not None:
            task["status"] = status
        if result is not None:
            task["result"] = result
        if agent is not None:
            task["owner"] = agent
        _atomic_write(team_dir, board)
        return dict(task)


def snapshot(team_dir: str | Path) -> dict[str, Any]:
    """Return a consistent read of the whole board."""

    with _BoardLock(team_dir):
        return _read(team_dir)


def save_board(team_dir: str | Path, board: Mapping[str, Any]) -> None:
    """Persist a board dict atomically (used after a round-end mutation)."""

    with _BoardLock(team_dir):
        _atomic_write(team_dir, board)


def detect_cycle(board: Mapping[str, Any]) -> list[str] | None:
    """Return a cycle path in the depends_on graph, or None if it is acyclic."""

    tasks = board["tasks"]
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {tid: WHITE for tid in tasks}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GRAY
        stack.append(node)
        for dep in tasks.get(node, {}).get("depends_on", []):
            if dep not in tasks:
                continue
            if color.get(dep) == GRAY:
                return stack[stack.index(dep):] + [dep]
            if color.get(dep) == WHITE:
                found = visit(dep)
                if found:
                    return found
        color[node] = BLACK
        stack.pop()
        return None

    for tid in tasks:
        if color[tid] == WHITE:
            cycle = visit(tid)
            if cycle:
                return cycle
    return None


def unblock_and_reap(board: dict[str, Any], rnd: int) -> dict[str, Any]:
    """Round-end sweep, mutating ``board`` in place.

    - Unblock tasks whose dependencies are all done.
    - Reap claimed tasks whose owner has been silent >= ``REAP_ROUNDS`` rounds,
      releasing them back to pending.
    - Detect a dependency cycle (deadlock); the returned ``cycle`` is the path.
    """

    cycle = detect_cycle(board)
    unblocked: list[str] = []
    reaped: list[str] = []

    for tid, task in board["tasks"].items():
        if task["status"] == "blocked" and _deps_done(board, task):
            task["status"] = "pending"
            unblocked.append(tid)
        elif (
            task["status"] == "claimed"
            and task.get("claimed_round") is not None
            and rnd - task["claimed_round"] >= REAP_ROUNDS
        ):
            task["status"] = "pending"
            task["owner"] = None
            task["claimed_round"] = None
            reaped.append(tid)

    board["round"] = rnd
    return {"unblocked": unblocked, "reaped": reaped, "cycle": cycle}


def all_done(board: Mapping[str, Any]) -> bool:
    return bool(board["tasks"]) and all(t["status"] == "done" for t in board["tasks"].values())
