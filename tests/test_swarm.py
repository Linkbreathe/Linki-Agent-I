"""Tests for the experimental swarm lab: board, mailbox, scheduler."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from Linki.core.state import create_runtime
from Linki.swarm import board, mailbox
from Linki.swarm.scheduler import run_swarm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _seed_workspace_agents(workspace: Path) -> None:
    src = PROJECT_ROOT / ".linki" / "agents"
    dst = workspace / ".linki" / "agents"
    dst.mkdir(parents=True, exist_ok=True)
    for md in src.glob("*.md"):
        dst.joinpath(md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")


# --------------------------------------------------------------------------- #
# board.py
# --------------------------------------------------------------------------- #


def test_atomic_claim_only_one_winner(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    board.init_board(team_dir, [{"id": "t1", "title": "x", "depends_on": []}])

    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        barrier.wait()  # maximize the race window
        ok = board.claim(team_dir, "t1", name, 0)
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker, args=(f"agent-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1, results
    assert results.count(False) == 1, results
    assert board.snapshot(team_dir)["tasks"]["t1"]["status"] == "claimed"


def test_dependency_unlock(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    board.init_board(
        team_dir,
        [{"id": "t1", "depends_on": []}, {"id": "t2", "depends_on": ["t1"]}],
    )
    assert board.snapshot(team_dir)["tasks"]["t2"]["status"] == "blocked"

    board.update(team_dir, "t1", status="done")
    snap = board.snapshot(team_dir)
    sweep = board.unblock_and_reap(snap, 1)

    assert "t2" in sweep["unblocked"]
    assert snap["tasks"]["t2"]["status"] == "pending"
    assert sweep["cycle"] is None


def test_timeout_reap_releases_stale_claim(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    board.init_board(team_dir, [{"id": "t1", "depends_on": []}])
    assert board.claim(team_dir, "t1", "agent-a", 0)

    # One round later: not yet stale.
    snap = board.snapshot(team_dir)
    assert "t1" not in board.unblock_and_reap(snap, 1)["reaped"]

    # Two rounds later: released back to pending.
    snap = board.snapshot(team_dir)
    sweep = board.unblock_and_reap(snap, 2)
    assert "t1" in sweep["reaped"]
    assert snap["tasks"]["t1"]["status"] == "pending"
    assert snap["tasks"]["t1"]["owner"] is None


def test_detect_cycle_returns_path(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    board.init_board(
        team_dir,
        [{"id": "t1", "depends_on": ["t2"]}, {"id": "t2", "depends_on": ["t1"]}],
    )
    snap = board.snapshot(team_dir)
    cycle = board.detect_cycle(snap)
    assert cycle is not None
    assert "t1" in cycle and "t2" in cycle


# --------------------------------------------------------------------------- #
# mailbox.py
# --------------------------------------------------------------------------- #


def test_mailbox_send_and_read_new_advances_cursor(tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    mailbox.send(team_dir, "bob", "alice", "hello")

    first = mailbox.read_new(team_dir, "bob")
    assert [m["text"] for m in first] == ["hello"]
    # Cursor advanced: nothing new until another message arrives.
    assert mailbox.read_new(team_dir, "bob") == []

    mailbox.send(team_dir, "bob", "alice", "again")
    second = mailbox.read_new(team_dir, "bob")
    assert [m["text"] for m in second] == ["again"]


# --------------------------------------------------------------------------- #
# scheduler.py
# --------------------------------------------------------------------------- #


class CyclicPlannerModel:
    """Decomposes into a cyclic board; agents then idle."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if "swarm planner" in str(getattr(messages[0], "content", "")).lower():
            return AIMessage(
                content=json.dumps(
                    {
                        "tasks": [
                            {"id": "t1", "title": "a", "depends_on": ["t2"]},
                            {"id": "t2", "title": "b", "depends_on": ["t1"]},
                        ]
                    }
                )
            )
        return AIMessage(content="idle")


class CompletingModel:
    """Decomposes into two independent tasks; agents claim then complete them."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if "swarm planner" in str(getattr(messages[0], "content", "")).lower():
            return AIMessage(
                content=json.dumps(
                    {
                        "tasks": [
                            {"id": "t1", "title": "a", "depends_on": []},
                            {"id": "t2", "title": "b", "depends_on": []},
                        ]
                    }
                )
            )
        last = messages[-1]
        if isinstance(last, ToolMessage):
            data = json.loads(last.content)
            if data.get("name") == "BoardClaimTool":
                if data.get("claimed"):
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "BoardUpdateTool",
                                "args": {"task_id": data["task_id"], "status": "done", "result": "ok"},
                                "id": "u",
                            }
                        ],
                    )
                return AIMessage(content="lost the claim")
            return AIMessage(content="done")
        match = re.search(r"- (t\d+):", str(getattr(last, "content", "")))
        if match:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "BoardClaimTool", "args": {"task_id": match.group(1)}, "id": "c"}
                ],
            )
        return AIMessage(content="nothing to do")


def test_cyclic_decomposition_terminates_as_deadlock(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    state = {"runtime": runtime, "model": CyclicPlannerModel()}

    result = run_swarm(state, "cyclic-team", "goal", agents=["search-agent"], max_rounds=5)

    assert result["status"] == "deadlock"
    assert result["cycle"]
    assert result["rounds"] == 1
    assert Path(result["log_path"]).exists()


def test_swarm_completes_independent_tasks(tmp_path: Path) -> None:
    _seed_workspace_agents(tmp_path)
    runtime = create_runtime(tmp_path)
    state = {"runtime": runtime, "model": CompletingModel()}

    result = run_swarm(
        state, "happy-team", "goal", agents=["reviewer", "doc-writer"], max_rounds=4
    )

    assert result["status"] == "done"
    assert all(t["status"] == "done" for t in result["board"]["tasks"].values())


# --------------------------------------------------------------------------- #
# CLI gating
# --------------------------------------------------------------------------- #


def test_swarm_cli_requires_experimental_flag() -> None:
    from typer.testing import CliRunner

    from Linki.cli.app import swarm_app

    runner = CliRunner()
    result = runner.invoke(swarm_app, ["do it", "--team", "X", "--agents", "a,b"])

    assert result.exit_code == 2
    assert "experimental" in result.output.lower()


def test_default_cli_still_reachable_alongside_swarm() -> None:
    """Adding swarm must not break the default `linki "<task>"` flow."""
    from typer.testing import CliRunner

    from Linki.cli.app import app

    runner = CliRunner()
    # A bad provider proves the default task command still parses its arguments.
    result = runner.invoke(app, ["some task", "--provider", "bogus"])
    assert "provider must be" in result.output
