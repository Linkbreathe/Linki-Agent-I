"""Round-based scheduler for the swarm lab.

Reuses the stage-nine ``run_subagent`` runtime to wake each agent in turn. Each
round: every agent is woken once with a board snapshot + its new mail and the
board/mailbox tools on top of its own allowlist. At round end the board is swept
(unblock dependencies, reap stale claims, detect cycles). The swarm ends when all
tasks are done or a dependency deadlock is detected. A per-round log is written
to ``log.md``.

Independent of ``Linki.graph`` by design — task decomposition uses the model
directly rather than the planner graph node.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from Linki.agents.registry import load_agent_registry
from Linki.providers.openai_provider import create_model
from Linki.swarm import board as board_mod
from Linki.swarm import mailbox as mailbox_mod
from Linki.tools.agent_tool import run_subagent

DEFAULT_MAX_ROUNDS = 8

DECOMPOSE_PROMPT = """You are the swarm planner. Break the user's task into 3 to 6 small
board tasks with dependencies. Respond with ONLY JSON of the form:
{"tasks": [{"id": "t1", "title": "...", "depends_on": []}, ...]}
Use short ids like t1, t2. depends_on lists ids that must finish first. Keep the
dependency graph acyclic unless the task genuinely requires a cycle."""


def _runtime(state: Any):
    if isinstance(state, Mapping):
        return state.get("runtime")
    return getattr(state, "runtime", None) or state


def _model(state: Any) -> Any:
    values = state if isinstance(state, Mapping) else {}
    if values.get("model") is not None:
        return values["model"]
    return create_model(provider=values.get("provider", "openai"), model=values.get("model_name"))


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def team_dir_for(runtime: Any, team: str) -> Path:
    return runtime.workspace / ".linki" / "swarm" / team


def decompose_task(state: Any, task: str) -> list[dict[str, Any]]:
    """Ask the model to split ``task`` into 3-6 dependency-linked board tasks."""

    model = _model(state)
    response = model.invoke(
        [SystemMessage(content=DECOMPOSE_PROMPT), HumanMessage(content=task)]
    )
    payload = _json_from_text(_message_text(response))
    tasks = payload.get("tasks")
    return tasks if isinstance(tasks, list) else []


class _BoardClaimInput(BaseModel):
    task_id: str = Field(description="Board task id to claim.")


class _BoardUpdateInput(BaseModel):
    task_id: str = Field(description="Board task id to update.")
    status: str = Field(description="New status, typically 'done'.")
    result: str = Field(default="", description="Short result note.")


class _SendMessageInput(BaseModel):
    to: str = Field(description="Teammate agent name to message.")
    text: str = Field(description="Message body.")


def _swarm_tools(team_dir: Path, agent: str, rnd: int) -> list[StructuredTool]:
    """Board + mailbox tools bound to one agent for the current round."""

    def board_claim_tool(task_id: str) -> dict[str, Any]:
        claimed = board_mod.claim(team_dir, task_id, agent, rnd)
        return {"ok": True, "name": "BoardClaimTool", "task_id": task_id, "claimed": claimed}

    def board_update_tool(task_id: str, status: str, result: str = "") -> dict[str, Any]:
        try:
            task = board_mod.update(team_dir, task_id, status=status, result=result, agent=agent)
        except KeyError as exc:
            return {"ok": False, "name": "BoardUpdateTool", "error": str(exc)}
        return {"ok": True, "name": "BoardUpdateTool", "task": task}

    def send_message_tool(to: str, text: str) -> dict[str, Any]:
        mailbox_mod.send(team_dir, to, agent, text)
        return {"ok": True, "name": "SendMessageTool", "to": to}

    return [
        StructuredTool.from_function(
            func=board_claim_tool,
            name="BoardClaimTool",
            description="Claim a board task by id. Returns claimed=true only if you won it.",
            args_schema=_BoardClaimInput,
        ),
        StructuredTool.from_function(
            func=board_update_tool,
            name="BoardUpdateTool",
            description="Update a board task you own, e.g. mark it done with a result.",
            args_schema=_BoardUpdateInput,
        ),
        StructuredTool.from_function(
            func=send_message_tool,
            name="SendMessageTool",
            description="Send a short message to a teammate's mailbox.",
            args_schema=_SendMessageInput,
        ),
    ]


def _agent_prompt(task: str, board: Mapping[str, Any], inbox: list[dict]) -> str:
    claimable = [
        f"- {tid}: {t['title']} (deps: {t['depends_on'] or 'none'})"
        for tid, t in board["tasks"].items()
        if t["status"] == "pending"
    ]
    mail = "\n".join(f"- from {m['from']}: {m['text']}" for m in inbox) or "- (no new mail)"
    board_json = json.dumps(board["tasks"], ensure_ascii=False, indent=2)
    return "\n\n".join(
        [
            f"Overall goal:\n{task}",
            f"Board:\n{board_json}",
            "Claimable tasks:\n" + ("\n".join(claimable) if claimable else "- (none right now)"),
            f"New mail:\n{mail}",
            (
                "Claim ONE claimable task with BoardClaimTool. If you win it, do the "
                "work, then call BoardUpdateTool to mark it 'done' with a short result. "
                "Message a teammate with SendMessageTool only if they need to know "
                "something. If nothing is claimable, reply briefly and stop."
            ),
        ]
    )


def run_swarm(
    state: Any,
    team: str,
    task: str,
    *,
    agents: list[str],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> dict[str, Any]:
    """Run the turn-based swarm until all tasks are done or a deadlock is found."""

    runtime = _runtime(state)
    team_dir = team_dir_for(runtime, team)
    registry = load_agent_registry(runtime)

    tasks = decompose_task(state, task)
    board_mod.init_board(team_dir, tasks)

    log: list[str] = [f"# Swarm log — team {team}", "", f"Goal: {task}", ""]
    status = "max_rounds"
    cycle: list[str] | None = None
    last_round = 0

    for rnd in range(1, max_rounds + 1):
        last_round = rnd
        log.append(f"## Round {rnd}")
        for agent in agents:
            spec = registry.get(agent)
            if spec is None:
                log.append(f"- {agent}: not a registered agent, skipped")
                continue
            snap = board_mod.snapshot(team_dir)
            inbox = mailbox_mod.read_new(team_dir, agent)
            summary = run_subagent(
                state,
                spec,
                _agent_prompt(task, snap, inbox),
                description=f"swarm {agent} r{rnd}",
                extra_tools=_swarm_tools(team_dir, agent, rnd),
            )
            log.append(f"- {agent}: {str(summary)[:160]}")

        board_snap = board_mod.snapshot(team_dir)
        sweep = board_mod.unblock_and_reap(board_snap, rnd)
        board_mod.save_board(team_dir, board_snap)
        log.append(
            f"- round-end: unblocked={sweep['unblocked']} reaped={sweep['reaped']} "
            f"cycle={sweep['cycle']}"
        )

        if sweep["cycle"]:
            status = "deadlock"
            cycle = sweep["cycle"]
            break
        if board_mod.all_done(board_snap):
            status = "done"
            break

    log_path = team_dir / "log.md"
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")

    return {
        "status": status,
        "rounds": last_round,
        "cycle": cycle,
        "log_path": str(log_path),
        "board": board_mod.snapshot(team_dir),
    }
