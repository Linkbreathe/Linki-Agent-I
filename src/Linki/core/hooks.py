"""Workspace hook policy loading, matching, and execution.

Hooks are configured in ``.linki/hooks.json`` inside the workspace and run as
local commands with a small JSON payload on stdin. They are policy code: a
broken JSON file fails startup, while individual hook runtime failures are
reported as trace warnings and otherwise fail open.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from Linki.core.state import RuntimeState

HOOKS_CONFIG_PATH = Path(".linki") / "hooks.json"
HOOK_TIMEOUT_SECONDS = 10.0
HOOK_OUTPUT_LIMIT = 4000

PRE_TOOL_USE = "PreToolUse"
POST_TOOL_USE = "PostToolUse"


@dataclass(frozen=True)
class HookEntry:
    matcher: str
    command: str


@dataclass(frozen=True)
class HookDecision:
    kind: Literal["allow", "deny", "ask"]
    reason: str = ""


def _workspace_path(workspace: str | Path | RuntimeState) -> Path:
    if isinstance(workspace, RuntimeState):
        return workspace.workspace
    return Path(workspace).expanduser().resolve()


def _emit_runtime_event(state: RuntimeState, event: dict[str, Any]) -> None:
    handler = getattr(state, "event_handler", None)
    if handler is not None:
        handler(dict(event))
        return

    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except (ImportError, RuntimeError, KeyError):
        return
    writer(dict(event))


def _trace_warn(state: RuntimeState, *, event: str, tool_name: str, hook: HookEntry, reason: str) -> None:
    _emit_runtime_event(
        state,
        {
            "type": "trace.warn",
            "event": event,
            "tool": tool_name,
            "hook": hook.command,
            "reason": reason,
        },
    )


def _emit_hook_decision(
    state: RuntimeState,
    *,
    event: str,
    tool_name: str,
    hook: HookEntry,
    decision: HookDecision,
) -> None:
    _emit_runtime_event(
        state,
        {
            "type": "hook_decision",
            "event": event,
            "tool": tool_name,
            "decision": decision.kind,
            "reason": decision.reason,
            "hook": hook.command,
        },
    )


def load_hooks_config(workspace: str | Path | RuntimeState) -> dict[str, list[HookEntry]]:
    """Load ``.linki/hooks.json`` from ``workspace``.

    Missing config returns an empty dict. Invalid JSON is intentionally allowed
    to propagate as ``json.JSONDecodeError`` so startup fails loudly when the
    policy file is corrupt.
    """

    path = _workspace_path(workspace) / HOOKS_CONFIG_PATH
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(".linki/hooks.json must contain an object")

    config: dict[str, list[HookEntry]] = {}
    for event, entries in raw.items():
        if not isinstance(entries, list):
            raise ValueError(f".linki/hooks.json entry for {event!r} must be a list")

        parsed_entries: list[HookEntry] = []
        for index, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(f".linki/hooks.json {event}[{index}] must be an object")
            matcher = item.get("matcher")
            command = item.get("command")
            if not isinstance(matcher, str) or not isinstance(command, str):
                raise ValueError(f".linki/hooks.json {event}[{index}] needs string matcher and command")
            parsed_entries.append(HookEntry(matcher=matcher, command=command))

        config[str(event)] = parsed_entries

    return config


def _matcher_pattern(matcher: str) -> str:
    return ".*" if matcher == "*" else matcher


def match_hooks(state: RuntimeState, event: str, tool_name: str) -> list[HookEntry]:
    """Return hooks whose matcher full-matches ``tool_name`` for ``event``."""

    config = load_hooks_config(state)
    matches: list[HookEntry] = []
    for hook in config.get(event, []):
        if re.fullmatch(_matcher_pattern(hook.matcher), tool_name):
            matches.append(hook)
    return matches


def _truncate_text(text: str, limit: int = HOOK_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _tool_result_for_hook(tool_result: dict[str, Any]) -> dict[str, Any]:
    output = tool_result.get("output")
    if output is None and tool_result.get("error"):
        output = tool_result.get("error")
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, default=str)

    return {
        "ok": bool(tool_result.get("ok")),
        "output": _truncate_text(output),
    }


def _hook_payload(
    state: RuntimeState,
    *,
    event: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "workspace": str(state.workspace),
    }
    if tool_result is not None:
        payload["tool_result"] = _tool_result_for_hook(tool_result)
    return payload


def _run_hook_process(
    state: RuntimeState,
    hook: HookEntry,
    payload: dict[str, Any],
    *,
    event: str,
    tool_name: str,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            hook.command,
            cwd=state.workspace,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=HOOK_TIMEOUT_SECONDS,
            shell=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _trace_warn(state, event=event, tool_name=tool_name, hook=hook, reason="hook timed out")
    except OSError as exc:
        _trace_warn(state, event=event, tool_name=tool_name, hook=hook, reason=f"hook failed: {exc}")
    return None


def _parse_pre_hook_stdout(stdout: str) -> HookDecision | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None

    decision = value.get("decision")
    if decision not in {"deny", "ask"}:
        return None
    return HookDecision(kind=decision, reason=str(value.get("reason") or ""))


def run_pre_tool_hooks(state: RuntimeState, tool_name: str, tool_input: dict[str, Any]) -> HookDecision:
    """Run matching ``PreToolUse`` hooks and return the effective decision."""

    effective = HookDecision(kind="allow", reason="")
    payload = _hook_payload(state, event=PRE_TOOL_USE, tool_name=tool_name, tool_input=tool_input)

    for hook in match_hooks(state, PRE_TOOL_USE, tool_name):
        completed = _run_hook_process(state, hook, payload, event=PRE_TOOL_USE, tool_name=tool_name)
        if completed is None:
            continue

        if completed.returncode == 2:
            reason = (completed.stderr or completed.stdout or "hook denied").strip()
            decision = HookDecision(kind="deny", reason=reason)
            _emit_hook_decision(state, event=PRE_TOOL_USE, tool_name=tool_name, hook=hook, decision=decision)
            return decision

        if completed.returncode != 0:
            _trace_warn(
                state,
                event=PRE_TOOL_USE,
                tool_name=tool_name,
                hook=hook,
                reason=f"hook exited {completed.returncode}: {(completed.stderr or '').strip()}",
            )
            continue

        decision = _parse_pre_hook_stdout(completed.stdout)
        if decision is None:
            decision = HookDecision(kind="allow", reason="")
        _emit_hook_decision(state, event=PRE_TOOL_USE, tool_name=tool_name, hook=hook, decision=decision)

        if decision.kind == "deny":
            return decision
        if decision.kind == "ask":
            effective = decision

    return effective


def run_post_tool_hooks(
    state: RuntimeState,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result: dict[str, Any],
) -> str:
    """Run matching ``PostToolUse`` hooks and concatenate successful stdout."""

    notes: list[str] = []
    payload = _hook_payload(
        state,
        event=POST_TOOL_USE,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_result=tool_result,
    )

    for hook in match_hooks(state, POST_TOOL_USE, tool_name):
        completed = _run_hook_process(state, hook, payload, event=POST_TOOL_USE, tool_name=tool_name)
        if completed is None:
            continue

        if completed.returncode != 0:
            _trace_warn(
                state,
                event=POST_TOOL_USE,
                tool_name=tool_name,
                hook=hook,
                reason=f"hook exited {completed.returncode}: {(completed.stderr or '').strip()}",
            )
            continue

        text = completed.stdout.strip()
        if text:
            notes.append(text)

    return "\n".join(notes)
