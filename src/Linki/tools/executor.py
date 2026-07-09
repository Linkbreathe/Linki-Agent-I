"""Single execution gate for Linki workspace tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from Linki.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    classify_command_risk,
    new_approval_request,
    normalize_approval_mode,
)
from Linki.core.hooks import run_post_tool_hooks, run_pre_tool_hooks
from Linki.core.state import RuntimeState
from Linki.tools.file_tools import protected_path_error

ToolCallable = Callable[..., Any]


def _tool_result(tool_name: str, ok: bool, *, output: Any = None, error: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"ok": ok, "name": tool_name}
    if error:
        result["error"] = error
    else:
        result["output"] = output
    return result


def _result_from_output(tool_name: str, output: Any) -> dict[str, Any]:
    ok = True
    result = _tool_result(tool_name, ok, output=output)

    if isinstance(output, Mapping):
        if isinstance(output.get("ok"), bool):
            result["ok"] = bool(output["ok"])
        for key in (
            "exit_code",
            "stdout",
            "stderr",
            "requires_approval",
            "approval_request_id",
            "risk_reason",
            "approved",
            "approval_reason",
        ):
            if key in output:
                result[key] = output[key]
        if output.get("error"):
            result["error"] = str(output["error"])

    return result


def is_tool_result(value: Any, tool_name: str) -> bool:
    """Return true when ``value`` is already an executor ToolResult."""

    return (
        isinstance(value, dict)
        and value.get("name") == tool_name
        and isinstance(value.get("ok"), bool)
        and ("output" in value or "error" in value)
    )


def _approval_command_text(tool_name: str, tool_input: Mapping[str, Any]) -> str:
    if tool_name == "BashTool":
        return str(tool_input.get("command") or "")
    return json.dumps(dict(tool_input), ensure_ascii=False, default=str)


def _approval_info(request: ApprovalRequest, decision: ApprovalDecision) -> dict[str, Any]:
    return {
        "requires_approval": True,
        "approval_request_id": request.id,
        "risk_reason": request.risk_reason,
        "approved": decision.approved,
        "approval_reason": decision.reason,
    }


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


def _emit_approval_requested(state: RuntimeState, request: ApprovalRequest) -> None:
    _emit_runtime_event(
        state,
        {
            "type": "approval_requested",
            "tool": request.tool_name,
            "reason": request.risk_reason,
            "command": request.command,
            "label": request.label,
            "approval_request_id": request.id,
        },
    )


def _emit_approval_decision(
    state: RuntimeState,
    request: ApprovalRequest,
    decision: ApprovalDecision,
) -> None:
    _emit_runtime_event(
        state,
        {
            "type": "approval_decision",
            "tool": request.tool_name,
            "approval_request_id": request.id,
            "approved": decision.approved,
            "reason": decision.reason,
            "label": request.label,
        },
    )


def _request_approval(
    state: RuntimeState,
    *,
    tool_name: str,
    tool_input: Mapping[str, Any],
    risk_reason: str,
    force_inline: bool = False,
) -> tuple[bool, dict[str, Any]]:
    mode = normalize_approval_mode(getattr(state, "approval_mode", None))

    request = new_approval_request(
        _approval_command_text(tool_name, tool_input),
        risk_reason,
        tool_name=tool_name,
    )
    label = str(getattr(state, "approval_label", "") or "")
    if label:
        request = replace(request, label=label)
    _emit_approval_requested(state, request)

    if mode == "deny":
        decision = ApprovalDecision(approved=False, reason="approval mode denied")
        _emit_approval_decision(state, request, decision)
        return False, _approval_info(request, decision)

    if mode == "auto" and not force_inline:
        decision = ApprovalDecision(approved=True, reason="approval mode auto")
        _emit_approval_decision(state, request, decision)
        return True, _approval_info(request, decision)

    handler = getattr(state, "approval_handler", None)
    if handler is None:
        decision = ApprovalDecision(approved=False, reason="approval handler unavailable")
        _emit_approval_decision(state, request, decision)
        return False, _approval_info(request, decision)

    decision = handler(request)
    _emit_approval_decision(state, request, decision)
    return bool(decision.approved), _approval_info(request, decision)


def _risk_reason(tool_name: str, tool_input: Mapping[str, Any]) -> str | None:
    if tool_name != "BashTool":
        return None
    return classify_command_risk(str(tool_input.get("command") or ""))


def _protected_path_result(state: RuntimeState, tool_name: str, tool_input: Mapping[str, Any]) -> dict[str, Any] | None:
    if tool_name not in {"FileWriteTool", "FileEditTool"}:
        return None

    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str):
        return None

    error = protected_path_error(state, file_path)
    if error:
        return _tool_result(tool_name, False, error=error)
    return None


def _merge_note(result: dict[str, Any], note: str) -> dict[str, Any]:
    if not note:
        return result
    existing = str(result.get("note") or "").strip()
    result["note"] = f"{existing}\n{note}".strip() if existing else note
    return result


def execute_tool(
    state: RuntimeState,
    tool_name: str,
    tool_input: dict[str, Any],
    action: ToolCallable,
) -> dict[str, Any]:
    """Run a workspace tool through hooks, risk classification, and approval."""

    protected = _protected_path_result(state, tool_name, tool_input)
    if protected is not None:
        return protected

    hook_decision = run_pre_tool_hooks(state, tool_name, tool_input)
    if hook_decision.kind == "deny":
        return _tool_result(tool_name, False, error=f"[hook denied] {hook_decision.reason}".strip())

    approval_reason = ""
    if hook_decision.kind == "ask":
        approval_reason = f"escalated by hook: {hook_decision.reason}"

    risk_reason = _risk_reason(tool_name, tool_input)
    if risk_reason:
        approval_reason = f"{approval_reason}; {risk_reason}" if approval_reason else risk_reason

    approval = {}
    if approval_reason:
        allowed, approval = _request_approval(
            state,
            tool_name=tool_name,
            tool_input=tool_input,
            risk_reason=approval_reason,
            force_inline=hook_decision.kind == "ask",
        )
        if not allowed:
            result = _tool_result(tool_name, False, error=f"approval denied: {approval.get('approval_reason', '')}".strip())
            result.update(approval)
            return result

    try:
        output = action(**tool_input)
        result = _result_from_output(tool_name, output)
    except Exception as exc:
        result = _tool_result(tool_name, False, error=str(exc))
        result["error_type"] = type(exc).__name__

    if approval:
        result.update(approval)

    note = run_post_tool_hooks(state, tool_name, tool_input, result)
    return _merge_note(result, note)
