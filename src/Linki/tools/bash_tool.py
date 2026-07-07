import subprocess
import shlex
import re
from collections.abc import Callable

from Linki.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    classify_command_risk,
    new_approval_request,
    normalize_approval_mode,
)
from Linki.core.paths import ensure_workspace
from Linki.core.state import RuntimeState


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _validate_workspace_command(command: str) -> None:
    if "\x00" in command:
        raise ValueError("command must not contain null bytes")
    if re.search(r"(^|[\s'\"(<>=])/(?!/)", command):
        raise PermissionError("Command contains an absolute path")
    if re.search(r"(^|[\s'\"(<>=])~", command):
        raise PermissionError("Command contains a home-directory path")
    if re.search(r"(^|[\s'\"(<>=])\.\.(?:/|$)", command) or "/../" in command:
        raise PermissionError("Command contains parent-directory traversal")

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Invalid shell command: {exc}") from exc

    for token in tokens:
        values = [token]
        if "=" in token:
            values.append(token.split("=", 1)[1])
        if token[:1] in {">", "<"}:
            values.append(token[1:])

        for value in values:
            if (
                value.startswith("/")
                or value.startswith("~")
                or value == ".."
                or value.startswith("../")
                or "/../" in value
                or value.endswith("/..")
                or "</" in value
                or ">/" in value
                or "=/" in value
            ):
                raise PermissionError("Command contains a path outside the workspace")


class BashTool:
    def __init__(
        self,
        state: RuntimeState,
        *,
        approval_mode: str | None = None,
        approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
    ) -> None:
        self.state = state
        runtime_approval_mode = getattr(state, "approval_mode", None)
        runtime_approval_handler = getattr(state, "approval_handler", None)
        self.approval_mode = normalize_approval_mode(
            approval_mode if approval_mode is not None else runtime_approval_mode
        )
        self.approval_handler = approval_handler or runtime_approval_handler

    def __call__(self, command: str, timeout_seconds: int = 30) -> dict:
        return self.run_bash(command, timeout_seconds=timeout_seconds)

    def run_bash(self, command: str, timeout_seconds: int = 30) -> dict:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        risk_reason = classify_command_risk(command)
        if risk_reason is None:
            return self._execute(command, timeout_seconds)

        if self.approval_mode == "deny":
            return {
                "ok": False,
                "requires_approval": True,
                "risk_reason": risk_reason,
            }

        if self.approval_mode == "auto":
            result = self._execute(command, timeout_seconds)
            result["requires_approval"] = True
            return result

        return self._run_with_inline_approval(command, timeout_seconds, risk_reason)

    def _run_with_inline_approval(self, command: str, timeout_seconds: int, risk_reason: str) -> dict:
        if self.approval_handler is None:
            raise RuntimeError("Inline approval mode requires an approval_handler to be configured")

        request = new_approval_request(command, risk_reason)
        decision = self.approval_handler(request)

        approval_info = {
            "requires_approval": True,
            "approval_request_id": request.id,
            "risk_reason": request.risk_reason,
            "approved": decision.approved,
            "approval_reason": decision.reason,
        }

        if not decision.approved:
            return {"ok": False, **approval_info}

        result = self._execute(command, timeout_seconds)
        result.update(approval_info)
        return result

    def _execute(self, command: str, timeout_seconds: int) -> dict:
        workspace = ensure_workspace(self.state)
        _validate_workspace_command(command)
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_timeout_output(exc.stdout)
            stderr = _decode_timeout_output(exc.stderr)
            output = "\n".join(part for part in [stdout, stderr] if part)
            raise TimeoutError(f"Command timed out after {timeout_seconds}s\n{output}") from exc

        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
