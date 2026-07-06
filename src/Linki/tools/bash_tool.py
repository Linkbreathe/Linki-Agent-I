import subprocess
import shlex
import re

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
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, command: str, timeout_seconds: int = 30) -> str:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

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

        output_parts = []
        if completed.stdout:
            output_parts.append(completed.stdout.rstrip())
        if completed.stderr:
            output_parts.append(completed.stderr.rstrip())
        output_parts.append(f"exit_code={completed.returncode}")
        return "\n".join(output_parts)
