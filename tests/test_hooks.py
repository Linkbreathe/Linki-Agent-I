"""Tests for workspace hook policy execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.messages import ToolMessage

from Linki.core.approval import ApprovalDecision, ApprovalRequest
from Linki.core.hooks import match_hooks, load_hooks_config
from Linki.core.agent import stream_session_events
from Linki.core.session import create_run_workspace, seed_policy_files
from Linki.core.state import create_runtime
from Linki.graph.nodes import _execute_call
from Linki.tools.registry import build_tools


def _write_hook(path: Path, body: str) -> str:
    hooks_dir = path / ".linki" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    script = hooks_dir / "hook.py"
    script.write_text(body, encoding="utf-8")
    return f"{sys.executable} .linki/hooks/hook.py"


def _write_config(path: Path, event: str, matcher: str, command: str) -> None:
    config_path = path / ".linki" / "hooks.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({event: [{"matcher": matcher, "command": command}]}),
        encoding="utf-8",
    )


def _tools(runtime):
    return {tool.name: tool for tool in build_tools(runtime)}


def test_pre_hook_exit_2_blocks_tool_and_tool_message_has_reason(tmp_path: Path) -> None:
    command = _write_hook(
        tmp_path,
        "from __future__ import annotations\n"
        "import sys\n"
        "print('blocked by policy', file=sys.stderr)\n"
        "raise SystemExit(2)\n",
    )
    _write_config(tmp_path, "PreToolUse", "FileWriteTool", command)
    runtime = create_runtime(tmp_path)

    result = _execute_call(
        {
            "id": "call-1",
            "name": "FileWriteTool",
            "args": {"file_path": "blocked.txt", "content": "nope"},
        },
        _tools(runtime),
    )
    message = ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id="call-1")

    assert result["ok"] is False
    assert "[hook denied] blocked by policy" in message.content
    assert not (tmp_path / "blocked.txt").exists()


def test_pre_hook_ask_forces_approval(tmp_path: Path) -> None:
    requests: list[ApprovalRequest] = []
    command = _write_hook(
        tmp_path,
        "from __future__ import annotations\n"
        "import json\n"
        "print(json.dumps({'decision': 'ask', 'reason': 'needs review'}))\n",
    )
    _write_config(tmp_path, "PreToolUse", "BashTool", command)

    def handler(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision(approved=True, reason="approved in test")

    runtime = create_runtime(tmp_path, approval_mode="inline", approval_handler=handler)
    result = _tools(runtime)["BashTool"].invoke({"command": "echo approved", "timeout_seconds": 5})

    assert result["ok"] is True
    assert requests
    assert requests[0].tool_name == "BashTool"
    assert requests[0].risk_reason == "escalated by hook: needs review"
    assert result["approved"] is True


def test_pre_hook_timeout_fails_open_and_records_trace_warn(tmp_path: Path, monkeypatch) -> None:
    from Linki.core import hooks

    events: list[dict] = []
    command = _write_hook(
        tmp_path,
        "from __future__ import annotations\n"
        "import time\n"
        "time.sleep(1)\n",
    )
    _write_config(tmp_path, "PreToolUse", "FileWriteTool", command)
    monkeypatch.setattr(hooks, "HOOK_TIMEOUT_SECONDS", 0.05)

    runtime = create_runtime(tmp_path, event_handler=events.append)
    result = _tools(runtime)["FileWriteTool"].invoke({"file_path": "ok.txt", "content": "written"})

    assert result["ok"] is True
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "written"
    assert any(event.get("type") == "trace.warn" and event.get("reason") == "hook timed out" for event in events)


def test_post_hook_stdout_is_returned_as_note(tmp_path: Path) -> None:
    command = _write_hook(
        tmp_path,
        "from __future__ import annotations\n"
        "import json\n"
        "import sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert payload['tool_result']['ok'] is True\n"
        "print('post hook note')\n",
    )
    _write_config(tmp_path, "PostToolUse", "FileWriteTool", command)
    runtime = create_runtime(tmp_path)

    result = _tools(runtime)["FileWriteTool"].invoke({"file_path": "noted.txt", "content": "written"})

    assert result["ok"] is True
    assert result["note"] == "post hook note"


def test_agent_cannot_write_hooks_config(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    result = _tools(runtime)["FileWriteTool"].invoke(
        {"file_path": ".linki/hooks.json", "content": "{}"}
    )

    assert result["ok"] is False
    assert result["error"] == "protected path: policy files are read-only to the agent"
    assert not (tmp_path / ".linki" / "hooks.json").exists()

    script_result = _tools(runtime)["FileWriteTool"].invoke(
        {"file_path": ".linki/hooks/new_policy.py", "content": "print('x')"}
    )
    assert script_result["ok"] is False
    assert script_result["error"] == "protected path: policy files are read-only to the agent"
    assert not (tmp_path / ".linki" / "hooks" / "new_policy.py").exists()


def test_invalid_hooks_json_raises(tmp_path: Path) -> None:
    config_path = tmp_path / ".linki" / "hooks.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{bad json", encoding="utf-8")

    try:
        load_hooks_config(tmp_path)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("invalid hooks JSON should raise")


def test_invalid_hooks_json_aborts_session_startup(tmp_path: Path) -> None:
    config_path = tmp_path / ".linki" / "hooks.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{bad json", encoding="utf-8")

    try:
        next(stream_session_events("hello", session_workspace=tmp_path, model=object()))
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("invalid hooks JSON should abort session startup")


def test_matcher_star_matches_any_tool_and_regex_is_fullmatch(tmp_path: Path) -> None:
    config_path = tmp_path / ".linki" / "hooks.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "PreToolUse": [
                    {"matcher": "*", "command": "echo any"},
                    {"matcher": "File.*", "command": "echo file"},
                    {"matcher": "Write", "command": "echo partial"},
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = create_runtime(tmp_path)

    hooks = match_hooks(runtime, "PreToolUse", "FileWriteTool")

    assert [hook.command for hook in hooks] == ["echo any", "echo file"]


def test_run_workspace_seeds_policy_and_hook_intercepts(tmp_path: Path) -> None:
    """Reproduce the real session layout: policy lives at the project root while
    tools execute inside an ephemeral ``run-<timestamp>`` workspace. Without
    seeding, ``load_hooks_config`` reads an empty config and hooks never fire.
    """

    project = tmp_path / "project"
    (project / ".linki" / "hooks").mkdir(parents=True)
    (project / ".linki" / "hooks" / "hook.py").write_text(
        "from __future__ import annotations\n"
        "import sys\n"
        "print('blocked by seeded policy', file=sys.stderr)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    (project / ".linki" / "hooks.json").write_text(
        json.dumps(
            {
                "PreToolUse": [
                    {
                        "matcher": "FileWriteTool",
                        "command": f"{sys.executable} .linki/hooks/hook.py",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_workspace = create_run_workspace(project / "workspace", policy_source=project)

    # Policy files must have been copied into the ephemeral run workspace.
    assert (run_workspace / ".linki" / "hooks.json").is_file()
    assert (run_workspace / ".linki" / "hooks" / "hook.py").is_file()
    assert run_workspace.name.startswith("run-")

    runtime = create_runtime(run_workspace)
    result = _tools(runtime)["FileWriteTool"].invoke(
        {"file_path": "blocked.txt", "content": "nope"}
    )

    assert result["ok"] is False
    assert "[hook denied] blocked by seeded policy" in result["error"]
    assert not (run_workspace / "blocked.txt").exists()


def test_seed_policy_files_does_not_overwrite_existing(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / ".linki").mkdir(parents=True)
    (source / ".linki" / "hooks.json").write_text('{"PreToolUse": []}', encoding="utf-8")

    workspace = tmp_path / "ws"
    (workspace / ".linki").mkdir(parents=True)
    existing = workspace / ".linki" / "hooks.json"
    existing.write_text('{"kept": true}', encoding="utf-8")

    seeded = seed_policy_files(workspace, source)

    assert seeded == []
    assert existing.read_text(encoding="utf-8") == '{"kept": true}'
