import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import typer

from Linki.core.approval import KIND_PLAN, KIND_QUESTION, ApprovalDecision, ApprovalRequest
from Linki.core.agent import _parse_graph_event, stream_agent_events, stream_session_events
from Linki.core.session import create_run_workspace

app = typer.Typer(no_args_is_help=True)
console = Console()

STATUS_ICONS = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "blocked": "🚫"}
TEXT_HEAVY_ARG_KEYS = {"content", "old_text", "new_text"}


def _json_block(value: object) -> Syntax:
    return Syntax(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        "json",
        word_wrap=True,
    )


def _truncate(text: str, limit: int = 200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _truncate_lines(text: str, *, line_limit: int = 10, char_limit: int = 1_200) -> str:
    text = text.strip()
    if not text:
        return "—"

    lines = text.splitlines()
    rendered = "\n".join(lines[:line_limit])
    omitted = max(len(lines) - line_limit, 0)
    rendered = _truncate(rendered, char_limit)
    if omitted:
        rendered = f"{rendered}\n… {omitted} more lines"
    return rendered


def _compact_json(value: Any, *, limit: int = 600) -> str:
    return _truncate(json.dumps(value, ensure_ascii=False, default=str), limit)


def _parse_json_mapping(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, str):
        return value if isinstance(value, Mapping) else None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _candidate_payloads(result: Any) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        return []

    payloads: list[Mapping[str, Any]] = [result]
    output = result.get("output")
    parsed_output = _parse_json_mapping(output)
    if parsed_output is not None:
        payloads.append(parsed_output)
    return payloads


def _tool_result_ok(result: Any) -> bool:
    payloads = _candidate_payloads(result)
    return not any(payload.get("ok") is False for payload in payloads)


def _compact_value(key: str, value: Any, *, limit: int = 180) -> str:
    if isinstance(value, str):
        if key in TEXT_HEAVY_ARG_KEYS:
            line_count = len(value.splitlines()) or 1
            return f"{len(value):,} chars across {line_count:,} lines"
        return _truncate_lines(value, line_limit=3, char_limit=limit)

    if isinstance(value, Mapping) or isinstance(value, list):
        return _compact_json(value, limit=limit)

    return _truncate(str(value), limit)


def _tool_args_table(args: Any) -> Table | str:
    if not args:
        return "No arguments."
    if not isinstance(args, Mapping):
        return _compact_value("args", args)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    for key, value in args.items():
        table.add_row(str(key), _compact_value(str(key), value))
    return table


def _tool_result_table(result: Any) -> Table | str:
    if not isinstance(result, Mapping):
        return _compact_value("result", result, limit=1_000)

    payloads = _candidate_payloads(result)
    effective = payloads[-1] if payloads else result
    ok = _tool_result_ok(result)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")

    status = "ok" if ok else "failed"
    if effective.get("exit_code") is not None:
        status = f"{status} (exit {effective.get('exit_code')})"
    table.add_row("Status", status)

    for payload in payloads:
        if payload.get("requires_approval"):
            approved = payload.get("approved")
            if approved is None:
                approval = "required"
            else:
                approval = "approved" if approved else "denied"
            table.add_row("Approval", approval)
            if payload.get("risk_reason"):
                table.add_row("Risk", _truncate(str(payload["risk_reason"]), 220))

    for payload in payloads:
        if payload.get("error"):
            error_type = payload.get("error_type")
            prefix = f"{error_type}: " if error_type else ""
            table.add_row("Error", _truncate_lines(prefix + str(payload["error"]), line_limit=5, char_limit=800))
            return table

    stdout = effective.get("stdout")
    stderr = effective.get("stderr")
    if stdout:
        table.add_row("stdout", _truncate_lines(str(stdout), line_limit=8, char_limit=1_000))
    if stderr:
        table.add_row("stderr", _truncate_lines(str(stderr), line_limit=8, char_limit=1_000))

    if stdout or stderr:
        return table

    output = result.get("output")
    parsed_output = _parse_json_mapping(output)
    if parsed_output is not None:
        output = parsed_output

    if isinstance(output, Mapping):
        updated = output.get("updated")
        if isinstance(updated, Mapping):
            table.add_row("Todo", _truncate(str(updated.get("content") or updated.get("id") or ""), 300))
            table.add_row("Todo status", str(updated.get("status") or ""))
            if updated.get("note"):
                table.add_row("Note", _truncate_lines(str(updated["note"]), line_limit=4, char_limit=500))
        else:
            table.add_row("Output", _compact_json(output, limit=900))
    elif output is not None:
        table.add_row("Output", _compact_value("output", output, limit=900))

    return table


def _search_results_renderable(result: Any) -> Group | Syntax | str:
    if not isinstance(result, Mapping):
        return _json_block(result)

    renderables: list[Any] = []
    if result.get("answer"):
        renderables.append(_truncate_lines(str(result["answer"]), line_limit=5, char_limit=700))

    if result.get("ok") is False:
        renderables.append(_compact_json(result, limit=1_000))
        return Group(*renderables)

    rows = result.get("results") or []
    if rows:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Title", overflow="fold")
        table.add_column("URL", overflow="fold")
        table.add_column("Score", justify="right", no_wrap=True)
        for index, item in enumerate(rows[:5], start=1):
            row = item if isinstance(item, Mapping) else {}
            score = row.get("score")
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else ""
            table.add_row(
                str(index),
                _truncate(str(row.get("title") or "Untitled"), 120),
                _truncate(str(row.get("url") or ""), 160),
                score_text,
            )
        renderables.append(table)
        if len(rows) > 5:
            renderables.append(f"… {len(rows) - 5} more results")
    else:
        renderables.append("No results.")

    return Group(*renderables)


def _memory_table(memory: dict) -> Table:
    working = memory.get("working_memory") or {}
    history = memory.get("history_summary_store") or {}

    todos = working.get("todos") or []
    status_counts: dict[str, int] = {}
    for todo in todos:
        status = todo.get("status", "pending") if isinstance(todo, dict) else "pending"
        status_counts[status] = status_counts.get(status, 0) + 1
    todos_summary = (
        ", ".join(f"{status_counts[status]} {status}" for status in STATUS_ICONS if status_counts.get(status))
        or "none"
    )

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")

    table.add_row("Plan", _truncate(str(working.get("plan_summary") or "—")))
    table.add_row("Todos", f"{len(todos)} ({todos_summary})")
    table.add_row("Acceptance criteria", str(len(working.get("acceptance_criteria") or [])))
    table.add_row("Verification commands", str(len(working.get("verification_commands") or [])))
    table.add_row("Sources", str(len(working.get("sources") or [])))
    table.add_row("Agent handoffs", str(len(working.get("agent_handoffs") or [])))
    table.add_row("Attempts", f"{working.get('attempts', 0)}/{working.get('max_attempts', 0)}")
    if working.get("last_error"):
        table.add_row("Last error", _truncate(str(working["last_error"])))
    table.add_row("Notepad", "present" if history.get("notepad_exists") else "empty")
    table.add_row("History summary", "present" if history.get("history_exists") else "empty")
    table.add_row("Compression events", str(len(history.get("compression_events") or [])))

    return table


def _todos_table(todos: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Todo", overflow="fold")
    table.add_column("Note", overflow="fold")
    for todo in todos:
        status = str(todo.get("status") or "pending")
        table.add_row(
            f"{STATUS_ICONS.get(status, '•')} {status}",
            str(todo.get("content") or ""),
            _truncate_lines(str(todo.get("note") or ""), line_limit=3, char_limit=400),
        )
    return table


def _checks_table(checks: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("", no_wrap=True)
    table.add_column("Check", overflow="fold")
    table.add_column("Detail", overflow="fold")
    for check in checks:
        table.add_row(
            "✅" if check.get("passed") else "❌",
            str(check.get("name") or ""),
            _truncate_lines(str(check.get("detail") or ""), line_limit=6, char_limit=900),
        )
    return table


def _cli_question_handler(request: ApprovalRequest) -> ApprovalDecision:
    renderable: Any = request.question
    if request.options:
        options_text = "\n".join(f"  {index}. {option}" for index, option in enumerate(request.options, start=1))
        renderable = Group(request.question, "", options_text)
    console.print(Panel(renderable, title="❓ Clarifying question", border_style="cyan"))

    answer = typer.prompt("Your answer", default="").strip()
    # A bare option number resolves to that option's text.
    if request.options and answer.isdigit():
        index = int(answer) - 1
        if 0 <= index < len(request.options):
            answer = request.options[index]

    return ApprovalDecision(approved=True, reason="answered via CLI", answer=answer)


def _cli_plan_handler(request: ApprovalRequest) -> ApprovalDecision:
    console.print(Panel(request.plan_text or "—", title="📋 Plan review", border_style="blue"))
    approved = typer.confirm("Approve this plan?", default=True)
    feedback = "" if approved else typer.prompt("Feedback for revision", default="").strip()
    return ApprovalDecision(
        approved=approved,
        reason="plan approved via CLI" if approved else "plan rejected via CLI",
        answer=feedback,
    )


def _cli_approval_handler(request: ApprovalRequest) -> ApprovalDecision:
    if request.kind == KIND_QUESTION:
        return _cli_question_handler(request)
    if request.kind == KIND_PLAN:
        return _cli_plan_handler(request)

    approved = typer.confirm(
        f"{request.tool_name} requires approval: {request.risk_reason}\nCommand: {request.command}",
        default=False,
    )
    return ApprovalDecision(
        approved=approved,
        reason="approved via CLI" if approved else "denied via CLI",
    )


def _print_event(event: dict, *, verbose: bool = False) -> None:
    event_type = event.get("type")

    if event_type == "custom_event":
        inner = event.get("event")
        if isinstance(inner, dict):
            _print_event(inner, verbose=verbose)
        return

    if event_type == "graph_event":
        for parsed in _parse_graph_event(("updates", event.get("event", {}))):
            _print_event(parsed, verbose=verbose)
        return

    if event_type == "trace_finished":
        trace = event.get("trace") or {}
        if trace:
            console.print(
                Panel(
                    f"Trace: {trace.get('trace_id')}\nStatus: {trace.get('status')}\nPath: {trace.get('path')}",
                    title="Trace Finished",
                    border_style="grey50",
                )
            )
        return

    if event_type == "interrupted":
        console.print(
            Panel(
                f"Workspace: {event.get('workspace')}\nResume: {event.get('resume_command')}",
                title="Interrupted",
                border_style="yellow",
            )
        )
        return

    if event_type == "checkpoint_resumed":
        console.print(
            Panel(
                f"Status: {event.get('status')}\nLatest node: {event.get('latest_node')}\nRestored: {event.get('restored')}",
                title="Checkpoint Resumed",
                border_style="yellow",
            )
        )
        return

    if event_type == "checkpoint_saved":
        if verbose:
            console.print(
                Panel(
                    f"Status: {event.get('status')}\nLatest node: {event.get('latest_node')}\nPath: {event.get('path')}",
                    title="Checkpoint Saved",
                    border_style="grey50",
                )
            )
        return

    if event_type == "session_saved":
        if verbose:
            console.print(
                Panel(
                    f"Session: {event.get('session_id')}\nTurn: {event.get('turn_index')}\nPath: {event.get('path')}",
                    title="Session Saved",
                    border_style="grey50",
                )
            )
        return

    if event_type == "intent_route":
        console.print(
            Panel(
                f"Route: {event.get('route')}\nConfidence: {event.get('confidence')}\nReason: {event.get('reason')}",
                title="🧭 Intent Route",
                border_style="cyan",
            )
        )
        return

    if event_type == "node_update":
        node = event.get("node")
        data = event.get("data", {})
        content = str(event.get("content", ""))

        if node == "intent_router":
            console.print(
                Panel(
                    f"Route: {data.get('intent_route')}\nConfidence: {data.get('intent_confidence')}\nReason: {data.get('intent_reason')}",
                    title="🧭 Intent Router",
                    border_style="cyan",
                )
            )
            return

        if node == "chat_responder":
            console.print(Panel(content or _json_block(data), title="💬 Chat", border_style="green"))
            return

        if node == "planner":
            renderables: list[Any] = [_todos_table(data["todos"])]
            if data.get("code_agent_summary"):
                renderables.append(f"Summary: {_truncate_lines(str(data['code_agent_summary']), line_limit=4, char_limit=700)}")
            footer = []
            if data.get("acceptance_criteria"):
                footer.append(f"Acceptance criteria: {len(data['acceptance_criteria'])}")
            if data.get("verification_commands"):
                footer.append(f"Verification commands: {len(data['verification_commands'])}")
            if footer:
                renderables.append(" | ".join(footer))
            console.print(
                Panel(
                    Group(*renderables),
                    title="📋 Planner",
                    subtitle=data["plan_summary"],
                    border_style="blue",
                )
            )
            return

        if node == "verifier":
            passed = bool(data["passed"])
            title = "✅ Verifier" if passed else "❌ Verifier"
            border_style = "green" if passed else "red"
            console.print(Panel(_checks_table(data["verification_checks"]), title=title, border_style=border_style))
            return

        if node == "final":
            console.print(Panel(content or _json_block(data), title="📝 Final", border_style="magenta"))
            return

        if node == "context_monitor":
            token_count = int(data.get("context_token_count", 0))
            should_compress = bool(data.get("context_should_compress"))
            next_node = data.get("context_next_node", "")
            status = "⚠️  compression needed" if should_compress else "✅ within budget"
            console.print(
                Panel(
                    f"Tokens: {token_count:,}\nStatus: {status}\nNext: {next_node}",
                    title="📈 Context Monitor",
                    border_style="yellow" if should_compress else "blue",
                )
            )
            return

        if node == "context_compressor":
            events = data.get("compression_events") or []
            last_event = events[-1] if events else {}
            before = int(last_event.get("token_count", 0))
            after = int(data.get("context_token_count", 0))
            summary = _truncate(str(last_event.get("summary", "")), 400)
            console.print(
                Panel(
                    f"Tokens: {before:,} → {after:,}\nSummary: {summary or '—'}",
                    title="🗜️  Context Compressor",
                    border_style="magenta",
                )
            )
            return

        console.print(Panel(_json_block(data), title=f"Node: {node}", border_style="white"))
        return

    if event_type == "ai_message":
        if not verbose:
            return
        content = str(event.get("content") or "").strip()
        if content:
            node = event.get("node")
            title = f"AI Message: {node}" if node else "AI Message"
            console.print(
                Panel(
                    _truncate_lines(content, line_limit=8, char_limit=1_000),
                    title=title,
                    border_style="grey50",
                )
            )
        return

    if event_type == "memory_snapshot":
        if not verbose:
            return
        memory = event.get("memory") or {}
        node = event.get("node", "")
        console.print(
            Panel(
                _memory_table(memory),
                title=f"🧠 Memory Snapshot: {node}",
                border_style="grey50",
            )
        )
        return

    if event_type == "handoff":
        console.print(
            Panel(
                str(event.get("instruction", "")),
                title=f"🤝 Handoff: {event.get('from')} → {event.get('to')}",
                border_style="yellow",
            )
        )
        return

    if event_type == "search_results":
        console.print(
            Panel(
                _search_results_renderable(event.get("result", {})),
                title=f"🔎 Search: {event.get('query')}",
                border_style="cyan",
            )
        )
        return

    if event_type == "tool_call":
        node = event.get("node")
        name = event.get("name")
        title = f"Tool Call: {node}/{name}" if node else f"Tool Call: {name}"
        console.print(
            Panel(
                _tool_args_table(event.get("args", {})),
                title=title,
                border_style="cyan",
            )
        )
        return

    if event_type == "tool_result":
        result = event.get("result", {})
        node = event.get("node")
        name = event.get("name")
        ok = _tool_result_ok(result)
        title = f"Tool Result: {node}/{name}" if node else f"Tool Result: {name}"
        console.print(
            Panel(
                _tool_result_table(result),
                title=title,
                border_style="green" if ok else "red",
            )
        )
        return

    if event_type == "final_answer":
        content = str(event.get("content", ""))
        if content:
            console.print(Panel(content, title="Final Answer", border_style="magenta"))
        return

    if verbose:
        console.print(Panel(_json_block(event), title=f"Event: {event_type}", border_style="grey50"))


@app.command()
def main(
    task: Annotated[str | None, typer.Argument(help="Task to send to the Linki model.")] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help=(
                "Base directory for run folders. Each start creates a fresh "
                "'run-<timestamp>' subfolder inside it so checkpoints are never "
                "overwritten. Defaults to 'workspace'."
            ),
        ),
    ] = None,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Model provider to use: openai or deepseek.",
        ),
    ] = "openai",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Override the provider default model.",
        ),
    ] = None,
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            help="Maximum planner/verifier attempts to run.",
        ),
    ] = 3,
    approval_mode: Annotated[
        str,
        typer.Option(
            "--approval-mode",
            help="Approval mode for risky shell commands: inline, auto, or deny.",
        ),
    ] = "inline",
    checkpoint_mode: Annotated[
        str,
        typer.Option(
            "--checkpoint-mode",
            help="Checkpoint mode: light, strict, or off.",
        ),
    ] = "light",
    trace_mode: Annotated[
        str,
        typer.Option(
            "--trace-mode",
            help="Trace mode: on or off.",
        ),
    ] = "on",
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Resume a workspace from its latest Linki checkpoint.",
        ),
    ] = None,
    plan: Annotated[
        bool,
        typer.Option(
            "--plan",
            help="Start in plan mode: read/research and submit a plan for approval before executing.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Print raw fallback events and checkpoint progress.",
        ),
    ] = False,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help="Launch the Textual terminal interface.",
        ),
    ] = False,
) -> None:
    provider_name = provider.lower()
    if provider_name not in {"openai", "deepseek"}:
        raise typer.BadParameter("provider must be 'openai' or 'deepseek'")
    if approval_mode not in {"inline", "auto", "deny"}:
        raise typer.BadParameter("approval-mode must be 'inline', 'auto', or 'deny'")
    if checkpoint_mode not in {"light", "strict", "off"}:
        raise typer.BadParameter("checkpoint-mode must be 'light', 'strict', or 'off'")
    if trace_mode not in {"on", "off"}:
        raise typer.BadParameter("trace-mode must be 'on' or 'off'")

    # Resuming targets an exact run folder; a fresh start gets a brand-new
    # run-<timestamp> folder under the base directory so checkpoints from a
    # previous run are never overwritten.
    active_workspace = resume if resume is not None else create_run_workspace(workspace)

    if tui:
        from Linki.cli.tui.app import LinkiTuiApp

        LinkiTuiApp(
            workspace=active_workspace,
            provider=provider_name,
            model_name=model,
            max_attempts=max_attempts,
            approval_mode=approval_mode,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
            initial_task=task,
            plan_mode=plan,
        ).run()
        return

    if task is None and resume is None:
        raise typer.BadParameter("task is required unless --resume is provided")

    approval_handler = _cli_approval_handler if approval_mode == "inline" else None
    if resume is None:
        console.print(f"[dim]workspace → {active_workspace}[/dim]")

    try:
        event_stream = (
            stream_agent_events(
                task or "",
                workspace=active_workspace,
                max_attempts=max_attempts,
                approval_mode=approval_mode,
                approval_handler=approval_handler,
                checkpoint_mode=checkpoint_mode,
                resume_workspace=resume,
                trace_mode=trace_mode,
                provider=provider_name,
                model_name=model,
                plan_mode=plan,
            )
            if resume is not None
            else stream_session_events(
                task or "",
                session_workspace=active_workspace,
                max_attempts=max_attempts,
                approval_mode=approval_mode,
                approval_handler=approval_handler,
                checkpoint_mode=checkpoint_mode,
                trace_mode=trace_mode,
                provider=provider_name,
                model_name=model,
                plan_mode=plan,
            )
        )
        for event in event_stream:
            _print_event(event, verbose=verbose)
    except KeyboardInterrupt as exc:
        raise typer.Exit(code=130) from exc
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
