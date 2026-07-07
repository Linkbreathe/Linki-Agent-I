import json
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import typer

from Linki.core.agent import stream_agent_events

app = typer.Typer(no_args_is_help=True)
console = Console()

STATUS_ICONS = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "blocked": "🚫"}


def _json_block(value: object) -> Syntax:
    return Syntax(
        json.dumps(value, indent=2, ensure_ascii=False),
        "json",
        word_wrap=True,
    )


def _todos_table(todos: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Status")
    table.add_column("Todo")
    table.add_column("Note")
    for todo in todos:
        table.add_row(f"{STATUS_ICONS[todo['status']]} {todo['status']}", todo["content"], todo["note"])
    return table


def _checks_table(checks: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("Check")
    table.add_column("Detail")
    for check in checks:
        table.add_row("✅" if check["passed"] else "❌", check["name"], check["detail"])
    return table


def _print_event(event: dict) -> None:
    event_type = event.get("type")

    if event_type == "node_update":
        node = event.get("node")
        data = event.get("data", {})
        content = str(event.get("content", ""))

        if node == "planner":
            console.print(
                Panel(
                    _todos_table(data["todos"]),
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

        console.print(Panel(_json_block(data), title=f"Node: {node}", border_style="white"))
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
                _json_block(event.get("result", {})),
                title=f"🔎 Search: {event.get('query')}",
                border_style="cyan",
            )
        )
        return

    if event_type == "tool_call":
        console.print(
            Panel(
                _json_block(event.get("args", {})),
                title=f"Tool Call: {event.get('name')}",
                border_style="cyan",
            )
        )
        return

    if event_type == "tool_result":
        console.print(
            Panel(
                _json_block(event.get("result", {})),
                title=f"Tool Result: {event.get('name')}",
                border_style="green",
            )
        )
        return

    if event_type == "final_answer":
        content = str(event.get("content", ""))
        if content:
            console.print(Panel(content, title="Final Answer", border_style="magenta"))


@app.command()
def main(
    task: Annotated[str, typer.Argument(help="Task to send to the Linki model.")],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace directory. Created automatically when it does not exist.",
        ),
    ] = Path.cwd(),
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
) -> None:
    provider_name = provider.lower()
    if provider_name not in {"openai", "deepseek"}:
        raise typer.BadParameter("provider must be 'openai' or 'deepseek'")

    try:
        for event in stream_agent_events(
            task,
            workspace=workspace,
            max_attempts=max_attempts,
            provider=provider_name,
            model_name=model,
        ):
            _print_event(event)
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
