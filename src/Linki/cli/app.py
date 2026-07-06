from pathlib import Path
from typing import Annotated

import typer

from Linki.core.paths import ensure_workspace
from Linki.core.state import RuntimeState
from Linki.providers.openai_provider import create_model
from Linki.tools.registry import build_tools

app = typer.Typer(no_args_is_help=True)


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
) -> None:
    state = RuntimeState(workspace=workspace)
    ensure_workspace(state, create=True)

    provider_name = provider.lower()
    if provider_name not in {"openai", "deepseek"}:
        raise typer.BadParameter("provider must be 'openai' or 'deepseek'")

    try:
        chat_model = create_model(provider=provider_name, model=model)
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    model_with_tools = chat_model.bind_tools(build_tools(state))
    response = model_with_tools.invoke(task)

    content = getattr(response, "content", response)
    typer.echo(content)
