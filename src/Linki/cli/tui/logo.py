from __future__ import annotations

import asyncio

from rich.text import Text


def build_logo(*, unicode: bool = True, status: str | None = None) -> Text:
    """Return the styled Linki startup logo."""

    if not unicode:
        logo = Text()
        logo.append("Linki\n", style="bold cyan")
        logo.append("----------------------------------\n", style="blue")
        logo.append(" Multi-Agent TUI", style="green")
        if status:
            logo.append(f" | {status}", style="bold green" if status == "ready" else "dim")
        logo.append("\n----------------------------------", style="blue")
        return logo

    logo = Text()
    logo.append(" Linki\n", style="bold rgb(119,239,216)")
    logo.append(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", style="blue")
    logo.append("  Multi-Agent TUI", style="rgb(179,194,213)")
    if status:
        logo.append(f"  {status}", style="bold green" if status == "ready" else "dim")
    logo.append("\n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", style="blue")
    return logo


async def animate_logo(target) -> None:
    """
    Render a short non-blocking startup animation into the target widget.
    """

    frames = [
        build_logo(),
        build_logo(status="loading"),
        build_logo(status="ready"),
    ]
    for frame in frames:
        target.update(frame)
        await asyncio.sleep(0.12)
