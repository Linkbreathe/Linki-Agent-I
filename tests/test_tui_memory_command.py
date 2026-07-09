from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from rich.console import Console
from textual.widgets import Input

from Linki.cli.tui.app import LinkiTuiApp
from Linki.core.memory_store import append_user_memory, existing_entries
from Linki.core.state import create_runtime


def test_hash_memory_write_echoes(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            events: list[str] = []
            app._write_event = lambda summary, detail=None, kind="system": events.append(summary)

            inp = app.query_one("#input", Input)
            inp.focus()
            inp.value = "# Prefer short answers"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert any("Saved to memory" in event for event in events)
            assert existing_entries(create_runtime(tmp_path))[0].text == "Prefer short answers"

    asyncio.run(impl())


def test_memory_rm_renumbers_listing(tmp_path: Path) -> None:
    async def impl() -> None:
        runtime = create_runtime(tmp_path)
        append_user_memory(runtime, "first")
        append_user_memory(runtime, "second")

        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as _pilot:
            events: list[tuple[str, object]] = []
            app._write_event = lambda summary, detail=None, kind="system": events.append((summary, detail))

            assert app._handle_slash_command("/memory rm 1") is True
            assert app._handle_slash_command("/memory") is True

            entries = existing_entries(runtime)
            assert [entry.text for entry in entries] == ["second"]
            console = Console(record=True, width=100)
            console.print(events[-1][1])
            listing = console.export_text()
            assert " 1 " in listing
            assert "second" in listing
            assert " 2 " not in listing

    asyncio.run(impl())
