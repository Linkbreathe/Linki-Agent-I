from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import Input

from Linki.cli.tui.app import LinkiTuiApp, _matching_commands


def test_matching_commands_includes_compact() -> None:
    names = [name for name, _ in _matching_commands("/c")]
    assert names == ["/compact"]


def test_compact_command_does_not_start_llm_turn(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            started = {"called": False}
            app._start_turn = lambda *a, **k: started.update(called=True)

            inp = app.query_one("#input", Input)
            inp.focus()
            inp.value = "/compact auth docs"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert started["called"] is False

    asyncio.run(impl())
