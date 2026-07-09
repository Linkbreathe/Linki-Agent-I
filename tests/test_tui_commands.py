"""Pilot tests for the TUI slash-command autocomplete and /plan wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Input, OptionList

from Linki.cli.tui.app import SLASH_COMMANDS, LinkiTuiApp, _matching_commands


def _run(coro) -> None:
    asyncio.run(coro)


def test_matching_commands_prefix_and_space() -> None:
    assert [n for n, _ in _matching_commands("/")] == ["/plan", "/compact", "/memory", "/resume"]
    assert [n for n, _ in _matching_commands("/p")] == ["/plan"]
    assert [n for n, _ in _matching_commands("/r")] == ["/resume"]
    # Once arguments begin (a space), suggestions stop.
    assert _matching_commands("/plan x") == []
    assert _matching_commands("hello") == []


def test_menu_shows_filters_and_accepts(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            inp = app.query_one("#input", Input)
            menu = app.query_one("#cmd-menu", OptionList)
            inp.focus()

            await pilot.press("/")
            await pilot.pause()
            assert menu.display is True
            assert menu.option_count == len(SLASH_COMMANDS)

            await pilot.press("p")
            await pilot.pause()
            assert menu.option_count == 1

            # Enter accepts the highlighted command instead of submitting.
            await pilot.press("enter")
            await pilot.pause()
            assert inp.value == "/plan "
            assert menu.display is False

    _run(impl())


def test_keyboard_navigation_and_escape(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            app.query_one("#input", Input).focus()
            menu = app.query_one("#cmd-menu", OptionList)

            await pilot.press("/")
            await pilot.pause()
            assert menu.highlighted == 0

            await pilot.press("down")
            await pilot.pause()
            assert menu.highlighted == 1

            await pilot.press("escape")
            await pilot.pause()
            assert menu.display is False

    _run(impl())


def test_plan_command_runs_turn_in_plan_mode(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            captured: dict = {}
            app._start_turn = lambda task, plan_mode=None: captured.update(task=task, plan_mode=plan_mode)

            inp = app.query_one("#input", Input)
            menu = app.query_one("#cmd-menu", OptionList)
            inp.focus()

            inp.value = "/plan build login"
            await pilot.pause()
            assert menu.display is False  # space present -> no suggestions

            await pilot.press("enter")
            await pilot.pause()
            assert captured == {"task": "build login", "plan_mode": True}

    _run(impl())


def test_resume_command_is_informational_only(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, initial_task=None)
        async with app.run_test() as pilot:
            started = {"called": False}
            app._start_turn = lambda *a, **k: started.update(called=True)

            inp = app.query_one("#input", Input)
            inp.focus()
            inp.value = "/resume "  # trailing space hides the menu
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            assert started["called"] is False

    _run(impl())
