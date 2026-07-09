"""Tests for the sidebar status panel: state, agent, verify attempts, context gauge."""

from __future__ import annotations

import asyncio
from pathlib import Path

from Linki.cli.tui.app import LinkiTuiApp


def _run(coro) -> None:
    asyncio.run(coro)


def _session_text(app: LinkiTuiApp) -> str:
    return app.query_one("#session").content.plain


def test_status_panel_shows_provider_and_turn(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path, provider="deepseek", model_name="deepseek-v4-flash")
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _session_text(app)
            assert "deepseek" in text
            assert "deepseek-v4-flash" in text

            app._handle_event({"type": "session_saved", "session_id": "abcd1234efgh", "turn_index": 3})
            await pilot.pause()
            text = _session_text(app)
            assert "abcd1234" in text
            assert "turn 3" in text

    _run(impl())


def test_status_panel_tracks_active_agent(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "codeAgent" not in _session_text(app)

            app._handle_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
            await pilot.pause()
            assert "codeAgent" in _session_text(app)

            app._handle_event({"type": "turn_finished"})
            await pilot.pause()
            assert "codeAgent" not in _session_text(app)

    _run(impl())


def test_status_panel_tracks_verify_attempts(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "verify" not in _session_text(app)

            app._handle_event(
                {
                    "type": "node_update",
                    "node": "verifier",
                    "data": {"passed": False, "attempts": 1, "max_attempts": 3},
                }
            )
            await pilot.pause()
            text = _session_text(app)
            assert "verify" in text
            assert "1/3" in text

            app._handle_event(
                {
                    "type": "node_update",
                    "node": "verifier",
                    "data": {"passed": True, "attempts": 2, "max_attempts": 3},
                }
            )
            await pilot.pause()
            assert "2/3" in _session_text(app)

    _run(impl())


def test_status_panel_shows_context_gauge(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "ctx" not in _session_text(app)

            # The real context_monitor node emits only token_count; the panel
            # must fall back to the default limit and still draw the gauge.
            app._handle_event(
                {
                    "type": "node_update",
                    "node": "context_monitor",
                    "data": {"context_token_count": 120_000},
                }
            )
            await pilot.pause()
            text = _session_text(app)
            assert "ctx" in text
            assert "120k" in text
            assert "400k" in text

    _run(impl())


def test_context_monitor_node_emits_limit(tmp_path: Path) -> None:
    """Regression: the node must publish the limit so the TUI gauge can render."""

    from Linki.core.state import create_runtime
    from Linki.graph.nodes import CONTEXT_TOKEN_LIMIT_DEFAULT, context_monitor_node

    # A model without token APIs forces the char-based estimate, so no provider
    # or API key is needed.
    out = context_monitor_node(
        {
            "messages": [],
            "context_next_node": "verifier",
            "model": object(),
            "runtime": create_runtime(tmp_path),
        }
    )
    assert out["context_token_limit"] == CONTEXT_TOKEN_LIMIT_DEFAULT
    assert "context_token_count" in out


def test_fmt_tokens() -> None:
    assert LinkiTuiApp._fmt_tokens(950) == "950"
    assert LinkiTuiApp._fmt_tokens(12_345) == "12.3k"
    assert LinkiTuiApp._fmt_tokens(400_000) == "400k"


def test_final_answer_rendered_as_markdown(tmp_path: Path) -> None:
    async def impl() -> None:
        app = LinkiTuiApp(workspace=tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_event({"type": "final_answer", "content": "# Done\n\n- item"})
            await pilot.pause()
            from rich.markdown import Markdown

            body = app.query_one("#final-body").content
            assert isinstance(body, Markdown)
            assert "Done" in body.markup

    _run(impl())
