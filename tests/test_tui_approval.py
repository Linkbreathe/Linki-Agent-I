"""Regression: the approval modal must render untrusted text literally.

Commands, risk reasons, clarifying questions, option labels, plan text, and the
parallel-dispatch job label routinely contain characters that look like Textual
markup (``[`` ``]`` ``=`` ``"``). Parsing them as markup raised MarkupError and
crashed the whole TUI. These cases must mount without raising.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App

from Linki.cli.tui.approval import ApprovalModal
from Linki.core.approval import KIND_PLAN, KIND_QUESTION, ApprovalRequest

MARKUP_CASES = {
    "command": ApprovalRequest(
        id="1", command='grep -P "[x=/dev/zero\\b\"),]" f', risk_reason="r", tool_name="BashTool"
    ),
    "risk_reason": ApprovalRequest(
        id="2", command="ls", risk_reason="matched [pattern=danger]", tool_name="BashTool"
    ),
    "job_label": ApprovalRequest(
        id="3", command="pip install x", risk_reason="install",
        tool_name="BashTool", label="[job-2 · reviewer]",
    ),
    "question_and_options": ApprovalRequest(
        id="4", kind=KIND_QUESTION, question="Pick [a=1]?",
        options=("[opt one]", "two [x=y]"), tool_name="AskUserQuestionTool",
    ),
    "plan_text": ApprovalRequest(
        id="5", kind=KIND_PLAN, plan_text="step [color=red] do x", tool_name="PlanReview"
    ),
}


@pytest.mark.parametrize("request_obj", MARKUP_CASES.values(), ids=list(MARKUP_CASES))
def test_approval_modal_renders_markup_like_text_without_crashing(request_obj) -> None:
    async def impl() -> None:
        class Harness(App):
            async def on_mount(self) -> None:
                await self.push_screen(ApprovalModal(request_obj, workspace="/tmp"))

        async with Harness().run_test(size=(120, 40)) as pilot:
            await pilot.pause()  # forces a layout/reflow — where markup was parsed

    asyncio.run(impl())


def test_activity_stream_renders_markup_like_tool_args_without_crashing(tmp_path) -> None:
    """A bash command with a regex char-class must not crash the event stream."""
    import tempfile

    from Linki.cli.tui.app import LinkiTuiApp

    async def impl() -> None:
        with tempfile.TemporaryDirectory() as d:
            app = LinkiTuiApp(workspace=d)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app._handle_event(
                    {
                        "type": "tool_call",
                        "node": "codeAgent",
                        "name": "BashTool",
                        "args": {"command": "grep -P '[a=b\"),]' file"},
                    }
                )
                app._handle_event(
                    {"type": "ai_message", "node": "codeAgent", "content": "regex [x=y] [/dev/zero]"}
                )
                await pilot.pause()

    asyncio.run(impl())
