"""Proactive clarification tool for the planner.

``AskUserQuestionTool`` lets the planner pause and ask the human a single
clarifying question when the task is ambiguous in a way that changes the plan
direction. It reuses the stage-five approval channel: a ``kind="question"``
request is handed to ``RuntimeState.approval_handler``, which blocks until the
human answers (terminal input in CLI mode, a modal in TUI mode).

Questions are strictly budgeted. The budget lives in graph state
(``LinkiGraphState.ask_budget``, initially 2) and is threaded in via the mutable
planner working dict so each successful ask decrements it and the remaining
budget survives planner re-entries within a run.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from langchain_core.tools import StructuredTool

from Linki.core.approval import new_question_request

DEFAULT_ASK_BUDGET = 2


def make_ask_user_question_tool(state: Any) -> StructuredTool:
    """Build the ``AskUserQuestionTool`` bound to the planner working state.

    ``state`` is the mutable planner working dict; it must carry ``runtime`` (for
    the approval handler) and ``ask_budget``. Each answered question decrements
    ``state['ask_budget']`` in place.
    """

    def ask_user_question_tool(
        question: str,
        options: list[str] | None = None,
        allow_free_text: bool = True,
    ) -> dict:
        values: MutableMapping[str, Any] = state if isinstance(state, MutableMapping) else {}
        budget = int(values.get("ask_budget", DEFAULT_ASK_BUDGET))

        if budget <= 0:
            return {
                "answer": "",
                "asked": False,
                "remaining_budget": 0,
                "reason": "Question budget exhausted; decide with the information you already have.",
            }

        runtime = values.get("runtime")
        handler = getattr(runtime, "approval_handler", None)
        if handler is None:
            # No interactive channel (auto/deny mode or a non-interactive run).
            return {
                "answer": "",
                "asked": False,
                "remaining_budget": budget,
                "reason": "No interactive channel available; proceed with your best judgment.",
            }

        request = new_question_request(question, options, allow_free_text=allow_free_text)
        decision = handler(request)

        remaining = budget - 1
        values["ask_budget"] = remaining

        return {
            "answer": str(getattr(decision, "answer", "") or ""),
            "asked": True,
            "remaining_budget": remaining,
        }

    return StructuredTool.from_function(
        func=ask_user_question_tool,
        name="AskUserQuestionTool",
        description=(
            "Ask the human ONE clarifying question before finalizing the plan, only when "
            "the task is ambiguous in a way that changes the plan direction. Pass optional "
            "'options' for a multiple-choice answer. Strictly budgeted (2 per run); never "
            "ask what read-only tools can determine."
        ),
    )
