"""Plan-mode tools for the planner.

Plan mode constrains the planner to reading and researching: it produces a
step-by-step plan and submits it for human review before any file writes or
commands happen. The flow reuses the stage-five approval channel:

- ``EnterPlanModeTool`` flips ``plan_mode`` on and records the current approval
  mode in ``pre_plan_approval_mode`` so it can be restored on exit. The planner
  may call it in normal mode; a run can also start in plan mode via ``--plan``.
- ``ExitPlanModeTool`` submits the Markdown plan through a ``kind="plan"``
  approval request. On approval the run leaves plan mode and proceeds; on
  rejection the reviewer's feedback is stored in ``plan_feedback`` and the
  planner stays in plan mode to revise.

Both tools mutate the planner working dict in place so ``planner_node`` returns
the updated flags to graph state.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from langchain_core.tools import StructuredTool

from Linki.core.approval import new_plan_request


def _values(state: Any) -> MutableMapping[str, Any]:
    return state if isinstance(state, MutableMapping) else {}


def make_enter_plan_mode_tool(state: Any) -> StructuredTool:
    """Build ``EnterPlanModeTool`` bound to the planner working state."""

    def enter_plan_mode_tool() -> dict:
        values = _values(state)
        if values.get("plan_mode"):
            return {"ok": True, "plan_mode": True, "note": "already in plan mode"}

        runtime = values.get("runtime")
        current_mode = getattr(runtime, "approval_mode", "inline")
        values["plan_mode"] = True
        values["pre_plan_approval_mode"] = current_mode
        return {"ok": True, "plan_mode": True, "pre_plan_approval_mode": current_mode}

    return StructuredTool.from_function(
        func=enter_plan_mode_tool,
        name="EnterPlanModeTool",
        description=(
            "Enter plan mode: stop writing/executing and switch to read-and-research only. "
            "Use this when a task should be planned and reviewed by the human before any changes."
        ),
    )


def handle_exit_plan(state: Any, plan: str) -> dict:
    """Submit ``plan`` for human review and update plan-mode flags accordingly."""

    values = _values(state)
    runtime = values.get("runtime")
    handler = getattr(runtime, "approval_handler", None)

    if handler is None:
        # No interactive channel (auto/deny mode or non-interactive run): accept
        # the plan and leave plan mode so the run can proceed.
        values["plan_mode"] = False
        values["plan_feedback"] = None
        return {
            "ok": True,
            "approved": True,
            "plan": plan,
            "reason": "No interactive channel available; plan auto-approved.",
        }

    decision = handler(new_plan_request(plan))

    if decision.approved:
        values["plan_mode"] = False
        values["plan_feedback"] = None
        return {"ok": True, "approved": True, "plan": plan}

    feedback = str(getattr(decision, "answer", "") or "") or "Plan rejected without specific feedback."
    values["plan_feedback"] = feedback
    # Stay in plan mode so the planner revises and resubmits.
    return {"ok": True, "approved": False, "feedback": feedback}


def make_exit_plan_mode_tool(state: Any) -> StructuredTool:
    """Build ``ExitPlanModeTool`` bound to the planner working state."""

    def exit_plan_mode_tool(plan: str) -> dict:
        return handle_exit_plan(state, plan)

    return StructuredTool.from_function(
        func=exit_plan_mode_tool,
        name="ExitPlanModeTool",
        description=(
            "Submit a finished step-by-step plan (Markdown) for human approval. "
            "If approved, plan mode ends and execution proceeds; if rejected, revise "
            "the plan using the returned feedback and submit again."
        ),
    )
