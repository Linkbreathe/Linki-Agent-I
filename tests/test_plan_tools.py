"""Tests for plan-mode tools, tool filtering, and planner wiring."""

from __future__ import annotations

from Linki.core.approval import KIND_PLAN, ApprovalDecision, ApprovalRequest
from Linki.core.state import create_runtime
from Linki.graph.nodes import _build_planner_tools
from Linki.tools.plan_tools import (
    handle_exit_plan,
    make_enter_plan_mode_tool,
    make_exit_plan_mode_tool,
)
from Linki.tools.registry import build_tools


def _rt(handler=None, approval_mode="inline"):
    return create_runtime(".", approval_mode=approval_mode, approval_handler=handler)


def _invoke(tool, **kwargs):
    return tool.invoke(kwargs)


# --- EnterPlanModeTool -------------------------------------------------------


def test_enter_plan_mode_sets_flags() -> None:
    state = {"runtime": _rt(approval_mode="auto")}
    result = _invoke(make_enter_plan_mode_tool(state))

    assert result["plan_mode"] is True
    assert state["plan_mode"] is True
    assert state["pre_plan_approval_mode"] == "auto"


def test_enter_plan_mode_idempotent() -> None:
    state = {"runtime": _rt(), "plan_mode": True, "pre_plan_approval_mode": "inline"}
    result = _invoke(make_enter_plan_mode_tool(state))
    assert result["plan_mode"] is True
    # Does not clobber the recorded pre-plan mode.
    assert state["pre_plan_approval_mode"] == "inline"


# --- ExitPlanModeTool / handle_exit_plan -------------------------------------


def test_exit_plan_approved_leaves_plan_mode() -> None:
    def handler(request: ApprovalRequest) -> ApprovalDecision:
        assert request.kind == KIND_PLAN
        assert request.plan_text == "1. do X\n2. do Y"
        return ApprovalDecision(approved=True)

    state = {"runtime": _rt(handler), "plan_mode": True}
    result = _invoke(make_exit_plan_mode_tool(state), plan="1. do X\n2. do Y")

    assert result["approved"] is True
    assert state["plan_mode"] is False
    assert state["plan_feedback"] is None


def test_exit_plan_rejected_stays_and_records_feedback() -> None:
    def handler(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=False, answer="split step 2 into two")

    state = {"runtime": _rt(handler), "plan_mode": True}
    result = handle_exit_plan(state, "1. do everything")

    assert result["approved"] is False
    assert result["feedback"] == "split step 2 into two"
    assert state["plan_feedback"] == "split step 2 into two"
    # Still in plan mode so the planner can revise.
    assert state["plan_mode"] is True


def test_exit_plan_without_handler_auto_approves() -> None:
    state = {"runtime": _rt(), "plan_mode": True}
    result = handle_exit_plan(state, "plan body")
    assert result["approved"] is True
    assert state["plan_mode"] is False


# --- build_tools filtering ---------------------------------------------------


def test_build_tools_normal_has_mutating_tools() -> None:
    names = {tool.name for tool in build_tools(_rt())}
    assert {"FileWriteTool", "FileEditTool", "BashTool"} <= names


def test_build_tools_plan_mode_is_read_only() -> None:
    names = {tool.name for tool in build_tools(_rt(), plan_mode=True)}
    assert names == {"FileReadTool", "GrepTool"}


# --- planner tool selection --------------------------------------------------


def test_planner_tools_normal_mode() -> None:
    working = {"runtime": _rt(), "ask_budget": 2}
    names = [tool.name for tool in _build_planner_tools(working, plan_mode=False, ask_budget_left=2)]
    assert "CallCodeAgentTool" in names
    assert "EnterPlanModeTool" in names
    assert "AskUserQuestionTool" in names
    assert "ExitPlanModeTool" not in names


def test_planner_tools_plan_mode() -> None:
    working = {"runtime": _rt(), "ask_budget": 2, "plan_mode": True}
    names = [tool.name for tool in _build_planner_tools(working, plan_mode=True, ask_budget_left=2)]
    assert "ExitPlanModeTool" in names
    assert "CallCodeAgentTool" not in names
    assert "EnterPlanModeTool" not in names


def test_planner_drops_ask_tool_when_budget_exhausted() -> None:
    working = {"runtime": _rt(), "ask_budget": 0}
    names = [tool.name for tool in _build_planner_tools(working, plan_mode=False, ask_budget_left=0)]
    assert "AskUserQuestionTool" not in names
