"""Tests for the proactive clarification tool and its approval-channel plumbing."""

from __future__ import annotations

from Linki.core.approval import (
    KIND_COMMAND,
    KIND_PLAN,
    KIND_QUESTION,
    ApprovalDecision,
    ApprovalRequest,
    new_approval_request,
    new_plan_request,
    new_question_request,
)
from Linki.core.state import create_runtime
from Linki.tools.ask_user_tool import make_ask_user_question_tool


def _runtime_with_handler(handler):
    return create_runtime(".", approval_handler=handler)


def _invoke(tool, **kwargs):
    return tool.invoke(kwargs)


def test_question_request_shape() -> None:
    request = new_question_request("Which DB?", ["postgres", "sqlite"], allow_free_text=False)
    assert request.kind == KIND_QUESTION
    assert request.question == "Which DB?"
    assert request.options == ("postgres", "sqlite")
    assert request.allow_free_text is False
    assert request.tool_name == "AskUserQuestionTool"


def test_command_and_plan_request_shapes() -> None:
    assert new_approval_request("rm x", "danger").kind == KIND_COMMAND
    assert new_plan_request("step 1\nstep 2").kind == KIND_PLAN
    # Decision keeps a default empty answer for backward compatibility.
    assert ApprovalDecision(approved=True).answer == ""


def test_ask_decrements_budget_and_returns_answer() -> None:
    captured: dict = {}

    def handler(request: ApprovalRequest) -> ApprovalDecision:
        captured["request"] = request
        return ApprovalDecision(approved=True, answer="use postgres")

    state = {"runtime": _runtime_with_handler(handler), "ask_budget": 2}
    tool = make_ask_user_question_tool(state)

    result = _invoke(tool, question="Which DB?", options=["postgres", "sqlite"])

    assert result["answer"] == "use postgres"
    assert result["asked"] is True
    assert result["remaining_budget"] == 1
    assert state["ask_budget"] == 1
    # Options were forwarded to the request.
    assert captured["request"].options == ("postgres", "sqlite")
    assert captured["request"].kind == KIND_QUESTION


def test_ask_twice_exhausts_budget_then_refuses() -> None:
    def handler(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, answer="ok")

    state = {"runtime": _runtime_with_handler(handler), "ask_budget": 2}
    tool = make_ask_user_question_tool(state)

    _invoke(tool, question="q1")
    _invoke(tool, question="q2")
    assert state["ask_budget"] == 0

    third = _invoke(tool, question="q3")
    assert third["asked"] is False
    assert third["remaining_budget"] == 0
    assert state["ask_budget"] == 0


def test_no_handler_does_not_consume_budget() -> None:
    state = {"runtime": create_runtime("."), "ask_budget": 2}
    tool = make_ask_user_question_tool(state)

    result = _invoke(tool, question="Which DB?")

    assert result["asked"] is False
    assert result["answer"] == ""
    # Budget preserved when nobody could answer.
    assert state["ask_budget"] == 2


def test_default_budget_when_absent() -> None:
    def handler(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, answer="a")

    state: dict = {"runtime": _runtime_with_handler(handler)}
    tool = make_ask_user_question_tool(state)

    result = _invoke(tool, question="q")
    # Defaults to 2, so one ask leaves 1.
    assert result["remaining_budget"] == 1
    assert state["ask_budget"] == 1
