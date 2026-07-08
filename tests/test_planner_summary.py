"""Regression test: a trailing sign-off must not erase the planner's answer.

Reproduces run-20260708-034644, where the planner produced the real analysis
and then a conversational "任务完成！" message. The last-message-wins capture
handed only the sign-off to the verifier as `last_actor_summary`, so the
verifier failed a turn that had actually answered the question.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from Linki.core.state import create_runtime
from Linki.graph.nodes import planner_node

ANALYSIS = "这个网站是艾欧泽亚售楼中心，用于查询 FFXIV 房屋出售/抽签信息。"
SIGN_OFF = "任务完成！有问题随时再问我~"


class _FakeAgent:
    """Emits a substantive answer (with a tool call) then a trailing sign-off."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content=ANALYSIS,
                tool_calls=[
                    {
                        "name": "TodoWriteTool",
                        "id": "call-1",
                        "args": {
                            "plan_summary": "查阅网站并汇报",
                            "todos": [],
                            "acceptance_criteria": [],
                            "verification_commands": [],
                        },
                    }
                ],
            )
        # No tool calls -> the ReAct loop ends here.
        return AIMessage(content=SIGN_OFF)


class _FakeModel:
    def bind_tools(self, tools):
        return _FakeAgent()


def test_last_actor_summary_keeps_substantive_answer() -> None:
    state = {
        "runtime": create_runtime("."),
        "model": _FakeModel(),
        "task": "这个网站是用来干嘛的",
        "ask_budget": 2,
    }

    result = planner_node(state)

    summary = result["last_actor_summary"]
    # The real answer must survive even though a sign-off came afterward.
    assert "艾欧泽亚售楼中心" in summary, f"answer was lost: {summary!r}"
    # The sign-off may still be present, but it must not be the whole thing.
    assert summary.strip() != SIGN_OFF
