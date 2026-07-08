from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from Linki.core.compact import compact_pipeline, enforce_acceptance_criteria, split_messages
from Linki.core.state import create_runtime
from Linki.tools.registry import build_tools


class FakeModel:
    def __init__(self, summary: str = "") -> None:
        self.summary = summary or "\n".join(
            [
                "# Task And Goal",
                "- continue",
                "# Current Plan",
                "# Acceptance Criteria",
                "- criterion one",
                "- criterion three",
                "# Completed Work",
                "# Open Work",
                "# Important Files",
                "# Tool Findings",
                "# Risks And Blockers",
            ]
        )
        self.invocations = 0

    def invoke(self, messages):
        self.invocations += 1
        return AIMessage(content=self.summary)

    def get_num_tokens_from_messages(self, messages):
        return sum(max(len(str(getattr(message, "content", ""))) // 4, 1) for message in messages)


def test_split_messages_retains_recent_five_text_messages() -> None:
    messages = [HumanMessage(content=f"old {i} " + ("x" * 800)) for i in range(8)]
    head, tail = split_messages(messages, min_tail_messages=5, min_tail_tokens=0)

    assert head == messages[:3]
    assert tail == messages[3:]


def test_split_messages_does_not_split_tool_call_pair() -> None:
    messages = [
        HumanMessage(content="old " + ("x" * 5000)),
        AIMessage(
            content="calling",
            tool_calls=[{"name": "FileReadTool", "args": {"file_path": "a.py"}, "id": "call-1"}],
        ),
        ToolMessage(content="result", tool_call_id="call-1"),
        HumanMessage(content="tail 1"),
        HumanMessage(content="tail 2"),
        HumanMessage(content="tail 3"),
        HumanMessage(content="tail 4"),
    ]

    head, tail = split_messages(messages, min_tail_messages=5, min_tail_tokens=0)

    assert messages[1] in tail
    assert messages[2] in tail
    assert messages[1] not in head
    assert messages[2] not in head


def test_acceptance_criteria_are_restored_and_warned(tmp_path: Path) -> None:
    events: list[dict] = []
    runtime = create_runtime(tmp_path, event_handler=events.append)
    summary = "# Acceptance Criteria\n- criterion one\n- criterion three\n"

    updated = enforce_acceptance_criteria(
        summary,
        ["criterion one", "criterion two", "criterion three"],
        runtime=runtime,
    )

    assert "criterion two" in updated
    assert any(
        event.get("type") == "trace.warn"
        and event.get("reason") == "acceptance criterion restored"
        and event.get("criterion") == "criterion two"
        for event in events
    )


def test_short_session_skips_compaction_without_message_changes(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    messages = [HumanMessage(content="short"), AIMessage(content="small")]
    model = FakeModel()

    updates = compact_pipeline(
        {"runtime": runtime, "messages": messages, "model": model, "compression_events": []},
        focus=None,
        trigger="manual",
    )

    assert "messages" not in updates
    assert model.invocations == 0


def test_compact_pipeline_writes_summary_and_recent_files_note(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    tools = {tool.name: tool for tool in build_tools(runtime)}
    tools["FileWriteTool"].invoke({"file_path": "app.py", "content": "print('x')\n"})
    messages = [HumanMessage(content=f"old {i} " + ("x" * 2000)) for i in range(8)]
    model = FakeModel()

    updates = compact_pipeline(
        {
            "runtime": runtime,
            "messages": messages,
            "model": model,
            "acceptance_criteria": ["criterion one", "criterion two", "criterion three"],
            "compression_events": [],
            "context_token_limit": 400_000,
        },
        focus="docs",
        trigger="manual",
    )

    content = updates["messages"][1].content
    assert "<compact_summary>" in content
    assert "Recently touched files" in content
    assert "app.py" in content
    assert "criterion two" in content


def test_recent_files_lru_evicts_oldest(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    tools = {tool.name: tool for tool in build_tools(runtime)}
    for index in range(9):
        tools["FileWriteTool"].invoke({"file_path": f"file{index}.txt", "content": str(index)})

    assert list(runtime.recent_files.keys()) == [f"file{index}.txt" for index in range(1, 9)]
