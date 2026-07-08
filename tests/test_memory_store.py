from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from Linki.core.context import assemble_project_context
from Linki.core.memory_store import (
    MemoryEntry,
    _write,
    append_user_memory,
    existing_entries,
    extract_run_memories,
    memory_path,
    upsert_agent_memory,
)
from Linki.core.state import create_runtime
from Linki.prompts.stage3 import PLANNER_PROMPT
from Linki.agents.code_agent import CODE_AGENT_PROMPT
from Linki.tools.registry import build_tools


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.invocations = 0

    def invoke(self, messages):
        self.invocations += 1
        return AIMessage(content=self.content)


def test_append_user_memory_format(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    append_user_memory(runtime, "Prefer concise summaries.")

    text = memory_path(runtime).read_text(encoding="utf-8")
    assert " · user] Prefer concise summaries." in text
    assert text.startswith("- [")


def test_extract_replaces_existing_entry_without_appending(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    _write(
        runtime,
        [MemoryEntry(date="07-08", source="extracted", run_id="run_old", text="Old preference")],
        added=1,
    )
    model = FakeModel('{"memories": [{"text": "New preference", "kind": "preference", "replaces": 1}]}')

    stats = extract_run_memories(
        {
            "runtime": runtime,
            "model": model,
            "session_summary": "User corrected the preferred style.",
            "run_id": "run_031",
        }
    )

    entries = existing_entries(runtime)
    assert stats == {"added": 0, "replaced": 1, "total": 1}
    assert len(entries) == 1
    assert entries[0].text == "New preference"
    assert entries[0].run_id == "run_031"


def test_extracted_memory_does_not_replace_user_entry(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    append_user_memory(runtime, "Always use pip.")
    model = FakeModel('{"memories": [{"text": "Always use uv.", "kind": "preference", "replaces": 1}]}')

    stats = extract_run_memories(
        {
            "runtime": runtime,
            "model": model,
            "session_summary": "The run mentioned uv.",
            "run_id": "run_031",
        }
    )

    entries = existing_entries(runtime)
    assert stats == {"added": 0, "replaced": 0, "total": 1}
    assert entries[0].source == "user"
    assert entries[0].text == "Always use pip."


def test_write_over_limit_triggers_consolidation(tmp_path: Path, monkeypatch) -> None:
    events: list[dict] = []
    runtime = create_runtime(tmp_path, event_handler=events.append)
    entries = [
        MemoryEntry(date="07-08", source="user", run_id=None, text=f"memory {index}")
        for index in range(151)
    ]

    def compact(items, state=None):
        return items[:2]

    monkeypatch.setattr("Linki.core.memory_store.consolidate_with_model", compact)
    _write(runtime, entries, added=151)

    assert len(existing_entries(runtime)) == 2
    assert any(event.get("type") == "memory_consolidate" and event.get("before") == 151 for event in events)
    assert any(event.get("type") == "memory_write" and event.get("total") == 2 for event in events)


def test_extract_empty_array_leaves_file_bytes_unchanged(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    _write(
        runtime,
        [MemoryEntry(date="07-08", source="user", run_id=None, text="Keep this")],
        added=1,
    )
    before = memory_path(runtime).read_bytes()

    stats = extract_run_memories(
        {
            "runtime": runtime,
            "model": FakeModel('{"memories": []}'),
            "session_summary": "Nothing durable.",
        }
    )

    assert stats == {"added": 0, "replaced": 0, "total": 1}
    assert memory_path(runtime).read_bytes() == before


def test_project_context_orders_rules_then_memory(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    (tmp_path / "LINKI.md").write_text("Rule layer", encoding="utf-8")
    append_user_memory(runtime, "Memory layer")

    context = assemble_project_context(runtime)

    assert context.index("<project_rules>") < context.index("<project_memory>")
    assert "Rule layer" in context
    assert "Memory layer" in context


def test_agent_cannot_edit_memory_store(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    path = tmp_path / ".linki" / "memory" / "MEMORY.md"
    path.parent.mkdir(parents=True)
    path.write_text("- [07-08 · user] keep\n", encoding="utf-8")
    tools = {tool.name: tool for tool in build_tools(runtime)}

    result = tools["FileEditTool"].invoke(
        {"file_path": ".linki/memory/MEMORY.md", "old_text": "keep", "new_text": "change"}
    )

    assert result["ok"] is False
    assert "protected path" in result["error"]
    assert "keep" in path.read_text(encoding="utf-8")


def test_memory_upsert_tool_saves_explicit_future_rule(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path, trace_id="run_031")
    tools = {tool.name: tool for tool in build_tools(runtime)}

    result = tools["MemoryUpsertTool"].invoke(
        {"text": "该项目统一使用 uv，不使用 pip。", "replaces": None}
    )

    assert result["status"] == "added"
    assert result["index"] == 1
    assert existing_entries(runtime)[0].source == "agent"
    assert "agent@run_031" in memory_path(runtime).read_text(encoding="utf-8")


def test_progress_and_unverified_guess_are_skipped(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)

    progress = upsert_agent_memory(runtime, "TODO: finish the login endpoint next.")
    guess = upsert_agent_memory(runtime, "Maybe Redis caused the failure.")

    assert progress["status"] == "skipped"
    assert guess["status"] == "skipped"
    assert existing_entries(runtime) == []


def test_verified_debugging_lesson_can_be_saved(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)

    result = upsert_agent_memory(runtime, "Integration tests require Redis to be running.")

    assert result["status"] == "added"
    assert existing_entries(runtime)[0].text == "Integration tests require Redis to be running."


def test_same_or_similar_agent_memory_is_skipped_without_replaces(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    upsert_agent_memory(runtime, "Always use uv for this project.")

    exact = upsert_agent_memory(runtime, "Always use uv for this project.")
    similar = upsert_agent_memory(runtime, "Use uv instead of pip in this project.")

    assert exact["status"] == "skipped"
    assert similar["status"] == "skipped"
    assert len(existing_entries(runtime)) == 1


def test_agent_replaces_in_place_without_increasing_count(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path, trace_id="run_031")
    upsert_agent_memory(runtime, "Use pip for installs.")

    result = upsert_agent_memory(runtime, "Use uv for installs.", replaces=1)

    entries = existing_entries(runtime)
    assert result["status"] == "replaced"
    assert len(entries) == 1
    assert entries[0].text == "Use uv for installs."
    assert entries[0].run_id == "run_031"


def test_agent_cannot_replace_user_memory_by_default(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    append_user_memory(runtime, "Always use pip.")

    result = upsert_agent_memory(runtime, "Always use uv.", replaces=1)

    assert result["status"] == "skipped"
    assert existing_entries(runtime)[0].text == "Always use pip."


def test_current_user_correction_can_replace_user_memory(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path, trace_id="run_031")
    append_user_memory(runtime, "Always use pip.")

    result = upsert_agent_memory(
        {"runtime": runtime, "allow_user_memory_replace": True},
        "Always use uv.",
        replaces=1,
    )

    entries = existing_entries(runtime)
    assert result["status"] == "replaced"
    assert entries[0].source == "agent"
    assert entries[0].text == "Always use uv."


def test_agent_memory_is_not_duplicated_by_run_end_extraction(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path, trace_id="run_031")
    upsert_agent_memory(runtime, "Use uv instead of pip in this project.")
    model = FakeModel('{"memories": [{"text": "Always use uv for this project.", "kind": "preference", "replaces": null}]}')

    stats = extract_run_memories(
        {
            "runtime": runtime,
            "model": model,
            "session_summary": "The user confirmed the project should use uv.",
            "run_id": "run_031",
        }
    )

    assert stats == {"added": 0, "replaced": 0, "total": 1}
    assert len(existing_entries(runtime)) == 1


def test_memory_instructions_and_tool_are_available_to_agents(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path)
    names = {tool.name for tool in build_tools(runtime)}

    assert "MemoryUpsertTool" in names
    assert "MemoryUpsertTool" in PLANNER_PROMPT
    assert "MemoryUpsertTool" in CODE_AGENT_PROMPT
    assert "Always use uv for this project" in CODE_AGENT_PROMPT
