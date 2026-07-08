"""Cross-session memory storage for Linki."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from Linki.core.session import SESSION_SUMMARY_FILE, SESSION_ROOT
from Linki.core.state import RuntimeState
from Linki.prompts.memory import CONSOLIDATE_PROMPT, EXTRACT_PROMPT

MEMORY_DIR = Path(".linki") / "memory"
MEMORY_FILENAME = "MEMORY.md"
MEMORY_PATH = MEMORY_DIR / MEMORY_FILENAME
MEMORY_MAX_LINES = 150
MEMORY_MAX_BYTES = 20 * 1024


@dataclass(frozen=True)
class MemoryEntry:
    date: str
    source: Literal["user", "agent", "extracted"]
    run_id: str | None
    text: str

    def to_line(self) -> str:
        source = self.source
        if self.source in {"agent", "extracted"} and self.run_id:
            source = f"{self.source}@{self.run_id}"  # type: ignore[assignment]
        return f"- [{self.date} · {source}] {self.text}"


def _runtime(state: RuntimeState | Mapping[str, Any]) -> RuntimeState:
    if isinstance(state, RuntimeState):
        return state
    runtime = state.get("runtime")
    if runtime is None:
        raise ValueError("state['runtime'] is required")
    return runtime


def memory_path(state: RuntimeState | Mapping[str, Any]) -> Path:
    return _runtime(state).workspace / MEMORY_PATH


def _today() -> str:
    return datetime.now().strftime("%m-%d")


def _run_id(state: RuntimeState | Mapping[str, Any]) -> str | None:
    if isinstance(state, Mapping):
        value = state.get("run_id") or state.get("trace_id") or state.get("session_id")
        if value:
            return str(value)
    runtime = _runtime(state)
    return runtime.trace_id or runtime.workspace.name


def _emit(state: RuntimeState | Mapping[str, Any], event: dict[str, Any]) -> None:
    runtime = _runtime(state)
    handler = runtime.event_handler
    if handler is not None:
        handler(dict(event))


_LINE_RE = re.compile(r"^- \[(?P<date>[^\]]+?) · (?P<source>[^\]]+)\] (?P<text>.*)$")


def _parse_line(line: str) -> MemoryEntry | None:
    match = _LINE_RE.match(line.strip())
    if not match:
        return None
    source_raw = match.group("source")
    source: Literal["user", "agent", "extracted"]
    run_id = None
    if source_raw == "user":
        source = "user"
    elif source_raw == "agent":
        source = "agent"
    elif source_raw.startswith("agent@"):
        source = "agent"
        run_id = source_raw.split("@", 1)[1] or None
    elif source_raw == "extracted":
        source = "extracted"
    elif source_raw.startswith("extracted@"):
        source = "extracted"
        run_id = source_raw.split("@", 1)[1] or None
    else:
        return None
    return MemoryEntry(
        date=match.group("date"),
        source=source,
        run_id=run_id,
        text=match.group("text"),
    )


def existing_entries(state: RuntimeState | Mapping[str, Any]) -> list[MemoryEntry]:
    path = memory_path(state)
    if not path.is_file():
        return []
    entries: list[MemoryEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = _parse_line(line)
        if entry is not None:
            entries.append(entry)
    return entries


def _render(entries: list[MemoryEntry]) -> str:
    return "\n".join(entry.to_line() for entry in entries) + ("\n" if entries else "")


def _over_limit(entries: list[MemoryEntry]) -> bool:
    rendered = _render(entries)
    return len(rendered.splitlines()) > MEMORY_MAX_LINES or len(rendered.encode("utf-8")) > MEMORY_MAX_BYTES


def _model(state: RuntimeState | Mapping[str, Any] | None = None) -> Any:
    if isinstance(state, Mapping) and state.get("model") is not None:
        return state["model"]
    from Linki.providers.openai_provider import create_model

    return create_model()


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def consolidate_with_model(
    entries: list[MemoryEntry],
    state: RuntimeState | Mapping[str, Any] | None = None,
) -> list[MemoryEntry]:
    numbered = "\n".join(f"{index + 1}. {entry.to_line()}" for index, entry in enumerate(entries))
    response = _model(state).invoke(
        [HumanMessage(content=f"{CONSOLIDATE_PROMPT}\n\nExisting memories:\n{numbered}")]
    )
    content = getattr(response, "content", "")
    payload = _json_from_text(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    raw_items = payload.get("memories") or []
    if not isinstance(raw_items, list):
        return entries

    compacted: list[MemoryEntry] = []
    for item in raw_items:
        text = str(item.get("text") if isinstance(item, Mapping) else item).strip()
        if text:
            compacted.append(MemoryEntry(date=_today(), source="extracted", run_id=None, text=text))
    return compacted or entries


def _write(
    state: RuntimeState | Mapping[str, Any],
    entries: list[MemoryEntry],
    *,
    added: int = 0,
) -> list[MemoryEntry]:
    if _over_limit(entries):
        before = len(entries)
        entries = consolidate_with_model(entries, state)
        _emit(state, {"type": "memory_consolidate", "before": before, "after": len(entries)})

    path = memory_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(entries), encoding="utf-8")
    _emit(state, {"type": "memory_write", "added": added, "total": len(entries)})
    return entries


def append_user_memory(state: RuntimeState | Mapping[str, Any], text: str) -> MemoryEntry:
    entry = MemoryEntry(date=_today(), source="user", run_id=None, text=text.strip())
    entries = [*existing_entries(state), entry]
    _write(state, entries, added=1)
    return entry


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


_STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "this",
    "that", "project", "use", "uses", "using", "should", "must", "always",
}
_DISTINCTIVE_TOPIC_WORDS = {
    "uv", "pip", "poetry", "redis", "postgres", "sqlite", "docker", "pnpm",
    "npm", "yarn", "pytest", "ruff", "black",
}


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w.-]+", _normalize_text(text))
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _same_topic(a: str, b: str) -> bool:
    a_words = _keywords(a)
    b_words = _keywords(b)
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    if overlap >= 1 and (a_words & b_words & _DISTINCTIVE_TOPIC_WORDS):
        return True
    return overlap >= 2 or overlap / max(min(len(a_words), len(b_words)), 1) >= 0.6


def _invalid_agent_memory_reason(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "memory text is empty"
    if len(stripped) > 500 or len(stripped.splitlines()) > 4:
        return "memory text must be short and self-contained"
    lowered = stripped.lower()
    if any(marker in lowered for marker in ("todo", "next step", "in progress", "completed task", "finished ")):
        return "do not save task progress or TODO state"
    if any(marker in lowered for marker in ("maybe", "might", "suspect", "hypothesis", "unverified", "guess")):
        return "do not save unverified hypotheses"
    if any(marker in lowered for marker in ("password", "token", "secret", "credential", "api key")):
        return "do not save secrets or credentials"
    return None


def _allow_user_replace(state: RuntimeState | Mapping[str, Any]) -> bool:
    return bool(isinstance(state, Mapping) and state.get("allow_user_memory_replace"))


def upsert_agent_memory(
    state: RuntimeState | Mapping[str, Any],
    text: str,
    replaces: int | None = None,
) -> dict[str, Any]:
    """Add or replace a durable memory from an agent-controlled tool."""

    reason = _invalid_agent_memory_reason(text)
    run_id = _run_id(state)
    if reason:
        result = {"status": "skipped", "index": None, "text": text.strip(), "reason": reason}
        _emit(state, {"type": "memory_agent_upsert", "action": "skipped", "index": None, "run_id": run_id, "reason": reason})
        return result

    entries = existing_entries(state)
    normalized = _normalize_text(text)

    if replaces is not None:
        if replaces < 1 or replaces > len(entries):
            reason = f"memory index out of range: {replaces}"
            _emit(state, {"type": "memory_agent_upsert", "action": "skipped", "index": replaces, "run_id": run_id, "reason": reason})
            return {"status": "skipped", "index": replaces, "text": text.strip(), "reason": reason}
        existing = entries[replaces - 1]
        if existing.source == "user" and not _allow_user_replace(state):
            reason = "cannot replace user-authored memory without explicit current-user correction"
            _emit(state, {"type": "memory_agent_upsert", "action": "skipped", "index": replaces, "run_id": run_id, "reason": reason})
            return {"status": "skipped", "index": replaces, "text": text.strip(), "reason": reason}
        entries[replaces - 1] = MemoryEntry(date=_today(), source="agent", run_id=run_id, text=text.strip())
        _write(state, entries, added=0)
        _emit(state, {"type": "memory_agent_upsert", "action": "replaced", "index": replaces, "run_id": run_id})
        return {"status": "replaced", "index": replaces, "text": text.strip()}

    for index, entry in enumerate(entries, start=1):
        if _normalize_text(entry.text) == normalized:
            _emit(state, {"type": "memory_agent_upsert", "action": "skipped", "index": index, "run_id": run_id, "reason": "duplicate"})
            return {"status": "skipped", "index": index, "text": text.strip(), "reason": "duplicate"}

    for index, entry in enumerate(entries, start=1):
        if _same_topic(entry.text, text):
            reason = "similar memory exists; specify replaces to update it"
            _emit(state, {"type": "memory_agent_upsert", "action": "skipped", "index": index, "run_id": run_id, "reason": reason})
            return {"status": "skipped", "index": index, "text": text.strip(), "reason": reason}

    entry = MemoryEntry(date=_today(), source="agent", run_id=run_id, text=text.strip())
    entries.append(entry)
    _write(state, entries, added=1)
    index = len(entries)
    _emit(state, {"type": "memory_agent_upsert", "action": "added", "index": index, "run_id": run_id})
    return {"status": "added", "index": index, "text": text.strip()}


def delete_memory(state: RuntimeState | Mapping[str, Any], index: int) -> int:
    entries = existing_entries(state)
    if index < 1 or index > len(entries):
        raise IndexError(f"memory index out of range: {index}")
    del entries[index - 1]
    _write(state, entries, added=0)
    return len(entries)


def _session_summary(state: RuntimeState | Mapping[str, Any]) -> str:
    if isinstance(state, Mapping):
        for key in ("session_summary", "final_answer", "last_actor_summary", "context_summary"):
            value = state.get(key)
            if value:
                return str(value)
    path = _runtime(state).workspace / SESSION_ROOT / SESSION_SUMMARY_FILE
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _merge_entries(
    existing: list[MemoryEntry],
    extracted: list[Mapping[str, Any]],
    *,
    run_id: str | None,
) -> tuple[list[MemoryEntry], int, int]:
    agent_texts = [entry.text for entry in existing if entry.source == "agent"]
    merged = list(existing)
    added = 0
    replaced = 0
    for item in extracted[:3]:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if any(_normalize_text(text) == _normalize_text(agent_text) or _same_topic(text, agent_text) for agent_text in agent_texts):
            continue
        entry = MemoryEntry(date=_today(), source="extracted", run_id=run_id, text=text)
        replaces = item.get("replaces")
        if isinstance(replaces, str) and replaces.isdigit():
            replaces = int(replaces)
        if isinstance(replaces, int) and 1 <= replaces <= len(merged):
            if merged[replaces - 1].source == "user":
                continue
            merged[replaces - 1] = entry
            replaced += 1
        elif replaces is None:
            merged.append(entry)
            added += 1
    return merged, added, replaced


def extract_run_memories(state: RuntimeState | Mapping[str, Any]) -> dict[str, int]:
    summary = _session_summary(state)
    if not summary.strip():
        return {"added": 0, "replaced": 0, "total": len(existing_entries(state))}

    entries = existing_entries(state)
    numbered = "\n".join(f"{index + 1}. {entry.to_line()}" for index, entry in enumerate(entries))
    response = _model(state).invoke(
        [
            HumanMessage(
                content=(
                    f"{EXTRACT_PROMPT}\n\nExisting entries:\n{numbered or '(none)'}"
                    f"\n\nSession summary:\n{summary}"
                )
            )
        ]
    )
    content = getattr(response, "content", "")
    payload = _json_from_text(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    raw_items = payload.get("memories") or []
    if not isinstance(raw_items, list) or not raw_items:
        return {"added": 0, "replaced": 0, "total": len(entries)}

    items = [item for item in raw_items if isinstance(item, Mapping)]
    merged, added, replaced = _merge_entries(entries, items, run_id=_run_id(state))
    if added or replaced:
        merged = _write(state, merged, added=added)
    return {"added": added, "replaced": replaced, "total": len(merged)}
