"""SkillTool: progressive disclosure of skill instructions.

The planner, codeAgent, and any subagent that whitelists ``SkillTool`` see only
each skill's one-line description in the ``<available_skills>`` prompt block.
When a task matches a description they call this tool to pull the skill's full
body into the conversation. The tool is lazy (body is only sent on demand),
deduplicated per run (a second call for the same skill returns a short notice
instead of re-sending the body), and forgiving (an unknown name returns the
available list rather than raising).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from Linki.core.state import RuntimeState
from Linki.skills.registry import SkillSpec, load_skill_registry
from Linki.tools.registry import SKILL_TOOL_NAME


class SkillToolInput(BaseModel):
    name: str = Field(description="Name of the skill whose full instructions to load.")


def _runtime(state: Any) -> RuntimeState | None:
    if isinstance(state, RuntimeState):
        return state
    if isinstance(state, Mapping):
        return state.get("runtime")
    return None


def _resolve_sink(runtime: RuntimeState | None):
    """Resolve where events are written: the runtime handler, or the LangGraph
    stream writer when running inside a graph node."""

    if runtime is not None and runtime.event_handler is not None:
        return runtime.event_handler
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (ImportError, RuntimeError, KeyError):
        return None


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for the skill_load trace event."""

    return max(1, len(text) // 4)


def _registry(runtime: RuntimeState | None) -> dict[str, SkillSpec]:
    if runtime is None:
        return {}
    # Prefer the run-start catalog; fall back to a fresh load when empty so the
    # tool still works in contexts that never populated runtime.skills.
    if runtime.skills:
        return dict(runtime.skills)
    return load_skill_registry(runtime)


def make_skill_tool(state: Any) -> StructuredTool:
    """Build the SkillTool bound to ``state``'s runtime."""

    runtime = _runtime(state)

    def skill_tool(name: str) -> dict[str, Any]:
        registry = _registry(runtime)
        spec = registry.get(name)
        if spec is None:
            available = ", ".join(sorted(registry)) or "(none)"
            return {
                "ok": True,
                "name": SKILL_TOOL_NAME,
                "output": f"Unknown skill: {name}. Available skills: {available}",
            }

        loaded = runtime.loaded_skills if runtime is not None else set()
        if name in loaded:
            return {
                "ok": True,
                "name": SKILL_TOOL_NAME,
                "skill": name,
                "output": f"Skill '{name}' already loaded earlier this run; its instructions are in context.",
            }

        loaded.add(name)

        sink = _resolve_sink(runtime)
        if sink is not None:
            sink({"type": "skill_load", "name": name, "tokens": _estimate_tokens(spec.body)})

        return {
            "ok": True,
            "name": SKILL_TOOL_NAME,
            "skill": name,
            "output": spec.body,
        }

    return StructuredTool.from_function(
        func=skill_tool,
        name=SKILL_TOOL_NAME,
        description=(
            "Load the full instructions for a named skill. Call this before "
            "performing a task that matches a skill's description in "
            "<available_skills>. The instructions are only sent once per run."
        ),
        args_schema=SkillToolInput,
    )
