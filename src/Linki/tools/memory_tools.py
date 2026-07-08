"""Dedicated persistent memory tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from Linki.core.memory_store import upsert_agent_memory


class MemoryUpsertInput(BaseModel):
    text: str = Field(description="Short, self-contained durable fact or rule to save.")
    replaces: int | None = Field(default=None, description="Existing memory number to replace, or null.")


def make_memory_upsert_tool(state: Any) -> StructuredTool:
    def memory_upsert_tool(text: str, replaces: int | None = None) -> dict[str, Any]:
        return upsert_agent_memory(state, text, replaces=replaces)

    return StructuredTool.from_function(
        func=memory_upsert_tool,
        name="MemoryUpsertTool",
        description=(
            "Save or update a short persistent project memory. Use only for stable "
            "future-facing facts, preferences, conventions, or verified lessons. "
            "Do not save TODOs, task progress, guesses, secrets, or long summaries."
        ),
        args_schema=MemoryUpsertInput,
    )
