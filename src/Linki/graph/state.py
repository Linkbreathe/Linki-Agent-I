from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from Linki.core.state import RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class VerificationResult(TypedDict):
    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class VerificationCheck(TypedDict):
    name: str
    passed: bool
    detail: str


class LinkiGraphState(TypedDict, total=False):
    task: str
    runtime: RuntimeState
    provider: str
    model_name: str | None
    model: Any
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    verification_checks: list[VerificationCheck]
    passed: bool
    attempts: int
    max_attempts: int
    last_error: str
    last_actor_summary: str
    final_answer: str
