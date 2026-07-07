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


class SourceItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float


class AgentHandoff(TypedDict, total=False):
    from_agent: str
    to_agent: str
    instruction: str
    result: str


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
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    passed: bool
    attempts: int
    max_attempts: int
    last_error: str
    last_actor_summary: str
    final_answer: str
