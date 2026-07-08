from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from Linki.core.state import RuntimeState
from Linki.graph.memory import CompressionEvent, LayeredMemory


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
    project_context: str
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
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    history_summary: str
    intent_route: str
    intent_reason: str
    intent_confidence: float
    chat_response: str
    session_id: str
    session_turn: int
    session_context: str
    # Proactive clarification / plan-review controls.
    ask_budget: int  # remaining human questions this run (initialized to 2)
    plan_mode: bool
    pre_plan_approval_mode: str | None
    plan_feedback: str | None
