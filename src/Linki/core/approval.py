import re
from dataclasses import dataclass
from uuid import uuid4

RISK_PATTERNS = [
    (
        r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b",
        "Python package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+add\b",
        "Project dependency change with uv add",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+sync\b",
        "Dependency synchronization with uv sync",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b",
        "Python package installation with uv pip",
    ),
    (
        r"(?:^|&&|\|\||;)\s*npm\s+install\b",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*pnpm\s+install\b",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b",
        "Network download command",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uvicorn\b",
        "Long-running development server",
    ),
    (
        r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b",
        "Long-running development server",
    ),
]


def classify_command_risk(command: str) -> str | None:
    """Return the matching risk-reason string when the command matches a risky
    command pattern.

    Return None when the command is considered safe.
    """

    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, command):
            return reason

    return None


# Approval channel "kinds". The command path is the original BashTool flow; the
# question/plan kinds reuse the same handler/gate plumbing for proactive
# clarification and plan review.
KIND_COMMAND = "command"
KIND_QUESTION = "question"
KIND_PLAN = "plan"


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    command: str = ""
    risk_reason: str = ""
    tool_name: str = "BashTool"
    kind: str = KIND_COMMAND
    # Optional presentation prefix, e.g. "[job-2 · reviewer]" for a request that
    # originated inside a parallel dispatch job. Shown in the approval UI title.
    label: str = ""
    # Populated for kind == "question".
    question: str = ""
    options: tuple[str, ...] = ()
    allow_free_text: bool = True
    # Populated for kind == "plan".
    plan_text: str = ""


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""
    # Free-text answer for question kind, or reviewer feedback for plan kind.
    answer: str = ""


def new_approval_request(command: str, risk_reason: str, *, tool_name: str = "BashTool") -> ApprovalRequest:
    """Build a command ApprovalRequest with a freshly generated request id."""

    return ApprovalRequest(
        id=f"approval-{uuid4().hex[:8]}",
        command=command,
        risk_reason=risk_reason,
        tool_name=tool_name,
        kind=KIND_COMMAND,
    )


def new_question_request(
    question: str,
    options: list[str] | tuple[str, ...] | None = None,
    *,
    allow_free_text: bool = True,
) -> ApprovalRequest:
    """Build a clarifying-question request routed through the approval channel."""

    return ApprovalRequest(
        id=f"question-{uuid4().hex[:8]}",
        tool_name="AskUserQuestionTool",
        kind=KIND_QUESTION,
        question=question,
        options=tuple(options or ()),
        allow_free_text=allow_free_text,
    )


def new_plan_request(plan_text: str) -> ApprovalRequest:
    """Build a plan-review request routed through the approval channel."""

    return ApprovalRequest(
        id=f"plan-{uuid4().hex[:8]}",
        tool_name="PlanReview",
        kind=KIND_PLAN,
        plan_text=plan_text,
    )


VALID_APPROVAL_MODES = {
    "inline",
    "auto",
    "deny",
}


def normalize_approval_mode(mode: str | None) -> str:
    """Normalize the approval mode.

    The default mode is "inline". Missing or invalid values must also fall
    back to "inline".
    """

    if mode in VALID_APPROVAL_MODES:
        return mode
    return "inline"
