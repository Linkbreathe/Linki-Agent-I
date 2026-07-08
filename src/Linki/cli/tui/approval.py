from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from Linki.core.approval import KIND_PLAN, KIND_QUESTION, ApprovalDecision, ApprovalRequest


class ApprovalGate:
    """
    Synchronize approval between a tool worker thread and the TUI.

    Reused across approval kinds: command approval (approved/denied), clarifying
    questions (a free-text/selected ``answer``), and plan review (approved plus
    optional reviewer ``answer`` feedback).
    """

    def __init__(self, request: ApprovalRequest):
        self.request = request
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._decision: ApprovalDecision | None = None

    def resolve(
        self,
        *,
        approved: bool,
        reason: str = "",
        answer: str = "",
    ) -> None:
        with self._lock:
            if self._decision is not None:
                return
            self._decision = ApprovalDecision(
                approved=approved,
                reason=reason or ("approved via TUI" if approved else "denied via TUI"),
                answer=answer,
            )
            self._event.set()

    def wait(self) -> ApprovalDecision:
        self._event.wait()
        with self._lock:
            if self._decision is None:
                return ApprovalDecision(approved=False, reason="approval gate closed without decision")
            return self._decision


class ApprovalRequestedMessage(Message):
    def __init__(
        self,
        gate: ApprovalGate,
        workspace: Path,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.workspace = workspace


class ApprovalModal(ModalScreen[object]):
    """
    Display an approval prompt whose layout depends on the request ``kind``.

    - command: risk reason + command with Approve/Deny buttons (dismisses bool).
    - question: question + option buttons + optional free-text (dismisses str).
    - plan: multi-line plan + feedback box + Approve/Reject (dismisses dict).
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, request: ApprovalRequest, workspace: Path) -> None:
        super().__init__()
        self.request = request
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        if self.request.kind == KIND_QUESTION:
            yield from self._compose_question()
        elif self.request.kind == KIND_PLAN:
            yield from self._compose_plan()
        else:
            yield from self._compose_command()

    def _compose_command(self) -> ComposeResult:
        yield Container(
            Vertical(
                Static("Approval Required", classes="approval-title"),
                Static(f"Tool: {self.request.tool_name}"),
                Static(f"Risk reason: {self.request.risk_reason}"),
                Static(f"Workspace: {self.workspace}"),
                Static(self.request.command, classes="approval-command"),
                Horizontal(
                    Button("Approve", variant="success", id="approve"),
                    Button("Deny", variant="error", id="deny"),
                    classes="approval-buttons",
                ),
                classes="approval-dialog",
            ),
            id="approval-modal",
        )

    def _compose_question(self) -> ComposeResult:
        option_buttons = [
            Button(option, variant="primary", id=f"opt-{index}")
            for index, option in enumerate(self.request.options)
        ]
        children: list[Any] = [
            Static("Clarifying Question", classes="approval-title"),
            Static(self.request.question, classes="approval-command"),
        ]
        if option_buttons:
            children.append(Vertical(*option_buttons, classes="approval-options"))
        if self.request.allow_free_text:
            children.append(
                Input(placeholder="Type an answer and press Enter…", id="free-text")
            )
        children.append(
            Horizontal(
                Button("Skip", variant="default", id="deny"),
                classes="approval-buttons",
            )
        )
        yield Container(
            Vertical(*children, classes="approval-dialog"),
            id="approval-modal",
        )

    def _compose_plan(self) -> ComposeResult:
        yield Container(
            Vertical(
                Static("Plan Review", classes="approval-title"),
                VerticalScroll(
                    Static(self.request.plan_text or "—"),
                    classes="approval-plan",
                ),
                Input(placeholder="Feedback (required to reject)…", id="feedback"),
                Horizontal(
                    Button("Approve", variant="success", id="plan-approve"),
                    Button("Reject", variant="error", id="plan-reject"),
                    classes="approval-buttons",
                ),
                classes="approval-dialog",
            ),
            id="approval-modal",
        )

    def _feedback_value(self) -> str:
        try:
            return self.query_one("#feedback", Input).value.strip()
        except Exception:
            return ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "approve":
            self.dismiss(True)
        elif button_id == "deny":
            # Deny (command) / Skip (question) both cancel with no answer.
            self.dismiss(None)
        elif button_id.startswith("opt-"):
            index = int(button_id.split("-", 1)[1])
            self.dismiss(self.request.options[index])
        elif button_id == "plan-approve":
            self.dismiss({"approved": True, "feedback": self._feedback_value()})
        elif button_id == "plan-reject":
            self.dismiss({"approved": False, "feedback": self._feedback_value()})

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the question free-text box submits the typed answer. The plan
        # feedback box requires an explicit Approve/Reject button press.
        if self.request.kind == KIND_QUESTION and event.input.id == "free-text":
            self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)
