from __future__ import annotations

import threading
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from Linki.core.approval import ApprovalDecision, ApprovalRequest


class ApprovalGate:
    """
    Synchronize approval between the BashTool worker thread and the TUI.
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
    ) -> None:
        with self._lock:
            if self._decision is not None:
                return
            self._decision = ApprovalDecision(
                approved=approved,
                reason=reason or ("approved via TUI" if approved else "denied via TUI"),
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


class ApprovalModal(ModalScreen[bool]):
    """
    Display a command approval prompt.
    """

    BINDINGS = [
        ("y", "approve", "Approve"),
        ("enter", "approve", "Approve"),
        ("n", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest, workspace: Path) -> None:
        super().__init__()
        self.request = request
        self.workspace = workspace

    def compose(self) -> ComposeResult:
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
