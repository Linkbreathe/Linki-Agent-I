from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, Input, Static

from Linki.cli.tui.approval import ApprovalGate, ApprovalModal, ApprovalRequestedMessage
from Linki.cli.tui.logo import animate_logo, build_logo
from Linki.core.agent import _parse_graph_event, stream_session_events
from Linki.core.approval import ApprovalDecision, ApprovalRequest
from Linki.core.session import load_or_create_session, resolve_session_workspace

# Rendering for each todo status: (icon, rich style).
TODO_RENDER = {
    "completed": ("✔", "bold green"),
    "in_progress": ("◐", "bold cyan"),
    "blocked": ("✗", "bold red"),
    "pending": ("○", "dim"),
}
PROGRESS_BAR_WIDTH = 18


class AgentEventMessage(Message):
    def __init__(self, event: dict) -> None:
        super().__init__()
        self.event = event


class LinkiTuiApp(App[None]):
    """
    Textual terminal interface for Linki multi-turn sessions.
    """

    TITLE = "Linki"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 46;
        min-width: 34;
        padding: 0 1;
        background: $panel;
        border-right: solid $accent;
    }

    #logo {
        height: 5;
        padding: 1 0 0 0;
    }

    #plan-wrap {
        height: 1fr;
        margin-top: 1;
        border: round $primary;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 1;
        background: $boost;
    }

    #session {
        height: auto;
        padding: 1 0 0 0;
        color: $text-muted;
    }

    #stream-col {
        width: 1fr;
        padding: 0 1;
    }

    #events {
        height: 1fr;
        border: round $panel;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 1;
    }

    #events Collapsible {
        padding-bottom: 0;
        background: transparent;
    }

    #events Collapsible.evt-user { background: $primary 25%; }
    #events Collapsible.evt-agent { background: $accent 18%; }
    #events Collapsible.evt-tool { background: $boost; }
    #events Collapsible.evt-plan { background: $warning 15%; }
    #events Collapsible.evt-success { background: $success 20%; }
    #events Collapsible.evt-error { background: $error 25%; }
    #events Collapsible.evt-handoff { background: $secondary 22%; }
    #events Collapsible.evt-system { background: $foreground 5%; }

    #final {
        min-height: 4;
        max-height: 10;
        margin-top: 1;
        border: round $success;
        border-title-color: $success;
        border-title-style: bold;
        padding: 0 1;
    }

    #input {
        height: 3;
        margin: 1 1 0 1;
        border: round $accent;
    }

    #approval-modal {
        align: center middle;
    }

    .approval-dialog {
        width: 80%;
        max-width: 100;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }

    .approval-title {
        text-style: bold;
        color: $warning;
    }

    .approval-command {
        border: solid $panel;
        padding: 1;
        margin: 1 0;
    }

    .approval-buttons {
        height: 3;
    }
    """

    def __init__(
        self,
        *,
        workspace: str | Path | None = None,
        provider: str = "openai",
        model_name: str | None = None,
        max_attempts: int = 3,
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
        trace_mode: str = "on",
        initial_task: str | None = None,
    ) -> None:
        super().__init__()
        self.theme = "nord"
        self.workspace = resolve_session_workspace(workspace)
        self.provider = provider
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.approval_mode = approval_mode
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self.initial_task = initial_task

        self._turn_thread: threading.Thread | None = None
        self._running_turn = False
        self._pending_gates: list[ApprovalGate] = []

        # Live plan state, kept in sync from streamed events.
        self._todos: list[dict[str, Any]] = []
        self._plan_summary: str = ""
        self._session_id: str = ""
        self._state_label: str = "starting"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static(build_logo(), id="logo")
                with VerticalScroll(id="plan-wrap"):
                    yield Static(self._render_plan(), id="plan")
                yield Static("", id="session")
            with Vertical(id="stream-col"):
                yield VerticalScroll(id="events")
                yield Static("Final answer will appear here.", id="final")
        yield Input(placeholder="Type a message for Linki", id="input")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#plan-wrap", VerticalScroll).border_title = "◐ Plan"
        self.query_one("#events", VerticalScroll).border_title = "▸ Activity"
        self.query_one("#final", Static).border_title = "✦ Final answer"

        session = load_or_create_session(self.workspace)
        self._session_id = str(session.get("session_id", ""))
        self._state_label = "idle"
        self._refresh_session_line()
        self._set_status(f"idle | workspace: {self.workspace}")

        logo = self.query_one("#logo", Static)
        await animate_logo(logo)
        if self.initial_task:
            self.call_later(self._start_turn, self.initial_task)

    def on_unmount(self) -> None:
        for gate in list(self._pending_gates):
            gate.resolve(approved=False, reason="TUI closed")

    def action_quit(self) -> None:
        for gate in list(self._pending_gates):
            gate.resolve(approved=False, reason="TUI closed")
        self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value or self._running_turn:
            return
        self.query_one("#input", Input).value = ""
        self._start_turn(value)

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        self._handle_event(message.event)

    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        def resolve(result: bool | None) -> None:
            approved = bool(result)
            message.gate.resolve(
                approved=approved,
                reason="approved via TUI" if approved else "denied via TUI",
            )
            if message.gate in self._pending_gates:
                self._pending_gates.remove(message.gate)

        self.push_screen(ApprovalModal(message.gate.request, message.workspace), resolve)

    def _set_status(self, text: str) -> None:
        self.sub_title = text

    def _set_state(self, label: str) -> None:
        self._state_label = label
        self._refresh_session_line()

    def _refresh_session_line(self) -> None:
        line = Text()
        line.append("● ", style="bold cyan")
        line.append(f"{self._state_label}\n", style="bold")
        short_id = self._session_id[:8] if self._session_id else "—"
        line.append(f"session {short_id}\n", style="dim")
        line.append(self.workspace.name, style="dim italic")
        self.query_one("#session", Static).update(line)

    def _write_event(self, summary: str, detail: Any = None, kind: str = "system") -> None:
        body = self._format_detail(detail) if detail is not None else summary
        collapsible = Collapsible(
            Static(Text(body)),
            title=summary,
            collapsed=True,
            classes=f"evt-{kind}",
        )
        events = self.query_one("#events", VerticalScroll)
        events.mount(collapsible)
        events.scroll_end(animate=False)

    def _refresh_plan(self) -> None:
        self.query_one("#plan", Static).update(self._render_plan())

    def _update_plan_from_node(self, data: dict[str, Any]) -> None:
        """Replace the plan with a full snapshot from a node update."""

        todos = data.get("todos")
        if isinstance(todos, list):
            self._todos = [dict(todo) for todo in todos if isinstance(todo, dict)]
        summary = data.get("plan_summary")
        if summary:
            self._plan_summary = str(summary)
        self._refresh_plan()

    def _sync_todos_from_tool(self, name: Any, result: Any) -> bool:
        """Update the live plan from a codeAgent/planner tool result.

        ``TodoWriteTool`` publishes/revises the whole plan; ``TodoUpdateTool``
        flips a single todo's status. Both stream long before the owning node
        finishes, so applying them here keeps the checklist continuously in
        sync instead of frozen until the node returns.
        """

        if not isinstance(result, dict):
            return False
        output = result.get("output")
        if not isinstance(output, dict):
            return False

        if name == "TodoWriteTool":
            plan = output.get("plan")
            if not isinstance(plan, dict):
                return False
            todos = plan.get("todos")
            if isinstance(todos, list):
                self._todos = [dict(todo) for todo in todos if isinstance(todo, dict)]
            if plan.get("plan_summary"):
                self._plan_summary = str(plan.get("plan_summary"))
            return True

        if name == "TodoUpdateTool":
            updated = output.get("updated")
            if isinstance(updated, dict):
                self._patch_todo(updated)
                return True

        return False

    def _patch_todo(self, updated: dict[str, Any]) -> None:
        todo_id = str(updated.get("id") or "")
        for todo in self._todos:
            if str(todo.get("id")) == todo_id:
                if updated.get("status"):
                    todo["status"] = updated["status"]
                if updated.get("content"):
                    todo["content"] = updated["content"]
                todo["note"] = updated.get("note", todo.get("note", ""))
                return
        self._todos.append(dict(updated))

    def _render_plan(self) -> Text:
        todos = self._todos
        text = Text()

        total = len(todos)
        done = sum(1 for todo in todos if todo.get("status") == "completed")
        filled = round((done / total) * PROGRESS_BAR_WIDTH) if total else 0
        bar_style = "bold green" if total and done == total else "bold cyan"
        text.append("▓" * filled, style=bar_style)
        text.append("░" * (PROGRESS_BAR_WIDTH - filled), style="dim")
        text.append(f"  {done}/{total}\n", style="bold" if total else "dim")

        if self._plan_summary:
            text.append(f"{self._plan_summary}\n", style="italic dim")
        text.append("\n")

        if not todos:
            text.append("Waiting for a plan…", style="dim italic")
            return text

        for todo in todos:
            status = str(todo.get("status") or "pending")
            icon, style = TODO_RENDER.get(status, TODO_RENDER["pending"])
            content = str(todo.get("content") or todo.get("id") or "")
            text.append(f"{icon} ", style=style)
            text.append(f"{content}\n", style=style if status != "pending" else "")
            if status == "blocked" and todo.get("note"):
                text.append(f"    ↳ {self._truncate(str(todo['note']), 80)}\n", style="dim red")
        return text

    @staticmethod
    def _truncate(text: str, limit: int = 200) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 1] + "…"

    @staticmethod
    def _format_detail(detail: Any) -> str:
        try:
            return json.dumps(detail, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(detail)

    def _start_turn(self, task: str) -> None:
        if self._running_turn:
            return
        self._running_turn = True
        self.query_one("#input", Input).disabled = True
        self.query_one("#final", Static).update("Running...")
        self._write_event(f"💬 User: {task}", detail={"task": task}, kind="user")
        self._set_state("running")
        self._set_status(f"running | workspace: {self.workspace}")

        self._turn_thread = threading.Thread(
            target=self._run_turn,
            args=(task,),
            daemon=True,
        )
        self._turn_thread.start()

    def _post_agent_event(self, event: dict[str, Any]) -> None:
        try:
            self.call_from_thread(self.post_message, AgentEventMessage(event))
        except RuntimeError:
            return

    def _approval_handler(self, request: ApprovalRequest) -> ApprovalDecision:
        gate = ApprovalGate(request)
        self._pending_gates.append(gate)
        try:
            self.call_from_thread(
                self.post_message,
                ApprovalRequestedMessage(gate=gate, workspace=self.workspace),
            )
        except RuntimeError:
            gate.resolve(approved=False, reason="TUI closed")
        return gate.wait()

    def _run_turn(self, task: str) -> None:
        try:
            for event in stream_session_events(
                task,
                session_workspace=self.workspace,
                max_attempts=self.max_attempts,
                approval_mode=self.approval_mode,
                approval_handler=self._approval_handler if self.approval_mode == "inline" else None,
                checkpoint_mode=self.checkpoint_mode,
                trace_mode=self.trace_mode,
                provider=self.provider,
                model_name=self.model_name,
            ):
                self._post_agent_event(event)
        except Exception as exc:
            self._post_agent_event(
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            self._post_agent_event({"type": "turn_finished"})

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "graph_event":
            for parsed in _parse_graph_event(("updates", event.get("event", {}))):
                self._handle_event(parsed)
            return

        if event_type == "custom_event":
            inner = event.get("event")
            if isinstance(inner, dict):
                self._handle_event(inner)
            return

        if event_type == "node_update":
            self._handle_node_update(event)
            return

        if event_type == "intent_route":
            self._write_event(
                f"🧭 Route: {event.get('route')} ({event.get('confidence')})", detail=event, kind="system"
            )
            return

        if event_type == "ai_message":
            content = str(event.get("content") or "").strip()
            if not content:
                return
            node = event.get("node")
            label = f"💭 {node}" if node else "💭 Agent"
            self._write_event(f"{label} → {self._truncate(content)}", detail=event, kind="agent")
            return

        if event_type == "tool_call":
            name = event.get("name")
            args = event.get("args") or {}
            detail = self._tool_detail(name, args)
            self._write_event(
                f"🔧 {name} → {detail}" if detail else f"🔧 {name}", detail=event, kind="tool"
            )
            return

        if event_type == "tool_result":
            result = event.get("result") or {}
            if self._sync_todos_from_tool(event.get("name"), result):
                self._refresh_plan()
            ok = self._tool_ok(result)
            self._write_event(
                "✅ Tool completed" if ok else "❌ Tool failed",
                detail=event,
                kind="success" if ok else "error",
            )
            return

        if event_type == "interrupted":
            self._write_event(
                f"⏸ Interrupted — resume with: {event.get('resume_command', '')}", detail=event, kind="error"
            )
            self.query_one("#final", Static).update("Run interrupted. Resume from the last checkpoint.")
            return

        if event_type == "handoff":
            self._write_event(
                f"🔄 Handoff: {event.get('from')} → {event.get('to')}", detail=event, kind="handoff"
            )
            return

        if event_type == "search_results":
            self._write_event(f"🔍 Search results received: {event.get('query')}", detail=event, kind="tool")
            return

        if event_type == "checkpoint_saved":
            self._write_event(f"💾 Checkpoint saved: {event.get('status')}", detail=event, kind="system")
            return

        if event_type == "session_saved":
            self._session_id = str(event.get("session_id") or self._session_id)
            self._refresh_session_line()
            self._set_status(f"session: {self._session_id[:8]} | turn: {event.get('turn_index')}")
            return

        if event_type == "trace_finished":
            trace = event.get("trace") or {}
            self._write_event(f"Trace finished: {trace.get('trace_id', '')}", detail=trace, kind="system")
            return

        if event_type == "final_answer":
            content = str(event.get("content") or "")
            route = event.get("route")
            prefix = f"💬 Final answer ({route})" if route else "💬 Final answer"
            self._write_event(prefix, detail=event, kind="success")
            self.query_one("#final", Static).update(content or "No final answer.")
            return

        if event_type == "error":
            self._write_event(f"❌ {event.get('error_type')}: {event.get('error')}", detail=event, kind="error")
            self.query_one("#final", Static).update(str(event.get("error") or "Error"))
            return

        if event_type == "turn_finished":
            self._running_turn = False
            self.query_one("#input", Input).disabled = False
            self.query_one("#input", Input).focus()
            self._set_state("idle")
            self._set_status(f"idle | workspace: {self.workspace}")
            return

    def _handle_node_update(self, event: dict[str, Any]) -> None:
        node = event.get("node")
        data = event.get("data") or {}

        if "todos" in data:
            self._update_plan_from_node(data)

        if node == "intent_router":
            self._write_event(f"🧭 Route: {data.get('intent_route')}", detail=data, kind="system")
            return

        if node == "chat_responder":
            self._write_event("💬 Chat response generated", detail=data, kind="agent")
            return

        if node == "planner":
            self._write_event("📋 Plan updated", detail=data, kind="plan")
            return

        if node == "context_compressor":
            self._write_event("🧠 Context compressed", detail=data, kind="system")
            return

        if node == "verifier":
            passed = bool(data.get("passed"))
            self._write_event(
                "✅ Verifier passed" if passed else "❌ Verifier failed",
                detail=data,
                kind="success" if passed else "error",
            )
            return

        if node == "final":
            content = str(data.get("final_answer") or "")
            self.query_one("#final", Static).update(content)
            return

    @staticmethod
    def _tool_detail(name: Any, args: Any) -> str:
        if not isinstance(args, dict):
            return ""
        if name == "BashTool":
            return str(args.get("command") or "")[:160]
        if args.get("file_path"):
            return str(args.get("file_path"))[:160]
        if args.get("query"):
            return str(args.get("query"))[:160]
        if args.get("pattern"):
            return str(args.get("pattern"))[:160]
        return str(args)[:160]

    @staticmethod
    def _tool_ok(result: Any) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("ok") is False:
            return False
        output = result.get("output")
        if isinstance(output, str):
            return '"ok": false' not in output and '"ok":false' not in output
        if isinstance(output, dict):
            return output.get("ok") is not False
        return True
