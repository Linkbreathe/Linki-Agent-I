from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from Linki.cli.tui.approval import ApprovalGate, ApprovalModal, ApprovalRequestedMessage
from Linki.cli.tui.logo import animate_logo, build_logo
from Linki.core.agent import _parse_graph_event, stream_session_events
from Linki.core.approval import KIND_PLAN, KIND_QUESTION, ApprovalDecision, ApprovalRequest
from Linki.core.compact import compact_pipeline
from Linki.core.memory_store import append_user_memory, delete_memory, existing_entries
from Linki.core.session import load_or_create_session, resolve_session_workspace
from Linki.core.state import create_runtime

# Rendering for each todo status: (icon, rich style).
TODO_RENDER = {
    "completed": ("✔", "bold green"),
    "in_progress": ("◐", "bold cyan"),
    "blocked": ("✗", "bold red"),
    "pending": ("○", "dim"),
}
PROGRESS_BAR_WIDTH = 18
CTX_BAR_WIDTH = 12
CTX_TOKEN_LIMIT_DEFAULT = 400_000

# Sidebar state-dot style per running state.
STATE_STYLES = {
    "starting": "bold cyan",
    "idle": "bold green",
    "running": "bold yellow",
}

# Slash commands offered by the input autocomplete: (name, description).
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/plan", "Run this turn in plan mode"),
    ("/compact", "Compress retained context now"),
    ("/memory", "List or remove saved memory"),
    ("/resume", "Resume the latest checkpoint"),
]


def _matching_commands(value: str) -> list[tuple[str, str]]:
    """Return slash commands matching the input while the command token is typed.

    Suggestions are only offered before the first space (i.e. while the command
    itself is being typed, not its arguments).
    """

    if not value.startswith("/") or " " in value:
        return []
    return [(name, desc) for name, desc in SLASH_COMMANDS if name.startswith(value)]


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
    #events Collapsible.evt-subagent { background: $secondary 14%; border-left: solid $secondary; }
    #events Collapsible.evt-hook { background: $warning 10%; border-left: solid $warning; }
    #events Collapsible.evt-system { background: $foreground 5%; }

    #final {
        min-height: 4;
        max-height: 16;
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

    #cmd-menu {
        height: auto;
        max-height: 6;
        margin: 0 1;
        border: round $accent;
        background: $panel;
        display: none;
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

    .approval-options {
        height: auto;
        margin: 1 0;
    }

    .approval-options Button {
        width: 100%;
        margin-bottom: 1;
    }

    .approval-plan {
        height: 12;
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
        plan_mode: bool = False,
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
        self.plan_mode = plan_mode

        self._turn_thread: threading.Thread | None = None
        self._running_turn = False
        self._pending_gates: list[ApprovalGate] = []

        # Live plan state, kept in sync from streamed events.
        self._todos: list[dict[str, Any]] = []
        self._plan_summary: str = ""
        self._session_id: str = ""
        self._state_label: str = "starting"
        self._last_messages: list[Any] = []

        # Cockpit status, kept in sync from streamed events.
        self._turn_index: int | None = None
        self._active_agent: str = ""
        self._attempts: int = 0
        self._max_attempts_seen: int = 0
        self._verify_passed: bool | None = None
        self._ctx_tokens: int = 0
        self._ctx_limit: int = CTX_TOKEN_LIMIT_DEFAULT

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
                with VerticalScroll(id="final"):
                    yield Static("Final answer will appear here.", id="final-body")
        yield Input(placeholder="Type a message for Linki", id="input")
        yield OptionList(id="cmd-menu")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#plan-wrap", VerticalScroll).border_title = "◐ Plan"
        self.query_one("#events", VerticalScroll).border_title = "▸ Activity"
        self.query_one("#final", VerticalScroll).border_title = "✦ Final answer"
        self.query_one("#cmd-menu", OptionList).display = False

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

    # --- Slash-command autocomplete ------------------------------------------

    def _cmd_menu(self) -> OptionList:
        return self.query_one("#cmd-menu", OptionList)

    def _cmd_menu_visible(self) -> bool:
        try:
            return bool(self._cmd_menu().display)
        except Exception:
            return False

    def _hide_cmd_menu(self) -> None:
        self._cmd_menu().display = False

    def _fill_command(self, name: str) -> None:
        """Complete the input with a command name and keep typing arguments."""

        input_widget = self.query_one("#input", Input)
        input_widget.value = f"{name} "
        input_widget.cursor_position = len(input_widget.value)
        self._hide_cmd_menu()
        input_widget.focus()

    def _accept_highlighted_command(self) -> None:
        menu = self._cmd_menu()
        index = menu.highlighted
        if index is None:
            self._hide_cmd_menu()
            return
        option = menu.get_option_at_index(index)
        self._fill_command(option.id or str(option.prompt))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "input":
            return
        matches = _matching_commands(event.value)
        menu = self._cmd_menu()
        if not matches:
            menu.display = False
            return
        menu.clear_options()
        menu.add_options([Option(f"{name}   {desc}", id=name) for name, desc in matches])
        menu.highlighted = 0
        menu.display = True

    def on_key(self, event: events.Key) -> None:
        # Only intercept navigation while the suggestion menu is open; Enter is
        # handled in on_input_submitted so it can also start a turn.
        if not self._cmd_menu_visible():
            return
        menu = self._cmd_menu()
        if event.key == "down":
            menu.action_cursor_down()
            event.stop()
            event.prevent_default()
        elif event.key == "up":
            menu.action_cursor_up()
            event.stop()
            event.prevent_default()
        elif event.key == "tab":
            self._accept_highlighted_command()
            event.stop()
            event.prevent_default()
        elif event.key == "escape":
            self._hide_cmd_menu()
            event.stop()
            event.prevent_default()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "cmd-menu":
            return
        self._fill_command(event.option.id or str(event.option.prompt))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter while the menu is open accepts the highlighted command instead of
        # submitting a turn.
        if self._cmd_menu_visible():
            self._accept_highlighted_command()
            return

        value = event.value.strip()
        if not value or self._running_turn:
            return
        self.query_one("#input", Input).value = ""

        if value.startswith("# "):
            text = value[2:].strip()
            if text:
                runtime = create_runtime(self.workspace)
                append_user_memory(runtime, text)
                self._write_event(f"✏️ 已写入记忆（user）：{text}", kind="system")
            return

        if value.startswith("/"):
            if self._handle_slash_command(value):
                return
            available = ", ".join(name for name, _ in SLASH_COMMANDS)
            self._write_event(f"Unknown command: {value.split()[0]} · available: {available}", kind="error")
            return

        self._start_turn(value, plan_mode=self.plan_mode)

    def _handle_slash_command(self, value: str) -> bool:
        if value == "/plan" or value.startswith("/plan "):
            task = value[len("/plan"):].strip()
            if not task:
                self._write_event("📝 Plan mode armed — type your task and press Enter.", kind="system")
                return True
            self._start_turn(task, plan_mode=True)
            return True

        if value == "/resume" or value.startswith("/resume"):
            self._write_event(
                "ℹ️ /resume is a startup flag — restart with: linki --resume <workspace>",
                kind="system",
            )
            return True

        if value == "/compact" or value.startswith("/compact "):
            focus = value[len("/compact"):].strip() or None
            self._run_manual_compact(focus)
            return True

        if value == "/memory":
            self._show_memory()
            return True

        if value.startswith("/memory rm "):
            raw_index = value[len("/memory rm "):].strip()
            try:
                remaining = delete_memory(create_runtime(self.workspace), int(raw_index))
            except (ValueError, IndexError) as exc:
                self._write_event(f"❌ Memory remove failed: {exc}", kind="error")
                return True
            self._write_event(f"🧠 记忆已删除，剩余 {remaining} 条", kind="system")
            return True

        return False

    def _show_memory(self) -> None:
        entries = existing_entries(create_runtime(self.workspace))
        if not entries:
            self._write_event("🧠 记忆为空", kind="system")
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Date", no_wrap=True)
        table.add_column("Source", no_wrap=True)
        table.add_column("Memory", overflow="fold")
        for index, entry in enumerate(entries, start=1):
            source = entry.source if entry.source == "user" or not entry.run_id else f"{entry.source}@{entry.run_id}"
            table.add_row(str(index), entry.date, source, entry.text)
        self._write_event("🧠 记忆", detail=table, kind="system")

    def _run_manual_compact(self, focus: str | None) -> None:
        events: list[dict[str, Any]] = []
        runtime = create_runtime(
            self.workspace,
            approval_mode=self.approval_mode,
            checkpoint_mode=self.checkpoint_mode,
            trace_mode=self.trace_mode,
            event_handler=events.append,
        )
        state = {
            "runtime": runtime,
            "messages": self._last_messages,
            "acceptance_criteria": [],
            "compression_events": [],
            "context_token_limit": 400_000,
            "context_token_count": 0,
        }
        updates = compact_pipeline(state, focus=focus, trigger="manual")
        messages = updates.get("messages")
        if isinstance(messages, list):
            self._last_messages = messages
        for event in events:
            self._handle_event(event)
        token_count = updates.get("context_token_count", 0)
        retained = 0
        compression_events = updates.get("compression_events") or []
        if compression_events:
            retained = int(compression_events[-1].get("tail_messages") or 0)
        self._write_event(
            f"🗜 Compact complete · tokens: {token_count} · retained: {retained}",
            detail=updates,
            kind="system",
        )

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        self._handle_event(message.event)

    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        kind = message.gate.request.kind

        def resolve(result: Any) -> None:
            if kind == KIND_QUESTION:
                answer = str(result or "")
                message.gate.resolve(
                    approved=True,
                    reason="answered via TUI" if answer else "skipped via TUI",
                    answer=answer,
                )
            elif kind == KIND_PLAN:
                data = result if isinstance(result, dict) else {}
                approved = bool(data.get("approved"))
                message.gate.resolve(
                    approved=approved,
                    reason="plan approved via TUI" if approved else "plan rejected via TUI",
                    answer=str(data.get("feedback") or ""),
                )
            else:
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
        line.append("● ", style=STATE_STYLES.get(self._state_label, "bold cyan"))
        line.append(self._state_label, style="bold")
        if self._turn_index is not None:
            line.append(f" · turn {self._turn_index}", style="dim")
        line.append("\n")
        short_id = self._session_id[:8] if self._session_id else "—"
        line.append(f"session {short_id} · ", style="dim")
        line.append(f"{self.workspace.name}\n", style="dim italic")
        line.append(f"{self.provider} · {self.model_name or 'default model'}\n", style="dim")
        if self._active_agent:
            line.append("🤖 ", style="bold")
            line.append(f"{self._active_agent}\n", style="bold magenta")
        if self._verify_passed is not None or self._attempts:
            icon, style = ("✔", "bold green") if self._verify_passed else ("⟳", "bold yellow")
            budget = f"/{self._max_attempts_seen}" if self._max_attempts_seen else ""
            line.append(f"{icon} verify {self._attempts}{budget}\n", style=style)
        if self._ctx_tokens and self._ctx_limit:
            used = min(self._ctx_tokens / self._ctx_limit, 1.0)
            filled = round(used * CTX_BAR_WIDTH)
            bar_style = "bold red" if used > 0.85 else "bold cyan"
            line.append("ctx ", style="dim")
            line.append("▓" * filled, style=bar_style)
            line.append("░" * (CTX_BAR_WIDTH - filled), style="dim")
            line.append(
                f" {self._fmt_tokens(self._ctx_tokens)}/{self._fmt_tokens(self._ctx_limit)}",
                style="dim",
            )
        self.query_one("#session", Static).update(line)

    def _sync_status_from_node(self, data: dict[str, Any]) -> None:
        """Pick up cockpit status (retry loop, context budget) from any node update."""

        changed = False
        if isinstance(data.get("attempts"), int):
            self._attempts = data["attempts"]
            changed = True
        if isinstance(data.get("max_attempts"), int):
            self._max_attempts_seen = data["max_attempts"]
            changed = True
        if isinstance(data.get("passed"), bool):
            self._verify_passed = data["passed"]
            changed = True
        if isinstance(data.get("context_token_count"), int):
            self._ctx_tokens = data["context_token_count"]
            changed = True
        if isinstance(data.get("context_token_limit"), int):
            self._ctx_limit = data["context_token_limit"]
            changed = True
        if changed:
            self._refresh_session_line()

    def _set_active_agent(self, agent: str) -> None:
        self._active_agent = agent
        self._refresh_session_line()

    @staticmethod
    def _fmt_tokens(count: int) -> str:
        if count < 1000:
            return str(count)
        if count % 1000 == 0:
            return f"{count // 1000}k"
        return f"{count / 1000:.1f}k"

    def _write_event(self, summary: str, detail: Any = None, kind: str = "system") -> None:
        if detail is not None and hasattr(detail, "__rich_console__"):
            renderable = detail
        else:
            body = self._format_detail(detail) if detail is not None else summary
            renderable = Text(body)
        collapsible = Collapsible(
            Static(renderable),
            title=summary,
            collapsed=True,
            classes=f"evt-{kind}",
        )
        events = self.query_one("#events", VerticalScroll)
        events.mount(collapsible)
        events.scroll_end(animate=False)

    def _subagent_prefix(self, event: dict[str, Any]) -> str:
        agent = event.get("agent")
        return f"   {agent} · " if agent else ""

    def _event_kind(self, event: dict[str, Any], fallback: str) -> str:
        return "subagent" if event.get("agent") and fallback in {"tool", "success"} else fallback

    @staticmethod
    def _hook_decision_label(event: dict[str, Any]) -> tuple[str, str]:
        decision = str(event.get("decision") or "allow")
        if decision == "deny":
            return "⛔ Hook denied", "error"
        if decision == "ask":
            return "⚠ Hook escalated", "hook"
        return "🪝 Hook allowed", "hook"

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
        if isinstance(detail, str):
            return detail
        try:
            return json.dumps(detail, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(detail)

    def _start_turn(self, task: str, plan_mode: bool | None = None) -> None:
        if self._running_turn:
            return
        turn_plan_mode = self.plan_mode if plan_mode is None else plan_mode
        self._running_turn = True
        self.query_one("#input", Input).disabled = True
        self.query_one("#final-body", Static).update("Running...")
        label = f"💬 User: {task}" + ("  · plan mode" if turn_plan_mode else "")
        self._write_event(label, detail={"task": task, "plan_mode": turn_plan_mode}, kind="user")
        self._set_state("running")
        self._set_status(f"running | workspace: {self.workspace}")

        self._turn_thread = threading.Thread(
            target=self._run_turn,
            args=(task, turn_plan_mode),
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

    def _run_turn(self, task: str, plan_mode: bool = False) -> None:
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
                plan_mode=plan_mode,
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
            prefix = self._subagent_prefix(event)
            self._write_event(
                f"{prefix}🔧 {name} → {detail}" if detail else f"{prefix}🔧 {name}",
                detail=event,
                kind=self._event_kind(event, "tool"),
            )
            return

        if event_type == "tool_result":
            result = event.get("result") or {}
            if self._sync_todos_from_tool(event.get("name"), result):
                self._refresh_plan()
            ok = self._tool_ok(result)
            prefix = self._subagent_prefix(event)
            name = event.get("name") or "Tool"
            if name == "AgentTool" and isinstance(result.get("output"), dict):
                output = result["output"]
                subagent = output.get("subagent_type") or "subagent"
                dispatch = output.get("description") or ""
                self._write_event(
                    f"{'✅' if ok else '❌'} AgentTool → {subagent}" + (f" · {dispatch}" if dispatch else ""),
                    detail=event,
                    kind="subagent" if ok else "error",
                )
                return
            self._write_event(
                f"{prefix}{'✅' if ok else '❌'} {name}",
                detail=event,
                kind=self._event_kind(event, "success") if ok else "error",
            )
            return

        if event_type == "interrupted":
            self._write_event(
                f"⏸ Interrupted — resume with: {event.get('resume_command', '')}", detail=event, kind="error"
            )
            self.query_one("#final-body", Static).update("Run interrupted. Resume from the last checkpoint.")
            return

        if event_type == "handoff":
            self._set_active_agent(str(event.get("to") or ""))
            self._write_event(
                f"🔄 Handoff: {event.get('from')} → {event.get('to')}", detail=event, kind="handoff"
            )
            return

        if event_type == "search_results":
            prefix = self._subagent_prefix(event)
            self._write_event(
                f"{prefix}🔍 Search results: {event.get('query')}",
                detail=event,
                kind=self._event_kind(event, "tool"),
            )
            return

        if event_type == "approval_requested":
            prefix = self._subagent_prefix(event)
            tool = event.get("tool") or "tool"
            reason = self._truncate(str(event.get("reason") or ""), 120)
            suffix = f" · {reason}" if reason else ""
            self._write_event(f"{prefix}⚠ Approval requested: {tool}{suffix}", detail=event, kind="error")
            return

        if event_type == "hook_decision":
            prefix = self._subagent_prefix(event)
            label, kind = self._hook_decision_label(event)
            tool = event.get("tool") or "tool"
            reason = self._truncate(str(event.get("reason") or ""), 120)
            suffix = f" · {reason}" if reason else ""
            self._write_event(f"{prefix}{label}: {tool}{suffix}", detail=event, kind=kind)
            return

        if event_type == "trace.warn" and event.get("event") in {"PreToolUse", "PostToolUse"}:
            prefix = self._subagent_prefix(event)
            tool = event.get("tool") or "tool"
            self._write_event(
                f"{prefix}⚠ Hook warning: {tool} · {event.get('reason')}",
                detail=event,
                kind="hook",
            )
            return

        if event_type == "subagent_start":
            agent = event.get("agent")
            self._set_active_agent(str(agent or ""))
            description = event.get("description") or ""
            tools = event.get("tools") or []
            suffix = f" · tools: {', '.join(str(tool) for tool in tools)}" if tools else ""
            self._write_event(f"🤖 {agent} · {description}{suffix}", detail=event, kind="subagent")
            return

        if event_type == "subagent_result":
            self._set_active_agent("")
            prefix = self._subagent_prefix(event)
            summary = self._truncate(str(event.get("summary") or ""))
            self._write_event(f"{prefix}↩ Conclusion: {summary}", detail=event, kind="subagent")
            return

        if event_type == "memory_extract":
            added = int(event.get("added") or 0)
            replaced = int(event.get("replaced") or 0)
            if added or replaced:
                self._write_event(f"🧠 记忆提取：新增 {added} 条 / 覆盖 {replaced} 条", detail=event, kind="system")
            return

        if event_type == "checkpoint_saved":
            self._write_event(f"💾 Checkpoint saved: {event.get('status')}", detail=event, kind="system")
            return

        if event_type == "session_saved":
            self._session_id = str(event.get("session_id") or self._session_id)
            turn_index = event.get("turn_index")
            if isinstance(turn_index, int):
                self._turn_index = turn_index
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
            self._update_final_body(content)
            return

        if event_type == "error":
            self._write_event(f"❌ {event.get('error_type')}: {event.get('error')}", detail=event, kind="error")
            self.query_one("#final-body", Static).update(str(event.get("error") or "Error"))
            return

        if event_type == "turn_finished":
            self._running_turn = False
            self.query_one("#input", Input).disabled = False
            self.query_one("#input", Input).focus()
            self._active_agent = ""
            self._set_state("idle")
            self._set_status(f"idle | workspace: {self.workspace}")
            return

    def _handle_node_update(self, event: dict[str, Any]) -> None:
        node = event.get("node")
        data = event.get("data") or {}

        if isinstance(data, dict):
            self._sync_status_from_node(data)

        if "todos" in data:
            self._update_plan_from_node(data)

        if node == "intent_router":
            self._write_event(f"🧭 Route: {data.get('intent_route')}", detail=data, kind="system")
            return

        if node == "chat_responder":
            self._write_event("💬 Chat response generated", detail=data, kind="agent")
            return

        if node == "planner":
            messages = data.get("messages")
            if isinstance(messages, list):
                self._last_messages = messages
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
            self._update_final_body(str(data.get("final_answer") or ""))
            return

    def _update_final_body(self, content: str) -> None:
        body = Markdown(content) if content.strip() else Text("No final answer.")
        self.query_one("#final-body", Static).update(body)

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
