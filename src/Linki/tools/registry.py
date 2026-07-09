from langchain_core.tools import StructuredTool

from Linki.core.state import RuntimeState
from Linki.tools.bash_tool import BashTool
from Linki.tools.executor import execute_tool
from Linki.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
from Linki.tools.grep_tool import GrepTool
from Linki.tools.memory_tools import make_memory_upsert_tool
from Linki.tools.notepad_tool import NotepadAppendTool, NotepadReadTool
from Linki.tools.web_search_tool import WebSearchTool


# The AgentTool dispatches subagents; it is registered onto the planner and
# codeAgent pools but is never handed to a subagent (see agent_tool.run_subagent).
AGENT_TOOL_NAME = "AgentTool"

# The SkillTool discloses a named skill's full instructions on demand. Unlike
# AgentTool it IS available to subagents that whitelist it in their definition.
SKILL_TOOL_NAME = "SkillTool"

# Every tool name an agent definition may reference. Agent definitions are
# validated against this set at registry-load time so an unknown tool aborts
# startup instead of failing silently at dispatch.
KNOWN_TOOL_NAMES = frozenset(
    {
        "FileReadTool",
        "FileWriteTool",
        "FileEditTool",
        "GrepTool",
        "BashTool",
        "WebSearchTool",
        "NotepadReadTool",
        "NotepadAppendTool",
        "MemoryUpsertTool",
        AGENT_TOOL_NAME,
        SKILL_TOOL_NAME,
    }
)

# Tools that mutate the workspace or run commands; withheld in plan mode.
_MUTATING_TOOLS = {"FileWriteTool", "FileEditTool", "BashTool"}


def build_tools(
    state: RuntimeState,
    *,
    plan_mode: bool = False,
    ask_budget_left: int | None = None,
) -> list[StructuredTool]:
    """Build Linki tools for model.bind_tools().

    In plan mode the mutating tools (write/edit/bash) are filtered out so the
    caller can only read and research. ``ask_budget_left`` is accepted for
    interface parity with the planner's budget-gated tool filtering; the code
    agent set has no question tool, so it currently gates nothing here.
    """

    file_read = FileReadTool(state)
    file_write = FileWriteTool(state)
    file_edit = FileEditTool(state)
    grep = GrepTool(state)
    bash = BashTool(state)

    def file_read_tool(file_path: str, offset: int = 0, limit: int | None = None) -> dict:
        return execute_tool(
            state,
            "FileReadTool",
            {"file_path": file_path, "offset": offset, "limit": limit},
            file_read,
        )

    def file_write_tool(file_path: str, content: str) -> dict:
        return execute_tool(
            state,
            "FileWriteTool",
            {"file_path": file_path, "content": content},
            file_write,
        )

    def file_edit_tool(file_path: str, old_text: str, new_text: str) -> dict:
        return execute_tool(
            state,
            "FileEditTool",
            {"file_path": file_path, "old_text": old_text, "new_text": new_text},
            file_edit,
        )

    def grep_tool(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 50,
        ignore_case: bool = False,
    ) -> dict:
        return execute_tool(
            state,
            "GrepTool",
            {
                "pattern": pattern,
                "path": path,
                "glob": glob,
                "head_limit": head_limit,
                "ignore_case": ignore_case,
            },
            grep,
        )

    def bash_tool(command: str, timeout_seconds: int = 30) -> dict:
        return execute_tool(
            state,
            "BashTool",
            {"command": command, "timeout_seconds": timeout_seconds},
            bash,
        )

    tools = [
        StructuredTool.from_function(
            func=file_read_tool,
            name="FileReadTool",
            description="Read a UTF-8 text file within the workspace. Supports line offset and limit.",
        ),
        StructuredTool.from_function(
            func=file_write_tool,
            name="FileWriteTool",
            description="Create or overwrite a UTF-8 text file within the workspace.",
        ),
        StructuredTool.from_function(
            func=file_edit_tool,
            name="FileEditTool",
            description="Replace a unique text fragment in a UTF-8 text file within the workspace.",
        ),
        StructuredTool.from_function(
            func=grep_tool,
            name="GrepTool",
            description="Search files within the workspace using a regular expression.",
        ),
        StructuredTool.from_function(
            func=bash_tool,
            name="BashTool",
            description="Run a bash command with the workspace as the working directory and a timeout.",
        ),
    ]

    if plan_mode:
        tools = [tool for tool in tools if tool.name not in _MUTATING_TOOLS]
        return tools

    return tools + [make_memory_upsert_tool(state)]


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build read-only Linki tools for verifier model.bind_tools()."""

    file_read = FileReadTool(state)
    grep = GrepTool(state)

    def file_read_tool(file_path: str, offset: int = 0, limit: int | None = None) -> dict:
        return execute_tool(
            state,
            "FileReadTool",
            {"file_path": file_path, "offset": offset, "limit": limit},
            file_read,
        )

    def grep_tool(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 50,
        ignore_case: bool = False,
    ) -> dict:
        return execute_tool(
            state,
            "GrepTool",
            {
                "pattern": pattern,
                "path": path,
                "glob": glob,
                "head_limit": head_limit,
                "ignore_case": ignore_case,
            },
            grep,
        )

    return [
        StructuredTool.from_function(
            func=file_read_tool,
            name="FileReadTool",
            description="Read a UTF-8 text file within the workspace. Supports line offset and limit.",
        ),
        StructuredTool.from_function(
            func=grep_tool,
            name="GrepTool",
            description="Search files within the workspace using a regular expression.",
        ),
    ]


def build_subagent_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build the full superset of workspace tools available to subagents.

    ``run_subagent`` filters this pool down to the tools named in the agent
    definition's allowlist. Unlike :func:`build_tools`, this includes research
    (WebSearchTool), durable-notes (NotepadReadTool/NotepadAppendTool), and the
    progressive-disclosure SkillTool, but never the AgentTool — subagents cannot
    dispatch further subagents. Every tool call still flows through
    :func:`execute_tool` so the hook, risk-classification, and approval pipeline
    stays active inside subagents.
    """

    # Local import avoids a circular import: skill_tool imports SKILL_TOOL_NAME
    # from this module at import time.
    from Linki.tools.skill_tool import make_skill_tool

    web_search = WebSearchTool()
    notepad_read = NotepadReadTool(state)
    notepad_append = NotepadAppendTool(state)

    def web_search_tool(query: str) -> dict:
        return execute_tool(state, "WebSearchTool", {"query": query}, web_search)

    def notepad_read_tool() -> dict:
        return execute_tool(state, "NotepadReadTool", {}, notepad_read)

    def notepad_append_tool(note: str) -> dict:
        return execute_tool(state, "NotepadAppendTool", {"note": note}, notepad_append)

    extra = [
        StructuredTool.from_function(
            func=web_search_tool,
            name="WebSearchTool",
            description="Search the web for factual information and return sources.",
        ),
        StructuredTool.from_function(
            func=notepad_read_tool,
            name="NotepadReadTool",
            description="Read the workspace durable notes (NOTEPAD.md).",
        ),
        StructuredTool.from_function(
            func=notepad_append_tool,
            name="NotepadAppendTool",
            description="Append a durable note to the workspace NOTEPAD.md.",
        ),
        make_skill_tool(state),
    ]

    return build_tools(state) + extra
