import json

from langchain_core.tools import StructuredTool

from Linki.core.state import RuntimeState
from Linki.tools.bash_tool import BashTool
from Linki.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
from Linki.tools.grep_tool import GrepTool


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

    def file_read_tool(file_path: str, offset: int = 0, limit: int | None = None) -> str:
        return file_read(file_path=file_path, offset=offset, limit=limit)

    def file_write_tool(file_path: str, content: str) -> str:
        return file_write(file_path=file_path, content=content)

    def file_edit_tool(file_path: str, old_text: str, new_text: str) -> str:
        return file_edit(file_path=file_path, old_text=old_text, new_text=new_text)

    def grep_tool(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 50,
        ignore_case: bool = False,
    ) -> str:
        return grep(
            pattern=pattern,
            path=path,
            glob=glob,
            head_limit=head_limit,
            ignore_case=ignore_case,
        )

    def bash_tool(command: str, timeout_seconds: int = 30) -> str:
        return json.dumps(bash(command=command, timeout_seconds=timeout_seconds), ensure_ascii=False)

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


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build read-only Linki tools for verifier model.bind_tools()."""

    file_read = FileReadTool(state)
    grep = GrepTool(state)

    def file_read_tool(file_path: str, offset: int = 0, limit: int | None = None) -> str:
        return file_read(file_path=file_path, offset=offset, limit=limit)

    def grep_tool(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 50,
        ignore_case: bool = False,
    ) -> str:
        return grep(
            pattern=pattern,
            path=path,
            glob=glob,
            head_limit=head_limit,
            ignore_case=ignore_case,
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
