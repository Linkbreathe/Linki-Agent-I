from pathlib import Path

from Linki.core.paths import resolve_workspace_path
from Linki.core.state import RuntimeState


def _display_path(state: RuntimeState, path: Path) -> str:
    return str(path.relative_to(state.workspace))


class FileReadTool:
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, offset: int = 0, limit: int | None = None) -> str:
        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0")
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to 0")

        path = resolve_workspace_path(self.state, file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[offset:] if limit is None else lines[offset : offset + limit]
        return "\n".join(selected)


class FileWriteTool:
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, content: str) -> str:
        path = resolve_workspace_path(self.state, file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {_display_path(self.state, path)}"


class FileEditTool:
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, old_text: str, new_text: str) -> str:
        if old_text == "":
            raise ValueError("old_text must not be empty")

        path = resolve_workspace_path(self.state, file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            raise ValueError("old_text was not found")
        if count > 1:
            raise ValueError("old_text appears multiple times; provide a unique fragment")

        path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {_display_path(self.state, path)}"
