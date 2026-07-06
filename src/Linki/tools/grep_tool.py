from pathlib import Path
import fnmatch
import re

from Linki.core.paths import resolve_workspace_path
from Linki.core.state import RuntimeState


class GrepTool:
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 50,
        ignore_case: bool = False,
    ) -> str:
        if head_limit < 0:
            raise ValueError("head_limit must be greater than or equal to 0")

        root = resolve_workspace_path(self.state, path)
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        matches: list[str] = []

        files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for file_path in files:
            relative = file_path.relative_to(self.state.workspace)
            if glob and not fnmatch.fnmatch(str(relative), glob):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{relative}:{line_number}:{line}")
                    if len(matches) >= head_limit:
                        return "\n".join(matches)

        return "\n".join(matches)
