"""Declarative agent registry.

Agents are defined as Markdown files with YAML frontmatter. Built-in agents ship
under ``src/Linki/agents/builtin/`` and workspace agents live under
``{workspace}/.linki/agents/``; workspace definitions override built-ins with the
same name. Every referenced tool is validated against ``KNOWN_TOOL_NAMES`` so an
unknown tool aborts startup rather than failing silently at dispatch time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Linki.core.frontmatter import parse_frontmatter_markdown as _parse_frontmatter_markdown
from Linki.core.state import RuntimeState
from Linki.tools.registry import KNOWN_TOOL_NAMES

BUILTIN_DIR = Path(__file__).parent / "builtin"
WORKSPACE_AGENTS_DIR = ".linki/agents"


@dataclass
class AgentSpec:
    name: str
    description: str
    tools: list[str]
    system_prompt: str


def parse_frontmatter_markdown(path: str | Path) -> AgentSpec:
    """Parse a Markdown agent definition with YAML frontmatter into an AgentSpec.

    Delegates frontmatter parsing and required-field validation (name/tools) to
    the shared :func:`Linki.core.frontmatter.parse_frontmatter_markdown`, then
    applies the agent-specific ``tools`` list check and builds the AgentSpec.
    """

    parsed = _parse_frontmatter_markdown(path, kind="agent")
    tools = parsed.meta.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"agent definition {Path(path)} 'tools' must be a list")

    return AgentSpec(
        name=str(parsed.meta["name"]),
        description=str(parsed.meta.get("description", "")),
        tools=[str(tool) for tool in tools],
        system_prompt=parsed.body.strip(),
    )


def _validate_tools(spec: AgentSpec, path: Path) -> None:
    for tool in spec.tools:
        if tool not in KNOWN_TOOL_NAMES:
            known = ", ".join(sorted(KNOWN_TOOL_NAMES))
            raise ValueError(
                f"agent definition {path} (agent '{spec.name}') references unknown "
                f"tool '{tool}'. Known tools: {known}"
            )


def _load_dir(directory: Path, registry: dict[str, AgentSpec]) -> None:
    if not directory.is_dir():
        return
    for md_path in sorted(directory.glob("*.md")):
        spec = parse_frontmatter_markdown(md_path)
        _validate_tools(spec, md_path)
        registry[spec.name] = spec  # later definitions override earlier ones


def load_agent_registry(state: RuntimeState) -> dict[str, AgentSpec]:
    """Load built-in then workspace agent definitions, keyed by agent name.

    Workspace definitions override built-ins with the same name. An unknown tool
    in any definition raises ``ValueError`` and aborts loading.
    """

    registry: dict[str, AgentSpec] = {}
    _load_dir(BUILTIN_DIR, registry)
    _load_dir(state.workspace / WORKSPACE_AGENTS_DIR, registry)
    return registry
