"""Declarative skill registry.

Skills are Markdown files with YAML frontmatter, one per directory, named
``SKILL.md``. Built-in skills ship under ``src/Linki/skills/builtin/<name>/`` and
workspace skills live under ``{workspace}/.linki/skills/<name>/``; a workspace
skill overrides a built-in of the same name.

Unlike agent definitions, a skill's ``description`` is capped at 120 characters
so the ``<available_skills>`` prompt block stays terse — an over-long description
aborts loading (i.e. startup) with a file-specific ``ValueError``. The full
instructions live in the body and are only disclosed on demand via SkillTool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Linki.core.frontmatter import parse_frontmatter_markdown
from Linki.core.state import RuntimeState

BUILTIN_DIR = Path(__file__).parent / "builtin"
WORKSPACE_SKILLS_DIR = ".linki/skills"
DESCRIPTION_MAX_CHARS = 120


@dataclass
class SkillSpec:
    name: str
    description: str
    body: str
    dir: Path


def _load_dir(directory: Path, registry: dict[str, SkillSpec]) -> None:
    if not directory.is_dir():
        return
    for skill_md in sorted(directory.glob("*/SKILL.md")):
        parsed = parse_frontmatter_markdown(skill_md, kind="skill")
        name = str(parsed.meta["name"])
        description = str(parsed.meta["description"])
        if len(description) > DESCRIPTION_MAX_CHARS:
            raise ValueError(
                f"skill definition {skill_md} description exceeds "
                f"{DESCRIPTION_MAX_CHARS} characters (got {len(description)})"
            )
        registry[name] = SkillSpec(
            name=name,
            description=description,
            body=parsed.body.strip(),
            dir=skill_md.parent,
        )  # later definitions override earlier ones


def load_skill_registry(state: RuntimeState) -> dict[str, SkillSpec]:
    """Load built-in then workspace skill definitions, keyed by skill name.

    Workspace skills override built-ins with the same name. A description longer
    than :data:`DESCRIPTION_MAX_CHARS` raises ``ValueError`` and aborts loading.
    """

    registry: dict[str, SkillSpec] = {}
    _load_dir(BUILTIN_DIR, registry)
    _load_dir(state.workspace / WORKSPACE_SKILLS_DIR, registry)
    return registry


def load_skills_into_runtime(state: RuntimeState) -> dict[str, SkillSpec]:
    """Populate ``runtime.skills`` from the registry and reset per-run load state.

    Called once at the start of each run: it clears any skills disclosed on a
    prior run (``loaded_skills``) and refreshes the available-skill catalog so
    prompt assembly and SkillTool read a consistent, current view.
    """

    registry = load_skill_registry(state)
    state.loaded_skills.clear()
    state.skills.clear()
    state.skills.update(registry)
    return registry


def render_available_skills(state: Any) -> str:
    """Render the ``<available_skills>`` prompt block: one terse line per skill.

    Only the name and description are shown — the full body is disclosed lazily
    via SkillTool. Returns an empty string when no skills are available.
    """

    runtime = state if isinstance(state, RuntimeState) else (
        state.get("runtime") if isinstance(state, Mapping) else None
    )
    if runtime is None:
        return ""

    registry: dict[str, SkillSpec] = dict(runtime.skills) if runtime.skills else load_skill_registry(runtime)
    if not registry:
        return ""

    lines = ["<available_skills>"]
    for name in sorted(registry):
        lines.append(f"· {name} — {registry[name].description}")
    lines.append("</available_skills>")
    return "\n".join(lines)
