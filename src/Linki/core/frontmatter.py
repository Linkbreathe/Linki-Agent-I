"""Shared YAML-frontmatter Markdown parser.

Both the agent registry and the skill registry describe their units as Markdown
files with a leading ``---`` YAML frontmatter block followed by a free-form body.
This module holds the single parser they share; the ``kind`` argument selects
which frontmatter fields are mandatory ("agent" requires ``name``/``tools``,
"skill" requires ``name``/``description``) and is woven into error messages so a
bad file names both its kind and its path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Required frontmatter fields per document kind.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "agent": ("name", "tools"),
    "skill": ("name", "description"),
}


@dataclass
class ParsedFrontmatter:
    """The YAML frontmatter mapping plus the Markdown body of a document."""

    meta: dict
    body: str


def parse_frontmatter_markdown(path: str | Path, kind: str) -> ParsedFrontmatter:
    """Parse a Markdown file with YAML frontmatter, validating required fields.

    ``kind`` must be one of :data:`REQUIRED_FIELDS`. Missing frontmatter,
    malformed delimiters, invalid YAML, a non-mapping frontmatter, or an absent
    required field each raise a ``ValueError`` that names the kind and the path.
    """

    if kind not in REQUIRED_FIELDS:
        raise ValueError(f"unknown frontmatter kind: {kind!r}")

    path = Path(path)
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{kind} definition {path} is missing YAML frontmatter")

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break

    if end_index is None:
        raise ValueError(f"{kind} definition {path} has malformed frontmatter delimiters")

    raw_yaml = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])

    try:
        meta = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"{kind} definition {path} has invalid YAML frontmatter: {exc}") from exc

    if not isinstance(meta, dict):
        raise ValueError(f"{kind} definition {path} frontmatter must be a mapping")

    for field_name in REQUIRED_FIELDS[kind]:
        value = meta.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{kind} definition {path} is missing required '{field_name}'")

    return ParsedFrontmatter(meta=meta, body=body)
