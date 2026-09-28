"""Shared utilities for the skill-library tooling (validate/router/graph/freshness/index).

Deliberately dependency-free (no PyYAML) so that `python scripts/whatever.py`
works on a bare Python 3.9+ install with no `pip install` step, per the
repo's "don't add unnecessary dependency managers" policy (CLAUDE.md /
docs/SKILL-SPEC.md). The frontmatter used across SKILL.md files is a small
subset of YAML (flat scalars, folded `>` block scalars, and flat string
lists) so a hand-rolled parser is sufficient and keeps the failure mode
obvious instead of hiding behind a general-purpose YAML library.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are NOT skills even though they may contain a SKILL.md
# (the template) or sit at repo root.
NON_SKILL_DIRS = {"_TEMPLATE", ".git", ".claude", "scripts", "schema", "tests", "docs", "evals", ".github"}


class FrontmatterError(ValueError):
    pass


@dataclass
class Skill:
    slug: str  # folder name
    path: Path  # path to SKILL.md
    frontmatter: dict[str, Any]
    body: str  # markdown after the closing `---`
    raw: str  # full file text

    def __getattr__(self, item):  # convenience: skill.type instead of skill.frontmatter["type"]
        try:
            return self.frontmatter[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def headers(self) -> list[str]:
        """Return every level-2 (`## `) markdown header in the body, in order."""
        return re.findall(r"^##\s+(.+?)\s*$", self.body, flags=re.MULTILINE)

    def bullets_under(self, header: str) -> list[str]:
        """Return top-level `- ` bullet lines under a given `## Header` section."""
        pattern = rf"^##\s+{re.escape(header)}\s*$(.*?)(?=^##\s+|\Z)"
        m = re.search(pattern, self.body, flags=re.MULTILINE | re.DOTALL)
        if not m:
            return []
        section = m.group(1)
        return re.findall(r"^-\s+.+$", section, flags=re.MULTILINE)

    def code_blocks(self, lang: str | None = "python") -> list[tuple[str, str]]:
        """Return (lang, code) for every fenced code block in the body.

        Code fences nested under a numbered-list step are themselves indented
        in the raw Markdown (e.g. 3 spaces to align with "1. "). That
        indentation is part of the Markdown, not the Python — dedent it here
        so downstream consumers (e.g. validate_skills.py's ast.parse check)
        see the code as it's meant to run, not as it's laid out on the page.
        """
        blocks = re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", self.body, flags=re.DOTALL)
        blocks = [(l, textwrap.dedent(c)) for l, c in blocks]
        if lang is None:
            return blocks
        return [(l, c) for l, c in blocks if l.lower() == lang]

    def markdown_links(self) -> list[tuple[str, str]]:
        """Return (link_text, url) for every markdown [text](url) in the body."""
        return re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+|[^\s)]+)\)", self.body)


def _split_frontmatter(raw: str) -> tuple[str, str]:
    if not raw.startswith("---"):
        raise FrontmatterError("file does not start with a `---` frontmatter fence")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise FrontmatterError("could not find closing `---` for frontmatter")
    return parts[1], parts[2].lstrip("\n")


def parse_frontmatter(fm_text: str) -> dict[str, Any]:
    """Parse the restricted YAML subset used by SKILL.md frontmatter.

    Supports:
      key: scalar value
      key: >          (folded block scalar -> joined with spaces)
      key: |          (literal block scalar -> joined with newlines)
      key:
        - item
        - item        (flat string list)
    Does NOT support nested maps, list-of-dicts, or flow collections
    ([a, b]) — none of those appear in this repo's frontmatter. If you need
    them, that's a signal the schema is growing beyond what belongs in a
    hand-authored skill card; reconsider before extending this parser.
    """
    lines = fm_text.split("\n")
    result: dict[str, Any] = {}
    i = 0
    n = len(lines)
    key_line_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
    list_item_re = re.compile(r"^\s{2,}-\s*(.*)$")

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = key_line_re.match(line)
        if not m:
            i += 1
            continue
        key, rest = m.group(1), m.group(2).strip()
        i += 1
        if rest in (">", "|"):
            block_lines = []
            while i < n and (lines[i].startswith("  ") or not lines[i].strip()):
                block_lines.append(lines[i])
                i += 1
            # de-indent by the minimum leading-space count among non-blank lines
            indents = [len(bl) - len(bl.lstrip(" ")) for bl in block_lines if bl.strip()]
            pad = min(indents) if indents else 0
            dedented = [bl[pad:] if len(bl) >= pad else bl for bl in block_lines]
            if rest == ">":
                # folded scalar: blank-separated "paragraphs" become one
                # space-joined string; we only ever have a single paragraph
                # in practice, so a plain space-join is sufficient.
                value = " ".join(dl.strip() for dl in dedented if dl.strip())
            else:
                value = "\n".join(dl for dl in dedented).rstrip("\n")
            result[key] = value
        elif rest == "":
            items = []
            while i < n and list_item_re.match(lines[i]):
                item = list_item_re.match(lines[i]).group(1).strip()
                item = item.strip('"').strip("'")
                items.append(item)
                i += 1
            if items:
                result[key] = items
            else:
                result[key] = None
        else:
            value = rest.strip('"').strip("'")
            result[key] = value
    return result


def load_skill(skill_dir: Path) -> Skill:
    skill_md = skill_dir / "SKILL.md"
    raw = skill_md.read_text(encoding="utf-8")
    fm_text, body = _split_frontmatter(raw)
    fm = parse_frontmatter(fm_text)
    return Skill(slug=skill_dir.name, path=skill_md, frontmatter=fm, body=body, raw=raw)


def iter_skill_dirs(root: Path = REPO_ROOT) -> list[Path]:
    dirs = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in NON_SKILL_DIRS or p.name.startswith("."):
            continue
        if (p / "SKILL.md").exists():
            dirs.append(p)
    return dirs


def load_all_skills(root: Path = REPO_ROOT) -> list[Skill]:
    skills = []
    for d in iter_skill_dirs(root):
        try:
            skills.append(load_skill(d))
        except FrontmatterError:
            # let callers (validate_skills.py) report this as a proper error
            # rather than crashing the whole run; re-raise with slug context.
            raise FrontmatterError(f"{d.name}: malformed frontmatter")
    return skills


def by_slug(skills: list[Skill]) -> dict[str, Skill]:
    return {s.slug: s for s in skills}
