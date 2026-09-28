"""Section-aware chunking that never splits fenced code blocks.

Each skill yields one "Card" chunk (frontmatter routing text) plus body-section chunks.
Body chunks carry a one-line "<display name> - <section>" header so a chunk like
"Gotchas" stays attributable to its skill for both BM25 and the embedding model.
"""
from __future__ import annotations
import re
from collections import Counter

from .models import Skill, Chunk

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
NEGATIVE_SCOPE_RE = re.compile(r"\bNOT for\b")
CARD_SECTION = "Card"


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_text), ...] for a markdown body."""
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        return [("Body", body.strip())]
    out = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            out.append((title, text))
    return out


def positive_scope(description: str) -> str:
    """Drop the trailing "NOT for X (see other-skill)" clause from a description.

    That clause names *other* skills' topics; indexing it as evidence for this skill
    would pull exactly the queries it disclaims toward the wrong skill.
    """
    return NEGATIVE_SCOPE_RE.split(description, maxsplit=1)[0].strip()


def card_chunk(skill: Skill) -> Chunk:
    """One chunk from the frontmatter: display name, positive-scope description, capabilities."""
    caps = ", ".join(c.replace("-", " ") for c in skill.capabilities)
    parts = [skill.display_name or skill.slug, positive_scope(skill.description),
             f"Capabilities: {caps}" if caps else ""]
    return Chunk(
        skill_slug=skill.slug,
        section=CARD_SECTION,
        text="\n".join(p for p in parts if p),
        chunk_id=f"{skill.slug}::{CARD_SECTION}::0",
    )


def _section_keys(titles: list[str], reserved: set[str]) -> list[str]:
    """Unique id keys for section titles: a repeated (or reserved) title gets "#2", "#3", ..."""
    seen = Counter(reserved)
    keys = []
    for t in titles:
        seen[t] += 1
        keys.append(t if seen[t] == 1 else f"{t}#{seen[t]}")
    return keys


def chunk_skill(skill: Skill, max_chars: int = 1200, include_card: bool = True) -> list[Chunk]:
    """Chunk a skill: optional Card chunk, then body sections with code fences kept intact.

    Chunk ids are unique within a skill even when a `##` title repeats or is itself "Card".
    """
    chunks: list[Chunk] = [card_chunk(skill)] if include_card else []
    title = skill.display_name or skill.slug
    sections = split_sections(skill.body)
    keys = _section_keys([t for t, _ in sections], {CARD_SECTION} if include_card else set())
    for (section, text), key in zip(sections, keys):
        for j, piece in enumerate(_split_keeping_code(text, max_chars)):
            chunks.append(Chunk(
                skill_slug=skill.slug,
                section=section,
                text=f"{title} - {section}\n{piece}",
                chunk_id=f"{skill.slug}::{key}::{j}",
            ))
    return chunks


def _split_keeping_code(text: str, max_chars: int) -> list[str]:
    """Split on blank lines but never inside a ``` fenced block."""
    blocks, buf, in_fence = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        buf.append(line)
        if not in_fence and sum(len(x) for x in buf) >= max_chars:
            blocks.append("\n".join(buf).strip())
            buf = []
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]
