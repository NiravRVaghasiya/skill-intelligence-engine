"""Section-aware chunking that never splits fenced code blocks."""
from __future__ import annotations
import re

from .models import Skill, Chunk

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


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


def chunk_skill(skill: Skill, max_chars: int = 1200) -> list[Chunk]:
    """Chunk a skill body by section, keeping code fences intact."""
    chunks: list[Chunk] = []
    for section, text in split_sections(skill.body):
        for j, piece in enumerate(_split_keeping_code(text, max_chars)):
            chunks.append(Chunk(
                skill_slug=skill.slug,
                section=section,
                text=piece,
                chunk_id=f"{skill.slug}::{section}::{j}",
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
