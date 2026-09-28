"""Shared dataclasses for the engine."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Skill:
    """A parsed SKILL.md: frontmatter metadata + body sections."""
    slug: str
    skill_type: str                 # "workflow" | "reference"
    domain: str
    level: str
    capabilities: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    risk_level: str = "low"
    evidence_level: str = "established-practice"
    body: str = ""                  # full markdown body (sections concatenated)
    display_name: str = ""          # human title, e.g. "RAG Pipeline"
    description: str = ""           # frontmatter "Use when ..." routing text


@dataclass
class Chunk:
    """A retrievable unit derived from a Skill body section."""
    skill_slug: str
    section: str                    # e.g. "Workflow", "Key Concepts", "Gotchas"
    text: str
    chunk_id: str


@dataclass
class Hit:
    """A ranked retrieval result."""
    skill_slug: str
    score: float
    section: str = ""
    snippet: str = ""
