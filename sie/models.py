"""Shared dataclasses for the engine: the one vocabulary every module (and the API) maps from."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Typed skill-to-skill relationships (see sie/graph/build.py for edge direction and semantics).
RELATIONSHIPS = ("requires", "recommended_before", "related", "conflicts",
                 "alternative_to", "specializes", "supersedes")


@dataclass
class Skill:
    """A parsed SKILL.md: frontmatter metadata + body, plus where it came from."""
    slug: str                       # the skill's own id (frontmatter id/slug/name, else folder)
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
    # extended relationships, each declared on this skill (direction: sie/graph/build.py)
    recommended_before: list[str] = field(default_factory=list)   # soft prerequisites of this skill
    alternative_to: list[str] = field(default_factory=list)
    specializes: list[str] = field(default_factory=list)          # this skill narrows these
    supersedes: list[str] = field(default_factory=list)           # this skill replaces these
    # provenance
    source: str = ""                # corpus name (corpus.toml `name`, else the corpus dir name)
    source_version: str = ""        # corpus version / commit; "" when unknown
    path: str = ""                  # SKILL.md path relative to the corpus root (posix)
    content_hash: str = ""          # sha256 of the file bytes, CRLF normalized to LF
    version: str = ""               # frontmatter `version`, if declared
    updated_at: str = ""            # frontmatter updated_at / last_verified (ISO), if declared
    declared: list[str] = field(default_factory=list)       # relationship keys present in frontmatter
    metadata: dict[str, Any] = field(default_factory=dict)  # other JSON-safe frontmatter keys

    @property
    def skill_id(self) -> str:
        """Corpus-independent identifier (the slug); `source` says which corpus supplied it."""
        return self.slug

    @property
    def name(self) -> str:
        return self.display_name or self.slug

    @property
    def provenance(self) -> str:
        """`<corpus>[@<version>]:<path>`, e.g. `ml-ai-skills@8328c60:rag-evaluation/SKILL.md`."""
        version = f"@{self.source_version}" if self.source_version else ""
        return f"{self.source or 'unknown'}{version}:{self.path or self.slug}"


@dataclass
class CorpusInfo:
    """Identity of a loaded corpus: what it is, which version, and a content fingerprint."""
    name: str
    root: str
    version: str = ""               # "" when neither corpus.toml nor the caller declares one
    source_url: str = ""
    license: str = ""
    manifest: str = ""              # path of the corpus.toml that was read, "" if none
    fingerprint: str = ""           # sha256 over sorted (slug, content_hash), first 16 hex
    n_skills: int = 0
    loaded_at: str = ""             # UTC ISO-8601 timestamp


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
    chunk_id: str = ""              # source chunk; lets the reranker score the full text


@dataclass(frozen=True)
class Edge:
    """One typed relationship between two skills, with where the claim came from."""
    source: str
    target: str
    relationship: str               # one of RELATIONSHIPS
    confidence: str = "declared"    # "declared" (corpus frontmatter) | "proposed" (overlay) | ...
    provenance: str = ""            # e.g. "ml-ai-skills@8328c60:rag-evaluation/SKILL.md#requires"


@dataclass
class MethodEvidence:
    """Why one method surfaced a skill: its best chunk for the query under that method."""
    method: str                     # "dense" | "bm25" | "rerank"
    rank: int                       # 1-based skill-level rank within this method's candidates
    score: float                    # cosine similarity | BM25 score | cross-encoder logit
    section: str = ""
    snippet: str = ""
    matched_terms: list[str] = field(default_factory=list)   # bm25: query terms in the chunk


@dataclass
class RankedSkill:
    """One skill in a routed ranking, with the per-method evidence behind it."""
    slug: str
    rank: int
    score: float
    score_type: str                 # "rrf" | "cosine" | "bm25" | "cross-encoder"
    section: str = ""
    snippet: str = ""
    chunk_id: str = ""              # internal; never exposed by the API
    methods: list[str] = field(default_factory=list)          # retrievers that found it
    evidence: list[MethodEvidence] = field(default_factory=list)

    def to_hit(self) -> Hit:
        return Hit(skill_slug=self.slug, score=self.score, section=self.section,
                   snippet=self.snippet, chunk_id=self.chunk_id)


@dataclass
class Confidence:
    """Heuristic, uncalibrated routing confidence (sie/confidence.py documents the rules)."""
    level: str                      # "high" | "medium" | "low" | "none"
    action: str                     # "route" | "clarify" | "abstain"
    ambiguous: bool = False
    reasons: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)      # skills contesting the top result
    signals: dict[str, Any] = field(default_factory=dict)     # the raw evidence the rules read


@dataclass
class RouteResult:
    """Everything one routing call decided, and how long each stage took."""
    query: str
    results: list[RankedSkill]
    confidence: Confidence
    mode: str = "hybrid"
    reranked: bool = False          # did the cross-encoder order `results`?
    rerank_error: str | None = None
    candidates: dict[str, int] = field(default_factory=dict)    # per-stage candidate counts
    timings_ms: dict[str, float] = field(default_factory=dict)  # per-stage latency + "total"

    @property
    def hits(self) -> list[Hit]:
        return [r.to_hit() for r in self.results]

    @property
    def top(self) -> RankedSkill | None:
        return self.results[0] if self.results else None


@dataclass
class Intent:
    """One candidate intent split out of a multi-part query."""
    index: int
    text: str                       # the segment as written
    query: str                      # what was routed (e.g. "it" resolved to the first object)
    result: RouteResult
    selected: str | None = None     # chosen skill; None when nothing matched well enough


@dataclass
class PlanStep:
    """One skill in a composed, dependency-ordered plan."""
    slug: str
    position: int
    role: str                       # "requested" | "prerequisite"
    intents: list[int] = field(default_factory=list)       # intents that asked for / need it
    required_by: list[str] = field(default_factory=list)   # plan skills that directly require it
    reason: str = ""


@dataclass
class Composition:
    """query -> candidate intents -> skills -> dependencies -> ordered plan."""
    query: str
    intents: list[Intent]
    steps: list[PlanStep]
    edges: list[Edge] = field(default_factory=list)          # relationships among plan skills
    conflicts: list[list[str]] = field(default_factory=list)  # conflicting pairs in the plan
    unmatched: list[int] = field(default_factory=list)        # intents with no good match
    notes: list[str] = field(default_factory=list)

    @property
    def multi_intent(self) -> bool:
        return len(self.intents) > 1
