"""The API's public JSON contract: pydantic v2 request/response models plus converters.

Field names here are the contract; internal dataclasses (`sie.models`) are never returned
as-is. Converters map them, drop internal chunk ids, round scores to 4 decimals, and attach
skill names, provenance and direct prerequisites from the engine. Error text they carry
(`rerank_error`, readiness reasons) shows paths under the repo root repo-relative
(`sie.engine.display_paths`). Confidence levels are heuristic (`calibrated` is always False;
see sie/confidence.py).

Only `sie.api` imports this module (pydantic is part of the `api` extra, not the core).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .engine import display_paths
from .models import Composition, Confidence, CorpusInfo, RankedSkill, RouteResult, Skill

MAX_QUERY_CHARS = 4000
MAX_BATCH = 32
QueryText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_QUERY_CHARS)]
Level = Literal["high", "medium", "low", "none"]
Action = Literal["route", "clarify", "abstain"]


class EngineLike(Protocol):
    """What the converters read from `sie.engine.Engine`."""

    @property
    def corpus(self) -> CorpusInfo | None: ...

    def skill_obj(self, slug: str) -> Skill: ...

    def prerequisites(self, slug: str) -> list[str]: ...


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")       # a misspelled field is a 422, not ignored


def _not_blank(query: str) -> str:
    if not query.strip():
        raise ValueError("query must not be blank")
    return query


# -- shared pieces ----------------------------------------------------------------------------

class ProvenanceOut(BaseModel):
    """Where a skill came from."""
    source: str = Field(description="corpus name")
    source_version: str = Field(description="corpus version / commit; '' when unknown")
    path: str = Field(description="skill file path relative to the corpus root")
    content_hash: str = Field(description="sha256 of the file (CRLF normalized to LF)")
    updated_at: str = Field(description="frontmatter updated_at / last_verified, if declared")
    version: str = Field(description="frontmatter version, if declared")
    provenance: str = Field(description="`<corpus>[@<version>]:<path>`")


class EvidenceOut(BaseModel):
    """Why one method surfaced a skill: its best chunk for the query under that method."""
    method: str = Field(description="dense | bm25 | rerank")
    rank: int = Field(description="1-based skill-level rank within this method")
    score: float = Field(description="cosine | BM25 score | cross-encoder logit (4 decimals)")
    section: str
    snippet: str
    matched_terms: list[str] = Field(description="bm25: query terms found in the chunk")


class SkillResult(BaseModel):
    """One ranked skill."""
    skill_id: str
    name: str
    rank: int
    score: float = Field(description="fused RRF score, or cross-encoder logit when reranked")
    score_type: str = Field(description="rrf | cross-encoder")
    section: str = Field(description="section of the best-matching chunk")
    snippet: str | None = Field(None, description="best chunk excerpt (explain only)")
    retrieval_methods: list[str] = Field(description="retrievers whose candidates contained it")
    evidence: list[EvidenceOut] = Field(description="per-method evidence (explain only, else [])")
    provenance: ProvenanceOut | None = Field(description="None if the skill is not in the corpus")
    prerequisites: list[str] = Field(description="direct `requires` of this skill, sorted")


class Alternative(BaseModel):
    """A skill a caller may want instead of the top result."""
    skill_id: str
    name: str
    rank: int | None = Field(description="rank in `results`; None when ranked below k")
    reason: str


class ConfidenceOut(BaseModel):
    """Heuristic, uncalibrated routing confidence and the evidence behind it."""
    level: Level
    action: Action = Field(description="advisory: route | clarify | abstain")
    ambiguous: bool
    reasons: list[str]
    competitors: list[str] = Field(description="skills contesting the top result")
    signals: dict[str, Any] = Field(description="raw signals the rules read")
    calibrated: Literal[False] = Field(False, description="always false: levels are heuristic")


class CorpusRef(BaseModel):
    """Which corpus answered."""
    name: str
    version: str
    fingerprint: str = Field(description="content fingerprint of the loaded skill files")


class EdgeOut(BaseModel):
    """One typed relationship, with where the claim came from."""
    source: str
    target: str
    relationship: str
    confidence: str = Field(description="declared (corpus frontmatter) | proposed (overlay) | ...")
    provenance: str


class ErrorResponse(BaseModel):
    detail: str


# -- composition ------------------------------------------------------------------------------

class IntentOut(BaseModel):
    index: int
    text: str = Field(description="the part as written")
    query: str = Field(description="what was routed (pronouns resolved)")
    selected: str | None = Field(description="chosen skill; None when nothing matched well enough")
    confidence: Level
    action: Action
    ambiguous: bool
    competitors: list[str]


class PlanStepOut(BaseModel):
    skill_id: str
    name: str
    position: int
    role: Literal["requested", "prerequisite"]
    intents: list[int]
    required_by: list[str]
    reason: str


class CompositionOut(BaseModel):
    """query -> intents -> skills -> dependencies -> ordered plan (opt-in, conservative)."""
    multi_intent: bool
    intents: list[IntentOut]
    plan: list[PlanStepOut]
    edges: list[EdgeOut] = Field(description="relationships among plan skills")
    conflicts: list[list[str]]
    unmatched: list[int] = Field(description="intents with no sufficiently good match")
    notes: list[str]


# -- routing ----------------------------------------------------------------------------------

class RouteRequest(_Request):
    query: QueryText
    k: int | None = Field(None, ge=1, le=50,
                          description="skills to return (default: the engine's top_k, i.e. 5)")
    pool: int | None = Field(None, ge=1, le=500,
                             description="candidate chunks per retriever (default: engine config)")
    rerank_k: int | None = Field(None, ge=1, le=500,
                                 description="top fused skills cross-encoded (default: the pool)")
    explain: bool = Field(True, description="include snippets and per-method evidence")
    multi_intent: bool = Field(False, description="also split the query into intents and "
                                                  "compose an ordered skill plan")
    max_intents: int = Field(5, ge=1, le=10, description="multi_intent: route at most N intents")

    _query_not_blank = field_validator("query")(_not_blank)


class RouteResponse(BaseModel):
    query: str
    confidence: ConfidenceOut
    results: list[SkillResult]
    alternatives: list[Alternative]
    reranked: bool = Field(description="did the cross-encoder order `results`?")
    rerank_error: str | None
    mode: str
    candidates: dict[str, int] = Field(description="per-stage candidate counts")
    timings_ms: dict[str, float] = Field(description="per-stage latency (wall clock)")
    corpus: CorpusRef | None
    composition: CompositionOut | None = Field(None, description="present when multi_intent")


class BatchRouteRequest(_Request):
    queries: list[QueryText] = Field(min_length=1, max_length=MAX_BATCH)
    k: int | None = Field(None, ge=1, le=50)
    pool: int | None = Field(None, ge=1, le=500)
    rerank_k: int | None = Field(None, ge=1, le=500)
    explain: bool = False

    @field_validator("queries")
    @classmethod
    def _queries_not_blank(cls, queries: list[str]) -> list[str]:
        blank = [i for i, q in enumerate(queries) if not q.strip()]
        if blank:
            raise ValueError(f"queries must not be blank (index {', '.join(map(str, blank))})")
        return queries


class BatchRouteResponse(BaseModel):
    results: list[RouteResponse] = Field(description="one per query, in request order")


class SearchHit(BaseModel):
    slug: str
    score: float
    section: str
    rank: int
    name: str
    retrieval_methods: list[str]


class ConfidenceBrief(BaseModel):
    level: Level
    action: Action
    ambiguous: bool


class SearchResponse(BaseModel):
    query: str
    reranked: bool
    pool: int
    results: list[SearchHit]
    confidence: ConfidenceBrief


# -- graph ------------------------------------------------------------------------------------

class LearningStep(BaseModel):
    skill: str
    position: int
    relation: Literal["target", "direct", "transitive", "recommended"]
    depth: int | None = Field(description="fewest requires hops to the target; None if recommended")
    required_by: list[str]
    reason: str
    provenance: list[str]


class RecommendedOut(BaseModel):
    skill: str
    before: str
    provenance: str


class LearningPathResponse(BaseModel):
    target: str
    path: list[str] = Field(description="prerequisites in dependency order, ending at target")
    related: list[str]
    conflicts: list[str]
    path_conflicts: list[list[str]]
    missing_prerequisites: list[list[str]] = Field(description="[skill, ref not in the corpus]")
    steps: list[LearningStep]
    direct: list[str]
    transitive: list[str]
    recommended: list[RecommendedOut]
    notes: list[str]
    complete: bool


class SkillDetail(BaseModel):
    slug: str
    skill_id: str
    name: str
    domain: str
    level: str
    skill_type: str
    display_name: str
    description: str
    capabilities: list[str]
    related: list[str]
    conflicts: list[str]
    requires: list[str]
    required_by: list[str]
    dangling: list[list[str]] = Field(description="[relationship, ref] not in the corpus")
    declared: list[str] = Field(description="relationship keys present in the frontmatter")
    provenance: ProvenanceOut | None
    relations: list[EdgeOut] = Field(description="every typed edge touching this skill")
    metadata: dict[str, Any] = Field(description="other frontmatter keys (JSON-safe)")


# -- service ----------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    skills: int
    corpus: CorpusRef | None = None
    detail: str | None = None


class CorpusStatus(BaseModel):
    name: str
    version: str
    source_url: str
    license: str
    manifest: str = Field(description="manifest file name, '' if none")
    fingerprint: str
    n_skills: int
    loaded_at: str


class GraphStatus(BaseModel):
    nodes: int
    edges_by_relationship: dict[str, int]
    overlays: list[str]
    cycles: list[list[str]]
    dangling: int


class ReadyResponse(BaseModel):
    ready: bool
    degraded: bool = Field(False, description="serving, but the cross-encoder fell back to RRF")
    reasons: list[str] = Field(description="why not ready ([] when ready)")
    errors: list[str] = Field(default_factory=list, description="failures of the latest start")
    version: str
    started_at: str | None = None
    corpus: CorpusStatus | None = None
    graph: GraphStatus | None = None
    router: dict[str, Any] | None = None


# -- converters -------------------------------------------------------------------------------

def _score(x: float) -> float:
    return round(float(x), 4)


def plain(value: Any) -> Any:
    """JSON-ready copy with builtin scalars only (numpy floats etc. become float/int/str)."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    item = getattr(value, "item", None)                # numpy scalar
    return plain(item()) if callable(item) else str(value)


def _display(value: Any) -> Any:
    """`value` (plain data) with repo paths in every string shown repo-relative."""
    if isinstance(value, str):
        return display_paths(value)
    if isinstance(value, dict):
        return {k: _display(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_display(v) for v in value]
    return value


class _Skills:
    """Per-response skill lookups (name, provenance, prerequisites), cached."""

    def __init__(self, engine: EngineLike):
        self.engine = engine
        self._cache: dict[str, Skill | None] = {}

    def get(self, slug: str) -> Skill | None:
        if slug not in self._cache:
            try:
                self._cache[slug] = self.engine.skill_obj(slug)
            except KeyError:
                self._cache[slug] = None
        return self._cache[slug]

    def name(self, slug: str) -> str:
        skill = self.get(slug)
        return skill.name if skill is not None else slug

    def provenance(self, slug: str) -> ProvenanceOut | None:
        s = self.get(slug)
        if s is None:
            return None
        return ProvenanceOut(source=s.source, source_version=s.source_version, path=s.path,
                             content_hash=s.content_hash, updated_at=s.updated_at,
                             version=s.version, provenance=s.provenance)


def corpus_ref(info: CorpusInfo | None) -> CorpusRef | None:
    return None if info is None else CorpusRef(name=info.name, version=info.version,
                                                fingerprint=info.fingerprint)


def confidence_out(conf: Confidence) -> ConfidenceOut:
    return ConfidenceOut(level=conf.level, action=conf.action, ambiguous=conf.ambiguous,
                         reasons=list(conf.reasons), competitors=list(conf.competitors),
                         signals=plain(conf.signals))


def skill_result(skills: _Skills, s: RankedSkill, explain: bool) -> SkillResult:
    evidence = [EvidenceOut(method=e.method, rank=e.rank, score=_score(e.score), section=e.section,
                            snippet=e.snippet, matched_terms=list(e.matched_terms))
                for e in s.evidence] if explain else []
    return SkillResult(skill_id=s.slug, name=skills.name(s.slug), rank=s.rank, score=_score(s.score),
                       score_type=s.score_type, section=s.section,
                       snippet=s.snippet if explain else None, retrieval_methods=list(s.methods),
                       evidence=evidence, provenance=skills.provenance(s.slug),
                       prerequisites=skills.engine.prerequisites(s.slug))


def _ranks(s: RankedSkill | None) -> str:
    """` (dense #2, bm25 #1)`: where each method ranked a skill; '' when unknown."""
    if s is None or not s.evidence:
        return ""
    return " (" + ", ".join(f"{e.method} #{e.rank}" for e in s.evidence) + ")"


def _competitor_reason(result: RouteResult, slug: str, s: RankedSkill | None) -> str:
    """Why a competitor contests the top result, from the confidence signals and evidence."""
    top = result.top
    leaders = result.confidence.signals.get("leaders") or {}
    methods = [m for m, lead in leaders.items() if lead == slug]
    if methods:
        return f"ranked first by {' and '.join(methods)}" + _ranks(s)
    rival = top.slug if top is not None else "the top result"
    if s is not None and top is not None and s.rank == 2:
        exact = round(s.score, 12) == round(top.score, 12)
        return f"{'score tie' if exact else 'near score tie'} with {rival}" + _ranks(s)
    return f"score (near-)tie with {rival}" + _ranks(s)


def alternatives(skills: _Skills, result: RouteResult) -> list[Alternative]:
    """The confidence's competitors if any, else the next two results by score."""
    ranked = {s.slug: s for s in result.results}
    if result.confidence.competitors:
        return [Alternative(skill_id=c, name=skills.name(c), rank=ranked[c].rank if c in ranked else None,
                            reason=_competitor_reason(result, c, ranked.get(c)))
                for c in result.confidence.competitors]
    return [Alternative(skill_id=s.slug, name=skills.name(s.slug), rank=s.rank,
                        reason="next best by score" + _ranks(s))
            for s in result.results[1:3]]


def composition_out(engine: EngineLike, c: Composition, skills: _Skills | None = None) -> CompositionOut:
    skills = skills or _Skills(engine)
    intents = [IntentOut(index=it.index, text=it.text, query=it.query, selected=it.selected,
                         confidence=it.result.confidence.level, action=it.result.confidence.action,
                         ambiguous=it.result.confidence.ambiguous,
                         competitors=list(it.result.confidence.competitors))
               for it in c.intents]
    plan = [PlanStepOut(skill_id=s.slug, name=skills.name(s.slug), position=s.position, role=s.role,
                        intents=list(s.intents), required_by=list(s.required_by), reason=s.reason)
            for s in c.steps]
    edges = [EdgeOut(source=e.source, target=e.target, relationship=e.relationship,
                     confidence=e.confidence, provenance=e.provenance) for e in c.edges]
    return CompositionOut(multi_intent=c.multi_intent, intents=intents, plan=plan, edges=edges,
                          conflicts=[list(p) for p in c.conflicts], unmatched=list(c.unmatched),
                          notes=list(c.notes))


def route_response(engine: EngineLike, result: RouteResult, explain: bool = True,
                   composition: Composition | None = None) -> RouteResponse:
    """A RouteResult (and optional Composition) as the API's RouteResponse.

    Args:
        engine: supplies skill names, provenance, prerequisites and the corpus identity.
        result: the routed query.
        explain: include snippets and per-method evidence.
        composition: `engine.compose` of the same query, for multi_intent requests.
    """
    skills = _Skills(engine)
    return RouteResponse(
        query=result.query, confidence=confidence_out(result.confidence),
        results=[skill_result(skills, s, explain) for s in result.results],
        alternatives=alternatives(skills, result),
        reranked=result.reranked, rerank_error=_display(result.rerank_error), mode=result.mode,
        candidates=plain(result.candidates), timings_ms=plain(result.timings_ms),
        corpus=corpus_ref(engine.corpus),
        composition=composition_out(engine, composition, skills) if composition is not None else None)


def search_response(engine: EngineLike, result: RouteResult, pool: int) -> SearchResponse:
    """The backward-compatible GET /search body."""
    skills = _Skills(engine)
    conf = result.confidence
    return SearchResponse(
        query=result.query, reranked=result.reranked, pool=pool,
        results=[SearchHit(slug=s.slug, score=_score(s.score), section=s.section, rank=s.rank,
                           name=skills.name(s.slug), retrieval_methods=list(s.methods))
                 for s in result.results],
        confidence=ConfidenceBrief(level=conf.level, action=conf.action, ambiguous=conf.ambiguous))


def learning_path_response(lp: dict[str, Any]) -> LearningPathResponse:
    return LearningPathResponse.model_validate(lp)


def skill_detail(data: dict[str, Any]) -> SkillDetail:
    return SkillDetail.model_validate(plain(data))


def ready_response(status: dict[str, Any]) -> ReadyResponse:
    """`Engine.status()` as the /ready body, with no absolute paths under the repo root."""
    return ReadyResponse.model_validate(_display(plain(status)))
