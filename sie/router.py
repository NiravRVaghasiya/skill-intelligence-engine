"""Top-level hybrid retrieval: dense + sparse -> RRF -> rerank -> dedup by skill.

`HybridRouter.route()` returns a `RouteResult`: the deduped skill ranking with per-retriever
evidence (rank, raw score, section, matched terms), a heuristic routing confidence
(sie/confidence.py), candidate counts and per-stage timings. `retrieve()` is the same
ranking as plain `Hit`s.

Usage:
    python -m sie.router --build                        # build indexes from data/skills
    python -m sie.router --skills DIR --persist-dir DIR --build   # another corpus / index dir
    python -m sie.router "your task description"        # query (hybrid + rerank)
    python -m sie.router "your task" --no-rerank        # ablation: RRF only
    python -m sie.router "your task" --mode dense       # ablation: one retriever only
    python -m sie.router "your task" --explain          # per-retriever evidence per result
    python -m sie.router "your task" --json             # the full RouteResult as JSON
    python -m sie.router "task A, then task B" --multi  # multi-intent composition (opt-in)
    python -m sie.router --path rag-evaluation          # GraphRAG learning path
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .confidence import DEFAULT_POLICY, ConfidencePolicy, assess, summary
from .ingest import load_corpus, load_corpus_report
from .chunking import chunk_skill
from .index.dense import DenseIndex
from .index.sparse import SparseIndex
from .index.fuse import best_per_skill, reciprocal_rank_fusion
from .observability import emit, query_fields
from .rerank import Reranker, RerankerUnavailable, display_model
from .models import (Chunk, Confidence, CorpusInfo, Hit, MethodEvidence, RankedSkill,
                     RouteResult, Skill)

MODES = ("hybrid", "dense", "sparse")
MODE_METHODS = {"hybrid": ("dense", "bm25"), "dense": ("dense",), "sparse": ("bm25",)}
SEED = 42
INDEX_SCHEMA = 2            # version of the metadata `build()` records on the dense index
SNIPPET_CHARS = 200


def set_seeds(seed: int = SEED) -> None:
    """Seed python/numpy, and torch if it is already loaded (never imports it)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Stable hash of chunk ids + texts; detects a dense index built from another corpus."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8") + b"\0" + c.text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def dedup_by_skill(hits: list[Hit], k: int) -> list[Hit]:
    """Keep the first (best-scored) hit per skill, up to k."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.skill_slug in seen:
            continue
        seen.add(h.skill_slug)
        out.append(h)
        if len(out) >= k:
            break
    return out


def _positive(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval knobs; the defaults are the benchmarked configuration.

    Attributes:
        top_k: skills returned when a call does not pass `k`.
        candidate_pool: chunks taken from each retriever when a call does not pass `pool`.
        rerank_k: how many top fused skills have their retrieved chunks cross-encoded;
            None = the call's pool (the original behavior: every fused skill in the pool).
        rrf_k: RRF smoothing constant, score = sum(1 / (rrf_k + rank)).
    """
    top_k: int = 5
    candidate_pool: int = 20
    rerank_k: int | None = None
    rrf_k: int = 60

    def __post_init__(self) -> None:
        _positive("top_k", self.top_k)
        _positive("candidate_pool", self.candidate_pool)
        if self.rerank_k is not None:
            _positive("rerank_k", self.rerank_k)
        _positive("rrf_k", self.rrf_k)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HybridRouter:
    """Routes a task description to skills: dense + BM25 -> RRF -> optional cross-encoder.

    Corpus loading and index validation happen once, lazily, under a lock, so one router can
    serve concurrent `route()` calls. Models load lazily on first use; call `warm_up()` at
    service start so the first request doesn't pay for it.
    """

    def __init__(self, skills_dir: str = "data/skills", use_reranker: bool = True,
                 mode: str = "hybrid", persist_dir: str = "data/chroma",
                 strict_rerank: bool = False, config: RetrievalConfig | None = None,
                 policy: ConfidencePolicy | None = None, reranker_model: str | None = None,
                 manifest: str | None = None):
        """
        Args:
            skills_dir: corpus root.
            use_reranker: cross-encode the fused candidates (False == `--no-rerank`).
            mode: "hybrid" (dense + BM25), or "dense" / "sparse" alone for ablations.
            persist_dir: ChromaDB directory.
            strict_rerank: raise if the cross-encoder can't load, instead of warning and
                falling back to RRF order.
            config: retrieval knobs (default `RetrievalConfig()`).
            policy: routing-confidence thresholds (default `confidence.DEFAULT_POLICY`).
            reranker_model: cross-encoder id or local dir (default `rerank.default_model()`).
            manifest: explicit corpus.toml (default `<skills_dir>/corpus.toml` if present).
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
        if config is not None and not isinstance(config, RetrievalConfig):
            raise TypeError(f"config must be a RetrievalConfig, got {type(config).__name__}")
        self.skills_dir = skills_dir
        self.manifest = manifest
        self.mode = mode
        self.config = config if config is not None else RetrievalConfig()
        self.policy = policy if policy is not None else DEFAULT_POLICY
        self.dense = DenseIndex(persist_dir=persist_dir)
        self.sparse = SparseIndex()
        self.reranker = Reranker(model_name=reranker_model) if use_reranker else None
        self.strict_rerank = strict_rerank
        self.rerank_error: str | None = None
        # Convenience for single-threaded callers: did the cross-encoder order the *last*
        # route()/retrieve() result? Shared across threads, so concurrent callers should read
        # RouteResult.reranked instead.
        self.last_reranked = False
        self.skills: dict[str, Skill] = {}
        self.corpus: CorpusInfo | None = None
        self._chunks: list[Chunk] | None = None
        self._texts: dict[str, str] = {}
        self._dense_gen: int | None = None   # dense.generation last verified fresh
        self._init_lock = threading.Lock()   # guards corpus loading and index (re)validation
        self._rerank_lock = threading.Lock() # guards disabling / restoring the shared reranker
        set_seeds()

    # -- corpus + indexes ---------------------------------------------------------------

    def _load_chunks(self) -> list[Chunk]:
        """Chunks of the whole corpus, loaded once (thread-safe)."""
        if self._chunks is None:
            with self._init_lock:
                return self._load_chunks_locked()
        return self._chunks

    def _load_chunks_locked(self) -> list[Chunk]:
        if self._chunks is None:
            try:
                report = load_corpus_report(self.skills_dir, self.manifest)
            except (OSError, ValueError) as e:     # missing / invalid explicit manifest
                raise RuntimeError(f"cannot load the corpus at {self.skills_dir}: {e}") from e
            for path, reason in report.skipped:
                print(f"[ingest] skipped {path}: {reason}", file=sys.stderr)
            if not report.skills:
                raise RuntimeError(f"no skills found under {self.skills_dir}")
            chunks = [c for s in report.skills for c in chunk_skill(s)]
            texts = {c.chunk_id: c.text for c in chunks}
            if len(texts) != len(chunks):
                raise RuntimeError("chunking produced duplicate chunk ids")
            self.skills = {s.slug: s for s in report.skills}
            self.corpus = report.corpus
            self._texts = texts
            self._chunks = chunks            # published last: the fast path keys on it
        return self._chunks

    def _index_metadata(self, chunks: list[Chunk]) -> dict:
        """What `build()` records on the dense index next to the chunk fingerprint."""
        corpus = self.corpus
        return {"index_schema": INDEX_SCHEMA, "embedder": self.dense.embedder,
                "model_name": self.dense.model_name,
                "corpus_name": corpus.name if corpus else "",
                "corpus_version": corpus.version if corpus else "",
                "corpus_fingerprint": corpus.fingerprint if corpus else "",
                "n_chunks": len(chunks), "n_skills": len(self.skills), "indexed_at": _utc_now()}

    def build(self) -> int:
        """Rebuild the persisted dense index and the in-memory BM25 index; returns #chunks."""
        with self._init_lock:
            chunks = self._load_chunks_locked()
            self.dense.build(chunks, fingerprint=corpus_fingerprint(chunks),
                             metadata=self._index_metadata(chunks))
            self.sparse.build(chunks)
            self._dense_gen = self.dense.generation
        return len(chunks)

    def _dense_info(self) -> dict | None:
        """The dense index's recorded metadata; None if never built or the backend has no info()."""
        info = getattr(self.dense, "info", None)
        return info() if callable(info) else None

    def _is_ready(self) -> bool:
        return (self._chunks is not None and self.sparse.is_built
                and (self.mode == "sparse" or self._dense_gen == self.dense.generation))

    def _ensure_ready(self) -> None:
        """Load the corpus and BM25; in dense modes verify the persisted index matches.

        Raises:
            RuntimeError: no skills, dense index missing, or stale (other corpus, embedder or
                embedding model than the one configured).
        """
        if self._is_ready():                 # fast path: no lock once verified
            return
        with self._init_lock:
            chunks = self._load_chunks_locked()
            if not self.sparse.is_built:
                self.sparse.build(chunks)
            if self.mode == "sparse" or self._dense_gen == self.dense.generation:
                return
            rebind = getattr(self.dense, "rebind", None)
            if callable(rebind):             # re-read the index as persisted now, not a cached
                rebind()                     # snapshot (it may have been rebuilt since)
            generation = self.dense.generation
            built = self.dense.fingerprint()
            if built is None:
                raise RuntimeError("dense index not built; run `python -m sie.router --build`")
            if built != corpus_fingerprint(chunks) or self.dense.built_with() != self.dense.embedder:
                raise RuntimeError("dense index is stale (corpus or embedder changed); "
                                   "run `python -m sie.router --build`")
            recorded = (self._dense_info() or {}).get("model_name")
            configured = getattr(self.dense, "model_name", None)
            if recorded is not None and configured is not None and recorded != configured:
                raise RuntimeError(f"dense index is stale (built with model '{recorded}', "
                                   f"configured '{configured}'); run `python -m sie.router --build`")
            self._dense_gen = generation

    def revalidate(self) -> None:
        """Re-check dense-index freshness if its handle was rebound since the last check.

        Cheap: reads the persisted index's fingerprint/metadata only; never loads a model or
        reloads the corpus. A no-op in sparse mode, before the corpus is loaded, before the
        index was ever verified, or while the verified handle is still current.

        Raises:
            RuntimeError: the dense index is now missing or stale, exactly as `_ensure_ready`.
        """
        if (self.mode == "sparse" or self._chunks is None or self._dense_gen is None
                or self._dense_gen == self.dense.generation):
            return
        self._ensure_ready()

    def _disable_reranker(self, reranker, error: Exception) -> None:
        """Stop using `reranker` after its load failed, unless another thread has loaded it."""
        with self._rerank_lock:
            if not getattr(reranker, "loaded", False) and self.reranker is reranker:
                self.rerank_error, self.reranker = str(error), None
        print(f"[router] WARNING: {error}; falling back to RRF order", file=sys.stderr)

    def _restore_reranker(self, reranker) -> None:
        """Put back a reranker that loaded after a concurrent load failure disabled it."""
        with self._rerank_lock:
            if self.reranker is None and getattr(reranker, "loaded", False):
                self.reranker, self.rerank_error = reranker, None

    def warm_up(self) -> dict:
        """Load what the first query would: corpus, BM25, embedding model, cross-encoder.

        Returns:
            `status()` afterwards.

        Raises:
            RuntimeError: the dense index is missing or stale (dense modes), as in `route()`.
            RerankerUnavailable: the cross-encoder can't load and `strict_rerank` is set;
                otherwise the router falls back to RRF order with a warning, as in `route()`.
        """
        self._ensure_ready()
        if self.mode != "sparse":
            self.dense.search("warm up", k=1)       # embeds one string: loads the embedder
            self.revalidate()                        # that search may have rebound the handle
        reranker = self.reranker
        load = getattr(reranker, "load", None)
        if callable(load):
            try:
                load()
            except RerankerUnavailable as e:
                if self.strict_rerank:
                    raise
                self._disable_reranker(reranker, e)
            else:
                self._restore_reranker(reranker)
        return self.status()

    def status(self) -> dict:
        """What is loaded and ready; never loads a model or embeds anything.

        In dense modes this reads the persisted index's metadata (`dense.info()`).
        """
        corpus = self.corpus
        required = self.mode != "sparse"
        dense: dict = {"required": required, "ready": False, "index": None}
        if required:
            dense["ready"] = (self._chunks is not None and self._dense_gen is not None
                              and self._dense_gen == self.dense.generation)
            try:
                dense["index"] = self._dense_info()
            except Exception as e:           # e.g. chromadb missing: report, don't raise
                dense["error"] = f"{type(e).__name__}: {e}"
        reranker = self.reranker
        model = getattr(reranker, "model_name", None)
        return {
            "mode": self.mode,
            "corpus": None if corpus is None else {
                "name": corpus.name, "version": corpus.version,
                "fingerprint": corpus.fingerprint, "n_skills": corpus.n_skills},
            "chunks": len(self._chunks) if self._chunks is not None else 0,
            "sparse": self.sparse.is_built,
            "dense": dense,
            "reranker": {"enabled": reranker is not None,
                         "loaded": bool(getattr(reranker, "loaded", False)),
                         "model": None if model is None else display_model(model),
                         "error": self.rerank_error},
        }

    # -- routing ------------------------------------------------------------------------

    def _text(self, hit: Hit) -> str:
        return self._texts.get(hit.chunk_id) or hit.snippet

    def _search(self, query: str, pool: int, timings: dict[str, float]) -> dict[str, list[Hit]]:
        """Chunk-level rankings per retriever that ran, in method order (dense, bm25)."""
        rankings: dict[str, list[Hit]] = {}
        if self.mode in ("hybrid", "dense"):
            start = time.perf_counter()
            rankings["dense"] = self.dense.search(query, k=pool)
            self.revalidate()      # the search rebound the handle (external rebuild): re-verify
            timings["dense"] = _ms(start)
        if self.mode in ("hybrid", "sparse"):
            start = time.perf_counter()
            rankings["bm25"] = self.sparse.search(query, k=pool)
            timings["bm25"] = _ms(start)
        return rankings

    @staticmethod
    def _rerank_candidates(rankings: dict[str, list[Hit]], top: list[Hit]) -> list[Hit]:
        """Every retrieved chunk (first occurrence, dense before bm25) of the `top` fused skills."""
        keep = {h.skill_slug for h in top}
        candidates: dict[str, Hit] = {}
        for ranking in rankings.values():
            for h in ranking:
                if h.skill_slug in keep:
                    candidates.setdefault(h.chunk_id, h)
        return list(candidates.values())

    def _rerank(self, reranker, query: str,
                candidates: list[Hit]) -> tuple[list[Hit] | None, str | None]:
        """(cross-encoded candidates best first, None), or (None, why) when unavailable (non-strict)."""
        try:
            scored = reranker.rerank(query, candidates, texts=self._texts)
        except RerankerUnavailable as e:
            if self.strict_rerank:
                raise
            self._disable_reranker(reranker, e)
            return None, str(e)
        self._restore_reranker(reranker)
        return scored, None

    def _explain(self, query: str, best: dict[str, list[Hit]], ranking: list[Hit],
                 reranked: bool) -> list[RankedSkill]:
        """One RankedSkill per ranked hit, with the evidence of every retriever that found it.

        Args:
            query: the routed query (for bm25 matched terms).
            best: method -> `best_per_skill` of its chunk ranking (skill-level ranks).
            ranking: the final deduped ranking (fused or cross-encoded).
            reranked: whether the cross-encoder produced `ranking`.
        """
        found = {m: {h.skill_slug: (rank, h) for rank, h in enumerate(hits, 1)}
                 for m, hits in best.items()}
        out: list[RankedSkill] = []
        for rank, hit in enumerate(ranking, 1):
            evidence: list[MethodEvidence] = []
            for method, by_skill in found.items():
                if hit.skill_slug not in by_skill:
                    continue
                m_rank, m_hit = by_skill[hit.skill_slug]
                terms = self.sparse.matched_terms(query, self._text(m_hit)) if method == "bm25" else []
                evidence.append(MethodEvidence(method=method, rank=m_rank, score=m_hit.score,
                                               section=m_hit.section,
                                               snippet=m_hit.snippet[:SNIPPET_CHARS],
                                               matched_terms=terms))
            methods = [e.method for e in evidence]
            if reranked:
                evidence.append(MethodEvidence(method="rerank", rank=rank, score=hit.score,
                                               section=hit.section,
                                               snippet=hit.snippet[:SNIPPET_CHARS]))
            out.append(RankedSkill(slug=hit.skill_slug, rank=rank, score=hit.score,
                                   score_type="cross-encoder" if reranked else "rrf",
                                   section=hit.section, snippet=hit.snippet,
                                   chunk_id=hit.chunk_id, methods=methods, evidence=evidence))
        return out

    def _coverage(self, query: str, rankings: dict[str, list[Hit]], slug: str) -> float | None:
        """Best lexical coverage of the query over `slug`'s retrieved chunks (any retriever)."""
        if not self.sparse.is_built:
            return None
        chunks = {h.chunk_id: h for ranking in rankings.values() for h in ranking
                  if h.skill_slug == slug}
        if not chunks:
            return None
        return max(self.sparse.coverage(query, self._text(h)) for h in chunks.values())

    def _empty(self, query: str, reason: str, start: float) -> RouteResult:
        return RouteResult(query=query, results=[],
                           confidence=Confidence(level="none", action="abstain", reasons=[reason]),
                           mode=self.mode, reranked=False, rerank_error=self.rerank_error,
                           candidates={}, timings_ms={"total": _ms(start)})

    def _route(self, query: str, k: int | None, pool: int | None,
               rerank_k: int | None) -> RouteResult:
        start = time.perf_counter()
        k = self.config.top_k if k is None else k
        pool = self.config.candidate_pool if pool is None else pool
        if rerank_k is None:
            rerank_k = self.config.rerank_k if self.config.rerank_k is not None else pool
        if pool < 1:
            raise ValueError("pool must be >= 1")
        if rerank_k < 1:
            raise ValueError("rerank_k must be >= 1")
        if not query.strip():
            return self._empty(query, "empty query", start)
        if k <= 0:
            return self._empty(query, "k <= 0", start)
        self._ensure_ready()
        timings: dict[str, float] = {}
        rankings = self._search(query, pool, timings)
        t = time.perf_counter()
        fused = reciprocal_rank_fusion(list(rankings.values()), k=self.config.rrf_k)
        timings["fuse"] = _ms(t)
        candidates = {"dense": len(rankings["dense"])} if "dense" in rankings else {}
        if "bm25" in rankings:
            candidates["bm25"] = len(rankings["bm25"])
        candidates["fused"] = len(fused)

        ordered, reranked, rerank_error = fused, False, None
        reranker = self.reranker             # snapshot: a failed load elsewhere may clear it
        if reranker is not None:
            pairs = self._rerank_candidates(rankings, fused[:rerank_k])
            if pairs:
                candidates["rerank"] = len(pairs)
                t = time.perf_counter()
                scored, rerank_error = self._rerank(reranker, query, pairs)
                timings["rerank"] = _ms(t)
                if scored is not None:
                    ordered, reranked = scored, True

        t = time.perf_counter()
        best = {m: best_per_skill(hits) for m, hits in rankings.items()}
        ranking = self._explain(query, best, dedup_by_skill(ordered, len(ordered)), reranked)
        methods = list(rankings)
        leaders = {m: hits[0].skill_slug for m, hits in best.items() if hits}
        coverage = self._coverage(query, rankings, ranking[0].slug) if ranking else None
        dense_scores = [h.score for h in rankings.get("dense", [])]
        confidence = assess(ranking, methods, leaders, coverage=coverage,
                            top_similarity=max(dense_scores) if dense_scores else None,
                            reranked=reranked, policy=self.policy)
        timings["explain"] = _ms(t)
        timings["total"] = _ms(start)
        if not reranked and rerank_error is None:
            rerank_error = self.rerank_error     # an earlier load failure disabled the reranker
        return RouteResult(query=query, results=ranking[:k], confidence=confidence,
                           mode=self.mode, reranked=reranked, rerank_error=rerank_error,
                           candidates=candidates, timings_ms=timings)

    def route(self, query: str, k: int | None = None, pool: int | None = None,
              rerank_k: int | None = None) -> RouteResult:
        """Rank skills for a query and explain the ranking.

        Args:
            query: free-text task description.
            k: skills to return (default `config.top_k`).
            pool: chunks taken from each retriever (default `config.candidate_pool`).
            rerank_k: top fused skills whose retrieved chunks are cross-encoded (default
                `config.rerank_k`, else `pool`).

        Returns:
            RouteResult whose `results` are the top k of the deduped ranking (one entry per
            skill, best section kept, highest score first). The confidence is assessed on
            the full ranking, so a runner-up counts even when k=1. A blank query or k <= 0
            returns no results (confidence "none") without touching any index.

        Raises:
            ValueError: pool or rerank_k < 1.
            RuntimeError: corpus missing, dense index missing/stale, or (strict) no reranker.
        """
        self.last_reranked = False
        try:
            result = self._route(query, k, pool, rerank_k)
        except Exception as e:
            _telemetry("route_error", query, lambda: {
                "error_type": type(e).__name__, "error": str(e), "mode": self.mode})
            raise
        self.last_reranked = result.reranked
        _telemetry("route", query, lambda: _route_event(result))
        return result

    def retrieve(self, query: str, k: int | None = None, pool: int | None = None,
                 rerank_k: int | None = None) -> list[Hit]:
        """Return up to k hits, one per skill (best section kept), highest score first.

        Args:
            query: free-text task description.
            k: number of skills to return (default `config.top_k`, i.e. 5).
            pool: chunks taken from each retriever (default `config.candidate_pool`, i.e. 20).
            rerank_k: fused skills passed to the reranker (default: the pool).

        The pool is fixed rather than scaled with k, so retrieve(q, 3) is always a prefix of
        retrieve(q, 10). Fewer than k hits come back when the pool covers fewer skills
        (about 10-16 at pool=20 on this corpus); raise `pool` for exhaustive listings.
        """
        return self.route(query, k=k, pool=pool, rerank_k=rerank_k).hits


def _route_event(result: RouteResult) -> dict:
    """The fields of a "route" observability event (see sie/observability.py)."""
    conf = result.confidence
    return {"mode": result.mode, "reranked": result.reranked, "rerank_error": result.rerank_error,
            "top_skill": result.top.slug if result.top else None, "confidence": conf.level,
            "action": conf.action, "ambiguous": conf.ambiguous, "n_results": len(result.results),
            "candidates": dict(result.candidates), "timings_ms": dict(result.timings_ms)}


def _telemetry(event: str, query: str, fields) -> None:
    """Emit `event` with `fields()` plus the query identity; never raises.

    Telemetry must neither discard a finished result nor replace the exception being
    re-raised on the error path, so a failure building or emitting the event is swallowed
    (and the query identity alone is dropped if it can't be computed).
    """
    try:
        record = fields()
        try:
            record.update(query_fields(query))
        except Exception:
            pass
        emit(event, **record)
    except Exception:
        pass


def route_result_dict(result: RouteResult) -> dict:
    """RouteResult as plain JSON-ready data, without internal chunk ids."""
    data = dataclasses.asdict(result)
    for item in data["results"]:
        item.pop("chunk_id", None)
    return data


_SCORE_LABEL = {"dense": "cosine", "bm25": "bm25", "rerank": "ce"}


def format_evidence(result: RouteResult, skill: RankedSkill) -> list[str]:
    """Indented `--explain` lines: each method's rank, raw score and section (bm25: terms)."""
    by_method = {e.method: e for e in skill.evidence}
    lines = []
    for method in MODE_METHODS[result.mode] + (("rerank",) if result.reranked else ()):
        e = by_method.get(method)
        if e is None:
            lines.append(f"      {method:6s}  -    not retrieved")
            continue
        line = f"      {method:6s} #{e.rank:<3d} {_SCORE_LABEL[method]}={e.score:.3f}  [{e.section}]"
        if method == "bm25":
            line += f"  terms: {', '.join(e.matched_terms) or '-'}"
        lines.append(line)
    return lines


def format_learning_path(skills_dir: str, target: str) -> str:
    """Human-readable learning path to `target`: ordered steps, why each one is there,
    see-also, conflicts, and notes on missing graph information.

    Raises:
        KeyError: unknown slug (message suggests close matches). ValueError: requires cycle.
    """
    import difflib
    from .graph.build import build_graph
    from .graph.paths import learning_path, render_learning_path
    g = build_graph(load_corpus(skills_dir))
    if target not in g:
        close = difflib.get_close_matches(target, list(g.nodes), n=3)
        raise KeyError(f"unknown skill '{target}'" + (f"; did you mean {', '.join(close)}?" if close else ""))
    return render_learning_path(g, learning_path(g, target))


def main() -> None:
    from .observability import configure_logging
    ap = argparse.ArgumentParser(description="Hybrid skill retrieval: dense + BM25 -> RRF -> rerank.")
    ap.add_argument("query", nargs="?", help="task description to route")
    ap.add_argument("--build", action="store_true", help="(re)build indexes")
    ap.add_argument("--skills", default="data/skills")
    ap.add_argument("--manifest", metavar="PATH", help="corpus.toml (default <skills>/corpus.toml)")
    ap.add_argument("--persist-dir", default="data/chroma", metavar="DIR",
                    help="ChromaDB directory of the dense index (default data/chroma)")
    ap.add_argument("-k", type=int, default=5, help="number of skills to return")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (ablation)")
    ap.add_argument("--mode", choices=MODES, default="hybrid", help="retrievers to use")
    ap.add_argument("--pool", type=int, default=20, help="candidate chunks per retriever")
    ap.add_argument("--rerank-k", type=int, default=None, metavar="N",
                    help="cross-encode the chunks of the top N fused skills (default: --pool)")
    ap.add_argument("--explain", action="store_true", help="per-retriever evidence for each result")
    ap.add_argument("--json", action="store_true", help="print the full RouteResult as JSON")
    ap.add_argument("--path", metavar="SLUG", help="print the learning path to SLUG (no index needed)")
    ap.add_argument("--multi", action="store_true",
                    help="split the query into intents and print a dependency-ordered skill plan")
    ap.add_argument("--overlay", action="append", default=[], metavar="PATH",
                    help="edge overlay JSON applied to the graph for --multi / --path (repeatable)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # never crash on a cp1252 console
    if os.getenv("SIE_LOG_LEVEL"):                 # opt-in: SIE_LOG_LEVEL=INFO prints routing events
        try:
            configure_logging()
        except ValueError as e:
            sys.exit(f"[router] SIE_LOG_LEVEL: {e}")
    if args.path:
        try:
            if args.overlay or args.manifest:      # the engine applies overlays / the manifest
                from .engine import Engine
                engine = Engine(skills_dir=args.skills, mode="sparse", use_reranker=False,
                                manifest=args.manifest, overlays=args.overlay)
                print(engine.render_learning_path(args.path))
            else:
                print(format_learning_path(args.skills, args.path))
        except (KeyError, ValueError, RuntimeError) as e:
            sys.exit(f"[path] {e.args[0] if e.args else e}")
        return
    from .compose import composition_summary
    from .engine import Engine, _missing_dependency, composition_dict
    engine = Engine(skills_dir=args.skills, persist_dir=args.persist_dir,
                    use_reranker=not args.no_rerank, mode=args.mode,
                    manifest=args.manifest, overlays=args.overlay)
    r = engine.router
    try:
        if args.build:
            print(f"[router] indexed {r.build()} chunks -> {r.dense.persist_dir}/",
                  file=sys.stderr if args.json else sys.stdout)
            if not args.query:
                return
        if not args.query:
            ap.error("provide a query, or use --build")
        if args.multi:                             # composite: one plan, not one ranking
            c = engine.compose(args.query, k=args.k, pool=args.pool, rerank_k=args.rerank_k)
        else:
            result = r.route(args.query, k=args.k, pool=args.pool, rerank_k=args.rerank_k)
    except ImportError as e:                       # core install: chromadb (retrieval extra) missing
        sys.exit(f"[router] {_missing_dependency(e)}")
    except (RuntimeError, ValueError) as e:
        sys.exit(f"[router] {e}")
    if args.multi:
        print(json.dumps(composition_dict(c), ensure_ascii=False, indent=2) if args.json
              else "\n".join(composition_summary(c)))
        return
    if args.json:
        print(json.dumps(route_result_dict(result), ensure_ascii=False, indent=2))
        return
    print(f"[router] mode={r.mode} ranking={'cross-encoder' if result.reranked else 'rrf'}")
    for i, s in enumerate(result.results, 1):
        print(f"{i:2d}. {s.slug:28s} score={s.score:.3f}  [{s.section}]")
        if args.explain:
            print("\n".join(format_evidence(result, s)))
    print(f"[router] confidence: {summary(result.confidence)}")


if __name__ == "__main__":
    main()
