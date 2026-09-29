"""Baseline (keyword) vs SIE on the labeled query sets -> benchmarks/.

Scores every system on every query set listed in eval/datasets.toml: routing accuracy,
confidence vs correctness, out-of-scope abstention and multi-intent composition. Writes the
`metrics` block of benchmarks/RESULTS.md, benchmarks/metrics.json (every metric, sorted keys,
6 decimals), benchmarks/results.png (+ a dark variant) and benchmarks/per_query.jsonl. Every
number in those files comes from here, and none of them holds a timing, so a rerun on the
same corpus, index and query sets reproduces them byte for byte.

Usage:
    python -m eval.run_eval                     # rebuild indexes, then score every system
    python -m eval.run_eval --no-build          # reuse the persisted data/chroma/ index
    python -m eval.run_eval --smoke             # CI: no index, no models, no writes; replays the
                                                # dense fixture and diffs against metrics.json
    python -m eval.run_eval --record-fixture    # store live dense rankings for --smoke (reads
                                                # the persisted index; never rebuilds it)
    python -m eval.run_eval --datasets other.toml --out some/dir    # any external query sets
    python -m eval.run_eval --systems bm25,hybrid --pool 40 --out some/dir
    python -m eval.run_eval --reranker path/to/weights --require-reranker
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sie.chunking import card_chunk
from sie.compose import compose
from sie.confidence import LEVELS
from sie.graph.build import build_graph
from sie.hub import looks_like_path
from sie.index.sparse import SparseIndex
from sie.ingest import load_corpus
from sie.models import Chunk, CorpusInfo, Hit, RouteResult, Skill
from sie.router import HybridRouter, RetrievalConfig, corpus_fingerprint
from sie.rerank import LOCAL_CE_DIR, LOCAL_CE_HINT, RerankerUnavailable
from eval.baseline import KeywordBaseline
from eval.datasets import DEFAULT_MANIFEST, ORIGIN_DESCRIPTIONS, ORIGINS, Dataset, golds, load_datasets
from eval.metrics import (ACTIONS, abstention, bootstrap_ci, first_relevant_rank, intent_metrics,
                          mean_metrics, paired_delta_ci, per_query_ks, selective)
from eval.replay import (FIXTURE, RERECORD, fixture_queries, load_fixture, record, replay_router,
                         write_fixture)
from eval.report import Bar, md_table, render_chart, replace_block

QUERIES = Path("eval/queries.jsonl")
SOURCE_QUERIES = Path("eval/queries_source_evals.jsonl")
OUT = Path("benchmarks")
README = Path("README.md")
K = 3
KS = (1, 3, 5)
DEPTH = 10          # ranks scored per query (MRR looks this deep)
METRICS = ["top-1", "recall@3", "recall@5", "mrr", "ndcg@3", "ndcg@5"]
CHART_METRICS = [f"recall@{K}", "mrr", f"ndcg@{K}"]
COMPOSE_K = 3       # compose(): skills routed per intent (its default)
MAX_INTENTS = 5     # compose(): intents routed per request (its default)
DECIMALS = 6        # metrics.json rounding
TOLERANCE = 1e-9    # --smoke: allowed difference after rounding
SCHEMA = 1          # metrics.json layout version
TIE_DECIMALS = 6    # raw single-retriever scores in the tie check: the dense fixture's precision,
                    # so a replayed ranking finds exactly the ties the live one has
REPO_ROOT = Path(__file__).resolve().parents[1]

# The first full run, of code that has since changed (see `first_measurement_note`). Kept as
# constants because that code no longer exists, so these numbers cannot be recomputed.
FIRST_BM25_IN_SCOPE_ABSTAIN = (62, 111)     # BM25-only abstentions / in-scope queries (main + source)
FIRST_COMPOSE_HYBRID = {"recall": 0.593, "precision": 0.803, "exact": 0.280, "count_match": 0.480}


@dataclass(frozen=True)
class EvalConfig:
    """Retrieval settings of one run; None keeps the router's defaults (the benchmarked config)."""
    pool: int | None = None          # candidate chunks per retriever
    rerank_k: int | None = None      # fused skills whose chunks are cross-encoded
    reranker: str | None = None      # cross-encoder id / local dir for the rerank system

    def retrieval(self) -> RetrievalConfig:
        default = RetrievalConfig()
        return RetrievalConfig(candidate_pool=default.candidate_pool if self.pool is None else self.pool,
                               rerank_k=self.rerank_k)

    @property
    def is_default(self) -> bool:
        """Pool and rerank_k are the benchmarked defaults (the reranker model may differ)."""
        return self.retrieval() == RetrievalConfig()


@dataclass
class System:
    key: str
    label: str
    family: str                      # "keyword" | "sie"
    make: Callable[..., object]      # make(cfg: EvalConfig) -> object with .retrieve(query, k)
                                     # (and .route(query, k) -> RouteResult for SIE routers)


class CardOnlyBM25:
    """Ablation: BM25 over one Card chunk per skill (display name + description + capability
    tags), i.e. roughly the frontmatter fields the keyword router scores, and no body text."""

    def __init__(self, skills_dir: str = "data/skills"):
        self.sparse = SparseIndex()
        self.sparse.build([card_chunk(s) for s in load_corpus(skills_dir)])

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        return self.sparse.search(query, k=k)       # one chunk per skill: already deduped


def _router(**kw) -> Callable[..., HybridRouter]:
    def make(cfg: EvalConfig | None = None) -> HybridRouter:
        return HybridRouter(config=(cfg or EvalConfig()).retrieval(), **kw)
    return make


def _rerank_router(cfg: EvalConfig | None = None) -> HybridRouter:
    cfg = cfg or EvalConfig()
    return HybridRouter(use_reranker=True, strict_rerank=True, config=cfg.retrieval(),
                        reranker_model=cfg.reranker)


SYSTEMS = [
    System("keyword", "Keyword router (baseline)", "keyword", lambda cfg=None: KeywordBaseline()),
    System("keyword-shipped", "Keyword route() as shipped", "keyword",
           lambda cfg=None: KeywordBaseline(shipped=True)),
    System("bm25-card", "SIE: BM25, Card chunks only", "sie", lambda cfg=None: CardOnlyBM25()),
    System("bm25", "SIE: BM25 only", "sie", _router(mode="sparse", use_reranker=False)),
    System("dense", "SIE: dense only", "sie", _router(mode="dense", use_reranker=False)),
    System("hybrid", "SIE: hybrid RRF (--no-rerank)", "sie", _router(use_reranker=False)),
    System("hybrid-rerank", "SIE: hybrid + cross-encoder", "sie", _rerank_router),
]
BASELINE = "keyword"
_LABELS = {s.key: s.label for s in SYSTEMS}
# --smoke: systems that need no index and no model, and the metrics.json system each must match
SMOKE_REFERENCE = {"keyword": "keyword", "keyword-shipped": "keyword-shipped",
                   "bm25-card": "bm25-card", "bm25": "bm25",
                   "hybrid-replay": "hybrid", "dense-replay": "dense"}


def smoke_systems(fixture: dict) -> list[System]:
    """The lightweight systems --smoke scores: keyword/BM25 ones plus dense-fixture replays."""
    def replay(mode: str) -> Callable[..., HybridRouter]:
        return lambda cfg=None: replay_router(fixture, mode=mode,
                                              config=(cfg or EvalConfig()).retrieval())
    by_key = {s.key: s for s in SYSTEMS}
    return [by_key["keyword"], by_key["keyword-shipped"], by_key["bm25-card"], by_key["bm25"],
            System("hybrid-replay", "SIE: hybrid RRF, dense replayed from the fixture", "sie",
                   replay("hybrid")),
            System("dense-replay", "SIE: dense only, replayed from the fixture", "sie",
                   replay("dense"))]


@dataclass
class Scored:
    """One system's output on one query set; `ranked` is None when the system could not run."""
    system: System
    ranked: list[list[str]] | None   # per query, top-DEPTH slugs
    note: str = ""
    scores: list[list[float]] | None = None   # parallel to `ranked`
    routes: bool = False             # scored through route(), i.e. has a routing confidence
    levels: list[str] | None = None           # confidence level per query (route() systems)
    actions: list[str] | None = None          # route / clarify / abstain per query (all systems;
                                              # without a confidence: abstain iff no result)
    ambiguous: list[bool] | None = None
    plans: list[list[str]] | None = None      # multi-intent: compose()'s requested skills
    parts: list[int] | None = None            # multi-intent: intents compose() routed
    model: dict | None = None                 # the cross-encoder it used (`reranker_identity`)


@dataclass
class Context:
    """What every run shares: the corpus (loaded once) and the query sets."""
    router: HybridRouter             # sparse-mode router that loaded the corpus (no models)
    corpus: CorpusInfo
    skills: dict[str, Skill]
    chunks: list[Chunk]
    datasets: list[Dataset]
    skipped: list[str]               # datasets not scored, with the reason
    manifest: Path

    def of_kind(self, kind: str) -> list[Dataset]:
        return [ds for ds in self.datasets if ds.kind == kind]


def load_queries(path: Path = QUERIES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_context(manifest: str | Path = DEFAULT_MANIFEST, skills_dir: str = "data/skills") -> Context:
    """Load the corpus (no index, no model) and every query set that applies to it.

    Raises:
        RuntimeError: the corpus is missing. ValueError: a query set is malformed or
        labels a skill the corpus does not have.
    """
    router = HybridRouter(skills_dir=skills_dir, mode="sparse", use_reranker=False)
    chunks = router._load_chunks()
    datasets, skipped = load_datasets(manifest, set(router.skills), router.corpus.name)
    return Context(router=router, corpus=router.corpus, skills=dict(router.skills), chunks=chunks,
                   datasets=datasets, skipped=skipped, manifest=Path(manifest))


# -- scoring ------------------------------------------------------------------------------

_SINGLE_RETRIEVER = {"sparse": "bm25", "dense": "dense"}     # route() mode -> its only method


def _tie_scores(result: RouteResult) -> list[float]:
    """The per-skill scores the tie check compares, parallel to `result.results`.

    A single-retriever, un-reranked ranking is RRF over one list, whose scores strictly
    decrease and so never tie; its ties are the retriever's raw scores (BM25 / cosine of each
    skill's best chunk, rounded to TIE_DECIMALS). Any other ranking: its own scores.
    """
    method = _SINGLE_RETRIEVER.get(result.mode)
    if method is None or result.reranked:
        return [r.score for r in result.results]
    raw = [next((e.score for e in r.evidence if e.method == method), None) for r in result.results]
    if any(score is None for score in raw):
        return [r.score for r in result.results]
    return [round(score, TIE_DECIMALS) for score in raw]


def _score_rows(system: System, runner, rows: list[dict]) -> Scored:
    """Top-DEPTH ranking of every row's query, through route() when the system has it."""
    routes = callable(getattr(runner, "route", None))
    s = Scored(system, ranked=[], scores=[], routes=routes, actions=[],
               levels=[] if routes else None, ambiguous=[] if routes else None)
    for q in rows:
        if routes:
            result = runner.route(q["query"], k=DEPTH)
            items = list(zip([r.slug for r in result.results], _tie_scores(result)))
            s.levels.append(result.confidence.level)
            s.actions.append(result.confidence.action)
            s.ambiguous.append(result.confidence.ambiguous)
        else:
            items = [(h.skill_slug, h.score) for h in runner.retrieve(q["query"], k=DEPTH)]
            s.actions.append("route" if items else "abstain")
        s.ranked.append([slug for slug, _ in items])
        s.scores.append([score for _, score in items])
    return s


def _compose_rows(runner, rows: list[dict], graph) -> tuple[list[list[str]], list[int]]:
    """compose() every row: (requested skills in plan order, intents routed) per query.

    Prerequisites the graph adds are not requested skills, so they are not scored as intents.
    """
    plans, parts = [], []
    for q in rows:
        c = compose(runner, graph, q["query"], k=COMPOSE_K, max_intents=MAX_INTENTS)
        plans.append([step.slug for step in c.steps if step.role == "requested"])
        parts.append(len(c.intents))
    return plans, parts


def _files_sha256(root: Path) -> str | None:
    """First 16 hex of a sha256 over every file under `root` (relative path, size, bytes)."""
    h = hashlib.sha256()
    try:
        files = sorted((p for p in root.rglob("*") if p.is_file()),
                       key=lambda p: p.relative_to(root).as_posix())
        for p in files:
            h.update(f"{p.relative_to(root).as_posix()}\0{p.stat().st_size}\0".encode("utf-8"))
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    except OSError:
        return None
    return h.hexdigest()[:16]


def reranker_identity(name: str | None) -> dict | None:
    """Report-safe identity of a cross-encoder, as metrics.json and RESULTS.md record it.

    Args:
        name: the resolved model id or local directory (`Reranker.model_name`).

    Returns:
        None without a name, else {"model": spelling}: a hub id as given, the bundled local
        weights directory as LOCAL_CE_HINT, another directory under the repo repo-relative,
        any other local path its last component only (no machine paths). A local directory
        that exists also gets "model_sha256", a hash of its files, so different weights under
        the same name are told apart.
    """
    if not name:
        return None
    path = Path(name).expanduser()
    if not (path.is_dir() or looks_like_path(name)):
        return {"model": name}
    resolved = path.resolve()
    if resolved == LOCAL_CE_DIR:
        spelled = LOCAL_CE_HINT
    elif resolved.is_relative_to(REPO_ROOT):
        spelled = resolved.relative_to(REPO_ROOT).as_posix()
    else:
        spelled = resolved.name or name
    out = {"model": spelled}
    digest = _files_sha256(resolved) if resolved.is_dir() else None
    if digest:
        out["model_sha256"] = digest
    return out


def _score_system(system: System, datasets: list[Dataset], cfg: EvalConfig) -> list[Scored]:
    """One Scored per dataset; a system missing its model is 'pending' on every set."""
    routes, model = False, None
    try:
        runner = system.make(cfg)
        routes = callable(getattr(runner, "route", None))
        model = reranker_identity(getattr(getattr(runner, "reranker", None), "model_name", None))
        out, graph = [], None
        for ds in datasets:
            scored = _score_rows(system, runner, ds.rows)
            scored.model = model
            if ds.kind == "multi_intent" and routes:
                if graph is None:
                    runner._load_chunks()
                    graph = build_graph(list(runner.skills.values()))
                scored.plans, scored.parts = _compose_rows(runner, ds.rows, graph)
            out.append(scored)
        return out
    except RerankerUnavailable as e:
        return [Scored(system, None, note=str(e), routes=routes, model=model) for _ in datasets]


def evaluate(systems: list[System], datasets: list[Dataset],
             cfg: EvalConfig | None = None) -> dict[str, list[Scored]]:
    """Score every system on every dataset: dataset name -> one Scored per system (in order)."""
    cfg = cfg or EvalConfig()
    results: dict[str, list[Scored]] = {ds.name: [] for ds in datasets}
    for system in systems:
        for ds, scored in zip(datasets, _score_system(system, datasets, cfg)):
            results[ds.name].append(scored)
    return results


def tie_aware_top1(scored: Scored, queries: list[dict]) -> tuple[int, float]:
    """(#queries whose top-1 is an exact score tie, expected top-1 under random tie order).

    Every system breaks exact ties deterministically (keyword and RRF rankings by slug, a
    single BM25 / dense retriever by its own order); this shows how much of its strict top-1
    that choice decides. Ties are compared at 9 decimals on `Scored.scores` (for
    single-retriever route() rankings, the raw scores `_tie_scores` stores).
    """
    tied, expected = 0, 0.0
    for ranked, scores, q in zip(scored.ranked, scores_of(scored), queries):
        if not ranked:
            continue
        top = round(scores[0], 9)
        group = [slug for slug, sc in zip(ranked, scores) if round(sc, 9) == top]
        gold = set(golds(q["gold"]))
        tied += len(group) > 1
        expected += sum(slug in gold for slug in group) / len(group)
    return tied, expected / len(queries)


def scores_of(scored: Scored) -> list[list[float]]:
    return scored.scores or [[0.0] * len(r) for r in scored.ranked]


def metric_values(scored: Scored, queries: list[dict]) -> dict[str, list[float]]:
    """Per-query recall@1/3/5, mrr, ndcg@1/3/5 and top-1 (= recall@1) of a routing set."""
    rows = [{"ranked": r, "gold": q["gold"]} for r, q in zip(scored.ranked, queries)]
    vals = per_query_ks(rows, ks=KS)
    vals["top-1"] = vals["recall@1"]
    return vals


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _action_rows(s: Scored, idx: list[int] | None = None) -> list[dict]:
    idx = range(len(s.actions)) if idx is None else idx
    return [{"action": s.actions[i], "level": s.levels[i] if s.levels is not None else None}
            for i in idx]


def _confidence_rows(s: Scored, queries: list[dict]) -> list[dict]:
    top1 = metric_values(s, queries)["top-1"]
    return [{"correct": c == 1.0, "level": lv, "action": a}
            for c, lv, a in zip(top1, s.levels, s.actions)]


def _intent_rows(s: Scored, queries: list[dict]) -> dict[str, list[dict]]:
    """Per-query intent metrics of a multi-intent set, per approach."""
    gold = [q["gold"] for q in queries]
    out = {"single_top1": [intent_metrics(r[:1], g) for r, g in zip(s.ranked, gold)],
           "top_n_oracle": [intent_metrics(r[:len(g)], g) for r, g in zip(s.ranked, gold)]}
    if s.plans is not None:
        out["compose"] = [{**intent_metrics(p, g), "split_match": float(n == len(g))}
                          for p, n, g in zip(s.plans, s.parts, gold)]
    return out


def _routing_metrics(s: Scored, ds: Dataset) -> dict:
    vals = metric_values(s, ds.rows)
    tied, expected = tie_aware_top1(s, ds.rows)
    out = {m: _avg(vals[m]) for m in METRICS}
    out["ties"] = {"tied_top1": tied, "top-1_expected": expected}
    out["actions"] = abstention(_action_rows(s))["by_action"]
    if s.routes:
        out["confidence"] = selective(_confidence_rows(s, ds.rows))
    return out


def _category_groups(ds: Dataset, field: str = "category") -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(ds.rows):
        groups.setdefault(str(q.get(field) or "-"), []).append(i)
    return dict(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def _oos_metrics(s: Scored, ds: Dataset) -> dict:
    out = abstention(_action_rows(s))
    out["by_category"] = {cat: abstention(_action_rows(s, idx))["by_action"]
                          for cat, idx in _category_groups(ds).items()}
    return out


def _intent_groups(ds: Dataset) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, q in enumerate(ds.rows):
        groups.setdefault(len(q["gold"]), []).append(i)
    return dict(sorted(groups.items()))


def pooled_recall(rows: list[dict], queries: list[dict]) -> float:
    """Covered gold intents / all gold intents of a set (each intent counts once).

    `intent_metrics` recall is per request, so its mean weighs a one-intent request like a
    four-intent one; this is the share over every gold intent of the set instead.
    """
    sizes = [len(q["gold"]) for q in queries]
    covered = sum(round(r["recall"] * n) for r, n in zip(rows, sizes))
    return covered / sum(sizes) if sum(sizes) else 0.0


def _multi_metrics(s: Scored, ds: Dataset) -> dict:
    per = _intent_rows(s, ds.rows)
    out: dict = {name: {**mean_metrics(rows), "pooled_recall": pooled_recall(rows, ds.rows)}
                 for name, rows in per.items()}
    out["by_n_intents"] = {str(n): {"n": len(idx), **{name: mean_metrics([rows[i] for i in idx])
                                                      for name, rows in per.items()}}
                           for n, idx in _intent_groups(ds).items()}
    return out


def system_metrics(s: Scored, ds: Dataset) -> dict | None:
    """Every metric of one system on one dataset (None when the system is pending)."""
    if s.ranked is None:
        return None
    if ds.kind == "routing":
        return _routing_metrics(s, ds)
    if ds.kind == "out_of_scope":
        return _oos_metrics(s, ds)
    return _multi_metrics(s, ds)


def _rounded(value):
    """Floats rounded to DECIMALS, recursively (bools/ints/None/str unchanged)."""
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def metrics_document(ctx: Context, results: dict[str, list[Scored]], cfg: EvalConfig) -> dict:
    """The deterministic metrics.json payload: corpus, config, datasets, systems, metrics.

    A system that reranks also records its cross-encoder (`reranker_identity`: `model`, plus
    `model_sha256` for a local directory), whether or not the weights loaded.
    """
    rcfg = cfg.retrieval()
    systems: dict[str, dict] = {}
    for scored in results.values():
        for s in scored:
            systems.setdefault(s.system.key, {"label": s.system.label, "family": s.system.family,
                                              "status": "ok" if s.ranked is not None else "pending",
                                              "confidence": s.routes, **(s.model or {})})
    return _rounded({
        "schema": SCHEMA,
        "generated_by": "python -m eval.run_eval",
        "corpus": {"name": ctx.corpus.name, "version": ctx.corpus.version,
                   "n_skills": ctx.corpus.n_skills, "n_chunks": len(ctx.chunks),
                   "fingerprint": ctx.corpus.fingerprint,
                   "chunk_fingerprint": corpus_fingerprint(ctx.chunks)},
        "config": {"depth": DEPTH, "pool": rcfg.candidate_pool, "rerank_k": rcfg.rerank_k,
                   "rrf_k": rcfg.rrf_k, "compose_k": COMPOSE_K, "max_intents": MAX_INTENTS},
        "datasets": {ds.name: {"path": ds.path, "kind": ds.kind, "n": ds.n, "sha256": ds.sha256,
                               "origins": ds.origins()} for ds in ctx.datasets},
        "systems": systems,
        "metrics": {ds.name: {s.system.key: system_metrics(s, ds) for s in results[ds.name]
                              if s.ranked is not None} for ds in ctx.datasets},
    })


def dumps_metrics(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# -- regression (--smoke, tests/test_regression.py) ----------------------------------------

_MISSING = "<missing>"


def _flatten(tree, prefix: str = "") -> dict[str, object]:
    """{"a": {"b": 1}} -> {"a.b": 1}; leaves are scalars."""
    if not isinstance(tree, dict):
        return {prefix: tree} if prefix else {}
    out = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _same(a, b) -> bool:
    """Numbers within TOLERANCE; everything else (bools included: True != 1) exactly."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= TOLERANCE
    return a == b


def compare_metrics(recorded: dict, current: dict,
                    systems: dict[str, str]) -> tuple[list[tuple[str, str, str, object, object]], int]:
    """Differences between a recorded and a fresh metrics document.

    Args:
        recorded: metrics.json as written by a full run.
        current: `metrics_document` of the systems just scored.
        systems: current system key -> the recorded system it must reproduce.

    Returns:
        ([(dataset, system, metric, recorded value, current value)], number of values
        compared). Also compared (dataset "-"): the corpus fingerprints, every `config` value
        (depth, pool, rerank_k, rrf_k, compose_k, max_intents), every field of each dataset
        entry (path, kind, n, sha256, origins), the `systems` entry of each reproduced system,
        and the recorded document's own consistency: a system is "ok" exactly when it has
        metrics on some dataset.
    """
    diffs: list[tuple[str, str, str, object, object]] = []
    n = 0
    for key in ("fingerprint", "chunk_fingerprint"):
        n += 1
        old, new = recorded.get("corpus", {}).get(key, _MISSING), current["corpus"][key]
        if old != new:
            diffs.append(("-", "-", f"corpus.{key}", old, new))
    n += _diff_fields(diffs, ("-", "-", "config."), recorded.get("config"), current.get("config"))
    for name in sorted(set(recorded.get("datasets", {})) - set(current["datasets"])):
        diffs.append((name, "-", "dataset", "scored", "not scored now"))
    for name, info in current["datasets"].items():
        old_info = recorded.get("datasets", {}).get(name)
        if old_info is None:
            n += 1
            diffs.append((name, "-", "dataset", _MISSING, info["sha256"][:12]))
            continue
        for field in sorted(set(old_info) | set(info)):
            n += 1
            a, b = old_info.get(field, _MISSING), info.get(field, _MISSING)
            if field == "sha256" and a != b:
                diffs.append((name, "-", "sha256", str(a)[:12], str(b)[:12]))
            elif field != "sha256" and not _same(a, b):
                diffs.append((name, "-", field, a, b))
        for cur_key, ref_key in systems.items():
            new = _flatten(current["metrics"].get(name, {}).get(cur_key))
            old = _flatten(recorded.get("metrics", {}).get(name, {}).get(ref_key))
            if not old and not new:
                continue
            if not old:
                diffs.append((name, cur_key, f"(all; no '{ref_key}' in metrics.json)", _MISSING, "scored"))
                continue
            for metric in sorted(set(old) | set(new)):
                n += 1
                a, b = old.get(metric, _MISSING), new.get(metric, _MISSING)
                if not _same(a, b):
                    diffs.append((name, cur_key, metric, a, b))
    for cur_key, ref_key in systems.items():
        old_sys = recorded.get("systems", {}).get(ref_key)
        new_sys = current.get("systems", {}).get(cur_key)
        if old_sys is None and new_sys is None:
            continue
        if new_sys is not None and cur_key != ref_key and ref_key in _LABELS:
            new_sys = {**new_sys, "label": _LABELS[ref_key]}     # a replay stands in for ref_key
        n += _diff_fields(diffs, ("-", cur_key, f"systems.{ref_key}."), old_sys or {}, new_sys or {})
    return diffs, n + _status_consistency(recorded, diffs)


def _diff_fields(diffs: list, where: tuple[str, str, str], old: dict | None, new: dict | None) -> int:
    """Append (dataset, system, prefix + field, old, new) for every differing field; count them.

    Nothing is compared when neither side has the mapping at all.
    """
    if old is None and new is None:
        return 0
    old, new = old or {}, new or {}
    dataset, system, prefix = where
    fields = sorted(set(old) | set(new))
    for field in fields:
        a, b = old.get(field, _MISSING), new.get(field, _MISSING)
        if not _same(a, b):
            diffs.append((dataset, system, f"{prefix}{field}", a, b))
    return len(fields)


def _status_consistency(recorded: dict, diffs: list) -> int:
    """metrics.json against itself: `systems.<k>.status` is "ok" exactly when some dataset has
    metrics for k (so a hand-edited status can't make a check skip). Returns systems checked."""
    if "systems" not in recorded:
        return 0
    scored = {key for per in recorded.get("metrics", {}).values() for key in (per or {})}
    keys = sorted(set(recorded["systems"]) | scored)
    for key in keys:
        status = (recorded["systems"].get(key) or {}).get("status", _MISSING)
        if (status == "ok") != (key in scored):
            diffs.append(("-", key, "systems.status", status,
                          "has metrics" if key in scored else "no metrics"))
    return len(keys)


def smoke(out: Path = OUT, manifest: str | Path = DEFAULT_MANIFEST,
          fixture_path: str | Path = FIXTURE) -> int:
    """Score the index-free systems (+ dense replays) and diff them against metrics.json.

    Returns:
        Process exit code: 0 when every compared value matches within TOLERANCE, else 1.
    """
    baseline = Path(out) / "metrics.json"
    if not baseline.exists():
        print(f"[smoke] FAIL: {baseline} not found; run a full `python -m eval.run_eval` first")
        return 1
    try:
        recorded = json.loads(baseline.read_text(encoding="utf-8"))
        config = recorded.get("config", {})
        cfg = EvalConfig(pool=config.get("pool"), rerank_k=config.get("rerank_k"))
        ctx = load_context(manifest)
        for reason in ctx.skipped:
            print(f"[smoke] {reason}")
        fixture = load_fixture(fixture_path)
        if fixture["corpus_fingerprint"] != corpus_fingerprint(ctx.chunks):   # fail fast
            raise RuntimeError(f"the dense fixture {Path(fixture_path).as_posix()} was recorded "
                               f"from another corpus; {RERECORD}")
        systems = smoke_systems(fixture)
        results = evaluate(systems, ctx.datasets, cfg)
    except (OSError, KeyError, RuntimeError, ValueError) as e:     # stale fixture, bad set, ...
        print(f"[smoke] FAIL: {e.args[0] if isinstance(e, KeyError) and e.args else e}")
        return 1
    current = metrics_document(ctx, results, cfg)
    diffs, n = compare_metrics(recorded, current, {s.key: SMOKE_REFERENCE[s.key] for s in systems})
    print(smoke_table(ctx, systems, diffs))
    print(f"[smoke] compared {n} values ({len(systems)} systems x {len(ctx.datasets)} datasets) "
          f"with {baseline.as_posix()}: {len(diffs)} mismatch{'es' if len(diffs) != 1 else ''}")
    return 1 if diffs else 0


def smoke_table(ctx: Context, systems: list[System], diffs: list[tuple]) -> str:
    """Per dataset x system: OK or the number of mismatches, then every mismatch."""
    bad = Counter((d[0], d[1]) for d in diffs)
    lines = [f"[smoke] {'dataset':14s} {'system':16s} {'reference':16s} result"]
    for ds in ctx.datasets:
        for s in systems:
            count = bad[(ds.name, s.key)]
            lines.append(f"[smoke] {ds.name:14s} {s.key:16s} {SMOKE_REFERENCE[s.key]:16s} "
                         f"{'OK' if not count else f'{count} MISMATCH'}")
    if diffs:
        lines.append(f"[smoke] {'dataset':14s} {'system':16s} {'metric':44s} {'metrics.json':>14s} {'now':>14s}")
        lines += [f"[smoke] {d:14s} {s:16s} {m:44s} {str(a):>14s} {str(b):>14s}" for d, s, m, a, b in diffs]
    return "\n".join(lines)


# -- report: routing sets (unchanged structure) --------------------------------------------

def _fmt_ci(vals: list[float]) -> str:
    lo, hi = bootstrap_ci(vals)
    return f"{sum(vals) / len(vals):.3f} <sub>[{lo:.2f}, {hi:.2f}]</sub>"


def headline_key(results: list[Scored]) -> str:
    """The full pipeline if its reranker ran, otherwise the best runnable SIE config."""
    rerank = {s.system.key: s for s in results}.get("hybrid-rerank")
    return "hybrid-rerank" if rerank is not None and rerank.ranked is not None else "hybrid"


def metrics_table(results: list[Scored], queries: list[dict], headline: str) -> str:
    rows = []
    for s in results:
        name = f"**{s.system.label}**" if s.system.key == headline else s.system.label
        if s.ranked is None:
            rows.append([name] + ["_pending_"] * len(METRICS))
            continue
        vals = metric_values(s, queries)
        rows.append([name] + [_fmt_ci(vals[m]) for m in METRICS])
    return md_table(["System"] + METRICS, rows)


def delta_table(results: list[Scored], queries: list[dict], headline: str) -> str:
    by_key = {s.system.key: s for s in results}
    a, b = metric_values(by_key[headline], queries), metric_values(by_key[BASELINE], queries)
    rows = []
    for m in METRICS:
        d, lo, hi = paired_delta_ci(a[m], b[m])
        rows.append([m, f"{sum(b[m]) / len(b[m]):.3f}", f"{sum(a[m]) / len(a[m]):.3f}",
                     f"{d:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]"])
    return md_table(["metric", "keyword", by_key[headline].system.label, "delta", "95% CI (paired)"], rows)


def win_loss(results: list[Scored], queries: list[dict], headline: str) -> str:
    by_key = {s.system.key: s for s in results}
    a = metric_values(by_key[headline], queries)["mrr"]
    b = metric_values(by_key[BASELINE], queries)["mrr"]
    wins, losses = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    return f"per-query reciprocal rank: SIE better on **{wins}**, keyword better on **{losses}**, tied on {len(a) - wins - losses}"


def breakdown(results: list[Scored], queries: list[dict], headline: str, field: str) -> str:
    by_key = {s.system.key: s for s in results}
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(queries):
        groups.setdefault(q.get(field, "-"), []).append(i)
    a, b = metric_values(by_key[headline], queries), metric_values(by_key[BASELINE], queries)
    rows = []
    for g, idx in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        mean = lambda v: sum(v[i] for i in idx) / len(idx)
        rows.append([g, str(len(idx)), f"{mean(b['top-1']):.2f}", f"{mean(a['top-1']):.2f}",
                     f"{mean(b['mrr']):.2f}", f"{mean(a['mrr']):.2f}"])
    return md_table([field, "n", "keyword top-1", "SIE top-1", "keyword MRR", "SIE MRR"], rows)


def _gold_text(gold) -> str:
    return gold if isinstance(gold, str) else " / ".join(gold)


def _cell(text: str) -> str:
    """Text safe inside a markdown table cell."""
    return " ".join(str(text).split()).replace("|", "/")


def misses(results: list[Scored], queries: list[dict], headline: str) -> str:
    """Every query where the headline system misses the top-3 — nothing hidden."""
    by_key = {s.system.key: s for s in results}
    rows = []
    for q, sie, kw in zip(queries, by_key[headline].ranked, by_key[BASELINE].ranked):
        if first_relevant_rank(sie[:K], q["gold"]) is None:
            rank = first_relevant_rank(sie, q["gold"])
            rows.append([q["query"].replace("|", "/"), f"`{_gold_text(q['gold'])}`", ", ".join(sie[:3]),
                         str(rank or f">{DEPTH}"), ", ".join(kw[:3])])
    if not rows:
        return "_none_"
    return md_table(["query", "gold", "SIE top-3", "SIE rank", "keyword top-3"], rows, align="lllcl")


def _version(package: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not installed"


def _portable(text: str) -> str:
    """A message with the absolute local cross-encoder path in its repo-relative spelling."""
    return text.replace(str(LOCAL_CE_DIR), LOCAL_CE_HINT)


def _cross_encoder_lines(results: list[Scored], rcfg: RetrievalConfig) -> list[str]:
    """One provenance line per reranking system: which cross-encoder, over what."""
    top = rcfg.rerank_k if rcfg.rerank_k is not None else rcfg.candidate_pool
    lines = []
    for s in results:
        if s.model is None:
            continue
        files = f" (files sha256 `{s.model['model_sha256']}`)" if s.model.get("model_sha256") else ""
        state = "" if s.ranked is not None else "; not run, see pending below"
        lines.append(f"- cross-encoder ({s.system.label}): `{s.model['model']}`{files} over the "
                     f"chunks of the top-{top} fused skills{state}")
    return lines


def provenance(ctx: Context, results: list[Scored], cfg: EvalConfig) -> str:
    """Where every number came from: corpus identity, query-set hashes, retrieval config and
    the cross-encoder a reranking system used."""
    c, dense, rcfg = ctx.corpus, ctx.router.dense, cfg.retrieval()
    version = f"@{c.version}" if c.version else ""
    pending = [s for s in results if s.ranked is None]
    lines = [
        f"- corpus: `{c.name}{version}` from `{Path(ctx.router.skills_dir).as_posix()}` "
        f"({c.n_skills} skills, {len(ctx.chunks)} chunks; content fingerprint `{c.fingerprint}`, "
        f"chunk fingerprint `{corpus_fingerprint(ctx.chunks)}`)",
        "- queries: " + ", ".join(f"`{ds.path}` sha256 `{ds.sha256[:12]}`" for ds in ctx.datasets),
        f"- dense: {dense.model_name.split('/')[-1]} via `{dense.embedder}` embedder, ChromaDB "
        f"{_version('chromadb')} (cosine HNSW, persisted to `{Path(dense.persist_dir).as_posix()}/`); "
        f"sparse: rank-bm25 {_version('rank-bm25')}; fusion: RRF k={rcfg.rrf_k} over the "
        f"top-{rcfg.candidate_pool} chunks of each retriever",
    ]
    lines += _cross_encoder_lines(results, rcfg)
    lines.append("- keyword baseline: vendored verbatim, see `eval/baseline/SOURCE.md`")
    lines += [f"- {reason}" for reason in ctx.skipped]
    lines += [f"- **{s.system.label}: pending** — {_portable(s.note)}" for s in pending]
    return "\n".join(lines)


def tie_table(results: list[Scored], queries: list[dict]) -> str:
    rows = []
    for s in results:
        if s.ranked is None:
            continue
        tied, expected = tie_aware_top1(s, queries)
        rows.append([s.system.label, f"{_mean(s, queries, 'top-1'):.3f}", f"{expected:.3f}", str(tied)])
    return md_table(["System", "top-1 (ties broken as ranked)", "top-1 (expected, random tie order)",
                     "queries with a tied top-1"], rows)


TIE_CAPTION = ("Tie sensitivity: the keyword and RRF rankings break exact score ties "
               "alphabetically by slug; the BM25 and dense rankings (Card-only BM25, BM25 only, "
               "dense only) keep the retriever's own order, and their ties are counted on its "
               f"raw scores (BM25 only and dense only: each skill's best chunk, at {TIE_DECIMALS} "
               "decimals), not on the single-list RRF scores, which never tie.")


def section(title: str, results: list[Scored], queries: list[dict], headline: str) -> str:
    return "\n\n".join([
        f"### {title} (n={len(queries)})",
        metrics_table(results, queries, headline),
        f"Headline vs baseline, paired over the same queries — {win_loss(results, queries, headline)}:",
        delta_table(results, queries, headline),
        TIE_CAPTION,
        tie_table(results, queries),
    ])


def write_per_query(ctx: Context, results: dict[str, list[Scored]], out: Path = OUT) -> None:
    """One JSON line per query and set: gold, and each system's rank / top-3 / confidence."""
    with (out / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for ds in ctx.datasets:
            for i, q in enumerate(ds.rows):
                row = {"set": ds.name, "query": q["query"], "gold": q["gold"]}
                if ds.kind == "routing":
                    row.update(domain=q.get("domain"), style=q.get("style") or q.get("category"))
                elif ds.kind == "out_of_scope":
                    row["category"] = q.get("category")
                else:
                    row.update(n_intents=len(q["gold"]), style=q.get("style"))
                for s in results[ds.name]:
                    if s.ranked is None:
                        continue
                    ranked = s.ranked[i]
                    cell = ({"rank": first_relevant_rank(ranked, q["gold"]), "top3": ranked[:3]}
                            if ds.kind == "routing" else {"top3": ranked[:3]})
                    if s.routes:
                        cell.update(confidence=s.levels[i], action=s.actions[i])
                    elif ds.kind == "out_of_scope":
                        cell["action"] = s.actions[i]
                    if s.plans is not None:
                        cell.update(compose=s.plans[i], intents=s.parts[i])
                    row[s.system.key] = cell
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bars(results: list[Scored], queries: list[dict], headline: str) -> list[Bar]:
    bars = []
    for s in results:
        if s.ranked is None:
            bars.append(Bar(s.system.label, s.system.family, None,
                            note="pending: cross-encoder weights unavailable"))
            continue
        vals = metric_values(s, queries)
        bars.append(Bar(s.system.label, s.system.family,
                        {m: (sum(vals[m]) / len(vals[m]), *bootstrap_ci(vals[m])) for m in CHART_METRICS},
                        labelled=s.system.key in (BASELINE, headline)))
    return bars


def chart(ctx: Context, results: dict[str, list[Scored]], headline: str, out: Path = OUT) -> None:
    rows = [(ds.chart_title, _bars(results[ds.name], ds.rows, headline))
            for ds in ctx.of_kind("routing")]
    for theme, name in (("light", "results.png"), ("dark", "results-dark.png")):
        render_chart(rows, CHART_METRICS, "Skill routing: keyword router vs SIE",
                     "bar = mean over queries, whisker = 95% bootstrap CI; higher is better",
                     out / name, theme=theme)


def _mean(s: Scored, queries: list[dict], metric: str) -> float:
    vals = metric_values(s, queries)[metric]
    return sum(vals) / len(vals)


def _style_note(ds: Dataset) -> str:
    """What the `-` style marks, computed (the main set: the scaffold's unlabeled originals)."""
    idx = [i for i, q in enumerate(ds.rows) if q.get("style", "-") == "-"]
    if idx and all(ds.row_origins[i] == "scaffold" for i in idx):
        return f"`-` marks the scaffold's original {len(idx)}"
    return f"`-` marks the {len(idx)} queries without a style"


def observations(ctx: Context, results: dict[str, list[Scored]], headline: str) -> str:
    """Takeaways computed from the same numbers as the tables, so the prose can't drift."""
    lines = []
    routing = ctx.of_kind("routing")
    for ds in routing:
        qs = ds.rows
        by = {s.system.key: s for s in results[ds.name] if s.ranked is not None}
        t1 = {key: _mean(by[key], qs, "top-1") for key in (BASELINE, "bm25-card", "bm25")}
        lines.append(
            f"- **{ds.label}, where the lexical gain comes from:** BM25 over the Card chunks "
            f"alone (about the fields the keyword router reads) takes top-1 from "
            f"{t1[BASELINE]:.3f} to {t1['bm25-card']:.3f} (better term weighting on the same "
            f"text). Indexing the body sections too "
            f"{'raises' if t1['bm25'] > t1['bm25-card'] else 'lowers'} it to {t1['bm25']:.3f}.")
    for ds in routing:
        qs = ds.rows
        by = {s.system.key: s for s in results[ds.name] if s.ranked is not None}
        best, other = sorted(("bm25", "dense"), key=lambda key: -_mean(by[key], qs, "mrr"))
        mrr = {key: _mean(by[key], qs, "mrr") for key in (best, other, headline, BASELINE)}
        lines.append(
            f"- **{ds.label}:** the stronger single retriever is "
            f"**{by[best].system.label.split(': ')[1]}** (MRR {mrr[best]:.3f} vs {mrr[other]:.3f}). "
            f"{by[headline].system.label} is {mrr[headline] - mrr[best]:+.3f} MRR against it and "
            f"{mrr[headline] - mrr[BASELINE]:+.3f} against the keyword router.")
    first = routing[0]
    if any("style" in q for q in first.rows):
        kw = metric_values({s.system.key: s for s in results[first.name]}[BASELINE], first.rows)["top-1"]
        by_style: dict[str, list[float]] = {}
        for q, v in zip(first.rows, kw):
            by_style.setdefault(q.get("style", "-"), []).append(v)
        means = {style: sum(v) / len(v) for style, v in by_style.items()}
        hi, lo = max(means, key=means.get), min(means, key=means.get)
        lines.append(f"- The keyword router is strongest on `{hi}` queries (top-1 {means[hi]:.2f}; "
                     f"{_style_note(first)}) and weakest on `{lo}` queries (top-1 {means[lo]:.2f}).")
    return "\n".join(lines)


# -- report: confidence, out-of-scope, multi-intent, query sets ----------------------------

def _acc_cell(group: dict) -> str:
    """`accuracy <sub>n=..</sub>` for a selective() group ("–" when empty)."""
    acc = "–" if group["accuracy"] is None else f"{group['accuracy']:.3f}"
    return f"{acc} <sub>n={group['n']}</sub>"


def _name(s: Scored, headline: str) -> str:
    return f"**{s.system.label}**" if s.system.key == headline else s.system.label


def _headline_scored(results: list[Scored], headline: str) -> Scored:
    return {s.system.key: s for s in results}[headline]


def confidence_section(ctx: Context, results: dict[str, list[Scored]], headline: str) -> str:
    """Per routing set: how often each heuristic confidence level is right, per SIE router."""
    header = (["System", "top-1"] + list(LEVELS)
              + ["routed", "top-1 when routed", "clarify", "abstain"])
    parts = [
        "### Confidence vs correctness (heuristic levels, uncalibrated)",
        "Every SIE router grades its top result `high` (action: route), `medium` or `low` "
        "(clarify) or `none` (abstain) with the rules in `sie/confidence.py`. Its thresholds and "
        "its hybrid and dense rules were fixed before any of these sets were scored and are not "
        "tuned on them; the BM25-only rule was changed after the first measurement (see "
        "*Changed after the first measurement* at the end of this section). "
        "The levels are not probabilities. A level cell is the top-1 accuracy of the queries at "
        "that level <sub>n = how many</sub>. *routed* is the share with action `route` (answered "
        "without a question); *top-1 when routed* is the accuracy a caller that acts only on "
        "`route` gets. Every query here has a correct skill, so each abstention is a false "
        "abstain. Keyword systems and the Card-only ablation have no confidence and are omitted.",
    ]
    for ds in ctx.of_kind("routing"):
        rows = []
        for s in results[ds.name]:
            if not s.routes:
                continue
            if s.ranked is None:
                rows.append([_name(s, headline)] + ["_pending_"] * (len(header) - 1))
                continue
            sel = selective(_confidence_rows(s, ds.rows))
            act = sel["by_action"]
            rows.append([_name(s, headline), f"{sel['accuracy']:.3f}"]
                        + [_acc_cell(sel["by_level"][lv]) for lv in LEVELS]
                        + [f"{sel['coverage']:.3f}", _acc_cell(act["route"]),
                           f"{act['clarify']['share']:.3f}", f"{act['abstain']['share']:.3f}"])
        parts += [f"#### {ds.label} (n={ds.n})", md_table(header, rows)]
        h = _headline_scored(results[ds.name], headline)
        sel = selective(_confidence_rows(h, ds.rows))
        routed = sel["by_action"]["route"]
        routed_acc = "–" if routed["accuracy"] is None else f"{routed['accuracy']:.3f}"
        parts.append(
            f"{h.system.label} routes {routed['n']} of {ds.n} queries without asking "
            f"(top-1 {routed_acc} on those, {sel['accuracy']:.3f} overall), asks on "
            f"{sel['by_action']['clarify']['n']} and abstains on {sel['by_action']['abstain']['n']}.")
    parts.append(first_measurement_note())
    return "\n\n".join(parts)


def first_measurement_note() -> str:
    """What changed after the first full run had scored it, with that run's numbers.

    Built from constants only (the earlier code is gone), so it is the same for every run.
    """
    n, total = FIRST_BM25_IN_SCOPE_ABSTAIN
    c = FIRST_COMPOSE_HYBRID
    intro = ("**Changed after the first measurement.** Two components were changed after the "
             "first full run had scored them on the default query sets (`eval/datasets.toml`), "
             "knowing how the earlier code did there. Their current numbers on those sets are "
             "therefore **not held-out measurements**. The figures quoted here are that first "
             "measurement, of the earlier code (constants in `eval/run_eval.py`; that code no "
             "longer runs).")
    return intro + "\n\n" + "\n".join([
        f"- *BM25-only confidence rule* (`sie/confidence.py`): weak lexical coverage used to make "
        f"a BM25-only router abstain (`none`); it now answers `low` (clarify). First measurement, "
        f"earlier code: BM25-only abstained on {n / total:.1%} of the in-scope queries ({n} of "
        f"{total}, main + source). Its clarify/abstain split, in scope and out of scope, is post "
        f"hoc.",
        f"- *`compose()` intent splitting* (`sie/compose.py`): a single-intent request is routed "
        f"verbatim; an intent whose pronoun was resolved to the first intent's object is routed "
        f"as written instead when the resolved text selects a skill an earlier intent already "
        f"has and the written text selects a new one, and a long substituted object phrase is "
        f"capped; noun homographs of action verbs (test, log, "
        f"train, index, ...) start an intent only before a verb-object word; \"et al.\" never "
        f"ends a sentence and \"etc.\" ends one unless a lowercase word follows. First measurement, earlier "
        f"code (SIE: hybrid RRF, mean over requests): intent recall {c['recall']:.3f}, precision "
        f"{c['precision']:.3f}, exact {c['exact']:.3f}, count match {c['count_match']:.3f}. Its "
        f"multi-intent numbers are post hoc.",
    ])


def _in_scope_actions(ctx: Context, results: dict[str, list[Scored]], key: str) -> list[str] | None:
    actions: list[str] = []
    for ds in ctx.of_kind("routing"):
        s = {x.system.key: x for x in results[ds.name]}.get(key)
        if s is None or s.ranked is None:
            return None
        actions += s.actions
    return actions


def _shares(actions: list[str]) -> dict[str, float]:
    return {a: v["share"] for a, v in abstention([{"action": x} for x in actions])["by_action"].items()}


_OOS_ORDER = ("abstain", "clarify", "route")      # out-of-scope tables: the right answer first


def oos_section(ctx: Context, results: dict[str, list[Scored]], headline: str) -> str:
    """Out-of-scope sets: how often each system declines, and what that costs in scope."""
    routing = ctx.of_kind("routing")
    n_in = sum(ds.n for ds in routing)
    scope = " + ".join(ds.name for ds in routing)
    parts = []
    for ds in ctx.of_kind("out_of_scope"):
        parts += [
            f"### Out-of-scope queries (n={ds.n}) — `{ds.path}`",
            "No skill in the corpus fits any of these queries (gold `null`), so the right answer "
            "is to abstain, or at least to ask. The in-scope columns show the other side of the "
            f"trade-off: the same system on the {n_in} in-scope queries ({scope}), where every "
            "abstention is a miss. † no routing confidence: the system answers whenever it "
            "returns anything, so only an empty result counts as abstain (the keyword router's "
            "full ranking is never empty; `route()` as shipped is empty when no skill clears its "
            "min score).",
        ]
        header = ["System", *_OOS_ORDER, f"in-scope abstain (n={n_in})", "in-scope clarify"]
        rows = []
        for s in results[ds.name]:
            name = _name(s, headline) + ("" if s.routes else " †")
            if s.ranked is None:
                rows.append([name] + ["_pending_"] * (len(header) - 1))
                continue
            oos = _shares(s.actions)
            ins = _in_scope_actions(ctx, results, s.system.key)
            ins = _shares(ins) if ins else None
            rows.append([name] + [f"{oos[a]:.3f}" for a in _OOS_ORDER]
                        + ([f"{ins['abstain']:.3f}", f"{ins['clarify']:.3f}"] if ins else ["–", "–"]))
        parts.append(md_table(header, rows))
        h = _headline_scored(results[ds.name], headline)
        cats = [[cat, str(len(idx))] + [f"{_shares([h.actions[i] for i in idx])[a]:.3f}"
                                        for a in _OOS_ORDER]
                for cat, idx in _category_groups(ds).items()]
        parts += [f"{h.system.label} by category:",
                  md_table(["category", "n"] + list(_OOS_ORDER), cats)]
        rows = [[_cell(q["query"]), str(q.get("category") or "-"), f"`{r[0]}`" if r else "-",
                 lv, a] for q, r, lv, a in zip(ds.rows, h.ranked, h.levels or ["-"] * ds.n, h.actions)]
        parts.append(f"<details><summary>Every out-of-scope query and what {h.system.label} "
                     f"does with it</summary>\n\n"
                     + md_table(["query", "category", "top-1", "level", "action"], rows, align="llllc")
                     + "\n\n</details>")
    return "\n\n".join(parts)


_APPROACHES = (("single_top1", "(a) route the whole query, take the top-1 skill"),
               ("top_n_oracle", "(b) top-n of that ranking, n = gold intent count (oracle)"),
               ("compose", "(c) `compose()`: split into intents, route each"))


def _pair(m: dict, *keys: str) -> str:
    return " / ".join(f"{m[k]:.2f}" for k in keys)


def multi_section(ctx: Context, results: dict[str, list[Scored]], headline: str) -> str:
    """Multi-intent sets: the headline router as single route, oracle top-n, and compose()."""
    parts = []
    for ds in ctx.of_kind("multi_intent"):
        h = _headline_scored(results[ds.name], headline)
        per = _intent_rows(h, ds.rows)
        n_gold = sum(len(q["gold"]) for q in ds.rows)
        c = FIRST_COMPOSE_HYBRID
        parts += [
            f"### Multi-intent composition (n={ds.n}) — `{ds.path}`",
            f"Each request asks for one or more things; gold is one list of acceptable skills per "
            f"intent. {h.system.label} is scored three ways. (b) uses the number of gold intents, "
            f"which no real caller knows: it is an upper reference for plain routing, not a "
            f"system. (c) is `sie.compose.compose()` with its defaults (k={COMPOSE_K}, "
            f"max_intents={MAX_INTENTS}): conservative splitting, one routed skill per intent, "
            f"intents that abstain select nothing; the prerequisites it adds from the graph are "
            f"not counted. *Intent recall* = share of a request's gold intents covered by a "
            f"returned skill, averaged over requests (every request weighs the same); *pooled "
            f"coverage* = covered gold intents / all {n_gold} gold intents of the set; "
            f"*precision* = share of returned skills that fit some intent; *exact* = both 1; "
            f"*count match* = as many distinct skills as gold intents.",
            f"`compose()` was changed after its first measurement on the default multi-intent "
            f"set (see *Changed after the first measurement* at the end of the confidence "
            f"section; first measurement, earlier code: intent recall {c['recall']:.3f}, precision "
            f"{c['precision']:.3f}, exact {c['exact']:.3f}, count match {c['count_match']:.3f}), "
            f"so (c) there is not a held-out measurement.",
            md_table(["approach", "intent recall", "pooled coverage", "precision", "exact match",
                      "count match"],
                     [[label, f"{mean_metrics(per[key])['recall']:.3f}",
                       f"{pooled_recall(per[key], ds.rows):.3f}",
                       *(f"{mean_metrics(per[key])[m]:.3f}" for m in ("precision", "exact", "count_match"))]
                      for key, label in _APPROACHES if key in per]),
        ]
        rows = []
        for n, idx in _intent_groups(ds).items():
            controls = all(ds.rows[i].get("style") == "control" for i in idx)
            m = {key: mean_metrics([per[key][i] for i in idx]) for key in per}
            rows.append([f"{n} (controls)" if controls else str(n), str(len(idx)),
                         _pair(m["single_top1"], "recall", "exact"),
                         _pair(m["top_n_oracle"], "recall", "exact"),
                         _pair(m["compose"], "recall", "precision", "exact"),
                         f"{m['compose']['split_match']:.2f}"])
        parts += ["By number of gold intents (recall / precision / exact):",
                  md_table(["gold intents", "n", "(a) recall / exact", "(b) recall / exact",
                            "(c) recall / precision / exact", "(c) intents split = gold count"], rows)]
        detail = [[_cell(q["query"]), " + ".join(f"`{_gold_text(g)}`" for g in q["gold"]),
                   ", ".join(plan) or "-", f"{n_parts}/{len(q['gold'])}",
                   "yes" if m["exact"] == 1.0 else "no"]
                  for q, plan, n_parts, m in zip(ds.rows, h.plans, h.parts, per["compose"])]
        parts.append("<details><summary>Every multi-intent request and its composed plan</summary>\n\n"
                     + md_table(["request", "gold intents", "compose() skills",
                                 "intents split / gold", "exact"], detail, align="llllc")
                     + "\n\n</details>")
    return "\n\n".join(parts)


def origin_section(ctx: Context) -> str:
    """Every query set: kind, size, who wrote it, file hash — and what none of them is."""
    rows = [[f"`{ds.name}`", ds.kind, str(ds.n),
             ", ".join(f"{o} {n}" for o, n in ds.origins().items()), f"`{ds.sha256[:12]}`",
             ds.spec.description.replace("|", "/")] for ds in ctx.datasets]
    counts = Counter(o for ds in ctx.datasets for o in ds.row_origins)
    used = [o for o in ORIGINS if counts[o]]
    human = counts["human-collected"]
    closing = ("No set here is human-collected production traffic: every query was written for "
               "evaluation, so these numbers say how the engine handles the query styles "
               "represented, not how it performs on real user traffic."
               if not human else
               f"{human} of {sum(counts.values())} queries are human-collected.")
    return "\n\n".join([
        "### Query sets and their origin",
        md_table(["set", "kind", "n", "origin", "sha256", "description"], rows, align="llclll"),
        "\n".join(f"- `{o}`: {ORIGIN_DESCRIPTIONS[o]}" for o in used),
        closing,
    ])


def report(ctx: Context, results: dict[str, list[Scored]], cfg: EvalConfig) -> str:
    routing = ctx.of_kind("routing")
    first = routing[0]
    headline = headline_key(results[first.name])
    label = {s.system.key: s.system.label for s in results[first.name]}[headline]
    blocks = [
        f"_Generated by `python -m eval.run_eval`; do not edit by hand. Headline system: "
        f"**{label}**. Cells are mean <sub>[95% bootstrap CI]</sub>; top-1 = recall@1._",
        "### Observations (computed)\n\n" + observations(ctx, results, headline),
        "![results](results.png)",
    ]
    blocks += [section(ds.title, results[ds.name], ds.rows, headline) for ds in routing]
    fields = [f for f in ("domain", "style", "set") if any(f in q for q in first.rows)]
    if fields:
        blocks.append(f"### Breakdown ({first.name} set)")
        blocks += [breakdown(results[first.name], first.rows, headline, f) for f in fields]
    blocks += [f"<details><summary>Every {ds.name}-set query where {label} misses the top-{K}</summary>\n\n"
               f"{misses(results[ds.name], ds.rows, headline)}\n\n</details>" for ds in routing]
    blocks.append(confidence_section(ctx, results, headline))
    if ctx.of_kind("out_of_scope"):
        blocks.append(oos_section(ctx, results, headline))
    if ctx.of_kind("multi_intent"):
        blocks.append(multi_section(ctx, results, headline))
    blocks += [origin_section(ctx),
               "### Provenance\n\n" + provenance(ctx, results[first.name], cfg)]
    return "\n\n".join(blocks)


# -- README blocks (default datasets only) -------------------------------------------------

def readme_headline(ctx: Context, results: dict[str, list[Scored]]) -> str:
    """The README's opening claim, computed (so adding reranker weights updates it too)."""
    main, source = ctx.of_kind("routing")[:2]
    headline = headline_key(results[main.name])
    m = {s.system.key: s for s in results[main.name]}
    src = {s.system.key: s for s in results[source.name]}
    pct = lambda s, qs: f"{100 * _mean(s, qs, 'top-1'):.1f}%"
    return (f"**Top-1 routing accuracy goes from {pct(m[BASELINE], main.rows)} to "
            f"{pct(m[headline], main.rows)}** on {main.n} labeled queries (keyword "
            f"router -> {m[headline].system.label}), and from {pct(src[BASELINE], source.rows)} to "
            f"{pct(src[headline], source.rows)} on the {source.n} queries taken from the "
            f"source repo's own eval cases. _Generated by `python -m eval.run_eval`._")


def _coverage_claim(ctx: Context, ds: Dataset) -> str:
    """`across all 8 domains (every skill covered)`, computed from the corpus and the labels."""
    domains = {s.domain for s in ctx.skills.values()}
    labelled = {g for q in ds.rows for g in golds(q["gold"])} & set(ctx.skills)
    hit = {ctx.skills[g].domain for g in labelled}
    where = (f"all {len(domains)} domains" if hit == domains
             else f"{len(hit)} of {len(domains)} domains")
    which = ("every skill covered" if len(labelled) == len(ctx.skills)
             else f"{len(labelled)} of {len(ctx.skills)} skills covered")
    return f"across {where} ({which})"


def _beyond_top1(ctx: Context, results: dict[str, list[Scored]], headline: str) -> str:
    """One computed paragraph on abstention and composition (the new RESULTS.md sections)."""
    lines = []
    for ds in ctx.of_kind("out_of_scope"):
        by = {s.system.key: s for s in results[ds.name]}
        h, shipped = by[headline], by.get("keyword-shipped")
        oos = _shares(h.actions)
        ins = _shares(_in_scope_actions(ctx, results, headline) or [])
        kw = ""
        if shipped is not None and shipped.ranked is not None:
            empty = _shares(shipped.actions)["abstain"]
            kw = ("; the keyword router has no way to abstain, and its `route()` as shipped "
                  + ("returned skills for every one" if not empty else
                     f"returned nothing for only {empty:.0%}"))
        lines.append(f"- **Out of scope ({ds.n} queries no skill fits):** {h.system.label} abstains "
                     f"on {oos['abstain']:.0%}, asks on {oos['clarify']:.0%} and routes "
                     f"{oos['route']:.0%} anyway{kw}. In scope it abstains on "
                     f"{ins.get('abstain', 0.0):.1%} of queries.")
    for ds in ctx.of_kind("multi_intent"):
        h = _headline_scored(results[ds.name], headline)
        rows = _intent_rows(h, ds.rows)
        per = {key: mean_metrics(r) for key, r in rows.items()}
        pooled = {key: pooled_recall(r, ds.rows) for key, r in rows.items()}
        n_gold = sum(len(q["gold"]) for q in ds.rows)
        lines.append(f"- **Multi-intent ({ds.n} requests, {n_gold} gold intents):** `compose()` "
                     f"covers on average {per['compose']['recall']:.0%} of each request's gold "
                     f"intents ({pooled['compose']:.0%} of all {n_gold} pooled), with "
                     f"{per['compose']['exact']:.0%} exact plans; routing the whole request and "
                     f"taking the top-1 covers on average {per['single_top1']['recall']:.0%} "
                     f"({pooled['single_top1']:.0%} pooled; {per['single_top1']['exact']:.0%} "
                     f"exact). The same ranking's top-n with the true intent count (an oracle no "
                     f"caller has) covers on average {per['top_n_oracle']['recall']:.0%} "
                     f"({pooled['top_n_oracle']:.0%} pooled; {per['top_n_oracle']['exact']:.0%} "
                     f"exact). `compose()` was revised after its first measurement on this set, "
                     f"so its numbers are not held-out (see RESULTS.md).")
    if lines:
        lines.append("- Confidence levels are heuristic and uncalibrated; how often each level is "
                     "right is measured per set in RESULTS.md.")
    return "\n".join(lines)


def readme_block(ctx: Context, results: dict[str, list[Scored]]) -> str:
    """Compact headline table + computed takeaways for README.md (same run as RESULTS.md)."""
    main, source = ctx.of_kind("routing")[:2]
    headline = headline_key(results[main.name])
    rows = []
    for s in results[main.name]:
        if s.system.key == "keyword-shipped":
            continue
        name = f"**{s.system.label}**" if s.system.key == headline else s.system.label
        if s.ranked is None:
            rows.append([name] + ["pending"] * len(METRICS))
        else:
            rows.append([name] + [f"{_mean(s, main.rows, m):.3f}" for m in METRICS])
    src = {s.system.key: s for s in results[source.name]}
    blocks = [
        f"{main.label}: {main.n} labeled queries {_coverage_claim(ctx, main)}.",
        md_table(["System"] + METRICS, rows),
        f"On the {source.n} queries taken from the source repo's own `evals/` cases: "
        f"keyword top-1 **{_mean(src[BASELINE], source.rows, 'top-1'):.3f}** / MRR "
        f"{_mean(src[BASELINE], source.rows, 'mrr'):.3f} vs SIE top-1 "
        f"**{_mean(src[headline], source.rows, 'top-1'):.3f}** / MRR "
        f"{_mean(src[headline], source.rows, 'mrr'):.3f}. "
        f"CIs, paired deltas, tie sensitivity, ablations and every miss: "
        f"[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). _Generated by `python -m eval.run_eval`._",
        "**What the ablations say (computed):**\n\n" + observations(ctx, results, headline),
    ]
    beyond = _beyond_top1(ctx, results, headline)
    if beyond:
        blocks.append("**Beyond top-1 (computed):**\n\n" + beyond)
    return "\n\n".join(blocks)


def print_summary(ctx: Context, results: dict[str, list[Scored]]) -> None:
    for ds in ctx.datasets:
        print(f"\n[eval] {ds.name} set (n={ds.n}, {ds.kind})")
        if ds.kind == "routing":
            cols = METRICS
        elif ds.kind == "out_of_scope":
            cols = list(ACTIONS)
        else:
            cols = ["a:recall", "a:exact", "b:recall", "b:exact", "c:recall", "c:exact"]
        print(f"  {'system':32s} " + " ".join(f"{m:>9s}" for m in cols))
        for s in results[ds.name]:
            if s.ranked is None:
                print(f"  {s.system.label:32s} pending ({s.note.split(';')[0]})")
                continue
            if ds.kind == "routing":
                vals = metric_values(s, ds.rows)
                cells = [sum(vals[m]) / len(vals[m]) for m in METRICS]
            elif ds.kind == "out_of_scope":
                cells = list(_shares(s.actions).values())
            else:
                per = {k: mean_metrics(v) for k, v in _intent_rows(s, ds.rows).items()}
                cells = [per[k][m] for k in ("single_top1", "top_n_oracle") for m in ("recall", "exact")]
                cells += [per["compose"][m] for m in ("recall", "exact")] if "compose" in per else []
            print(f"  {s.system.label:32s} " + " ".join(f"{v:9.3f}" for v in cells))


# -- CLI ----------------------------------------------------------------------------------

def _select(keys: str | None) -> list[System]:
    if not keys:
        return list(SYSTEMS)
    wanted = [k.strip() for k in keys.split(",") if k.strip()]
    known = {s.key for s in SYSTEMS}
    unknown = [k for k in wanted if k not in known]
    if unknown or not wanted:
        raise ValueError(f"unknown system(s) {', '.join(unknown) or '(none given)'}; "
                         f"choose from {', '.join(s.key for s in SYSTEMS)}")
    return [s for s in SYSTEMS if s.key in wanted]


def _same_path(a: str | Path, b: str | Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def record_fixture(manifest: str | Path, path: str | Path, cfg: EvalConfig) -> dict:
    """Record the persisted dense index's rankings for every query the harness routes.

    Reads the index as it is (never rebuilds it); raises RuntimeError if it is missing or stale.
    """
    ctx = load_context(manifest)
    router = HybridRouter(use_reranker=False, config=cfg.retrieval())
    router._ensure_ready()
    fixture = record(router.dense, fixture_queries(ctx.datasets, max_intents=MAX_INTENTS),
                     corpus_fingerprint=router.dense.fingerprint(),
                     pool=router.config.candidate_pool)
    write_fixture(fixture, path)
    return fixture


def _write(ctx: Context, results: dict[str, list[Scored]], cfg: EvalConfig, out: Path,
           partial: bool, readme: bool) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "per_query.jsonl", out / "metrics.json"]
    write_per_query(ctx, results, out)
    (out / "metrics.json").write_text(dumps_metrics(metrics_document(ctx, results, cfg)),
                                      encoding="utf-8", newline="\n")
    if partial:
        return written
    chart(ctx, results, headline_key(results[ctx.of_kind("routing")[0].name]), out)
    replace_block(out / "RESULTS.md", "metrics", report(ctx, results, cfg))
    written += [out / "RESULTS.md", out / "results.png"]
    if readme:
        replace_block(README, "headline", readme_headline(ctx, results))
        replace_block(README, "benchmarks", readme_block(ctx, results))
        written.append(README)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score the keyword baseline and SIE variants.")
    ap.add_argument("--no-build", action="store_true", help="reuse the persisted dense index")
    ap.add_argument("--datasets", default=str(DEFAULT_MANIFEST), metavar="PATH",
                    help="query-set manifest (default eval/datasets.toml)")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="output directory (default benchmarks/, which only a full run with the "
                         "default datasets and config writes, README blocks included; any other "
                         "run prints only unless DIR is somewhere else)")
    ap.add_argument("--systems", default=None, metavar="K1,K2",
                    help="score only these systems (a partial run: prints, and writes "
                         "metrics.json + per_query.jsonl only to an explicit --out)")
    ap.add_argument("--pool", type=int, default=None, metavar="N", help="candidate chunks per retriever")
    ap.add_argument("--rerank-k", type=int, default=None, metavar="N",
                    help="cross-encode the chunks of the top N fused skills (default: the pool)")
    ap.add_argument("--reranker", default=None, metavar="MODEL",
                    help="cross-encoder id or local dir for the rerank system (as SIE_RERANKER)")
    ap.add_argument("--require-reranker", action="store_true",
                    help="exit 1 (before writing anything) if the rerank system is pending")
    ap.add_argument("--record-fixture", action="store_true",
                    help="record the persisted dense index's rankings for --smoke, then exit")
    ap.add_argument("--fixture", default=str(FIXTURE), metavar="PATH",
                    help="dense-rankings fixture (default tests/fixtures/dense_rankings.json)")
    ap.add_argument("--smoke", action="store_true",
                    help="no index, no models, no writes: score keyword/BM25 systems and dense "
                         "replays, diff against <out>/metrics.json, exit 1 on any mismatch")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")       # never crash on a cp1252 console
    out = Path(args.out) if args.out else OUT
    try:
        cfg = EvalConfig(pool=args.pool, rerank_k=args.rerank_k, reranker=args.reranker)
        cfg.retrieval()
        systems = _select(args.systems)
    except ValueError as e:
        ap.error(str(e))
    if args.smoke:
        sys.exit(smoke(out, args.datasets, args.fixture))
    if args.record_fixture:
        try:
            fixture = record_fixture(args.datasets, args.fixture, cfg)
        except (OSError, RuntimeError, ValueError) as e:        # index missing / stale, bad set
            sys.exit(f"[eval] cannot record the dense fixture: {e}")
        print(f"[eval] recorded the top {fixture['depth']} dense chunks for "
              f"{len(fixture['queries'])} queries -> {Path(args.fixture).as_posix()}")
        return
    partial = len(systems) < len(SYSTEMS)
    to_benchmarks = args.out is None or _same_path(out, OUT)
    canonical = _same_path(args.datasets, DEFAULT_MANIFEST) and cfg.is_default
    print_only = (partial or not canonical) and to_benchmarks
    if print_only:
        print("[eval] partial or non-default run (--systems, --datasets, --pool or --rerank-k): "
              "printing only; benchmarks/ holds the full default run. Pass --out DIR (not "
              "benchmarks/) to write this run's results")
    if not args.no_build:
        print("[eval] building indexes ...")
        HybridRouter(use_reranker=False, config=cfg.retrieval()).build()
    try:
        ctx = load_context(args.datasets)
    except (OSError, RuntimeError, ValueError) as e:            # corpus missing, bad query set
        sys.exit(f"[eval] {e}")
    for reason in ctx.skipped:
        print(f"[eval] {reason}")
    if not ctx.of_kind("routing"):
        sys.exit("[eval] no routing dataset to score")
    results = evaluate(systems, ctx.datasets, cfg)
    print_summary(ctx, results)
    if args.require_reranker:
        rerank = [s for s in results[ctx.datasets[0].name] if s.system.key == "hybrid-rerank"]
        if not rerank or rerank[0].ranked is None:
            why = rerank[0].note if rerank else "the hybrid-rerank system was not selected"
            sys.exit(f"[eval] FAIL: --require-reranker: {why}")
    if print_only:
        return
    readme = not partial and to_benchmarks and canonical
    written = _write(ctx, results, cfg, out, partial, readme)
    print(f"\n[eval] wrote {', '.join(p.as_posix() for p in written)}")


if __name__ == "__main__":
    main()
