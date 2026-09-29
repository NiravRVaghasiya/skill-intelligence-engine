"""eval.perf with fakes: aggregation, synthetic replication, measurement plumbing, rendering.

No models and no chromadb: routers are scripted fakes, or real BM25 over the real corpus
with the dense index replaced by a stand-in. Latency values here are synthetic.
"""
import argparse
import gc
import json
import weakref
from pathlib import Path

import pytest

from eval import perf
from eval.datasets import load_datasets
from eval.metrics import percentile
from eval.perf import Query, Setting
from sie.chunking import chunk_skill
from sie.ingest import load_corpus
from sie.models import Chunk, Confidence, Hit, RankedSkill, RouteResult
from sie.rerank import RerankerUnavailable
from sie.router import HybridRouter, corpus_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"
SHAP = "explain individual predictions with shap values"


def _corpus_chunks() -> list[Chunk]:
    return [c for s in load_corpus(CORPUS) for c in chunk_skill(s)]


class FakeDense:
    """Dense-index stand-in: records build(); search returns the built chunks in corpus order."""
    embedder = "onnx"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    ef_search = 400
    made: list[str] = []

    def __init__(self, persist_dir="unused"):
        self.persist_dir, self.generation, self.chunks, self._fp, self.builds = persist_dir, 0, [], None, 0
        FakeDense.made.append(persist_dir)

    def build(self, chunks, fingerprint="", metadata=None):
        self.chunks, self._fp, self.builds = list(chunks), fingerprint, self.builds + 1
        self.generation += 1

    def fingerprint(self):
        return self._fp

    def built_with(self):
        return None if self._fp is None else "onnx"

    def info(self):
        if self._fp is None:
            return None
        return {"embedder": "onnx", "fingerprint": self._fp, "model_name": self.model_name,
                "count": len(self.chunks)}

    def _embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def search(self, query, k=10):
        self._embed([query])
        return [Hit(c.skill_slug, 1.0 - i / 1000, c.section, c.text[:200], c.chunk_id)
                for i, c in enumerate(self.chunks[:k])]


class FailingReranker:
    model_name = "fake-ce"

    def load(self):
        raise RerankerUnavailable("no cross-encoder weights (test)")


def _result(slugs, timings, candidates, reranked=False) -> RouteResult:
    ranked = [RankedSkill(slug=s, rank=i, score=1.0 / i, score_type="rrf") for i, s in enumerate(slugs, 1)]
    return RouteResult(query="q", results=ranked, confidence=Confidence("high", "route"),
                       reranked=reranked, candidates=candidates, timings_ms=timings)


class ScriptedRouter:
    """route() answers from a table; the n-th call's dense time is n ms, so passes are traceable."""
    reranker = None

    def __init__(self, rankings, drift_after=None):
        self.rankings, self.drift_after, self.calls = rankings, drift_after, []

    def warm_up(self):
        return {}

    def route(self, query, k=None, pool=None, rerank_k=None):
        self.calls.append((query, pool, rerank_k))
        n = len(self.calls)
        ranking = self.rankings[query]
        if self.drift_after is not None and n > self.drift_after:
            ranking = ranking[::-1]
        timings = {"dense": float(n), "bm25": 0.5, "fuse": 0.01, "explain": 0.2, "total": n + 1.0}
        candidates = {"dense": pool, "bm25": pool // 2, "fused": 3}
        if rerank_k is not None:
            timings["rerank"], candidates["rerank"] = 7.0, rerank_k
        return _result(ranking[:k], timings, candidates, reranked=rerank_k is not None)


QUERIES = [Query("main", "q1", "a"), Query("main", "q2", "b"), Query("source", "q3", ["c", "z"])]
RANKINGS = {"q1": ["a", "b", "c"], "q2": ["a", "b", "c"], "q3": ["x", "y", "z"]}


# --- aggregation ------------------------------------------------------------------------------

def test_stage_stats_nearest_rank_per_stage_over_samples_that_ran_it():
    timings = [{"total": 3.0, "dense": 1.0}, {"total": 1.0, "rerank": 5.0}, {"zeta": 1.0, "total": 2.0}]
    stats = perf.stage_stats(timings)
    assert list(stats) == ["dense", "rerank", "total", "zeta"]     # STAGES order, unknown last
    assert stats["total"] == {"n": 3, "p50": 2.0, "p95": 3.0, "mean": 2.0}
    assert stats["rerank"] == {"n": 1, "p50": 5.0, "p95": 5.0, "mean": 5.0}
    values = [0.1 * i for i in range(1, 101)]
    big = perf.stage_stats([{"total": v} for v in reversed(values)])["total"]
    assert big["p50"] == percentile(values, 50) and big["p95"] == percentile(values, 95)
    assert perf.stage_stats([]) == {}


def test_mean_candidates_counts_a_missing_stage_as_zero():
    results = [_result([], {}, {"dense": 20, "fused": 10, "rerank": 8, "x": 1}),
               _result([], {}, {"dense": 10, "fused": 6})]
    out = perf.mean_candidates(results)
    assert out == {"dense": 15.0, "fused": 8.0, "rerank": 4.0, "x": 0.5}
    assert list(out) == ["dense", "fused", "rerank", "x"]
    with pytest.raises(ValueError):
        perf.mean_candidates([])


def test_quality_means_and_list_golds():
    q = perf.quality([["a", "b"], ["b", "a"], ["x", "y", "z"]], ["a", "a", ["z", "q"]])
    assert q == {"recall@1": round(1 / 3, 6), "recall@3": 1.0, "recall@5": 1.0,
                 "mrr": round((1 + 0.5 + 1 / 3) / 3, 6)}
    with pytest.raises(ValueError):
        perf.quality([["a"]], [])
    with pytest.raises(ValueError):
        perf.quality([], [])


# --- measurement ------------------------------------------------------------------------------

def test_measure_excludes_the_warm_up_and_interleaves_timed_passes():
    router = ScriptedRouter(RANKINGS)
    rows = perf.measure(router, QUERIES, [Setting(5), Setting(20)], repeat=2, k=10)
    pools = [pool for _, pool, _ in router.calls]
    assert pools == [5] * 3 + [20] * 3 + [5] * 3 + [20] * 3 + [5] * 3 + [20] * 3
    first = rows[0]
    assert (first["pool"], first["rerank_k"], first["samples"]) == (5, None, 6)
    # timed pool-5 calls are #7-9 and #13-15; the warm-up (#1-3) is not in the sample
    assert first["latency_ms"]["dense"] == {"n": 6, "p50": 9.0, "p95": 15.0, "mean": 11.0}
    assert list(first["latency_ms"]) == ["dense", "bm25", "fuse", "explain", "total"]
    assert first["quality"] == {
        "main": {"n": 2, "recall@1": 0.5, "recall@3": 1.0, "recall@5": 1.0, "mrr": 0.75},
        "source": {"n": 1, "recall@1": 0.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": round(1 / 3, 6)}}
    assert first["candidates"]["main"] == {"dense": 5.0, "bm25": 2.0, "fused": 3.0}
    assert rows[1]["candidates"]["source"] == {"dense": 20.0, "bm25": 10.0, "fused": 3.0}
    assert all(r["deterministic"] and r["reranked"] == 0 for r in rows)


def test_measure_flags_rankings_that_change_between_passes():
    rows = perf.measure(ScriptedRouter(RANKINGS, drift_after=6), QUERIES, [Setting(5), Setting(20)],
                        repeat=1)
    assert [r["deterministic"] for r in rows] == [False, False]
    assert rows[0]["quality"]["main"]["recall@1"] == 0.5         # still the warm-up's rankings


def test_measure_arms_interleaves_timed_passes_across_routers():
    log = []

    class Logged(ScriptedRouter):
        def __init__(self, name):
            super().__init__(RANKINGS)
            self.name = name

        def route(self, query, k=None, pool=None, rerank_k=None):
            log.append(self.name)
            return super().route(query, k=k, pool=pool, rerank_k=rerank_k)

    rows = perf.measure_arms([(Logged("x"), Setting(20)), (Logged("y"), Setting(20))], QUERIES[:1],
                             repeat=2)
    assert log == ["x", "y", "x", "y", "x", "y"]          # warm-ups, then pass 1, pass 2
    assert [r["samples"] for r in rows] == [2, 2]


def test_measure_embedding_times_each_query_after_a_warm_up():
    seen = []
    stats = perf.measure_embedding(seen.append, QUERIES, repeat=2)
    assert seen == ["q1", "q2", "q3"] * 3                  # warm-up + 2 timed passes
    assert stats["n"] == 6 and set(stats) == {"n", "p50", "p95", "mean"}
    assert 0.0 <= stats["p50"] <= stats["p95"]
    with pytest.raises(ValueError):
        perf.measure_embedding(seen.append, QUERIES, repeat=0)
    with pytest.raises(ValueError):
        perf.measure_embedding(seen.append, [], repeat=1)


class Embedder:
    """Embedding-function stand-in: logs every call; `bump` changes one element's value."""

    def __init__(self, name, log, bump=0.0, dims=3):
        self.name, self.log, self.bump, self.dims = name, log, bump, dims

    def __call__(self, texts):
        self.log.append((self.name, list(texts)))
        return [[float(len(t)), 0.5 + self.bump, 1.0][:self.dims] for t in texts]


def test_compare_embedding_calls_checks_vectors_then_alternates_timed_passes():
    log = []
    out = perf.compare_embedding_calls(Embedder("before", log), Embedder("held", log), QUERIES, repeat=2)
    assert out["measured"] is True and out["identical"] is True and out["max_abs_diff"] == 0.0
    assert (out["queries"], out["repeat"]) == (3, 2)
    assert out["before"]["n"] == out["held"]["n"] == 6 and set(out["held"]) == {"n", "p50", "p95", "mean"}
    assert all(len(texts) == 1 for _, texts in log)                   # one query per call
    names = [name for name, _ in log]
    assert names[:6] == ["before", "held"] * 3                        # untimed vector check
    assert names[6:] == ["before"] * 3 + ["held"] * 3 + ["before"] * 3 + ["held"] * 3
    json.dumps(out, allow_nan=False)                                   # performance.json-safe


def test_compare_embedding_calls_reports_different_vectors():
    log = []
    out = perf.compare_embedding_calls(Embedder("before", log), Embedder("held", log, bump=1e-4),
                                       QUERIES[:1], repeat=1)
    assert out["identical"] is False and out["max_abs_diff"] == pytest.approx(1e-4)
    other = perf.compare_embedding_calls(Embedder("before", log), Embedder("held", log, dims=2),
                                         QUERIES[:1], repeat=1)
    assert other["identical"] is False and other["max_abs_diff"] is None
    with pytest.raises(ValueError):
        perf.compare_embedding_calls(Embedder("b", log), Embedder("h", log), QUERIES, repeat=0)
    with pytest.raises(ValueError):
        perf.compare_embedding_calls(Embedder("b", log), Embedder("h", log), [], repeat=1)


def test_embedding_calls_not_measured_without_onnx_or_chromadb():
    st = perf.measure_embedding_calls(type("D", (), {"embedder": "sentence-transformers"})(), QUERIES, 1)
    assert st["measured"] is False and "onnx backend" in st["reason"]
    blocked = perf.measure_embedding_calls(FakeDense(), QUERIES, 1)   # chromadb is blocked in tests
    assert blocked["measured"] is False and "ImportError" in blocked["reason"]


def test_measure_validates_its_inputs():
    with pytest.raises(ValueError, match="repeat"):
        perf.measure(ScriptedRouter(RANKINGS), QUERIES, [Setting(5)], repeat=0)
    with pytest.raises(ValueError):
        perf.measure(ScriptedRouter(RANKINGS), [], [Setting(5)], repeat=1)
    with pytest.raises(ValueError):
        perf.measure(ScriptedRouter(RANKINGS), QUERIES, [], repeat=1)


def test_reranker_that_cannot_load_is_reported_not_measured():
    class Unloadable(ScriptedRouter):
        reranker = FailingReranker()

        def warm_up(self):
            self.reranker.load()

    router = Unloadable(RANKINGS)
    out = perf.measure_reranker(lambda: router, QUERIES, [5, 10], pool=20, repeat=1)
    assert out == {"measured": False, "reason": "no cross-encoder weights (test)",
                   "model": "fake-ce", "pool": 20}
    assert router.calls == []


def test_reranker_rows_per_rerank_k_at_the_given_pool():
    router = ScriptedRouter(RANKINGS)
    out = perf.measure_reranker(lambda: router, QUERIES, [5, 10], pool=20, repeat=1)
    assert out["measured"] is True and out["pool"] == 20
    assert [(r["pool"], r["rerank_k"], r["reranked"]) for r in out["rows"]] == [(20, 5, 3), (20, 10, 3)]
    assert {pool for _, pool, _ in router.calls} == {20}
    assert out["rows"][1]["candidates"]["main"]["rerank"] == 10.0
    assert out["rows"][0]["latency_ms"]["rerank"]["p50"] == 7.0


# --- synthetic scaling ------------------------------------------------------------------------

def test_replicate_chunks_suffixes_slugs_and_ids_and_keeps_texts():
    chunks = [Chunk("a", "Card", "alpha", "a::Card::0"), Chunk("a", "Workflow", "beta", "a::Workflow::0"),
              Chunk("b", "Body", "gamma", "custom-id")]
    one = perf.replicate_chunks(chunks, 1)
    assert one == chunks and one is not chunks
    three = perf.replicate_chunks(chunks, 3)
    assert len(three) == 9 and three[:3] == chunks
    assert three[3] == Chunk("a~2", "Card", "alpha", "a~2::Card::0")
    assert [c.chunk_id for c in three[5::3]] == ["custom-id~2", "custom-id~3"]
    assert {c.skill_slug for c in three} == {"a", "b", "a~2", "b~2", "a~3", "b~3"}
    assert [(c.text, c.section) for c in three] == [(c.text, c.section) for c in chunks] * 3
    assert perf.replicate_chunks(chunks, 3) == three                # deterministic


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_replicate_chunks_rejects_bad_factors(factor):
    with pytest.raises(ValueError):
        perf.replicate_chunks([Chunk("a", "S", "t", "a::S::0")], factor)


def test_replicate_chunks_refuses_colliding_ids():
    with pytest.raises(ValueError, match="not unique"):
        perf.replicate_chunks([Chunk("a", "S", "t", "x"), Chunk("b", "S", "t", "x~2")], 2)


def test_replicating_the_real_corpus_keeps_chunk_ids_unique():
    chunks = _corpus_chunks()
    four = perf.replicate_chunks(chunks, 4)
    assert len({c.chunk_id for c in four}) == len(four) == 4 * len(chunks)
    assert len({c.skill_slug for c in four}) == 4 * len({c.skill_slug for c in chunks})


def test_dedup_dense_embeds_each_distinct_text_once_per_call(tmp_path):
    seen = []

    def embed(texts):
        seen.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    index = perf.DedupDenseIndex(persist_dir=str(tmp_path))
    index._embed_fn = embed
    assert index._embed(["aa", "b", "aa"]) == [[2.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    assert index._embed(["aa"]) == [[2.0, 1.0]]
    assert seen == [["aa", "b"], ["aa"]]          # nothing cached across calls: queries re-embed


def test_chunk_router_indexes_and_routes_exactly_the_given_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "DedupDenseIndex", FakeDense)
    chunks = _corpus_chunks()
    r = perf.ChunkRouter(perf.replicate_chunks(chunks, 2), persist_dir=str(tmp_path), mode="sparse")
    assert r.build() == 2 * len(chunks)
    assert isinstance(r.dense, FakeDense) and r.dense._fp == corpus_fingerprint(r.dense.chunks)
    assert r.dense.chunks[len(chunks)].skill_slug == chunks[0].skill_slug + "~2"
    slugs = [s.slug for s in r.route(SHAP, k=10).results]
    assert slugs[:2] == ["explainability", "explainability~2"]    # identical copies, original first
    assert all(s.removesuffix("~2") in {c.skill_slug for c in chunks} for s in slugs)


def test_chunk_router_rejects_duplicate_chunk_ids(tmp_path):
    c = Chunk("a", "S", "t", "a::S::0")
    with pytest.raises(RuntimeError, match="duplicate"):
        perf.ChunkRouter([c, c], persist_dir=str(tmp_path))._load_chunks()


def test_measure_scale_times_each_factor_and_records_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "DedupDenseIndex", FakeDense)
    chunks = _corpus_chunks()
    n, skills = len(chunks), len({c.skill_slug for c in chunks})
    made = []

    def make(replicated, path):
        if len(replicated) == 3 * n:
            raise ValueError("batch too large")
        made.append(path)
        return perf.ChunkRouter(replicated, persist_dir=str(path))

    queries = [Query("main", SHAP, "explainability"), Query("main", "forecast a time series", "time-series")]
    rows = perf.measure_scale(chunks, [1, 2, 3], queries, repeat=2, workdir=tmp_path, pool=20,
                              make_router=make)
    assert [(r["factor"], r["chunks"], r["skills"]) for r in rows] == [
        (1, n, skills), (2, 2 * n, 2 * skills), (3, 3 * n, 3 * skills)]
    assert made == [tmp_path / "scale-1", tmp_path / "scale-2"]
    for r in rows[:2]:
        assert r["samples"] == 4 and r["deterministic"] is True and r["ef_search"] == 400
        assert list(r["latency_ms"]) == ["dense", "bm25", "fuse", "explain", "total"]
    assert rows[2]["error"] == "ValueError: batch too large" and "latency_ms" not in rows[2]


def test_measure_scale_builds_every_index_before_timing_them_interleaved(tmp_path):
    log = []

    class Scaled(ScriptedRouter):
        def __init__(self, chunks, path):
            super().__init__({SHAP: ["explainability"]})
            self.n, self.dense = len(chunks), FakeDense(str(path))

        def build(self):
            log.append(("build", self.n))

        def warm_up(self):
            log.append(("warm", self.n))

        def route(self, query, k=None, pool=None, rerank_k=None):
            log.append(("route", self.n))
            return super().route(query, k=k, pool=pool, rerank_k=rerank_k)

    rows = perf.measure_scale([Chunk("a", "S", "t", "a::S::0")], [1, 2],
                              [Query("main", SHAP, "explainability")], repeat=1, workdir=tmp_path,
                              pool=20, make_router=Scaled)
    assert log == [("build", 1), ("warm", 1), ("build", 2), ("warm", 2),
                   ("route", 1), ("route", 2), ("route", 1), ("route", 2)]
    assert [(r["chunks"], r["samples"]) for r in rows] == [(1, 1), (2, 1)]


@pytest.mark.parametrize("fail_on", [None, "build", "make"])
def test_measure_scale_drops_every_router_before_releasing_handles(monkeypatch, tmp_path, fail_on):
    """A router still referenced at release time keeps its index files open (they leak on Windows)."""
    refs, alive = [], []

    class Scaled(ScriptedRouter):
        def __init__(self, chunks, path):
            if fail_on == "make" and len(chunks) == 3:
                raise MemoryError("no room")
            super().__init__({SHAP: ["explainability"]})
            self.n, self.dense = len(chunks), FakeDense(str(path))
            refs.append(weakref.ref(self))

        def build(self):
            if fail_on == "build" and self.n == 3:
                raise ValueError("batch too large")

    def release():
        gc.collect()
        alive.append([ref() is not None for ref in refs])

    monkeypatch.setattr(perf, "_release_handles", release)
    rows = perf.measure_scale([Chunk("a", "S", "t", "a::S::0")], [1, 2, 3],
                              [Query("main", SHAP, "explainability")], repeat=1, workdir=tmp_path,
                              pool=20, make_router=Scaled)
    assert len(alive) == 1 and not any(alive[0]) and len(refs) == (2 if fail_on == "make" else 3)
    assert ("error" in rows[2]) == (fail_on is not None)


def test_run_scale_uses_a_temporary_directory_and_removes_it(monkeypatch):
    monkeypatch.setattr(perf, "DedupDenseIndex", FakeDense)
    FakeDense.made = []
    out = perf.run_scale(_corpus_chunks(), [1, 2], [Query("main", SHAP, "explainability")],
                         repeat=1, pool=20)
    assert (out["queries"], out["pool"], out["k"], out["temp_dir_removed"]) == (1, 20, perf.DEPTH, True)
    assert [r["factor"] for r in out["rows"]] == [1, 2] and not any("error" in r for r in out["rows"])
    dirs = [Path(p) for p in FakeDense.made if Path(p).name.startswith("scale-")]
    assert [d.name for d in dirs] == ["scale-1", "scale-2"]
    assert dirs[0].parent.name.startswith("sie-perf-") and not dirs[0].parent.exists()


# --- rendering --------------------------------------------------------------------------------

def test_fmt_ms_keeps_about_three_significant_digits():
    assert [perf.fmt_ms(v) for v in (0.0, 0.0421, 1.234, 12.34, 123.4, 1234.5)] == [
        "0.000", "0.042", "1.23", "12.3", "123", "1234"]
    assert perf.fmt_latency(None) == perf.fmt_latency({}) == "-"
    assert perf.fmt_latency({"p50": 1, "p95": 2, "mean": 1.5, "n": 3}) == "1.00 / 2.00 / 1.50"


def _report(pools, reranker, scale):
    return {
        "command": "python -m eval.perf --pools 5,20 --repeat 2 --scale 1 --rerank-k 5,10",
        "note": perf.NOTE, "stamp": "2026-01-02",
        "environment": {"platform": "TestOS-1", "processor": "TestCPU", "cpu_count": 4,
                        "python": "3.11.0", "versions": {"rank-bm25": "0.2.2", "chromadb": "1.0"}},
        "setup": {"corpus": {"name": "demo", "version": "abc123", "n_skills": 3, "n_chunks": 9,
                             "content_fingerprint": "c" * 16, "chunk_fingerprint": "d" * 16},
                  "dense": {"persist_dir": "data/chroma", "embedder": "onnx", "model": "mini",
                            "count": 9},
                  "queries": {"source": {"path": "eval/s.jsonl", "n": 1, "sha256": "5" * 12},
                              "main": {"path": "eval/m.jsonl", "n": 2, "sha256": "m" * 12}},
                  "k": 10, "repeat": 2, "rrf_k": 60, "default_pool": 20},
        "pools": pools, "embedding": {"n": 6, "p50": 4.0, "p95": 6.5, "mean": 4.25},
        "reranker": reranker, "scale": scale,
    }


def _pool_rows():
    return perf.measure(ScriptedRouter(RANKINGS), QUERIES, [Setting(5), Setting(20)], repeat=2)


def test_render_header_pool_tables_and_unmeasured_reranker():
    reranker = {"measured": False, "reason": "no weights here", "model": "ce", "pool": 20}
    md = perf.render(_report(_pool_rows(), reranker, None))
    assert md.startswith("# Router performance\n\n_Generated by `python -m eval.perf --pools 5,20")
    assert "**Machine-dependent; not byte-reproducible:**" in md
    assert md.count("2026-01-02") == 1                              # the only wall-clock value
    assert "TestOS-1; TestCPU; 4 logical CPUs" in md
    assert "then 2 timed passes interleaved across configurations, so n = queries x 2" in md
    assert "Python 3.11.0, chromadb 1.0, rank-bm25 0.2.2" in md       # sorted, not dict order
    assert "`demo@abc123`: 3 skills, 9 chunks" in md
    assert ("`main` = `eval/m.jsonl` (n=2, sha256 `mmmmmmmmmmmm`), "
            "`source` = `eval/s.jsonl` (n=1, sha256 `555555555555`)") in md
    assert ("| pool | set | n | recall@1 | recall@3 | recall@5 | MRR | dense chunks | bm25 chunks "
            "| fused skills |") in md
    assert "| 20 (default) | main | 2 | 0.500 | 1.000 | 1.000 | 0.750 | 20.0 | 10.0 | 3.0 |" in md
    assert "| 5 | source | 1 | 0.000 | 1.000 | 1.000 | 0.333 | 5.0 | 2.0 | 3.0 |" in md
    assert "| pool | n | dense | bm25 | fuse | explain | total |" in md
    assert "| 5 | 6 | 9.00 / 15.0 / 11.0 | 0.500 / 0.500 / 0.500 |" in md
    assert "Rankings were identical in the warm-up and every timed pass." in md
    assert "Query embedding alone, timed separately on the same queries (n=6): **4.00 / 6.50 / 4.25**" in md
    assert "## Cross-encoder reranker\n\nnot measured: no weights here" in md
    assert "not run (pass e.g. `--scale 1,4,16`)." in md
    assert md.endswith("\n") and md == perf.render(_report(_pool_rows(), reranker, None))
    assert "Embedding call comparison" not in md          # reports from before it existed


def test_render_embedding_call_comparison():
    reranker = {"measured": False, "reason": "no weights here", "model": "ce", "pool": 20}
    report = _report(_pool_rows(), reranker, None)
    stats = lambda p50: {"n": 6, "p50": p50, "p95": p50 * 2, "mean": p50 * 1.5}
    report["embedding_calls"] = {"measured": True, "queries": 3, "repeat": 2, "identical": True,
                                 "max_abs_diff": 0.0, "before": stats(500.0), "held": stats(4.0)}
    md = perf.render(report)
    assert "**Embedding call comparison, one query per call.** The same 3 queries" in md
    assert "then 2 timed passes alternating between the two" in md
    assert "| embedding call | n | p50 / p95 / mean (ms) |" in md
    assert ("| before this change: chromadb `DefaultEmbeddingFunction()` (builds a new ONNX session "
            "on every call) | 6 | 500 / 1000 / 750 |") in md
    assert ("| now: one held `ONNXMiniLM_L6_V2` instance (what `DenseIndex` calls) | 6 | "
            "4.00 / 8.00 / 6.00 |") in md
    assert "The same 3 queries (the first 3 of the pool sweep's)" in md
    assert "Vectors identical on all 3 queries: **yes** (largest element difference 0)." in md
    assert md.index("Embedding call comparison") < md.index("## Cross-encoder reranker")
    report["embedding_calls"].update(identical=False, max_abs_diff=None)
    assert "**no** (dimensions differ)" in perf.render(report)
    report["embedding_calls"] = {"measured": False, "reason": "ImportError: no chromadb"}
    assert "Query embedding alone" in perf.render(report)
    assert "not measured: ImportError: no chromadb" in perf.render(report)


def test_render_measured_reranker_scale_rows_and_failures():
    rr = perf.measure_reranker(lambda: ScriptedRouter(RANKINGS), QUERIES, [5, 10], pool=20, repeat=1)
    drifting = perf.measure(ScriptedRouter(RANKINGS, drift_after=3), QUERIES, [Setting(5)], repeat=1)
    ok = {"factor": 2, "chunks": 18, "skills": 6, "ef_search": 400, "samples": 4,
          "deterministic": True, "latency_ms": perf.stage_stats([{"dense": 3.0, "total": 5.0}] * 4)}
    failed = {"factor": 3, "chunks": 27, "skills": 9, "ef_search": 400, "error": "ValueError: too big"}
    scale = {"queries": 2, "pool": 20, "k": 10, "rows": [ok, failed], "temp_dir_removed": True}
    md = perf.render(_report(drifting, rr, scale))
    assert "**Rankings changed between passes for pool 5**" in md
    assert "Cross-encoder over the hybrid candidates at pool 20" in md and "| rerank_k | set | n |" in md
    assert "| cross-encoded chunks |" in md and "| 5 | main | 2 |" in md
    assert "| rerank_k | n | dense | bm25 | fuse | rerank | explain | total |" in md
    assert "quality on these duplicated corpora is not meaningful" in md
    assert "HNSW ef_search=400" in md and "first 2 queries" in md
    assert "| scale | chunks | skills | n | dense | total |" in md
    assert "| 2x | 18 | 6 | 4 | 3.00 / 3.00 / 3.00 | 5.00 / 5.00 / 5.00 |" in md
    assert "- 3x (27 chunks): not measured: ValueError: too big" in md
    assert "could not be removed" not in md


# --- CLI --------------------------------------------------------------------------------------

def test_argument_types():
    assert perf.int_list("20,5,20") == [5, 20] and perf.int_list("7") == [7]
    for bad in ("", "a,b", "0,5", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            perf.int_list(bad)
    assert perf.positive_int("3") == 3
    for bad in ("0", "1,2", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            perf.positive_int(bad)
    assert perf.iso_date("2026-09-28") == "2026-09-28"
    with pytest.raises(argparse.ArgumentTypeError):
        perf.iso_date("yesterday")


def test_command_line_spells_out_every_knob_and_only_non_default_paths():
    args = argparse.Namespace(pools=[5, 20], repeat=1, scale=[1], rerank_k=[5], scale_queries=50,
                              compare_queries=perf.COMPARE_QUERIES,
                              datasets=Path("eval/datasets.toml"), skills="data/skills", manifest=None,
                              persist_dir="data/chroma", reranker=None)
    assert perf.command_line(args) == "python -m eval.perf --pools 5,20 --repeat 1 --scale 1 --rerank-k 5"
    args.persist_dir, args.scale_queries, args.datasets = "other/chroma", 10, Path("x/sets.toml")
    args.compare_queries = 7
    assert perf.command_line(args).endswith(
        "--rerank-k 5 --scale-queries 10 --compare-queries 7 --datasets x/sets.toml "
        "--persist-dir other/chroma")


# --- query sets from the datasets manifest ----------------------------------------------------

def _write_manifest(tmp_path, corpus="ml-ai-skills"):
    """A datasets.toml with two routing sets and an out-of-scope set (which perf must skip)."""
    rows = {"main": [(SHAP, "explainability"), ("forecast a time series with seasonality", "time-series"),
                     ("impute missing values before training", "data-preprocessing")],
            "source": [("evaluate a rag pipeline", ["rag-evaluation", "rag-pipeline"]),
                       ("tune hyperparameters", "hyperparameter-tuning")],
            "oos": [("bake sourdough bread", None)]}
    tables = []
    for name, items in rows.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"query": q, "gold": g}) + "\n" for q, g in items),
                        encoding="utf-8")
        kind = "out_of_scope" if name == "oos" else "routing"
        tables.append("\n".join(["[[dataset]]", f'name = "{name}"', f'path = "{path.as_posix()}"',
                                 f'kind = "{kind}"', f'corpus = "{corpus}"',
                                 'origin = "synthetic-llm"', ""]))
    manifest = tmp_path / "datasets.toml"
    manifest.write_text("\n".join(tables), encoding="utf-8")
    return manifest


def test_routing_queries_and_facts_come_from_routing_sets_only(tmp_path):
    datasets, skipped = load_datasets(_write_manifest(tmp_path))
    assert skipped == [] and [d.kind for d in datasets] == ["routing", "routing", "out_of_scope"]
    queries = perf.routing_queries(datasets)
    assert [q.set_name for q in queries] == ["main"] * 3 + ["source"] * 2
    assert queries[0] == Query("main", SHAP, "explainability")
    assert queries[3].gold == ["rag-evaluation", "rag-pipeline"]
    facts = perf.dataset_facts(datasets)
    assert list(facts) == ["main", "source"]
    assert facts["main"] == {"path": (tmp_path / "main.jsonl").as_posix(), "n": 3,
                             "sha256": datasets[0].sha256[:12]}


def _fake_hybrid(built=True):
    def make(skills_dir, use_reranker, persist_dir, manifest, strict_rerank=False, reranker_model=None):
        r = HybridRouter(skills_dir=skills_dir, use_reranker=False, persist_dir=persist_dir,
                         manifest=manifest)
        chunks = r._load_chunks()
        r.dense = FakeDense(persist_dir)
        if built:
            r.dense._fp, r.dense.chunks = corpus_fingerprint(chunks), chunks
        if use_reranker:
            r.reranker, r.strict_rerank = FailingReranker(), strict_rerank
        return r
    return make


def test_main_writes_both_files_without_models(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "HybridRouter", _fake_hybrid())
    monkeypatch.setattr(perf, "DedupDenseIndex", FakeDense)
    log = []
    monkeypatch.setattr(perf, "embedding_call_functions",
                        lambda dense: (Embedder("before", log), Embedder("held", log)))
    out, manifest = tmp_path / "out", _write_manifest(tmp_path)
    perf.main(["--pools", "20,5", "--repeat", "1", "--scale", "1,2", "--scale-queries", "2",
               "--compare-queries", "2", "--rerank-k", "5", "--datasets", str(manifest),
               "--skills", str(CORPUS), "--out", str(out), "--stamp", "2026-01-02"])
    data = json.loads((out / "performance.json").read_text(encoding="utf-8"))
    assert data["stamp"] == "2026-01-02" and data["note"] == perf.NOTE
    assert data["command"].startswith("python -m eval.perf --pools 5,20 --repeat 1 --scale 1,2 "
                                      "--rerank-k 5 --scale-queries 2 --compare-queries 2 --datasets ")
    assert [r["pool"] for r in data["pools"]] == [5, 20]
    assert data["pools"][1]["quality"]["main"]["n"] == 3 and data["pools"][1]["samples"] == 5
    assert data["embedding"]["n"] == 5
    calls = data["embedding_calls"]
    assert calls["measured"] is True and calls["identical"] is True and calls["queries"] == 2
    assert calls["before"]["n"] == calls["held"]["n"] == 2
    assert {texts[0] for _, texts in log} == {SHAP, "forecast a time series with seasonality"}
    assert data["reranker"] == {"measured": False, "reason": "no cross-encoder weights (test)",
                                "model": "fake-ce", "pool": 20}
    n = len(_corpus_chunks())
    assert [(r["factor"], r["chunks"]) for r in data["scale"]["rows"]] == [(1, n), (2, 2 * n)]
    assert data["scale"]["queries"] == 2 and data["scale"]["temp_dir_removed"] is True
    setup = data["setup"]
    assert setup["corpus"]["name"] == "ml-ai-skills" and setup["corpus"]["n_chunks"] == n
    assert setup["dense"]["count"] == n and setup["default_pool"] == 20
    assert {name: s["n"] for name, s in setup["queries"].items()} == {"main": 3, "source": 2}
    md = (out / "PERFORMANCE.md").read_text(encoding="utf-8")
    assert md == perf.render(data)                   # the markdown is a pure function of the JSON
    assert "then 1 timed pass interleaved" in md


def test_main_never_builds_a_missing_dense_index(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "HybridRouter", _fake_hybrid(built=False))
    with pytest.raises(SystemExit, match=r"\[perf\] dense index not built"):
        perf.main(["--datasets", str(_write_manifest(tmp_path)), "--skills", str(CORPUS),
                   "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_main_exits_when_no_routing_set_fits_the_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "HybridRouter", _fake_hybrid())
    with pytest.raises(SystemExit, match="no routing dataset for corpus 'ml-ai-skills'"):
        perf.main(["--datasets", str(_write_manifest(tmp_path, corpus="other-corpus")),
                   "--skills", str(CORPUS), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
