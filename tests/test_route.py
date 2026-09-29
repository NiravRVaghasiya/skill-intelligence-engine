"""HybridRouter.route(): explainable RouteResults over the real corpus BM25, dense mocked."""
import re
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from sie import router as router_mod
from sie.confidence import ConfidencePolicy
from sie.index.fuse import best_per_skill, reciprocal_rank_fusion
from sie.index.sparse import tokenize
from sie.models import Hit, RouteResult
from sie.rerank import RerankerUnavailable
from sie.router import (MODES, HybridRouter, RetrievalConfig, corpus_fingerprint, dedup_by_skill,
                        route_result_dict)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"
SHAP = "explain individual predictions with shap values"
QUERIES = [SHAP, "time series forecasting seasonality", "shap values", "bake sourdough bread at home",
           "impute missing values before training", "zzzz qqqq"]


class FakeDense:
    """Dense index stand-in: fixed hits, a recorded fingerprint/metadata, build() capture."""
    embedder = "onnx"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    persist_dir = "unused"

    def __init__(self, hits, fingerprint, recorded=None):
        self.hits, self._fp, self.calls, self.generation = hits, fingerprint, 0, 0
        self.recorded = dict(recorded or {})
        self.built = None

    def fingerprint(self):
        return self._fp

    def built_with(self):
        return "onnx"

    def info(self):
        if self._fp is None:
            return None
        return {"embedder": "onnx", "fingerprint": self._fp, **self.recorded, "count": len(self.hits)}

    def search(self, query, k=10):
        self.calls += 1
        return self.hits[:k]

    def build(self, chunks, fingerprint="", metadata=None):
        self.built = (list(chunks), fingerprint, dict(metadata or {}))
        self._fp, self.recorded = fingerprint, dict(metadata or {})
        self.generation += 1


class FakeReranker:
    """Cross-encoder stand-in with load(); scores chunks from a table, records what it saw."""
    model_name = "fake-ce"

    def __init__(self, scores=None, fail=None):
        self.scores, self.fail, self.seen, self.loaded = scores or {}, fail, [], False

    def load(self):
        if self.fail:
            raise RerankerUnavailable(self.fail)
        self.loaded = True

    def rerank(self, query, hits, top_k=None, texts=None):
        self.load()
        self.seen = [h.chunk_id for h in hits]
        out = [replace(h, score=self.scores.get(h.chunk_id, 0.0)) for h in hits]
        out.sort(key=lambda h: (-h.score, h.skill_slug, h.chunk_id))
        return out if top_k is None else out[:top_k]


def _hit(slug, section="Workflow", j=0, score=0.5):
    return Hit(skill_slug=slug, score=score, section=section, chunk_id=f"{slug}::{section}::{j}")


DENSE = [_hit("explainability", "Card"), _hit("explainability", "Workflow"),
         _hit("ai-ethics-fairness", "Card"), _hit("model-evaluation", "Workflow")]
DISAGREE = [_hit("time-series", "Card"), _hit("rnn-sequence", "Card"), _hit("recommender-systems", "Card")]
CE_SCORES = {"model-evaluation::Workflow::0": 9.0, "explainability::Card::0": 5.0,
             "explainability::Workflow::0": 4.0}


def _router(dense_hits=DENSE, reranker=None, mode="hybrid", fingerprint=None, strict=False,
            recorded=None, **kw):
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode=mode, strict_rerank=strict, **kw)
    fp = corpus_fingerprint(r._load_chunks()) if fingerprint is None else fingerprint
    r.dense = FakeDense(dense_hits, fp, recorded)
    r.reranker = reranker
    return r


def _legacy_retrieve(r, query, k, pool):
    """The pre-route() retrieve pipeline, inlined: route().hits must reproduce it exactly."""
    if k <= 0 or not query.strip():
        return []
    r._ensure_ready()
    rankings = []
    if r.mode in ("hybrid", "dense"):
        rankings.append(r.dense.search(query, k=pool))
    if r.mode in ("hybrid", "sparse"):
        rankings.append(r.sparse.search(query, k=pool))
    fused = reciprocal_rank_fusion(rankings)
    if r.reranker is not None:
        keep = {h.skill_slug for h in fused[:pool]}
        candidates = {}
        for ranking in rankings:
            for h in ranking:
                if h.skill_slug in keep:
                    candidates.setdefault(h.chunk_id, h)
        return dedup_by_skill(r.reranker.rerank(query, list(candidates.values()), texts=r._texts), k)
    return dedup_by_skill(fused, k)


def _stable(result: RouteResult) -> dict:
    """RouteResult as data minus wall-clock timings."""
    data = route_result_dict(result)
    data.pop("timings_ms")
    return data


# --- route() == retrieve() == the original pipeline ------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rerank", [False, True])
def test_route_hits_equal_retrieve_and_the_original_pipeline(mode, rerank):
    r = _router(reranker=FakeReranker(CE_SCORES) if rerank else None, mode=mode)
    for q in QUERIES:
        for k, pool in [(1, 20), (3, 3), (10, 20), (5, 40)]:
            hits = r.route(q, k=k, pool=pool).hits
            assert hits == r.retrieve(q, k=k, pool=pool) == _legacy_retrieve(r, q, k, pool), (q, k, pool)


def test_retrieve_defaults_are_the_benchmarked_k5_pool20():
    r = _router()
    assert r.retrieve(SHAP) == r.retrieve(SHAP, k=5, pool=20) == _legacy_retrieve(r, SHAP, 5, 20)


def test_results_are_ranked_skills_one_per_skill():
    res = _router().route(SHAP, k=10)
    slugs = [s.slug for s in res.results]
    assert len(slugs) == len(set(slugs)) and [s.rank for s in res.results] == list(range(1, len(slugs) + 1))
    assert [s.score for s in res.results] == sorted((s.score for s in res.results), reverse=True)
    assert {s.score_type for s in res.results} == {"rrf"} and res.reranked is False
    assert res.top is res.results[0] and res.query == SHAP and res.mode == "hybrid"


def test_confidence_is_assessed_on_the_full_ranking_not_the_top_k():
    r = _router()
    one, ten = r.route(SHAP, k=1), r.route(SHAP, k=10)
    assert len(one.results) == 1 and one.results[0] == ten.results[0]
    assert one.confidence == ten.confidence
    assert one.confidence.signals["margin"] is not None      # the runner-up was visible


# --- evidence ---------------------------------------------------------------------------------

def test_evidence_methods_ranks_scores_and_matched_terms():
    r = _router()
    res = r.route(SHAP, k=10)
    top = res.results[0]
    assert top.slug == "explainability" and top.methods == ["dense", "bm25"]
    dense_ev, bm25_ev = top.evidence
    assert (dense_ev.method, dense_ev.rank, dense_ev.score, dense_ev.section) == ("dense", 1, 0.5, "Card")
    assert dense_ev.matched_terms == []
    bm25_best = best_per_skill(r.sparse.search(SHAP, k=20))
    by_slug = {h.skill_slug: (i, h) for i, h in enumerate(bm25_best, 1)}
    q_terms = tokenize(SHAP)
    for skill in res.results:
        for ev in skill.evidence:
            assert len(ev.snippet) <= 200
            if ev.method != "bm25":
                continue
            rank, hit = by_slug[skill.slug]
            assert (ev.rank, ev.score, ev.section) == (rank, hit.score, hit.section)
            assert ev.matched_terms == r.sparse.matched_terms(SHAP, r._texts[hit.chunk_id])
            assert ev.matched_terms and ev.matched_terms == [t for t in dict.fromkeys(q_terms)
                                                            if t in ev.matched_terms]
    assert "shap" in bm25_ev.matched_terms
    bm25_only = [s for s in res.results if s.methods == ["bm25"]]
    assert bm25_only and all(len(s.evidence) == 1 for s in bm25_only)


def test_single_retriever_modes_only_report_their_method():
    assert {m for s in _router(mode="sparse").route(SHAP, k=10).results for m in s.methods} == {"bm25"}
    assert {m for s in _router(mode="dense").route(SHAP, k=10).results for m in s.methods} == {"dense"}


def test_reranked_results_carry_cross_encoder_evidence():
    res = _router(reranker=FakeReranker(CE_SCORES)).route(SHAP, k=3)
    assert res.reranked is True and [s.slug for s in res.results][:2] == ["model-evaluation", "explainability"]
    for s in res.results:
        assert s.score_type == "cross-encoder" and "rerank" not in s.methods
        ce = s.evidence[-1]
        assert (ce.method, ce.rank, ce.score, ce.section) == ("rerank", s.rank, s.score, s.section)
    assert res.results[1].section == "Card"                      # best CE chunk of the skill
    assert res.confidence.reasons[0] == "cross-encoder ranks model-evaluation first"


# --- confidence wiring ------------------------------------------------------------------------

def test_agreeing_retrievers_route_with_high_confidence():
    c = _router().route(SHAP).confidence
    assert (c.level, c.action, c.ambiguous) == ("high", "route", False)
    assert c.signals["leaders"] == {"dense": "explainability", "bm25": "explainability"}
    assert c.signals["top_similarity"] == 0.5 and 0.0 < c.signals["coverage"] <= 1.0
    assert "explainability" in c.reasons[0]


def test_disagreeing_retrievers_ask_to_clarify():
    res = _router(DISAGREE).route(SHAP)
    c = res.confidence
    assert c.signals["leaders"] == {"dense": "time-series", "bm25": "explainability"}
    assert c.action == "clarify" and c.ambiguous is True
    assert {"time-series", "explainability"} - {res.top.slug} <= set(c.competitors)
    assert any("ranks explainability first" in reason for reason in c.reasons)


def test_unsupported_query_without_dense_evidence_asks_instead_of_abstaining():
    # BM25 alone can't tell out-of-scope from a long paraphrased request: low, not none
    res = _router(mode="sparse").route("bake sourdough bread at home")
    assert res.results and (res.confidence.level, res.confidence.action) == ("low", "clarify")
    assert res.confidence.signals["coverage"] < 0.4


def test_unsupported_query_abstains_when_dense_evidence_is_weak_too():
    weak = [_hit("time-series", "Card", score=0.08), _hit("rnn-sequence", "Card", score=0.05)]
    res = _router(weak).route("bake sourdough bread at home")
    assert res.results and (res.confidence.level, res.confidence.action) == ("none", "abstain")
    assert res.confidence.signals["coverage"] < 0.4 and res.confidence.signals["top_similarity"] == 0.08


def test_coverage_is_the_best_over_the_top_skills_retrieved_chunks():
    r = _router()
    res = r.route(SHAP)
    chunks = {h.chunk_id for ranking in (r.dense.search(SHAP, 20), r.sparse.search(SHAP, 20))
              for h in ranking if h.skill_slug == res.top.slug}
    best = max(r.sparse.coverage(SHAP, r._texts[c]) for c in chunks)
    assert res.confidence.signals["coverage"] == round(best, 4)


def test_policy_is_used():
    default = _router(mode="sparse").route(SHAP).confidence
    strict = _router(mode="sparse", policy=ConfidencePolicy(min_coverage=0.99)).route(SHAP).confidence
    assert default.level == "high" and strict.level == "low"


# --- rerank_k, config -------------------------------------------------------------------------

def test_rerank_k_limits_cross_encoder_candidates_independently_of_pool():
    rr = FakeReranker()
    r = _router(DISAGREE, reranker=rr)
    r._ensure_ready()
    rankings = [r.dense.search(SHAP, k=20), r.sparse.search(SHAP, k=20)]
    fused = reciprocal_rank_fusion(rankings)
    full = r.route(SHAP, k=10, pool=20)
    assert {c.split("::")[0] for c in rr.seen} == {h.skill_slug for h in fused}
    limited = r.route(SHAP, k=10, pool=20, rerank_k=2)
    top2 = {h.skill_slug for h in fused[:2]}
    assert {c.split("::")[0] for c in rr.seen} == top2
    assert set(rr.seen) == {h.chunk_id for ranking in rankings for h in ranking if h.skill_slug in top2}
    assert limited.candidates["rerank"] == len(rr.seen) < full.candidates["rerank"]
    for key in ("dense", "bm25", "fused"):                     # the pool is untouched
        assert limited.candidates[key] == full.candidates[key]
    assert {s.slug for s in limited.results} == top2


def test_rerank_k_from_config_and_call_override():
    rr = FakeReranker()
    r = _router(DISAGREE, reranker=rr, config=RetrievalConfig(rerank_k=1))
    r.route(SHAP)
    assert len({c.split("::")[0] for c in rr.seen}) == 1
    r.route(SHAP, rerank_k=3)
    assert len({c.split("::")[0] for c in rr.seen}) == 3


def test_config_defaults_drive_k_pool_and_rrf_k():
    r = _router(mode="sparse", config=RetrievalConfig(top_k=2, candidate_pool=10, rrf_k=10))
    res = r.route("time series forecasting seasonality")
    assert res.candidates["bm25"] == 10 and res.candidates["fused"] > 2
    assert len(res.results) == 2 and len(r.retrieve("time series forecasting seasonality")) == 2
    assert res.results[0].score == pytest.approx(1 / 11)
    assert len(r.route("time series forecasting seasonality", k=3, pool=20).results) == 3


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"candidate_pool": 0}, {"rerank_k": 0}, {"rrf_k": 0},
                                    {"top_k": True}, {"candidate_pool": 2.5}])
def test_retrieval_config_validation(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig(**kwargs)


def test_retrieval_config_defaults_and_type_check():
    c = RetrievalConfig()
    assert (c.top_k, c.candidate_pool, c.rerank_k, c.rrf_k) == (5, 20, None, 60)
    with pytest.raises(TypeError):
        HybridRouter(use_reranker=False, config={"top_k": 3})


def test_invalid_pool_or_rerank_k_raise():
    r = _router()
    with pytest.raises(ValueError, match="pool"):
        r.route(SHAP, pool=0)
    with pytest.raises(ValueError, match="rerank_k"):
        r.route(SHAP, rerank_k=0)


# --- timings, candidates, empty results -------------------------------------------------------

@pytest.mark.parametrize("mode,rerank,timing_keys,candidate_keys", [
    ("hybrid", False, {"dense", "bm25", "fuse", "explain", "total"}, {"dense", "bm25", "fused"}),
    ("hybrid", True, {"dense", "bm25", "fuse", "rerank", "explain", "total"}, {"dense", "bm25", "fused", "rerank"}),
    ("dense", False, {"dense", "fuse", "explain", "total"}, {"dense", "fused"}),
    ("sparse", False, {"bm25", "fuse", "explain", "total"}, {"bm25", "fused"}),
    ("sparse", True, {"bm25", "fuse", "rerank", "explain", "total"}, {"bm25", "fused", "rerank"}),
])
def test_timings_and_candidates(mode, rerank, timing_keys, candidate_keys):
    res = _router(reranker=FakeReranker() if rerank else None, mode=mode).route(SHAP, pool=20)
    assert set(res.timings_ms) == timing_keys and set(res.candidates) == candidate_keys
    assert all(isinstance(v, float) and v >= 0 and round(v, 3) == v for v in res.timings_ms.values())
    assert res.timings_ms["total"] >= max(v for key, v in res.timings_ms.items() if key != "total")
    if "dense" in candidate_keys:
        assert res.candidates["dense"] == len(DENSE)
    if "bm25" in candidate_keys:
        assert res.candidates["bm25"] == 20
    assert res.candidates["fused"] >= len(res.results)


def test_no_rerank_attempt_when_nothing_was_retrieved():
    rr = FakeReranker()
    res = _router(mode="sparse", reranker=rr).route("zzzz qqqq")
    assert res.results == [] and res.reranked is False and rr.seen == []
    assert "rerank" not in res.candidates and "rerank" not in res.timings_ms
    assert (res.confidence.level, res.confidence.reasons) == ("none", ["no candidate skills retrieved"])


@pytest.mark.parametrize("query,k,reason", [("   ", 5, "empty query"), ("", 5, "empty query"),
                                           ("shap values", 0, "k <= 0"), ("shap values", -1, "k <= 0")])
def test_empty_query_or_nonpositive_k(query, k, reason):
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False)
    r.dense = FakeDense(DENSE, "never-checked")
    res = r.route(query, k=k)
    assert res.results == [] and res.hits == [] and res.top is None
    c = res.confidence
    assert (c.level, c.action, c.reasons, c.ambiguous) == ("none", "abstain", [reason], False)
    assert res.candidates == {} and set(res.timings_ms) == {"total"} and res.reranked is False
    assert r.dense.calls == 0 and not r.sparse.is_built and r._chunks is None   # nothing touched


def test_last_reranked_mirrors_the_last_result():
    r = _router(reranker=FakeReranker())
    assert r.route(SHAP).reranked is True and r.last_reranked is True
    assert r.route("  ").reranked is False and r.last_reranked is False


def test_route_result_dict_drops_chunk_ids_and_is_json_ready():
    import json
    data = route_result_dict(_router(reranker=FakeReranker(CE_SCORES)).route(SHAP, k=3))
    text = json.dumps(data, ensure_ascii=False)
    assert "chunk_id" not in text and json.loads(text) == data
    assert set(data) == {"query", "results", "confidence", "mode", "reranked", "rerank_error",
                         "candidates", "timings_ms"}
    assert data["results"][0]["evidence"][0]["method"] == "dense"


# --- corpus, build, freshness -----------------------------------------------------------------

def test_corpus_identity_is_loaded_with_the_chunks():
    r = _router()
    assert set(r.skills) == {c.skill_slug for c in r._load_chunks()} and len(r.skills) == 38
    assert r.corpus.name == "ml-ai-skills" and r.corpus.n_skills == 38
    assert re.fullmatch(r"[0-9a-f]{16}", r.corpus.fingerprint)


def test_build_records_index_metadata():
    r = _router()
    fake = r.dense
    assert r.build() == len(r._load_chunks())
    chunks, fingerprint, meta = fake.built
    assert fingerprint == corpus_fingerprint(chunks) and chunks == r._load_chunks()
    assert set(meta) == {"index_schema", "embedder", "model_name", "corpus_name", "corpus_version",
                         "corpus_fingerprint", "n_chunks", "n_skills", "indexed_at"}
    assert meta["index_schema"] == 2 and meta["embedder"] == "onnx"
    assert meta["model_name"] == FakeDense.model_name
    assert (meta["corpus_name"], meta["corpus_version"]) == (r.corpus.name, r.corpus.version)
    assert meta["corpus_fingerprint"] == r.corpus.fingerprint != fingerprint
    assert (meta["n_chunks"], meta["n_skills"]) == (len(chunks), 38)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["indexed_at"])
    assert r.sparse.is_built and r.status()["dense"]["ready"] is True
    assert r.route(SHAP).results                                 # fresh: no stale error


def test_embedding_model_mismatch_is_stale():
    r = _router(recorded={"model_name": "sentence-transformers/all-mpnet-base-v2"})
    with pytest.raises(RuntimeError, match="stale.*all-mpnet-base-v2"):
        r.route(SHAP)


def test_old_index_without_model_name_is_still_compatible():
    r = _router(recorded={})                   # built before metadata was recorded
    assert r.route(SHAP).results


def test_explicit_manifest_names_the_corpus(tmp_path):
    corpus = tmp_path / "elsewhere"
    for slug, body in [("alpha", "manifest routing works"), ("beta", "something else entirely")]:
        (corpus / slug).mkdir(parents=True)       # >= 2 skills: BM25 idf is 0 on a 2-chunk corpus
        (corpus / slug / "SKILL.md").write_text(
            f"---\ntype: workflow\ndomain: d\nlevel: beginner\ndescription: Use when {slug}\n---\n"
            f"## Overview\n{body}\n", encoding="utf-8")
    (corpus / "custom.toml").write_text('name = "custom-corpus"\nversion = "7"\n', encoding="utf-8")
    r = HybridRouter(skills_dir=str(tmp_path / "unused"), use_reranker=False, mode="sparse",
                     manifest=str(corpus / "custom.toml"))
    res = r.route("manifest routing")
    assert res.top.slug == "alpha"
    assert (r.corpus.name, r.corpus.version, r.corpus.n_skills) == ("custom-corpus", "7", 2)


def test_missing_manifest_is_a_runtime_error(tmp_path):
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode="sparse",
                     manifest=str(tmp_path / "nope.toml"))
    with pytest.raises(RuntimeError, match="cannot load the corpus"):
        r.route(SHAP)


# --- status() and warm_up() -------------------------------------------------------------------

def test_status_of_a_fresh_sparse_router_loads_nothing():
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode="sparse")
    r.dense = None                                 # sparse status must not touch dense
    assert r.status() == {
        "mode": "sparse", "corpus": None, "chunks": 0, "sparse": False,
        "dense": {"required": False, "ready": False, "index": None},
        "reranker": {"enabled": False, "loaded": False, "model": None, "error": None}}
    after = r.warm_up()
    assert after["sparse"] is True and after["chunks"] == len(r._load_chunks())
    assert after["corpus"] == {"name": "ml-ai-skills", "version": r.corpus.version,
                               "fingerprint": r.corpus.fingerprint, "n_skills": 38}


def test_warm_up_loads_embedder_and_reranker_and_status_never_embeds():
    rr = FakeReranker()
    r = _router(reranker=rr)
    before = r.status()
    assert before["dense"]["required"] is True and before["dense"]["ready"] is False
    assert before["reranker"] == {"enabled": True, "loaded": False, "model": "fake-ce", "error": None}
    status = r.warm_up()
    assert r.dense.calls == 1 and rr.loaded                     # one dummy embedding
    assert status["dense"] == {"required": True, "ready": True, "index": r.dense.info()}
    assert status["reranker"] == {"enabled": True, "loaded": True, "model": "fake-ce", "error": None}
    r.status()
    assert r.dense.calls == 1


def test_warm_up_reranker_failure_falls_back_unless_strict(capsys):
    r = _router(reranker=FakeReranker(fail="no weights"))
    status = r.warm_up()
    assert r.reranker is None and "no weights" in r.rerank_error
    assert status["reranker"] == {"enabled": False, "loaded": False, "model": None, "error": "no weights"}
    assert "falling back to RRF" in capsys.readouterr().err
    res = r.route(SHAP)
    assert res.reranked is False and res.rerank_error == "no weights"
    with pytest.raises(RerankerUnavailable):
        _router(reranker=FakeReranker(fail="no weights"), strict=True).warm_up()


def test_warm_up_raises_like_route_on_missing_or_stale_dense_index():
    with pytest.raises(RuntimeError, match="stale"):
        _router(fingerprint="0000").warm_up()
    r = _router()
    r.dense._fp = None
    with pytest.raises(RuntimeError, match="not built"):
        r.warm_up()
    assert r.status()["dense"]["index"] is None


def test_status_reports_an_unreadable_dense_index_instead_of_raising():
    r = _router()

    def broken():
        raise ImportError("chromadb missing")

    r.dense.info = broken
    status = r.status()
    assert status["dense"]["index"] is None and "chromadb missing" in status["dense"]["error"]


# --- concurrency ------------------------------------------------------------------------------

def test_concurrent_routes_initialize_once_and_agree(monkeypatch):
    calls = []
    real = router_mod.load_corpus_report

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(router_mod, "load_corpus_report", counting)
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode="sparse")
    queries = ["time series forecasting seasonality", SHAP, "impute missing values"]
    start, errors, outputs = threading.Barrier(8), [], [[] for _ in range(8)]

    def worker(i):
        try:
            start.wait()
            for j in range(20):
                q = queries[j % len(queries)]
                outputs[i].append((q, _stable(r.route(q, k=5))))
        except Exception as e:                     # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(calls) == 1
    reference = {q: _stable(HybridRouter(skills_dir=str(CORPUS), use_reranker=False,
                                         mode="sparse").route(q, k=5)) for q in queries}
    assert all(result == reference[q] for out in outputs for q, result in out)
    assert sum(len(out) for out in outputs) == 160


# --- reranker disable / restore races (per-request rerank_error) -----------------------------

class LoadedElsewhere(FakeReranker):
    """This request's load fails, but another thread finished loading the model meanwhile."""

    def rerank(self, query, hits, top_k=None, texts=None):
        self.loaded = True
        raise RerankerUnavailable("hub probe timed out")


class RacingReranker(FakeReranker):
    """While request 1 is cross-encoding, request 2's load fails transiently (and disables it);
    request 1 then finishes loading and scores."""

    def __init__(self, router_fn):
        super().__init__(CE_SCORES)
        self.router_fn, self.depth, self.inner = router_fn, 0, None

    def rerank(self, query, hits, top_k=None, texts=None):
        self.depth += 1
        try:
            if self.depth == 1:
                self.inner = self.router_fn().route(query)   # the concurrent request
                return super().rerank(query, hits, top_k, texts)
            raise RerankerUnavailable("hub probe timed out")
        finally:
            self.depth -= 1


def test_a_load_failure_never_disables_a_reranker_another_thread_loaded():
    rr = LoadedElsewhere(CE_SCORES)
    r = _router(reranker=rr)
    res = r.route(SHAP)
    assert res.reranked is False and res.rerank_error == "hub probe timed out"   # this request
    assert r.reranker is rr and r.rerank_error is None                           # still in use
    assert r.status()["reranker"] == {"enabled": True, "loaded": True, "model": "fake-ce", "error": None}


def test_a_reranker_loaded_after_a_concurrent_failure_is_restored(capsys):
    r = _router()
    rr = RacingReranker(lambda: r)
    r.reranker = rr
    res = r.route(SHAP, k=3)
    inner = rr.inner
    assert inner.reranked is False and inner.rerank_error == "hub probe timed out"
    assert res.reranked is True and res.rerank_error is None                     # not contradictory
    assert [s.slug for s in res.results][:2] == ["model-evaluation", "explainability"]
    assert r.reranker is rr and r.rerank_error is None                           # back in use
    assert r.status()["reranker"] == {"enabled": True, "loaded": True, "model": "fake-ce", "error": None}
    again = r.route(SHAP, k=3)
    assert again.reranked is True and again.rerank_error is None
    assert "falling back to RRF" in capsys.readouterr().err


def test_a_reranked_result_never_carries_an_old_router_error():
    rr = FakeReranker(CE_SCORES)
    r = _router(reranker=rr)
    r.rerank_error = "an earlier failure"                  # e.g. left by a racing thread
    res = r.route(SHAP)
    assert res.reranked is True and res.rerank_error is None


# --- revalidate(): cheap re-verification after the dense handle was rebound ------------------

class CountingDense(FakeDense):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fingerprint_reads = 0

    def fingerprint(self):
        self.fingerprint_reads += 1
        return super().fingerprint()


def _counting_router(**kw):
    r = _router(**kw)
    r.dense = CountingDense(DENSE, corpus_fingerprint(r._load_chunks()))
    return r


def test_revalidate_is_a_no_op_until_there_is_something_to_recheck():
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode="sparse")
    r.dense = None                                         # sparse: dense never touched
    r.revalidate()
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False)
    r.dense = CountingDense(DENSE, "never-read")
    r.revalidate()                                         # corpus not loaded
    assert r._chunks is None and r.dense.fingerprint_reads == 0
    r = _counting_router()
    r.revalidate()                                         # never verified: nothing to re-check
    r.route(SHAP)
    reads = r.dense.fingerprint_reads
    r.revalidate()                                         # verified handle still current
    assert r.dense.fingerprint_reads == reads


def test_revalidate_after_a_rebind_rechecks_without_loading_anything(monkeypatch):
    r = _counting_router()
    r.route(SHAP)
    loads = []
    monkeypatch.setattr(router_mod, "load_corpus_report", lambda *a, **kw: loads.append(a))
    r.dense.generation += 1                                # another process rebuilt; handle rebound
    assert r.status()["dense"]["ready"] is False
    calls = r.dense.calls
    r.revalidate()
    assert r.status()["dense"]["ready"] is True
    assert r.dense.calls == calls and not loads            # no embedding, no corpus reload


def test_revalidate_raises_like_ensure_ready_when_stale_or_missing():
    r = _counting_router()
    r.route(SHAP)
    r.dense._fp, r.dense.generation = "rebuilt-from-other-corpus", 1
    with pytest.raises(RuntimeError, match="dense index is stale"):
        r.revalidate()
    assert r.status()["dense"]["ready"] is False
    r.dense._fp = None
    with pytest.raises(RuntimeError, match="dense index not built"):
        r.revalidate()
    r.dense._fp = corpus_fingerprint(r._load_chunks())    # rebuilt again from this corpus
    r.revalidate()
    assert r.status()["dense"]["ready"] is True


class RebindingDense(FakeDense):
    """search() finds the collection replaced (chromadb NotFoundError) and rebinds, like DenseIndex."""

    def __init__(self, *a, new_fp=None, **kw):
        super().__init__(*a, **kw)
        self.new_fp = new_fp

    def search(self, query, k=10):
        if self.new_fp is not None:
            self._fp, self.new_fp = self.new_fp, None
            self.generation += 1
        return super().search(query, k)


def test_the_request_that_rebinds_is_not_served_from_an_unverified_index():
    r = _router()
    fp = corpus_fingerprint(r._load_chunks())
    r.dense = RebindingDense(DENSE, fp)
    r.route(SHAP)                                          # verified
    r.dense.new_fp = "rebuilt-from-other-corpus"
    with pytest.raises(RuntimeError, match="stale"):
        r.route(SHAP)
    r = _router()
    r.dense = RebindingDense(DENSE, fp)
    r.route(SHAP)
    r.dense.new_fp = fp                                    # rebuilt from the same corpus
    assert r.route(SHAP).results and r.status()["dense"]["ready"] is True


# --- no machine paths in status / messages ---------------------------------------------------

def test_status_shows_report_safe_reranker_model_names(tmp_path):
    from sie.rerank import LOCAL_CE_DIR, LOCAL_CE_HINT, Reranker
    r = _router()
    r.reranker = Reranker(model_name=str(LOCAL_CE_DIR))
    assert r.status()["reranker"]["model"] == LOCAL_CE_HINT
    r.reranker = Reranker(model_name=str(tmp_path / "models" / "my-ce"))
    assert r.status()["reranker"]["model"] == "my-ce"
    r.reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert r.status()["reranker"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_corpus_errors_show_the_path_as_passed(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    r = HybridRouter(skills_dir="relative/corpus", use_reranker=False, mode="sparse")
    with pytest.raises(RuntimeError) as exc:
        r.route(SHAP)
    assert str(exc.value) == "no skills found under relative/corpus"
    r = HybridRouter(skills_dir="relative/corpus", use_reranker=False, mode="sparse",
                     manifest="missing.toml")
    with pytest.raises(RuntimeError) as exc:
        r.route(SHAP)
    assert str(exc.value).startswith("cannot load the corpus at relative/corpus: ")
    assert "missing.toml" in str(exc.value)
    assert str(tmp_path) not in str(exc.value) and os.getcwd() not in str(exc.value)


def test_concurrent_routes_with_one_transient_load_failure_keep_the_reranker():
    class FlakyLoad(FakeReranker):
        """The first load attempt fails (e.g. a hub probe timeout); every later one succeeds."""

        def __init__(self):
            super().__init__(CE_SCORES)
            self.attempts, self.lock = 0, threading.Lock()

        def load(self):
            with self.lock:
                if self.loaded:
                    return
                self.attempts += 1
                if self.attempts == 1:
                    raise RerankerUnavailable("hub probe timed out")
                self.loaded = True

    rr = FlakyLoad()
    r = _router(reranker=rr)
    r._ensure_ready()
    start, results, errors = threading.Barrier(8), [], []

    def worker():
        try:
            start.wait()
            for _ in range(5):
                results.append(r.route(SHAP, k=3))
        except Exception as e:                     # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 40
    assert all(res.rerank_error is None for res in results if res.reranked)
    assert sum(not res.reranked for res in results) >= 1              # the failed request
    if any(res.reranked for res in results):       # a thread loaded it: it must stay in use
        assert r.reranker is rr and r.rerank_error is None and rr.loaded
        assert r.route(SHAP, k=3).reranked is True
    else:                                          # the failure came first: RRF from then on
        assert r.reranker is None and r.rerank_error == "hub probe timed out"
    status = r.status()["reranker"]
    assert status["enabled"] == status["loaded"] == (r.reranker is not None)


class SnapshotDense(FakeDense):
    """Like a chroma handle: metadata is a snapshot until rebind() drops the cached handle."""

    def __init__(self, hits, fingerprint):
        super().__init__(hits, fingerprint)
        self.persisted, self.rebinds = fingerprint, 0

    def rebind(self):
        self.rebinds += 1
        self._fp = self.persisted


def test_stale_index_recovers_after_an_external_rebuild_without_restart():
    r = _router()
    good = r.dense._fp
    r.dense = SnapshotDense(DENSE, "built-from-another-corpus")
    with pytest.raises(RuntimeError, match="stale"):
        r.route(SHAP)
    r.dense.persisted = good                 # another process rebuilt the index correctly
    assert r.route(SHAP).results and r.dense.rebinds == 2
