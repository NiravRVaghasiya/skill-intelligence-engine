"""HybridRouter logic with the dense index and cross-encoder mocked (no model downloads)."""
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sie.models import Hit
from sie.rerank import RerankerUnavailable
from sie.index.fuse import reciprocal_rank_fusion
from sie.router import HybridRouter, corpus_fingerprint, dedup_by_skill

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"


class FakeDense:
    embedder = "onnx"
    generation = 0

    def __init__(self, hits, fingerprint):
        self.hits, self._fp, self.calls = hits, fingerprint, 0

    def fingerprint(self):
        return self._fp

    def built_with(self):
        return "onnx"

    def search(self, query, k=10):
        self.calls += 1
        return self.hits[:k]


class FakeReranker:
    """Scores chunks from a fixed table; records what text it was asked to score."""

    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def rerank(self, query, hits, top_k=None, texts=None):
        self.seen = [(h.chunk_id, (texts or {}).get(h.chunk_id)) for h in hits]
        out = [replace(h, score=self.scores.get(h.chunk_id, 0.0)) for h in hits]
        out.sort(key=lambda h: (-h.score, h.skill_slug, h.chunk_id))
        return out if top_k is None else out[:top_k]


class BrokenReranker:
    def rerank(self, *a, **kw):
        raise RerankerUnavailable("no weights")


def _hit(slug, section="Workflow", j=0, score=0.5):
    return Hit(skill_slug=slug, score=score, section=section, chunk_id=f"{slug}::{section}::{j}")


def _router(dense_hits, reranker=None, mode="hybrid", fingerprint=None, strict=False):
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode=mode, strict_rerank=strict)
    fp = corpus_fingerprint(r._load_chunks()) if fingerprint is None else fingerprint
    r.dense = FakeDense(dense_hits, fp)
    r.reranker = reranker
    return r


DENSE = [_hit("explainability", "Card"), _hit("explainability", "Workflow"),
         _hit("ai-ethics-fairness", "Card"), _hit("model-evaluation", "Workflow")]


def test_retrieve_is_deduped_and_sorted():
    hits = _router(DENSE).retrieve("explain individual predictions with shap values", k=5)
    slugs = [h.skill_slug for h in hits]
    assert len(slugs) == len(set(slugs)) and 0 < len(slugs) <= 5
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_no_rerank_prefers_consensus_of_dense_and_bm25():
    hits = _router(DENSE).retrieve("explain individual predictions with shap values", k=3)
    assert hits[0].skill_slug == "explainability"


def test_reranker_scores_full_chunk_text_not_snippet():
    rr = FakeReranker({})
    r = _router(DENSE, reranker=rr)
    r.retrieve("shap values", k=3)
    texts = dict(rr.seen)
    assert texts["explainability::Workflow::0"] == r._texts["explainability::Workflow::0"]
    assert len(texts["explainability::Workflow::0"]) > 200


def test_reranker_order_wins_and_k_applies_after_dedup():
    rr = FakeReranker({"model-evaluation::Workflow::0": 9.0,
                       "explainability::Card::0": 5.0, "explainability::Workflow::0": 4.0})
    hits = _router(DENSE, reranker=rr).retrieve("shap values", k=3)
    assert [h.skill_slug for h in hits][:2] == ["model-evaluation", "explainability"]
    assert len(hits) == 3 and len({h.skill_slug for h in hits}) == 3
    assert hits[1].section == "Card"            # best-scoring section kept


def test_reranker_candidates_limited_to_top_fused_skills():
    # dense top-2 and BM25 top-2 come from three different skills; only the top-2 fused
    # skills may reach the cross-encoder, with every one of their retrieved chunks
    dense = [_hit("time-series", "Card"), _hit("rnn-sequence", "Card"), _hit("recommender-systems", "Card")]
    r = _router(dense, reranker=FakeReranker({}))
    q = "explain individual predictions with shap values"
    r._ensure_ready()
    rankings = [r.dense.search(q, k=3), r.sparse.search(q, k=3)]
    fused = reciprocal_rank_fusion(rankings)
    assert len(fused) >= 4                      # so the pool cut really excludes a skill
    r.retrieve(q, k=3, pool=3)
    seen = {cid.split("::")[0] for cid, _ in r.reranker.seen}
    assert seen == {h.skill_slug for h in fused[:3]} and len(seen) == 3
    expected = {h.chunk_id for ranking in rankings for h in ranking if h.skill_slug in seen}
    assert {cid for cid, _ in r.reranker.seen} == expected


def test_hybrid_uses_both_retrievers():
    # a dense-only skill (BM25 can't match "zzz") and BM25-only skills must both surface
    dense = [_hit("reinforcement-learning", "Card")]
    r = _router(dense)
    hits = r.retrieve("time series forecasting seasonality zzz", k=10)
    slugs = [h.skill_slug for h in hits]
    assert "reinforcement-learning" in slugs and "time-series" in slugs
    assert r.dense.calls == 1
    bm25_only = {h.skill_slug for h in r.sparse.search("time series forecasting seasonality zzz", k=20)}
    assert set(slugs) - {"reinforcement-learning"} <= bm25_only


def test_unavailable_reranker_falls_back_to_rrf_with_warning(capsys):
    r = _router(DENSE, reranker=BrokenReranker())
    expected = [h.skill_slug for h in _router(DENSE).retrieve("shap values", k=3)]
    assert [h.skill_slug for h in r.retrieve("shap values", k=3)] == expected
    assert r.reranker is None and "no weights" in r.rerank_error
    assert "falling back to RRF" in capsys.readouterr().err


def test_strict_rerank_raises():
    with pytest.raises(RerankerUnavailable):
        _router(DENSE, reranker=BrokenReranker(), strict=True).retrieve("shap values")


def test_sparse_mode_never_touches_dense():
    r = _router(DENSE, mode="sparse")
    r.dense = None                                 # any dense access would raise
    assert r.retrieve("time series forecasting seasonality", k=3)[0].skill_slug == "time-series"


def test_dense_mode_ignores_bm25():
    hits = _router(DENSE, mode="dense").retrieve("anything at all", k=10)
    assert [h.skill_slug for h in hits] == ["explainability", "ai-ethics-fairness", "model-evaluation"]


def test_stale_or_missing_dense_index_raises():
    with pytest.raises(RuntimeError, match="stale"):
        _router(DENSE, fingerprint="0000").retrieve("x y z")
    r = _router(DENSE)
    r.dense._fp = None
    with pytest.raises(RuntimeError, match="not built"):
        r.retrieve("x y z")


def test_empty_query_or_zero_k_returns_nothing():
    r = _router(DENSE)
    assert r.retrieve("   ") == [] and r.retrieve("shap", k=0) == []
    assert r.dense.calls == 0


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        HybridRouter(mode="colbert", use_reranker=False)


def test_dedup_by_skill_keeps_first_and_caps_k():
    hits = [_hit("a", j=0, score=3), _hit("a", j=1, score=2), _hit("b", score=1), _hit("c", score=0)]
    out = dedup_by_skill(hits, k=2)
    assert [(h.skill_slug, h.chunk_id) for h in out] == [("a", "a::Workflow::0"), ("b", "b::Workflow::0")]


def test_fingerprint_changes_with_text():
    r = _router(DENSE)
    chunks = r._load_chunks()
    assert corpus_fingerprint(chunks) == corpus_fingerprint(list(chunks))
    assert corpus_fingerprint(chunks) != corpus_fingerprint([replace(chunks[0], text="x")] + chunks[1:])


def test_importing_engine_loads_no_heavy_libraries():
    code = ("import sys, sie.router, sie.api, sie.rerank, sie.index.dense; "
            "heavy = {'chromadb', 'sentence_transformers', 'torch', 'onnxruntime'} & set(sys.modules); "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_pool_must_be_positive():
    with pytest.raises(ValueError):
        _router(DENSE).retrieve("shap", pool=0)


def test_rebind_after_external_rebuild_rechecks_freshness():
    r = _router(DENSE)
    assert r.retrieve("shap values", k=1)
    r.dense._fp, r.dense.generation = "rebuilt-from-other-corpus", 1   # another process rebuilt
    with pytest.raises(RuntimeError, match="stale"):
        r.retrieve("shap values", k=1)


def test_retrieve_is_prefix_consistent_across_k():
    r = _router(DENSE)
    q = "explain individual predictions with shap values"
    top10 = [h.skill_slug for h in r.retrieve(q, k=10)]
    assert [h.skill_slug for h in r.retrieve(q, k=3)] == top10[:3]


def test_last_reranked_reports_what_actually_ordered_the_result():
    r = _router(DENSE, reranker=FakeReranker({}))
    r.retrieve("shap values", k=2)
    assert r.last_reranked is True
    r.retrieve("   ", k=2)                     # early return: nothing was reranked
    assert r.last_reranked is False
    broken = _router(DENSE, reranker=BrokenReranker())
    broken.retrieve("shap values", k=2)
    assert broken.last_reranked is False
