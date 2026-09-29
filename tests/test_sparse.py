"""SparseIndex tokenization and lexical evidence (idf, coverage, matched terms); no models."""
import math
from pathlib import Path

import pytest

from sie.chunking import chunk_skill
from sie.ingest import load_corpus
from sie.index import sparse as sparse_mod
from sie.index.sparse import SparseIndex, tokenize
from sie.models import Chunk

CORPUS = Path(__file__).resolve().parents[1] / "data" / "skills"


def _chunk(slug, text):
    return Chunk(skill_slug=slug, section="S", text=text, chunk_id=f"{slug}::S::0")


# "alpha" is in every chunk (rank_bm25 floors its negative idf); the rest are rarer
TOY = [_chunk("a", "alpha beta gamma"), _chunk("b", "alpha delta beta"),
       _chunk("c", "alpha epsilon zeta"), _chunk("d", "alpha eta theta"), _chunk("e", "alpha iota")]


@pytest.fixture(scope="module")
def toy() -> SparseIndex:
    idx = SparseIndex()
    idx.build(TOY)
    return idx


@pytest.fixture(scope="module")
def real() -> SparseIndex:
    idx = SparseIndex()
    idx.build([c for s in load_corpus(CORPUS) for c in chunk_skill(s)])
    return idx


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("Fine-tune BERT (v2.0) on GPUs!") == ["fine", "tune", "bert", "v2", "0", "on", "gpus"]
    assert tokenize("  ...  ") == [] and tokenize("") == []
    assert sparse_mod._tok is tokenize                      # backward-compatible alias


def test_terms_are_unique_in_first_seen_order():
    assert SparseIndex.terms("b a B c a") == ["b", "a", "c"]
    assert SparseIndex().terms("x y x") == ["x", "y"]        # needs no built index


def test_idf_requires_a_built_index():
    with pytest.raises(RuntimeError, match="not built"):
        SparseIndex().idf("alpha")


def test_idf_matches_bm25_and_oov_gets_the_maximum(toy):
    assert toy.idf("beta") == pytest.approx(toy._bm25.idf["beta"])
    assert toy.idf("gamma") > toy.idf("beta") > toy.idf("alpha") > 0   # rarer -> heavier
    assert toy.idf("never-seen") == toy.idf("zzzz") == max(toy._bm25.idf.values())


def test_coverage_bounds_and_idf_weighting(toy):
    assert toy.coverage("beta gamma", "beta gamma extra") == 1.0
    assert toy.coverage("beta gamma", "nothing here") == 0.0
    assert toy.coverage("", "alpha") == 0.0 and toy.coverage("?!", "alpha") == 0.0
    both = toy.idf("alpha") + toy.idf("gamma")
    assert toy.coverage("alpha gamma", "gamma") == pytest.approx(toy.idf("gamma") / both)
    # the rare term carries more of the query's weight than the everywhere-term
    assert toy.coverage("alpha gamma", "gamma") > 0.5 > toy.coverage("alpha gamma", "alpha")
    # repeated query terms count once; case and punctuation don't matter
    assert toy.coverage("Gamma, gamma GAMMA beta", "gamma") == toy.coverage("gamma beta", "gamma")


def test_coverage_falls_back_to_unweighted_share_when_all_weights_are_zero(toy, monkeypatch):
    monkeypatch.setattr(toy, "idf", lambda term: 0.0)
    assert toy.coverage("beta gamma delta zeta", "beta gamma") == 0.5


def test_coverage_is_deterministic_and_bounded_on_the_real_corpus(real):
    queries = ["time series forecasting seasonality", "explain individual predictions with shap values",
               "bake sourdough bread at home", "the the the", "quantize a model for edge deployment"]
    texts = [c.text for c in real._chunks[:60]]
    first = [[real.coverage(q, t) for t in texts] for q in queries]
    assert first == [[real.coverage(q, t) for t in texts] for q in queries]
    assert all(0.0 <= v <= 1.0 and not math.isnan(v) for row in first for v in row)
    rebuilt = SparseIndex()
    rebuilt.build(real._chunks)
    assert first == [[rebuilt.coverage(q, t) for t in texts] for q in queries]


def test_matched_terms_keep_query_order_and_are_unique(toy):
    assert toy.matched_terms("zeta beta ZETA nope alpha", "alpha beta zeta") == ["zeta", "beta", "alpha"]
    assert toy.matched_terms("nope", "alpha") == [] and toy.matched_terms("", "alpha") == []


def test_search_behavior_unchanged(toy, real):
    hits = toy.search("gamma beta", k=10)
    assert [h.skill_slug for h in hits] == ["a", "b"]          # c/d share no term: never returned
    assert hits[0].score > hits[1].score and toy.search("!!!") == []
    with pytest.raises(RuntimeError, match="not built"):
        SparseIndex().search("x")
    top = real.search("time series forecasting seasonality", k=3)
    assert top[0].skill_slug == "time-series" and len(top[0].snippet) <= 200
    assert real.matched_terms("time series forecasting seasonality",
                              next(c.text for c in real._chunks if c.chunk_id == top[0].chunk_id))
