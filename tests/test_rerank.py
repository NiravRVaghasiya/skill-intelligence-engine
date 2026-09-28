"""Reranker behaviour with a stub model; no weights are ever loaded."""
import sys

import pytest

from sie import hub, rerank
from sie.models import Hit
from sie.rerank import Reranker, RerankerUnavailable


class StubModel:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs, show_progress_bar=False):
        self.pairs = pairs
        return [float(len(text)) for _, text in pairs]      # longer text -> higher score


def _stubbed() -> Reranker:
    r = Reranker(model_name="stub")
    r._model = StubModel()
    return r


def test_rerank_uses_full_text_and_does_not_mutate_inputs():
    hits = [Hit("a", 0.9, chunk_id="a::0", snippet="short"), Hit("b", 0.1, chunk_id="b::0", snippet="s")]
    out = _stubbed().rerank("q", hits, texts={"b::0": "a much longer full chunk text"})
    assert [h.skill_slug for h in out] == ["b", "a"]
    assert hits[0].score == 0.9 and hits[1].score == 0.1


def test_rerank_top_k_and_empty():
    hits = [Hit(s, 0.0, chunk_id=s, snippet=s * n) for s, n in [("x", 1), ("y", 3), ("z", 2)]]
    assert [h.skill_slug for h in _stubbed().rerank("q", hits, top_k=2)] == ["y", "z"]
    assert _stubbed().rerank("q", []) == []


def test_missing_weights_fail_fast_without_importing_torch(monkeypatch):
    monkeypatch.setattr(hub, "cached_locally", lambda name: False)
    monkeypatch.setattr(hub, "hub_reachable", lambda timeout=3.0: False)
    already = "sentence_transformers" in sys.modules
    # the fast-fail reason, not "not installed": conftest blocks the import anyway, so only
    # this message proves the decision was made before sentence_transformers was touched
    with pytest.raises(RerankerUnavailable, match="hub is unreachable.*--no-rerank"):
        Reranker(model_name="cross-encoder/none").rerank("q", [Hit("a", 0.0, chunk_id="a")])
    assert already or "sentence_transformers" not in sys.modules


def test_default_model_honors_env(monkeypatch):
    monkeypatch.setenv("SIE_RERANKER", "/models/ce")
    assert rerank.default_model() == "/models/ce"
    monkeypatch.delenv("SIE_RERANKER")
    assert rerank.default_model() in {rerank._DEFAULT_CE, str(rerank.LOCAL_CE_DIR)}


def test_nonexistent_local_path_is_unavailable_not_a_crash(tmp_path):
    with pytest.raises(RerankerUnavailable, match="local model path not found"):
        Reranker(model_name=str(tmp_path / "no-such-model")).rerank("q", [Hit("a", 0.0, chunk_id="a")])
