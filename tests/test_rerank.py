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
    monkeypatch.setattr(rerank, "_st_installed", lambda: True)     # the library is "installed"
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


def test_nonexistent_local_path_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "_st_installed", lambda: True)
    with pytest.raises(RerankerUnavailable, match="local model path not found: no-such-model") as exc:
        Reranker(model_name=str(tmp_path / "no-such-model")).rerank("q", [Hit("a", 0.0, chunk_id="a")])
    assert str(tmp_path) not in str(exc.value)                      # no machine path in the message
    assert "cannot load cross-encoder 'no-such-model'" in str(exc.value)


# --- report-safe model names (no machine paths in messages / status) ------------------------

@pytest.mark.parametrize("name,shown", [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),   # hub id
    ("stub", "stub"), (r"C:\Users\me\models\my-ce", "my-ce"), ("C:/Users/me/models/my-ce/", "my-ce"),
    ("/home/me/models/my-ce", "my-ce"), ("~/models/my-ce", "my-ce"), ("./my-ce", "my-ce"),
])
def test_display_model(name, shown):
    assert rerank.display_model(name) == shown


def test_display_model_shows_the_default_local_dir_repo_relative():
    assert rerank.display_model(str(rerank.LOCAL_CE_DIR)) == rerank.LOCAL_CE_HINT
    assert rerank.display_model(str(rerank.LOCAL_CE_DIR) + "/") == rerank.LOCAL_CE_HINT


def test_default_local_weights_that_fail_to_load_are_reported_repo_relative(tmp_path, monkeypatch):
    # the weights dir exists (so no hub probe) but loading fails, e.g. a partial copy
    import types
    local = tmp_path / "machine" / "specific" / "ms-marco-MiniLM-L-6-v2"
    local.mkdir(parents=True)
    monkeypatch.setattr(rerank, "LOCAL_CE_DIR", local)

    class CrossEncoder:
        def __init__(self, name, **kwargs):
            raise OSError(f"incomplete weights in {name}")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=CrossEncoder))
    with pytest.raises(RerankerUnavailable) as exc:
        Reranker(model_name=str(local)).load()
    message = str(exc.value)
    assert message.startswith(f"cannot load cross-encoder '{rerank.LOCAL_CE_HINT}' (OSError)")
    assert str(tmp_path) not in message and "machine" not in message


# --- sentence-transformers not installed: say so, and don't probe the hub --------------------

def test_missing_library_is_reported_without_probing_the_hub(monkeypatch):
    monkeypatch.setattr(rerank, "unavailable_reason", lambda name: pytest.fail("probed weights / hub"))
    with pytest.raises(RerankerUnavailable) as exc:
        Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2").load()
    message = str(exc.value)
    assert "(sentence-transformers not installed)" in message and "'reranker' extra" in message
    assert "--no-rerank" in message
    assert "copy the weights" not in message and "SIE_RERANKER" not in message
    assert "sentence_transformers" not in sys.modules                # never imported


def test_library_check_never_imports_it(monkeypatch):
    import types
    assert rerank._st_installed() is False                          # conftest blocks the import
    assert "sentence_transformers" not in sys.modules
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace())
    assert rerank._st_installed() is True                           # already imported (or stubbed)
