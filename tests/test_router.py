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


# --- CLI: `python -m sie.router "<query>"` (sparse mode: no models) ---------------------------

class ReverseReranker:
    """Stand-in for sie.router.Reranker in CLI tests: scores chunks by reverse chunk-id order."""
    model_name = "cli-fake"

    def __init__(self, model_name=None):
        pass

    def rerank(self, query, hits, top_k=None, texts=None):
        ordered = sorted(hits, key=lambda h: h.chunk_id, reverse=True)
        out = [replace(h, score=float(-i)) for i, h in enumerate(ordered)]
        return out if top_k is None else out[:top_k]


def _cli(monkeypatch, capsys, *args):
    from sie.router import main
    monkeypatch.setattr(sys, "argv", ["sie.router", "--skills", str(CORPUS), *args])
    main()
    return capsys.readouterr()


def test_cli_prints_ranking_and_confidence(monkeypatch, capsys):
    out = _cli(monkeypatch, capsys, "time series forecasting seasonality", "--mode", "sparse",
               "--no-rerank", "-k", "3").out.splitlines()
    assert out[0] == "[router] mode=sparse ranking=rrf"
    assert out[1].split()[:2] == ["1.", "time-series"] and "score=" in out[1] and out[1].endswith("[Card]")
    assert [line.split()[0] for line in out[1:4]] == ["1.", "2.", "3."]
    assert out[4] == "[router] confidence: high (route): bm25 ranks time-series first"
    assert len(out) == 5


def test_cli_explain_prints_evidence_under_each_result(monkeypatch, capsys):
    out = _cli(monkeypatch, capsys, "time series forecasting seasonality", "--mode", "sparse",
               "--no-rerank", "-k", "2", "--explain").out.splitlines()
    assert out[1].split()[:2] == ["1.", "time-series"]
    method, rank, score = out[2].split()[:3]
    assert (method, rank) == ("bm25", "#1") and score.startswith("bm25=")
    assert out[2].endswith("[Card]  terms: time, series, forecasting, seasonality")
    assert out[3].split()[0] == "2." and out[4].split()[:2] == ["bm25", "#2"]
    assert out[5].startswith("[router] confidence: ") and len(out) == 6


def test_cli_json_is_the_route_result_without_chunk_ids(monkeypatch, capsys):
    import json
    captured = _cli(monkeypatch, capsys, "time series forecasting seasonality", "--mode", "sparse",
                    "--no-rerank", "-k", "2", "--json")
    data = json.loads(captured.out)
    assert data["query"] == "time series forecasting seasonality" and data["mode"] == "sparse"
    assert data["results"][0]["slug"] == "time-series" and len(data["results"]) == 2
    assert data["confidence"]["level"] == "high" and "chunk_id" not in captured.out
    assert data["results"][0]["evidence"][0]["matched_terms"] == ["time", "series", "forecasting", "seasonality"]


def test_cli_ranking_line_reports_what_actually_ordered_the_result(monkeypatch, capsys):
    import sie.router as router_mod
    monkeypatch.setattr(router_mod, "Reranker", ReverseReranker)
    out = _cli(monkeypatch, capsys, "time series forecasting", "--mode", "sparse").out.splitlines()
    assert out[0] == "[router] mode=sparse ranking=cross-encoder"
    # a reranker is configured, but nothing was retrieved, so nothing was cross-encoded
    out = _cli(monkeypatch, capsys, "zzzz qqqq", "--mode", "sparse").out.splitlines()
    assert out == ["[router] mode=sparse ranking=rrf",
                   "[router] confidence: none (abstain): no candidate skills retrieved"]


def test_cli_rerank_k_limits_candidates(monkeypatch, capsys):
    import json
    import sie.router as router_mod
    monkeypatch.setattr(router_mod, "Reranker", ReverseReranker)
    data = json.loads(_cli(monkeypatch, capsys, "time series forecasting", "--mode", "sparse",
                           "--rerank-k", "1", "-k", "10", "--json").out)
    assert data["reranked"] is True and len(data["results"]) == 1
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, capsys, "time series", "--mode", "sparse", "--rerank-k", "0")
    assert "rerank_k must be >= 1" in str(exc.value)


def test_cli_manifest_errors_exit_cleanly(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, capsys, "time series", "--mode", "sparse", "--no-rerank",
             "--manifest", str(tmp_path / "missing.toml"))
    assert "cannot load the corpus" in str(exc.value)


def test_importing_new_engine_modules_loads_no_heavy_libraries():
    code = ("import sys, sie.router, sie.observability, sie.confidence, sie.index.sparse; "
            "heavy = {'chromadb', 'sentence_transformers', 'torch', 'onnxruntime'} & set(sys.modules); "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# --- backend knobs: Reranker(batch_size, max_length, load), DenseIndex(metadata, info) -------

def test_reranker_validates_knobs_and_exposes_load_state():
    from sie.rerank import Reranker
    for bad in ({"batch_size": 0}, {"batch_size": True}, {"max_length": 0}):
        with pytest.raises(ValueError):
            Reranker(model_name="stub", **bad)
    r = Reranker(model_name="stub", batch_size=8, max_length=256)
    assert (r.batch_size, r.max_length, r.loaded) == (8, 256, False)
    r._model = object()
    assert r.loaded is True
    r.load()                                     # already loaded: a no-op that imports nothing
    r._lazy()                                    # backward-compatible alias


def _fake_cross_encoder_module(created):
    import types

    class CrossEncoder:
        def __init__(self, name, **kwargs):
            created.append((name, kwargs))
            self.calls = []

        def predict(self, pairs, **kwargs):
            self.calls.append(kwargs)
            return [float(len(text)) for _, text in pairs]

    return types.SimpleNamespace(CrossEncoder=CrossEncoder)


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """A stub `sentence_transformers.CrossEncoder` that records how it was built and called."""
    import sie.rerank as rerank_mod
    created = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_cross_encoder_module(created))
    monkeypatch.setattr(rerank_mod, "unavailable_reason", lambda name: None)
    monkeypatch.setattr(rerank_mod, "cached_locally", lambda name: True)
    return created


def test_reranker_passes_max_length_and_batch_size_to_the_model(fake_cross_encoder):
    from sie.rerank import Reranker
    r = Reranker(model_name="local-ce", batch_size=4, max_length=128)
    r.load()
    assert fake_cross_encoder == [("local-ce", {"local_files_only": True, "max_length": 128})] and r.loaded
    hits = [Hit("a", 0.0, chunk_id="a", snippet="xx"), Hit("b", 0.0, chunk_id="b", snippet="x")]
    assert [h.skill_slug for h in r.rerank("q", hits)] == ["a", "b"]
    assert r._model.calls == [{"show_progress_bar": False, "batch_size": 4}]
    default = Reranker(model_name="local-ce")
    default.rerank("q", [Hit("a", 0.0, chunk_id="a", snippet="x")])
    assert fake_cross_encoder[-1] == ("local-ce", {"local_files_only": True})   # no max_length
    assert default._model.calls == [{"show_progress_bar": False}]               # library default


def test_reranker_load_is_thread_safe(fake_cross_encoder):
    import threading
    from sie.rerank import Reranker
    r = Reranker(model_name="local-ce")
    threads = [threading.Thread(target=r.load) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake_cross_encoder) == 1


class _FakeCollection:
    def __init__(self, name, configuration, metadata):
        self.name, self.configuration, self.metadata, self.n = name, configuration, metadata, 0

    def add(self, ids, embeddings, documents, metadatas):
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        self.batches = getattr(self, "batches", []) + [list(ids)]
        self.n += len(ids)

    def count(self):
        return self.n


class _FakeClient:
    def __init__(self, max_batch=None):
        self.created = []
        if max_batch is not None:
            self.get_max_batch_size = lambda: max_batch

    def create_collection(self, name, configuration=None, embedding_function=None, metadata=None):
        coll = _FakeCollection(name, configuration, metadata)
        self.created.append(coll)
        return coll


def test_dense_build_records_metadata_and_info_reports_it(monkeypatch, tmp_path):
    from sie.index.dense import DenseIndex
    from sie.models import Chunk
    d = DenseIndex(persist_dir=str(tmp_path), embedder="onnx", ef_search=123)
    client = _FakeClient()
    monkeypatch.setattr(d, "_embed", lambda texts: [[0.0, 1.0]] * len(texts))
    monkeypatch.setattr(d, "_lazy_client", lambda: client)
    monkeypatch.setattr(d, "_drop", lambda name: None)
    monkeypatch.setattr(d, "_swap_in", lambda staging: None)
    chunks = [Chunk("a", "Card", "alpha", "a::Card::0"), Chunk("b", "Card", "beta", "b::Card::0")]
    d.build(chunks, fingerprint="fp123", metadata={
        "n_chunks": 2, "exact": True, "ratio": 0.5, "corpus_name": "c", "tags": ["x", "y"],
        "missing": None, "embedder": "not-me", "fingerprint": "not-me"})
    (coll,) = client.created
    assert coll.metadata == {"n_chunks": 2, "exact": True, "ratio": 0.5, "corpus_name": "c",
                             "tags": "['x', 'y']", "embedder": "onnx", "fingerprint": "fp123"}
    assert coll.configuration == {"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 123}}
    assert d.info() == {**coll.metadata, "count": 2} and d.generation == 1
    monkeypatch.setattr(d, "_collection", lambda: None)
    assert d.info() is None


def test_dense_info_of_a_missing_index_creates_nothing(tmp_path):
    from sie.index.dense import DenseIndex
    d = DenseIndex(persist_dir=str(tmp_path / "never-built"), embedder="onnx")
    assert d.info() is None and not (tmp_path / "never-built").exists()   # no chromadb import


def test_dense_defaults_and_ef_search_validation():
    from sie.index.dense import DenseIndex
    d = DenseIndex(embedder="onnx")
    assert d.ef_search == 400 and d.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    for bad in (0, -5, True, 1.5):
        with pytest.raises(ValueError):
            DenseIndex(embedder="onnx", ef_search=bad)


def test_dense_build_splits_adds_at_the_clients_max_batch(monkeypatch, tmp_path):
    # one collection.add() is capped by chromadb (5461 rows on 1.5); larger corpora must batch
    from sie.index.dense import DenseIndex
    from sie.models import Chunk
    d = DenseIndex(persist_dir=str(tmp_path), embedder="onnx")
    client = _FakeClient(max_batch=3)
    monkeypatch.setattr(d, "_embed", lambda texts: [[float(i)] for i, _ in enumerate(texts)])
    monkeypatch.setattr(d, "_lazy_client", lambda: client)
    monkeypatch.setattr(d, "_drop", lambda name: None)
    monkeypatch.setattr(d, "_swap_in", lambda staging: None)
    chunks = [Chunk(f"s{i}", "Card", f"text {i}", f"s{i}::Card::0") for i in range(7)]
    d.build(chunks, fingerprint="fp")
    (coll,) = client.created
    assert [len(b) for b in coll.batches] == [3, 3, 1]
    assert [cid for b in coll.batches for cid in b] == [c.chunk_id for c in chunks] and coll.n == 7


def test_onnx_embedder_is_one_held_session(monkeypatch):
    # chromadb's DefaultEmbeddingFunction builds a new ONNX session per call; the index must
    # hold one model instance instead (no chromadb import: a stub module stands in for it)
    import sys
    import types
    from sie.index.dense import DenseIndex
    made = []

    class StubONNX:
        def __init__(self):
            made.append(self)

        def __call__(self, texts):
            return [[1.0, 0.0] for _ in texts]

    names = ["chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
             "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].ONNXMiniLM_L6_V2 = StubONNX
    d = DenseIndex(embedder="onnx")
    assert d._embed(["a"]) == [[1.0, 0.0]] and d._embed(["b", "c"]) == [[1.0, 0.0]] * 2
    assert len(made) == 1


def test_concurrent_cold_embeds_load_the_embedding_model_once(monkeypatch):
    import threading
    import time
    import types
    from sie.index.dense import DenseIndex
    made, calls = [], []

    class SlowONNX:
        def __init__(self):
            time.sleep(0.05)                         # widen the race window
            made.append(self)

        def __call__(self, texts):
            calls.append(list(texts))
            return [[1.0, 0.0] for _ in texts]

    names = ["chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
             "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].ONNXMiniLM_L6_V2 = SlowONNX
    d = DenseIndex(embedder="onnx")
    start, out = threading.Barrier(8), []

    def worker(i):
        start.wait()
        out.append(d._embed([f"q{i}"]))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and out == [[[1.0, 0.0]]] * 8
    assert calls[0] == ["warm up"] and calls.count(["warm up"]) == 1   # first use inside the lock


def test_an_embedder_whose_first_call_fails_is_not_kept(monkeypatch):
    import types
    from sie.index.dense import DenseIndex
    made = []

    class FlakyONNX:
        def __init__(self):
            made.append(self)
            self.first = len(made) == 1

        def __call__(self, texts):
            if self.first:
                raise OSError("model download interrupted")
            return [[0.5] for _ in texts]

    names = ["chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
             "chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].ONNXMiniLM_L6_V2 = FlakyONNX
    d = DenseIndex(embedder="onnx")
    with pytest.raises(RuntimeError, match="embedding with the onnx backend failed"):
        d._embed(["a"])
    assert d._embed_fn is None                      # a failed load is never published
    assert d._embed(["a"]) == [[0.5]] and len(made) == 2


# --- CLI: --persist-dir, missing optional dependencies ---------------------------------------

class RecordingDense:
    """Stand-in for sie.router.DenseIndex: records its persist_dir; builds are kept per dir."""
    embedder, model_name = "onnx", "sentence-transformers/all-MiniLM-L6-v2"
    built: dict = {}
    made: list = []

    def __init__(self, persist_dir="data/chroma", **kwargs):
        self.persist_dir, self.generation = persist_dir, 0
        RecordingDense.made.append(self)

    def build(self, chunks, fingerprint="", metadata=None):
        RecordingDense.built[self.persist_dir] = fingerprint
        self.generation += 1

    def fingerprint(self):
        return RecordingDense.built.get(self.persist_dir)

    def built_with(self):
        return "onnx"

    def info(self):
        return None

    def search(self, query, k=10):
        return DENSE[:k]


@pytest.fixture
def recording_dense(monkeypatch):
    import sie.router as router_mod
    monkeypatch.setattr(RecordingDense, "built", {})
    monkeypatch.setattr(RecordingDense, "made", [])
    monkeypatch.setattr(router_mod, "DenseIndex", RecordingDense)
    return RecordingDense


def test_cli_persist_dir_is_used_for_build_query_and_multi(monkeypatch, capsys, tmp_path, recording_dense):
    mine = str(tmp_path / "chroma-mine")
    out = _cli(monkeypatch, capsys, "--persist-dir", mine, "--no-rerank", "--build",
               "explain individual predictions with shap values").out.splitlines()
    assert out[0].startswith("[router] indexed ") and out[0].endswith(f" chunks -> {mine}/")
    assert out[1] == "[router] mode=hybrid ranking=rrf" and out[2].split()[:2] == ["1.", "explainability"]
    assert [d.persist_dir for d in recording_dense.made] == [mine]
    assert set(recording_dense.built) == {mine}
    out = _cli(monkeypatch, capsys, "--persist-dir", mine, "--no-rerank", "shap values").out
    assert "explainability" in out                                 # the index built there is found
    out = _cli(monkeypatch, capsys, "--persist-dir", mine, "--no-rerank", "--multi",
               "explain predictions with shap values").out
    assert "explainability" in out
    assert [d.persist_dir for d in recording_dense.made] == [mine] * 3
    with pytest.raises(SystemExit) as exc:                         # the default dir was never built
        _cli(monkeypatch, capsys, "--no-rerank", "shap values")
    assert "dense index not built" in str(exc.value)
    assert recording_dense.made[-1].persist_dir == "data/chroma"


def test_cli_missing_chromadb_exits_with_the_install_hint(monkeypatch, capsys):
    import sie.router as router_mod

    class NoChroma(RecordingDense):
        def fingerprint(self):
            raise ModuleNotFoundError("No module named 'chromadb'", name="chromadb")

    monkeypatch.setattr(router_mod, "DenseIndex", NoChroma)
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, capsys, "--no-rerank", "evaluate my RAG answers")
    message = str(exc.value)
    assert message.startswith("[router] missing dependency: No module named 'chromadb'")
    assert "'retrieval' extra" in message and "SIE_MODE=sparse" in message
