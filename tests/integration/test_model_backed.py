"""Model-backed checks against the real persisted index (opt-in; CI never runs them).

    SIE_INTEGRATION=1 pytest tests/integration -q

Needs a built `data/chroma/` index and the embedding model already cached locally: the network
stays blocked (tests/conftest.py), so nothing is downloaded. With SIE_INTEGRATION=1 the conftest
heavy-import blocker is off and only tests marked `integration` run.

What they pin down: the live dense index matches the corpus; live hybrid / dense metrics on the
routing sets equal benchmarks/metrics.json; the dense fixture that `--smoke` replays equals what
the live index returns; and, only where cross-encoder weights load, the reranker really reranks.
"""
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from eval import run_eval
from eval.replay import FIXTURE, load_fixture
from sie.rerank import Reranker, RerankerUnavailable
from sie.router import HybridRouter, corpus_fingerprint

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("SIE_INTEGRATION") != "1",
                                 reason="model-backed: opt in with SIE_INTEGRATION=1")]

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = 10


@pytest.fixture(scope="module")
def repo_cwd():
    old = os.getcwd()
    os.chdir(ROOT)
    yield ROOT
    os.chdir(old)


@pytest.fixture(scope="module")
def recorded():
    return json.loads((ROOT / "benchmarks" / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_router(repo_cwd):
    router = HybridRouter(use_reranker=False)
    router._ensure_ready()          # raises if the index is missing or stale
    return router


def test_live_dense_index_is_fresh(live_router):
    status = live_router.status()
    assert status["dense"]["ready"] is True
    assert live_router.dense.fingerprint() == corpus_fingerprint(live_router._load_chunks())


def test_live_routing_metrics_equal_metrics_json(repo_cwd, recorded):
    ctx = run_eval.load_context()
    routing = replace(ctx, datasets=ctx.of_kind("routing"))
    config = recorded["config"]
    cfg = run_eval.EvalConfig(pool=config["pool"], rerank_k=config["rerank_k"])
    systems = [s for s in run_eval.SYSTEMS if s.key in ("dense", "hybrid")]
    current = run_eval.metrics_document(routing, run_eval.evaluate(systems, routing.datasets, cfg), cfg)
    names = {ds.name for ds in routing.datasets}
    subset = {**recorded, "datasets": {k: v for k, v in recorded["datasets"].items() if k in names}}
    diffs, n = run_eval.compare_metrics(subset, current, {"dense": "dense", "hybrid": "hybrid"})
    assert n > 100
    assert not diffs, "\n".join(f"{m} on {d} ({s}): metrics.json {a}, live {b}" for d, s, m, a, b in diffs)


def test_fixture_rankings_equal_live_dense_rankings(live_router):
    fixture = load_fixture(ROOT / FIXTURE)
    queries = sorted(fixture["queries"])
    sample = queries[::max(1, len(queries) // SAMPLE)][:SAMPLE]
    for q in sample:
        live = live_router.dense.search(q, k=fixture["depth"])
        stored = fixture["queries"][q]
        assert [h.chunk_id for h in live] == [cid for cid, _ in stored], q
        assert [h.score for h in live] == pytest.approx([s for _, s in stored], abs=1e-5), q


def test_reranker_reranks_when_weights_load(repo_cwd):
    try:
        Reranker().load()
    except RerankerUnavailable as e:
        pytest.skip(f"cross-encoder weights unavailable: {e}")
    router = HybridRouter(use_reranker=True, strict_rerank=True)
    result = router.route("evaluate the faithfulness of my RAG answers", k=3)
    assert result.reranked and result.rerank_error is None
    assert all(r.score_type == "cross-encoder" for r in result.results)
    assert result.candidates["rerank"] > 0
