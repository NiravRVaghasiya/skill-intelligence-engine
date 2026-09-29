"""Every endpoint via TestClient over a real Engine with the dense index / reranker faked.

No index, no models: BM25 runs over the vendored corpus, dense hits are scripted.
"""
import json
import threading

import pytest
from fastapi.testclient import TestClient

import sie
from sie import api as api_mod
from sie.engine import Engine
from tests.test_engine import DENSE, ROOT, FakeReranker, make_engine, write_skill

RAG = "evaluate my rag answers"


def _client(monkeypatch, engine, **kw) -> TestClient:
    monkeypatch.setattr(api_mod, "get_engine", lambda: engine)
    c = TestClient(api_mod.api, **kw)
    c.engine = engine
    return c


@pytest.fixture
def client(monkeypatch):
    """Hybrid engine: real BM25, scripted dense hits, no reranker."""
    return _client(monkeypatch, make_engine())


@pytest.fixture
def sparse(monkeypatch):
    return _client(monkeypatch, make_engine(mode="sparse"))


def _no_chunk_ids(response, engine: Engine) -> None:
    assert "chunk_id" not in response.text
    assert not any(cid in response.text for cid in engine.router._texts)


# -- /health ----------------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    fingerprint = body["corpus"].pop("fingerprint")
    assert body == {"status": "ok", "version": sie.__version__, "skills": 38,
                    "corpus": {"name": "ml-ai-skills", "version": "8328c60"}}
    assert len(fingerprint) == 16
    assert client.engine.router.dense.searches == []          # liveness never embeds anything


def test_health_is_degraded_when_the_reranker_fell_back(client):
    client.engine.router.rerank_error = "weights unavailable"
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded"


# -- /ready -----------------------------------------------------------------------------------

def test_ready_starts_a_never_started_engine_once(sparse):
    engine, starts = sparse.engine, []
    real = engine._start_locked
    engine._start_locked = lambda eager: starts.append(eager) or real(eager)
    r = sparse.get("/ready")
    assert r.status_code == 200 and starts == [True]
    body = r.json()
    assert body["ready"] is True and body["reasons"] == [] and body["degraded"] is False
    assert body["version"] == sie.__version__ and body["started_at"]
    assert body["corpus"]["name"] == "ml-ai-skills" and "root" not in body["corpus"]
    assert body["graph"]["nodes"] == 38 and body["router"]["mode"] == "sparse"
    assert sparse.get("/ready").status_code == 200 and starts == [True]


def test_ready_is_503_with_reasons_when_the_dense_index_is_missing(monkeypatch):
    c = _client(monkeypatch, make_engine(fingerprint=None))
    r = c.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False and "dense index not built" in body["errors"][0]
    assert body["reasons"][0] == body["errors"][0] and body["corpus"]["n_skills"] == 38


def test_ready_reports_a_reranker_fallback_as_degraded(monkeypatch):
    c = _client(monkeypatch, make_engine(reranker=FakeReranker(fail="weights unavailable")))
    body = c.get("/ready").json()
    assert body["ready"] is True and body["degraded"] is True
    assert body["router"]["reranker"]["error"] == "weights unavailable"


def test_ready_strict_reranker_failure_is_503(monkeypatch):
    engine = make_engine(reranker=FakeReranker(fail="weights unavailable"), strict_rerank=True)
    r = _client(monkeypatch, engine).get("/ready")
    assert r.status_code == 503 and "RerankerUnavailable" in r.json()["errors"][0]


def test_ready_recovers_after_an_external_index_rebuild(client):
    assert client.get("/ready").status_code == 200
    client.engine.router.dense.generation += 1             # the next search rebound its handle
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["reasons"] == []


def test_ready_says_stale_after_an_external_rebuild_of_another_corpus(client):
    assert client.get("/ready").status_code == 200
    dense = client.engine.router.dense
    dense._fp, dense.generation = "other-corpus", dense.generation + 1
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["reasons"][0].startswith("dense index is stale")


def test_invalid_configuration_is_503_everywhere(monkeypatch):
    def broken():
        raise RuntimeError("invalid engine configuration: SIE_MODE must be one of ...")

    monkeypatch.setattr(api_mod, "get_engine", broken)
    c = TestClient(api_mod.api)
    r = c.get("/ready")
    assert r.status_code == 503 and r.json()["ready"] is False and "SIE_MODE" in r.json()["reasons"][0]
    assert c.get("/health").status_code == 503
    assert c.get("/search", params={"q": "x"}).status_code == 503
    assert c.post("/route", json={"query": "x"}).status_code == 503
    assert c.get("/skill/rag-evaluation").status_code == 503


# -- lifespan / get_engine ------------------------------------------------------------------------

def test_lifespan_starts_the_engine_eagerly(monkeypatch):
    engine, starts = make_engine(mode="sparse"), []
    real = engine.start
    engine.start = lambda eager=True: starts.append(eager) or real(eager)
    monkeypatch.setattr(api_mod, "configure_logging", lambda level=None: None)
    monkeypatch.delenv("SIE_EAGER_INIT", raising=False)
    monkeypatch.setattr(api_mod, "get_engine", lambda: engine)
    with TestClient(api_mod.api) as c:
        assert starts == [True] and engine.started_at is not None
        assert c.get("/ready").status_code == 200 and starts == [True]


def test_lifespan_defers_start_when_eager_init_is_off(monkeypatch):
    engine, starts = make_engine(mode="sparse"), []
    real = engine._start_locked
    engine._start_locked = lambda eager: starts.append(eager) or real(eager)
    monkeypatch.setattr(api_mod, "configure_logging", lambda level=None: None)
    monkeypatch.setenv("SIE_EAGER_INIT", "0")
    monkeypatch.setattr(api_mod, "get_engine", lambda: engine)
    with TestClient(api_mod.api) as c:
        assert starts == [] and engine.started_at is None
        assert c.get("/ready").status_code == 200 and starts == [True]   # the first /ready starts it


@pytest.mark.parametrize("value", ["false", "off", "no", "FALSE", " 0 "])
def test_lifespan_defers_start_for_every_off_spelling(monkeypatch, value):
    engine, starts = make_engine(mode="sparse"), []
    real = engine._start_locked
    engine._start_locked = lambda eager: starts.append(eager) or real(eager)
    monkeypatch.setattr(api_mod, "configure_logging", lambda level=None: None)
    monkeypatch.setenv("SIE_EAGER_INIT", value)
    monkeypatch.setattr(api_mod, "get_engine", lambda: engine)
    with TestClient(api_mod.api) as c:
        assert starts == [] and engine.started_at is None
        assert c.get("/ready").status_code == 200 and starts == [True]


def test_lifespan_invalid_eager_init_warms_up_logs_and_is_a_ready_reason(monkeypatch, caplog):
    engine, starts = make_engine(mode="sparse"), []
    real = engine.start
    engine.start = lambda eager=True: starts.append(eager) or real(eager)
    monkeypatch.setattr(api_mod, "configure_logging", lambda level=None: None)
    monkeypatch.setenv("SIE_EAGER_INIT", "maybe")
    monkeypatch.setattr(api_mod, "get_engine", lambda: engine)
    with caplog.at_level("ERROR", logger="sie.api"), TestClient(api_mod.api) as c:
        assert starts == [True]                                   # the default: eager start
        r = c.get("/ready")
        assert c.get("/health").status_code == 200               # everything else still serves
    assert any("SIE_EAGER_INIT" in rec.getMessage() and "'maybe'" in rec.getMessage()
               for rec in caplog.records)
    body = r.json()
    assert r.status_code == 503 and body["ready"] is False and body["errors"] == []
    assert body["reasons"] == ["invalid configuration: SIE_EAGER_INIT must be 1 or 0 (or true/false, "
                               "yes/no, on/off), got 'maybe'; the default (eager start) applies"]


def test_lifespan_survives_a_failing_start(monkeypatch, tmp_path):
    engine = Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False)   # empty corpus
    monkeypatch.setattr(api_mod, "configure_logging", lambda level=None: None)
    monkeypatch.delenv("SIE_EAGER_INIT", raising=False)
    monkeypatch.setattr(api_mod, "get_engine", lambda: engine)
    with TestClient(api_mod.api) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "no skills found" in r.json()["errors"][0]


def test_get_engine_defaults_are_cwd_independent(monkeypatch, tmp_path):
    for name in ("SIE_SKILLS_DIR", "SIE_PERSIST_DIR", "SIE_MODE", "SIE_RERANK", "SIE_EDGE_OVERLAYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_mod, "_engine", None)
    engine = api_mod.get_engine()
    assert engine is api_mod.get_engine()                     # one engine per process
    assert engine.router.skills_dir == str(ROOT / "data" / "skills")
    assert engine.router.dense.persist_dir == str(ROOT / "data" / "chroma")


def test_get_engine_invalid_env_is_a_runtime_error_and_not_cached(monkeypatch):
    monkeypatch.setattr(api_mod, "_engine", None)
    monkeypatch.setenv("SIE_MODE", "fast")
    with pytest.raises(RuntimeError, match="SIE_MODE"):
        api_mod.get_engine()
    monkeypatch.setenv("SIE_MODE", "sparse")
    assert api_mod.get_engine().mode == "sparse"


# -- /search (backward compatible) --------------------------------------------------------------

def test_search_returns_ranked_results(client):
    r = client.get("/search", params={"q": RAG, "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"query", "reranked", "pool", "results", "confidence"}
    assert body["query"] == RAG and body["reranked"] is False and body["pool"] == 20
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert set(first) == {"slug", "score", "section", "rank", "name", "retrieval_methods"}
    assert first["rank"] == 1 and first["score"] == round(first["score"], 4)
    assert first["retrieval_methods"] and set(first["retrieval_methods"]) <= {"dense", "bm25"}
    assert set(body["confidence"]) == {"level", "action", "ambiguous"}
    assert client.engine.router.dense.searches == [(RAG, 20)]
    expected = client.engine.route(RAG, k=2)
    assert [h["slug"] for h in body["results"]] == [s.slug for s in expected.results]
    assert first["name"] == client.engine.skill_obj(first["slug"]).name
    _no_chunk_ids(r, client.engine)


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 999},
                                    {"q": "x" * 4001}, {"q": "x", "rerank_k": 0}])
def test_search_validates_params(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_search_passes_pool_and_rerank_k(client):
    body = client.get("/search", params={"q": "x", "k": 2, "pool": 40}).json()
    assert body["pool"] == 40 and client.engine.router.dense.searches[-1] == ("x", 40)
    assert client.get("/search", params={"q": "x", "pool": 0}).status_code == 422
    assert client.get("/search", params={"q": "x", "rerank_k": 3}).status_code == 200


def test_search_unbuilt_index_is_503(monkeypatch):
    r = _client(monkeypatch, make_engine(fingerprint=None)).get("/search", params={"q": "x"})
    assert r.status_code == 503 and "not built" in r.json()["detail"]


# -- POST /route --------------------------------------------------------------------------------

def test_route_explained(client):
    r = client.post("/route", json={"query": RAG})
    assert r.status_code == 200
    body = r.json()
    result = client.engine.route(RAG)
    assert body["query"] == RAG and body["mode"] == "hybrid" and body["reranked"] is False
    assert [s["skill_id"] for s in body["results"]] == [s.slug for s in result.results]
    assert [s["rank"] for s in body["results"]] == list(range(1, len(body["results"]) + 1))
    conf = body["confidence"]
    assert conf["calibrated"] is False and conf["level"] == result.confidence.level
    assert conf["reasons"] == result.confidence.reasons and "leaders" in conf["signals"]
    top = body["results"][0]
    assert top["score_type"] == "rrf" and isinstance(top["snippet"], str)
    assert {e["method"] for e in top["evidence"]} == set(top["retrieval_methods"])
    bm25 = [e for e in top["evidence"] if e["method"] == "bm25"]
    assert all(isinstance(e["matched_terms"], list) for e in bm25)
    assert top["provenance"]["provenance"].startswith("ml-ai-skills@8328c60:")
    assert set(body["candidates"]) >= {"dense", "bm25", "fused"} and "total" in body["timings_ms"]
    assert body["corpus"]["name"] == "ml-ai-skills" and body["composition"] is None
    _no_chunk_ids(r, client.engine)


def test_route_prerequisites_and_alternatives(client):
    body = client.post("/route", json={"query": RAG, "k": 10}).json()
    by_id = {s["skill_id"]: s for s in body["results"]}
    assert by_id["rag-evaluation"]["prerequisites"] == ["rag-pipeline"]
    assert by_id["rag-pipeline"]["prerequisites"] == []
    alts = body["alternatives"]
    if body["confidence"]["competitors"]:
        assert [a["skill_id"] for a in alts] == body["confidence"]["competitors"]
    else:
        assert [a["skill_id"] for a in alts] == [s["skill_id"] for s in body["results"][1:3]]
        assert all(a["reason"].startswith("next best by score") for a in alts)
    assert all(a["name"] and a["reason"] for a in alts)


def test_route_alternatives_name_the_competing_retriever(monkeypatch):
    # dense ranks rag-evaluation first, BM25 ranks time-series first: the two leaders disagree
    engine = make_engine(dense_hits=[DENSE[0]])
    body = _client(monkeypatch, engine).post("/route", json={"query": "time series forecasting"}).json()
    conf = body["confidence"]
    assert conf["ambiguous"] is True and conf["competitors"]
    first = body["alternatives"][0]
    assert first["skill_id"] == conf["competitors"][0]
    assert first["reason"].startswith(("ranked first by", "score tie", "near score tie", "score (near-)tie"))


def test_route_without_explain_drops_evidence_and_snippets(client):
    body = client.post("/route", json={"query": RAG, "explain": False}).json()
    assert body["results"] and all(s["evidence"] == [] and s["snippet"] is None for s in body["results"])
    assert all(s["retrieval_methods"] for s in body["results"])


def test_route_multi_intent_adds_a_composition(sparse):
    q = "Build a RAG pipeline, evaluate it, deploy it, and monitor it"
    r = sparse.post("/route", json={"query": q, "multi_intent": True, "k": 3})
    assert r.status_code == 200
    body = r.json()
    comp = body["composition"]
    assert body["results"] and body["confidence"]["level"]              # the whole query, too
    assert comp["multi_intent"] is True and len(comp["intents"]) == 4
    assert comp["intents"][1]["query"] == "evaluate a RAG pipeline"
    plan = [s["skill_id"] for s in comp["plan"]]
    assert [s["position"] for s in comp["plan"]] == list(range(1, len(plan) + 1))
    if "rag-pipeline" in plan and "rag-evaluation" in plan:
        assert plan.index("rag-pipeline") < plan.index("rag-evaluation")
    expected = sparse.engine.compose(q, k=3)
    assert plan == [s.slug for s in expected.steps] and comp["notes"] == expected.notes
    assert all(set(e) == {"source", "target", "relationship", "confidence", "provenance"}
               for e in comp["edges"])
    _no_chunk_ids(r, sparse.engine)


def test_route_max_intents_truncates(sparse):
    q = "Build a RAG pipeline, evaluate it, deploy it, and monitor it"
    comp = sparse.post("/route", json={"query": q, "multi_intent": True, "max_intents": 2}).json()["composition"]
    assert len(comp["intents"]) == 2 and "2 more intents were not routed (max_intents=2)" in comp["notes"]


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "   "}, {"query": "x" * 4001}, {"query": "x", "k": 0},
    {"query": "x", "k": 51}, {"query": "x", "pool": 501}, {"query": "x", "rerank_k": 0},
    {"query": "x", "max_intents": 11}, {"query": "x", "top_k": 3}, {}, {"query": 5}])
def test_route_validates_the_request(client, payload):
    assert client.post("/route", json=payload).status_code == 422


def test_route_error_mapping(monkeypatch, client):
    def raise_(error):
        def fn(*args, **kwargs):
            raise error
        return fn

    engine = client.engine
    monkeypatch.setattr(engine, "route", raise_(RuntimeError("dense index is stale")))
    r = client.post("/route", json={"query": "x"})
    assert r.status_code == 503 and r.json()["detail"] == "dense index is stale"
    monkeypatch.setattr(engine, "route", raise_(ValueError("pool must be >= 1")))
    assert client.post("/route", json={"query": "x"}).status_code == 422
    monkeypatch.setattr(engine, "route", raise_(KeyError("bug")))        # a bug is not a 404
    lenient = TestClient(api_mod.api, raise_server_exceptions=False)
    assert lenient.post("/route", json={"query": "x"}).status_code == 500


def test_route_on_an_unbuilt_index_is_503(monkeypatch):
    r = _client(monkeypatch, make_engine(fingerprint=None)).post("/route", json={"query": "x"})
    assert r.status_code == 503 and "not built" in r.json()["detail"]


def test_multi_intent_cycle_is_409(monkeypatch, tmp_path):
    write_skill(tmp_path, "alpha", "requires:\n  - beta", "configure alpha widgets carefully")
    write_skill(tmp_path, "beta", "requires:\n  - alpha", "tune beta gadgets")
    write_skill(tmp_path, "gamma", "", "unrelated gardening notes")
    c = _client(monkeypatch, Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False))
    assert c.post("/route", json={"query": "configure alpha widgets"}).status_code == 200
    r = c.post("/route", json={"query": "configure alpha widgets", "multi_intent": True})
    assert r.status_code == 409 and "requires cycle" in r.json()["detail"]


# -- POST /batch-route --------------------------------------------------------------------------

def test_batch_route_in_order(client):
    queries = [RAG, "time series forecasting"]
    r = client.post("/batch-route", json={"queries": queries, "k": 2})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["query"] for x in results] == queries
    single = client.post("/route", json={"query": queries[1], "k": 2, "explain": False}).json()
    assert [s["skill_id"] for s in results[1]["results"]] == [s["skill_id"] for s in single["results"]]
    assert all(s["evidence"] == [] and s["snippet"] is None for x in results for s in x["results"])
    _no_chunk_ids(r, client.engine)


def test_batch_route_limits(sparse):
    assert sparse.post("/batch-route", json={"queries": ["time series"] * 32}).status_code == 200
    for payload in ({"queries": ["time series"] * 33}, {"queries": []}, {"queries": ["ok", "  "]},
                    {"queries": ["x" * 4001]}, {"queries": "time series"}, {"queries": ["x"], "k": 0}):
        assert sparse.post("/batch-route", json=payload).status_code == 422, payload


def test_batch_route_explain(sparse):
    body = sparse.post("/batch-route", json={"queries": ["time series forecasting"], "explain": True}).json()
    assert body["results"][0]["results"][0]["evidence"]


# -- /learning-path, /skill ---------------------------------------------------------------------

def test_learning_path(client):
    r = client.get("/learning-path", params={"target": "rag-evaluation"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == ["rag-pipeline", "rag-evaluation"] and "llm-evaluation" in body["related"]
    assert body == client.engine.learning_path("rag-evaluation")          # every key, unchanged
    assert body["steps"][0]["relation"] == "direct" and body["direct"] == ["rag-pipeline"]
    assert body["complete"] is True
    r = client.get("/learning-path", params={"target": "rag-evaluation", "include_recommended": True})
    assert r.status_code == 200 and r.json()["path"] == body["path"]
    assert client.engine.router.dense.searches == []          # graph queries never touch the index


def test_learning_path_unknown_target_is_404(client):
    r = client.get("/learning-path", params={"target": "nope"})
    assert r.status_code == 404 and "unknown skill" in r.json()["detail"]
    assert client.get("/learning-path").status_code == 422


def test_learning_path_cycle_is_409(monkeypatch, tmp_path):
    write_skill(tmp_path, "a", "requires:\n  - b")
    write_skill(tmp_path, "b", "requires:\n  - a")
    c = _client(monkeypatch, Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False))
    r = c.get("/learning-path", params={"target": "a"})
    assert r.status_code == 409 and "requires cycle" in r.json()["detail"]


def test_skill_detail(client):
    body = client.get("/skill/rag-evaluation").json()
    assert body["slug"] == body["skill_id"] == "rag-evaluation" and body["domain"] == "llm"
    assert body["requires"] == ["rag-pipeline"] and body["required_by"] == []
    assert body["description"].startswith("Use when") and body["name"] == "RAG Evaluation"
    assert body["provenance"]["path"] == "rag-evaluation/SKILL.md"
    assert body["metadata"]["lifecycle"] and "requires" in body["declared"]
    assert any(e["relationship"] == "requires" and e["source"] == "rag-pipeline" for e in body["relations"])
    assert client.get("/skill/rag-pipeline").json()["required_by"] == ["rag-evaluation"]


def test_skill_unknown_is_404(client):
    r = client.get("/skill/nope")
    assert r.status_code == 404 and "unknown skill 'nope'" in r.json()["detail"]


@pytest.fixture
def corpus_dir(monkeypatch, tmp_path):
    """The API over a scratch corpus (BM25 only)."""
    engine = Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False)
    c = _client(monkeypatch, engine)
    c.root = tmp_path
    return c


def test_missing_corpus_is_503_and_not_cached(corpus_dir):
    c = corpus_dir
    r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded" and r.json()["skills"] == 0
    assert "no skills found" in r.json()["detail"]
    assert str(c.root) in r.json()["detail"]                   # outside the repo: as given
    assert c.get("/learning-path", params={"target": "a"}).status_code == 503
    assert c.get("/skill/a").status_code == 503
    assert c.post("/route", json={"query": "hi"}).status_code == 503
    write_skill(c.root, "a")                                   # corpus appears: no restart needed
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["skills"] == 1 and body["version"] == sie.__version__


def test_skill_relationships_match_learning_path(corpus_dir):
    c = corpus_dir
    write_skill(c.root, "t", "related:\n  - x\n  - y\n  - y\n  - nobody\nrequires:\n  - ghost")
    write_skill(c.root, "x", "conflicts:\n  - t")
    write_skill(c.root, "y", "")
    body = c.get("/skill/t").json()
    assert body["conflicts"] == ["x"] and body["related"] == ["y"]
    assert body["dangling"] == [["related", "nobody"], ["requires", "ghost"]]
    lp = c.get("/learning-path", params={"target": "t"}).json()
    assert lp["conflicts"] == body["conflicts"] and lp["related"] == body["related"]
    assert lp["missing_prerequisites"] == [["t", "ghost"]] and lp["complete"] is False


# -- no machine paths in response bodies ------------------------------------------------------

def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _no_repo_paths(response) -> None:
    roots = (str(ROOT).lower(), ROOT.as_posix().lower())
    leaked = [s for s in _strings(response.json()) if any(r in s.lower() for r in roots)]
    assert not leaked, leaked


def test_repo_corpus_paths_are_repo_relative_in_error_bodies(monkeypatch):
    missing = ROOT / "data" / "no-such-corpus"                 # never created
    assert not missing.exists()
    c = _client(monkeypatch, Engine(skills_dir=missing, mode="sparse", use_reranker=False))
    health, route, ready = c.get("/health"), c.post("/route", json={"query": "x"}), c.get("/ready")
    assert health.status_code == route.status_code == ready.status_code == 503
    assert "data/no-such-corpus" in health.json()["detail"]
    assert route.json()["detail"] == health.json()["detail"]
    body = ready.json()
    assert body["errors"] and "data/no-such-corpus" in body["errors"][0]
    assert body["reasons"][0] == body["errors"][0]
    for r in (health, route, ready):
        _no_repo_paths(r)
    assert str(missing) in c.engine.errors[0]                  # the server side keeps the full path


def test_repo_overlay_paths_are_repo_relative_in_error_bodies(monkeypatch):
    overlay = ROOT / "docs" / "no-such.edges.json"
    assert not overlay.exists()
    c = _client(monkeypatch, make_engine(mode="sparse", overlays=[overlay]))
    ready, skill = c.get("/ready"), c.get("/skill/rag-evaluation")
    assert ready.status_code == skill.status_code == 503
    assert ready.json()["errors"][0].startswith(
        "load failed: RuntimeError: cannot load edge overlay docs/no-such.edges.json")
    assert skill.json()["detail"].startswith("cannot load edge overlay docs/no-such.edges.json")
    _no_repo_paths(ready)
    _no_repo_paths(skill)


def test_repo_model_paths_are_repo_relative_in_ready_and_route_bodies(monkeypatch):
    local = ROOT / "data" / "models" / "ms-marco-MiniLM-L-6-v2"
    reranker = FakeReranker(fail=f"cannot load cross-encoder '{local}' (weights incomplete)")
    reranker.model_name = str(local)
    c = _client(monkeypatch, make_engine(reranker=reranker))
    ready = c.get("/ready")
    assert ready.status_code == 200 and ready.json()["degraded"] is True
    shown = "cannot load cross-encoder 'data/models/ms-marco-MiniLM-L-6-v2' (weights incomplete)"
    assert ready.json()["router"]["reranker"]["error"] == shown
    route = c.post("/route", json={"query": RAG})
    assert route.status_code == 200 and route.json()["rerank_error"] == shown
    _no_repo_paths(ready)
    _no_repo_paths(route)


# -- contract ---------------------------------------------------------------------------------

def test_plain_converts_numpy_scalars_to_builtins():
    import numpy as np
    from sie.schemas import plain
    out = plain({"a": np.float64(1.5), "b": [np.int64(2), (np.bool_(True), None)], 3: "x"})
    assert out == {"a": 1.5, "b": [2, [True, None]], "3": "x"}
    assert type(out["a"]) is float and type(out["b"][0]) is int and type(out["b"][1][0]) is bool
    json.dumps(out)


def test_openapi_schema_documents_every_endpoint(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["version"] == sie.__version__
    paths = spec["paths"]
    for path, method in [("/health", "get"), ("/ready", "get"), ("/search", "get"), ("/route", "post"),
                         ("/batch-route", "post"), ("/learning-path", "get"), ("/skill/{slug}", "get")]:
        op = paths[path][method]
        assert op["summary"] and op["description"], path
    assert {"404", "409", "503"} <= set(paths["/learning-path"]["get"]["responses"])
    assert "503" in paths["/ready"]["get"]["responses"]
    schemas = spec["components"]["schemas"]
    assert {"RouteRequest", "RouteResponse", "BatchRouteRequest", "LearningPathResponse",
            "SkillDetail", "ConfidenceOut", "CompositionOut"} <= set(schemas)
    assert "chunk_id" not in json.dumps(schemas)


def test_concurrent_requests_agree(sparse):
    expected = sparse.post("/route", json={"query": "time series forecasting", "explain": False}).json()
    expected.pop("timings_ms")
    seen, errors = [], []

    def work():
        try:
            for _ in range(5):
                body = sparse.post("/route", json={"query": "time series forecasting",
                                                   "explain": False}).json()
                body.pop("timings_ms")
                seen.append(body)
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(seen) == 30 and all(b == expected for b in seen)
