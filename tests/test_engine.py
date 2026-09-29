"""sie.engine.Engine: env config, one corpus load, start/status readiness, overlays, passthrough.

No models: the dense index and reranker are fakes, or the engine runs in sparse (BM25) mode.
Also covers `python -m sie.router --multi / --overlay`, packaging metadata and the demo app.
"""
import json
import os
import runpy
import sys
import threading
import tomllib
import types
from dataclasses import replace
from pathlib import Path

import pytest

import sie
from sie import engine as engine_mod
from sie import observability as obs
from sie import router as router_mod
from sie.compose import compose
from sie.engine import (DEFAULT_PERSIST_DIR, DEFAULT_SKILLS_DIR, CycleError, Engine,
                        UnknownSkillError, composition_dict)
from sie.graph.paths import learning_path
from sie.models import RELATIONSHIPS, Hit
from sie.rerank import RerankerUnavailable
from sie.router import RetrievalConfig, corpus_fingerprint, route_result_dict

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"
OVERLAY = ROOT / "docs" / "proposed_requires.edges.json"
SIE_VARS = ("SIE_SKILLS_DIR", "SIE_PERSIST_DIR", "SIE_CORPUS_MANIFEST", "SIE_MODE", "SIE_RERANK",
            "SIE_RERANKER", "SIE_STRICT_RERANK", "SIE_EDGE_OVERLAYS", "SIE_POOL", "SIE_RERANK_K",
            "SIE_TOP_K", "SIE_EMBEDDER", "SIE_EAGER_INIT")


# -- fakes ------------------------------------------------------------------------------------

class FakeDense:
    """Dense index stand-in: fixed hits and a recorded fingerprint; records search k values."""
    embedder = "onnx"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    persist_dir = "unused"

    def __init__(self, hits, fingerprint):
        self.hits, self._fp, self.generation, self.searches = hits, fingerprint, 0, []

    def fingerprint(self):
        return self._fp

    def built_with(self):
        return "onnx"

    def info(self):
        return None if self._fp is None else {"embedder": "onnx", "fingerprint": self._fp,
                                              "count": len(self.hits)}

    def search(self, query, k=10):
        self.searches.append((query, k))
        return self.hits[:k]


class FakeReranker:
    model_name = "fake-ce"

    def __init__(self, fail=None):
        self.fail, self.loaded = fail, False

    def load(self):
        if self.fail:
            raise RerankerUnavailable(self.fail)
        self.loaded = True

    def rerank(self, query, hits, top_k=None, texts=None):
        self.load()
        out = sorted((replace(h, score=float(len(h.chunk_id))) for h in hits),
                     key=lambda h: (-h.score, h.chunk_id))
        return out if top_k is None else out[:top_k]


def _hit(slug, section="Card", j=0, score=0.5):
    return Hit(skill_slug=slug, score=score, section=section, chunk_id=f"{slug}::{section}::{j}")


DENSE = [_hit("rag-evaluation", score=0.62), _hit("llm-evaluation", score=0.55),
         _hit("rag-pipeline", score=0.51)]


def make_engine(mode="hybrid", dense_hits=DENSE, fingerprint="auto", reranker=None,
                skills_dir=CORPUS, **kw) -> Engine:
    """An Engine over a corpus with the dense index / reranker replaced by fakes."""
    e = Engine(skills_dir=skills_dir, persist_dir="unused-chroma", mode=mode, use_reranker=False, **kw)
    if mode != "sparse":
        fp = corpus_fingerprint(e.router._load_chunks()) if fingerprint == "auto" else fingerprint
        e.router.dense = FakeDense(dense_hits, fp)
    e.router.reranker = reranker
    return e


def write_skill(root: Path, slug: str, fm: str = "", body: str = "hi") -> None:
    (root / slug).mkdir(parents=True, exist_ok=True)
    (root / slug / "SKILL.md").write_text(
        f"---\ntype: workflow\ndomain: d\nlevel: intermediate\n{fm}\n---\n## Overview\n{body}\n",
        encoding="utf-8")


def _stable(result) -> dict:
    data = route_result_dict(result)
    data.pop("timings_ms")
    return data


def _stable_composition(c) -> dict:
    data = composition_dict(c)
    for it in data["intents"]:
        it["result"].pop("timings_ms")
    return data


@pytest.fixture
def clean_env(monkeypatch):
    for name in SIE_VARS:
        monkeypatch.delenv(name, raising=False)


# -- from_env ---------------------------------------------------------------------------------

def test_from_env_defaults_are_repo_paths_independent_of_cwd(clean_env, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    e = Engine.from_env()
    assert e.router.skills_dir == str(ROOT / "data" / "skills") == str(DEFAULT_SKILLS_DIR)
    assert e.router.dense.persist_dir == str(ROOT / "data" / "chroma") == str(DEFAULT_PERSIST_DIR)
    assert e.mode == "hybrid" and e.router.reranker is not None and e.strict_rerank is False
    assert e.overlays == () and e.router.manifest is None and e.config == RetrievalConfig()
    assert e.started_at is None and e.errors == []


def test_from_env_reads_every_variable(tmp_path):
    sep = os.pathsep
    env = {"SIE_SKILLS_DIR": str(tmp_path / "skills"), "SIE_PERSIST_DIR": str(tmp_path / "db"),
           "SIE_CORPUS_MANIFEST": str(tmp_path / "corpus.toml"), "SIE_MODE": " Sparse ",
           "SIE_RERANK": "1", "SIE_RERANKER": "my/cross-encoder", "SIE_STRICT_RERANK": "1",
           "SIE_EDGE_OVERLAYS": f"a.json{sep}{sep} b.json ", "SIE_POOL": "40", "SIE_RERANK_K": "7",
           "SIE_TOP_K": "3", "SIE_EMBEDDER": "sentence-transformers"}
    e = Engine.from_env(env)
    assert e.router.skills_dir == env["SIE_SKILLS_DIR"] and e.router.manifest == env["SIE_CORPUS_MANIFEST"]
    assert e.router.dense.persist_dir == env["SIE_PERSIST_DIR"]
    assert e.mode == "sparse" and e.strict_rerank and e.router.strict_rerank
    assert e.router.reranker.model_name == "my/cross-encoder"
    assert e.overlays == ("a.json", "b.json")
    assert e.config == RetrievalConfig(top_k=3, candidate_pool=40, rerank_k=7)
    assert e.router.dense.embedder == "sentence-transformers"


def test_from_env_rerank_off_and_empty_values_mean_default(clean_env):
    e = Engine.from_env({"SIE_RERANK": "0", "SIE_MODE": "", "SIE_POOL": " ", "SIE_RERANKER": ""})
    assert e.router.reranker is None and e.mode == "hybrid" and e.config.candidate_pool == 20
    assert Engine.from_env({"SIE_RERANK": "false"}).router.reranker is None


@pytest.mark.parametrize("name,value", [
    ("SIE_MODE", "fast"), ("SIE_RERANK", "maybe"), ("SIE_STRICT_RERANK", "2"), ("SIE_POOL", "abc"),
    ("SIE_POOL", "0"), ("SIE_TOP_K", "-1"), ("SIE_RERANK_K", "1.5"), ("SIE_EMBEDDER", "bert")])
def test_from_env_invalid_values_name_the_variable(name, value):
    with pytest.raises(ValueError, match=name):
        Engine.from_env({name: value})


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), (" Yes ", True), ("ON", True), ("0", False), ("false", False),
    ("No", False), ("off", False), ("", None), ("  ", None)])
def test_env_bool_accepts_the_documented_spellings(raw, expected):
    for default in (True, False):
        want = default if expected is None else expected
        assert engine_mod.env_bool({"SIE_X": raw}, "SIE_X", default) is want
    assert engine_mod.env_bool({}, "SIE_X", True) is True


def test_env_bool_rejects_anything_else_naming_the_variable():
    with pytest.raises(ValueError, match="SIE_EAGER_INIT must be 1 or 0 .*got 'maybe'"):
        engine_mod.env_bool({"SIE_EAGER_INIT": "maybe"}, "SIE_EAGER_INIT", True)


def test_engine_accepts_a_single_overlay_path():
    assert Engine(skills_dir=CORPUS, mode="sparse", use_reranker=False, overlays=str(OVERLAY)).overlays \
        == (str(OVERLAY),)


# -- load -------------------------------------------------------------------------------------

def test_corpus_is_parsed_once_and_shared_with_the_router(monkeypatch):
    calls = []
    real = router_mod.load_corpus_report

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(router_mod, "load_corpus_report", counting)
    e = make_engine(mode="sparse")
    assert not e.loaded and e.corpus is None
    e.load()
    e.load()
    e.route("evaluate my rag answers")
    e.compose("build a rag pipeline, then evaluate it")
    e.learning_path("rag-evaluation")
    e.skill("rag-evaluation")
    assert len(calls) == 1
    assert sorted(e.graph.nodes) == sorted(e.router.skills) and len(e.router.skills) == 38
    assert e.corpus is e.router.corpus and e.corpus.name == "ml-ai-skills"


def test_concurrent_loads_parse_once(monkeypatch):
    calls = []
    real = router_mod.load_corpus_report
    monkeypatch.setattr(router_mod, "load_corpus_report",
                        lambda *a, **kw: calls.append(a) or real(*a, **kw))
    e = make_engine(mode="sparse")
    threads = [threading.Thread(target=e.load) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and e.loaded


def test_missing_corpus_raises_and_is_not_cached(tmp_path):
    e = Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False)
    with pytest.raises(RuntimeError, match="no skills found"):
        e.load()
    assert not e.loaded
    write_skill(tmp_path, "a")
    e.load()
    assert list(e.graph.nodes) == ["a"]


def test_invalid_manifest_is_a_runtime_error(tmp_path):
    e = Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False, manifest=tmp_path / "nope.toml")
    with pytest.raises(RuntimeError, match="cannot load the corpus"):
        e.load()


# -- start / status ---------------------------------------------------------------------------

def test_never_started_status_says_why_without_loading_anything():
    e = make_engine(mode="sparse")
    s = e.status()
    assert s["ready"] is False and s["started_at"] is None and s["corpus"] is None and s["graph"] is None
    assert s["reasons"] == ["corpus not loaded", "BM25 index not built"]
    assert s["version"] == sie.__version__ and not e.loaded


def test_sparse_mode_is_ready_without_a_dense_index():
    e = make_engine(mode="sparse")
    s = e.start()
    assert s["ready"] is True and s["reasons"] == [] and s["errors"] == [] and s["degraded"] is False
    assert s["started_at"] and s["router"]["dense"] == {"required": False, "ready": False, "index": None}
    assert s["corpus"]["name"] == "ml-ai-skills" and s["corpus"]["n_skills"] == 38
    assert "root" not in s["corpus"] and s["corpus"]["manifest"] == "corpus.toml"
    assert s["graph"]["nodes"] == 38 and s["graph"]["cycles"] == [] and s["graph"]["overlays"] == []
    assert set(s["graph"]["edges_by_relationship"]) == set(RELATIONSHIPS)
    json.dumps(s)                                              # JSON-ready


def test_hybrid_ready_after_warm_up_with_a_fresh_index():
    e = make_engine(reranker=FakeReranker())
    s = e.start()
    assert s["ready"] is True and s["router"]["dense"]["ready"] is True
    assert s["router"]["reranker"] == {"enabled": True, "loaded": True, "model": "fake-ce", "error": None}
    assert e.router.dense.searches == [("warm up", 1)]         # warm-up embedded one string


def test_start_records_failures_without_raising_and_resets_them():
    e = make_engine(fingerprint=None)                           # dense index never built
    s = e.start()
    assert s["ready"] is False and len(e.errors) == 1
    assert e.errors[0].startswith("warm-up failed: RuntimeError: dense index not built")
    assert s["reasons"] == e.errors + ["dense index not verified (missing, stale, or not warmed up)"]
    assert s["corpus"] is not None and s["graph"] is not None  # the corpus did load
    e.router.dense._fp = corpus_fingerprint(e.router._load_chunks())   # index built meanwhile
    s = e.start()
    assert s["ready"] is True and e.errors == []


def test_start_with_a_missing_corpus_records_the_load_error(tmp_path):
    e = Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False)
    s = e.start()
    assert s["ready"] is False and e.errors[0].startswith("load failed: RuntimeError: no skills found")
    assert "corpus not loaded" in s["reasons"] and s["started_at"]


def test_loaded_but_not_warmed_is_not_ready():
    e = make_engine()
    s = e.start(eager=False)
    assert s["ready"] is False and s["errors"] == [] and e.loaded
    assert s["reasons"] == ["BM25 index not built",
                            "dense index not verified (missing, stale, or not warmed up)"]
    e.route("evaluate my rag answers")                           # first query initializes lazily
    assert e.status()["ready"] is True


def test_status_reverifies_a_dense_index_rebuilt_externally():
    # `python -m sie.router --build` elsewhere: the next dense search rebinds its handle
    # (generation += 1); readiness must recover without waiting for another routing request
    e = make_engine()
    assert e.start()["ready"] is True
    searches = list(e.router.dense.searches)
    e.router.dense.generation += 1
    s = e.status()
    assert s["ready"] is True and s["reasons"] == [] and s["router"]["dense"]["ready"] is True
    assert e.router.dense.searches == searches                 # metadata only: nothing embedded


def test_status_reports_an_externally_rebuilt_stale_index_as_stale():
    e = make_engine()
    assert e.start()["ready"] is True
    e.router.dense._fp, e.router.dense.generation = "other-corpus", e.router.dense.generation + 1
    s = e.status()
    assert s["ready"] is False and s["errors"] == []
    assert s["reasons"] == ["dense index is stale (corpus or embedder changed); "
                            "run `python -m sie.router --build`"]
    e.router.dense._fp = corpus_fingerprint(e.router._load_chunks())   # rebuilt correctly
    assert e.status()["ready"] is True


def test_status_never_raises_when_the_dense_recheck_fails():
    e = make_engine()
    e.start()

    def broken():
        raise OSError("disk gone")

    e.router.dense.fingerprint = broken
    e.router.dense.generation += 1
    s = e.status()
    assert s["ready"] is False and s["reasons"] == ["dense index check failed: OSError: disk gone"]


def test_reranker_fallback_is_degraded_not_unready(capsys):
    e = make_engine(reranker=FakeReranker(fail="weights unavailable"))
    s = e.start()
    assert s["ready"] is True and s["degraded"] is True and e.degraded
    assert s["router"]["reranker"]["error"] == "weights unavailable"
    assert "falling back to RRF order" in capsys.readouterr().err


def test_strict_reranker_failure_is_not_ready():
    e = make_engine(reranker=FakeReranker(fail="weights unavailable"), strict_rerank=True)
    s = e.start()
    assert s["ready"] is False and s["degraded"] is False
    assert s["errors"] == ["warm-up failed: RerankerUnavailable: weights unavailable"]
    assert "cross-encoder required (strict_rerank) but not loaded" in s["reasons"]


def test_missing_retrieval_backend_is_reported_as_a_runtime_error(tmp_path):
    # the real DenseIndex imports chromadb, which the test suite blocks (like a core-only install)
    e = Engine(skills_dir=CORPUS, persist_dir=tmp_path / "chroma", use_reranker=False)
    with pytest.raises(RuntimeError, match="missing dependency"):
        e.route("evaluate my rag answers")
    s = e.start()
    assert s["ready"] is False and "missing dependency" in s["errors"][0]


def test_ensure_started_starts_only_once():
    e = make_engine(mode="sparse")
    starts = []
    real = e._start_locked
    e._start_locked = lambda eager: starts.append(eager) or real(eager)
    assert e.ensure_started()["ready"] is True
    first = e.started_at
    assert e.ensure_started()["ready"] is True and e.started_at == first and starts == [True]


def test_start_emits_an_engine_start_event():
    events = []
    obs.add_hook(events.append)
    try:
        make_engine(mode="sparse").start()
    finally:
        obs.remove_hook(events.append)
    (event,) = [ev for ev in events if ev["event"] == "engine_start"]
    assert event["ready"] is True and event["mode"] == "sparse" and event["corpus"] == "ml-ai-skills"
    assert event["reasons"] == [] and event["duration_ms"] >= 0


# -- overlays ---------------------------------------------------------------------------------

def _overlay(path: Path, edges: list[dict]) -> Path:
    path.write_text(json.dumps({"edges": edges}), encoding="utf-8")
    return path


def test_overlays_change_the_graph_learning_path_and_skill(tmp_path):
    corpus = tmp_path / "skills"
    write_skill(corpus, "a")
    write_skill(corpus, "b")
    overlay = _overlay(tmp_path / "extra.json", [{"source": "a", "target": "b", "relationship": "requires"}])
    plain = Engine(skills_dir=corpus, mode="sparse", use_reranker=False)
    e = Engine(skills_dir=corpus, mode="sparse", use_reranker=False, overlays=[overlay])
    assert plain.learning_path("b")["path"] == ["b"]
    lp = e.learning_path("b")
    assert lp["path"] == ["a", "b"] and lp["steps"][0]["provenance"] == ["overlay:extra.json [proposed]"]
    assert e.skill("b")["requires"] == ["a"] and e.prerequisites("b") == ["a"]
    assert e.skill("a")["relations"] == [{"source": "a", "target": "b", "relationship": "requires",
                                          "confidence": "proposed", "provenance": "overlay:extra.json"}]
    graph = e.start()["graph"]
    assert graph["overlays"] == ["overlay:extra.json"] and graph["edges_by_relationship"]["requires"] == 1


def test_proposed_overlay_applies_to_the_real_corpus_without_cycles():
    plain = make_engine(mode="sparse")
    e = make_engine(mode="sparse", overlays=[OVERLAY])
    before = plain.start()["graph"]["edges_by_relationship"]["requires"]
    status = e.start()
    assert status["ready"] and status["graph"]["cycles"] == []
    assert status["graph"]["edges_by_relationship"]["requires"] == before + 9
    assert status["graph"]["overlays"] == ["docs/PROPOSED_REQUIRES.md"]
    assert "neural-net-fundamentals" in e.learning_path("cnn-vision")["path"]
    assert "neural-net-fundamentals" not in plain.learning_path("cnn-vision")["path"]


@pytest.mark.parametrize("content", ["{not json", json.dumps({"edges": [{"source": "a"}]})])
def test_invalid_overlay_fails_load_and_is_recorded_by_start(tmp_path, content):
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")
    e = make_engine(mode="sparse", overlays=[bad])
    with pytest.raises(RuntimeError, match="cannot load edge overlay"):
        e.load()
    s = e.start()
    assert s["ready"] is False and "cannot load edge overlay" in s["errors"][0]
    assert "skill graph not built" in s["reasons"] and s["corpus"]["n_skills"] == 38 and s["graph"] is None


def test_missing_overlay_file_is_a_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="cannot load edge overlay"):
        make_engine(mode="sparse", overlays=[tmp_path / "missing.json"]).load()


# -- passthrough ------------------------------------------------------------------------------

def test_route_and_batch_pass_through_to_the_router():
    e = make_engine()
    queries = ["evaluate my rag answers", "time series forecasting", "   "]
    direct = [_stable(e.router.route(q, k=3, pool=10)) for q in queries]
    assert [_stable(r) for r in e.batch(queries, k=3, pool=10)] == direct
    assert _stable(e.route(queries[0], k=3, pool=10)) == direct[0]
    assert e.router.dense.searches[-1] == ("evaluate my rag answers", 10)
    with pytest.raises(TypeError):
        e.batch("one query")
    with pytest.raises(ValueError, match="pool must be >= 1"):
        e.route("x", pool=0)


def test_route_defaults_come_from_the_engine_config():
    e = make_engine(config=RetrievalConfig(top_k=2, candidate_pool=7))
    result = e.route("evaluate my rag answers")
    assert len(result.results) == 2 and e.router.dense.searches[-1][1] == 7


def test_compose_passes_through_with_pool_and_rerank_k():
    e = make_engine(mode="sparse")
    query = "Build a RAG pipeline, evaluate it, deploy it, and monitor it"
    expected = compose(e.router, e.graph, query, k=3, max_intents=5)
    got = e.compose(query)
    assert _stable_composition(got) == _stable_composition(expected)
    hybrid = make_engine()
    hybrid.compose("build a rag pipeline, then evaluate it", pool=7)
    assert {k for _, k in hybrid.router.dense.searches} == {7}
    with pytest.raises(ValueError, match="k must be"):
        e.compose(query, k=0)


def test_compose_cycle_is_a_cycle_error(tmp_path):
    write_skill(tmp_path, "alpha", "requires:\n  - beta", "configure alpha widgets carefully")
    write_skill(tmp_path, "beta", "requires:\n  - alpha", "tune beta gadgets")
    write_skill(tmp_path, "gamma", "", "unrelated gardening notes")
    e = Engine(skills_dir=tmp_path, mode="sparse", use_reranker=False)
    assert e.start()["graph"]["cycles"] == [["alpha", "beta"]]
    with pytest.raises(CycleError, match="requires cycle"):
        e.compose("configure alpha widgets")
    with pytest.raises(CycleError, match="requires cycle"):
        e.learning_path("alpha")


def test_learning_path_passthrough_and_unknown_target():
    e = make_engine(mode="sparse")
    assert e.learning_path("rag-evaluation") == learning_path(e.graph, "rag-evaluation")
    assert e.learning_path("rag-evaluation", include_recommended=True)["path"] == ["rag-pipeline",
                                                                                  "rag-evaluation"]
    with pytest.raises(KeyError) as exc:
        e.learning_path("rag-pipelin")
    assert isinstance(exc.value, UnknownSkillError)
    assert str(exc.value).startswith("unknown skill 'rag-pipelin'; did you mean rag-pipeline")
    text = e.render_learning_path("rag-evaluation").splitlines()
    assert text[0].startswith("[path] learning path to rag-evaluation (2 steps")


def test_skill_detail_and_provenance():
    e = make_engine(mode="sparse")
    d = e.skill("rag-evaluation")
    assert d["slug"] == d["skill_id"] == "rag-evaluation" and d["name"] == "RAG Evaluation"
    assert d["requires"] == ["rag-pipeline"] and d["required_by"] == [] and d["dangling"] == []
    assert d["provenance"]["provenance"] == "ml-ai-skills@8328c60:rag-evaluation/SKILL.md"
    assert len(d["provenance"]["content_hash"]) == 64 and "lifecycle" in d["metadata"]
    assert {"source": "rag-pipeline", "target": "rag-evaluation", "relationship": "requires",
            "confidence": "declared",
            "provenance": "ml-ai-skills@8328c60:rag-evaluation/SKILL.md#requires"} in d["relations"]
    assert "requires" in d["declared"]
    assert e.skill("rag-pipeline")["required_by"] == ["rag-evaluation"]
    json.dumps(d)
    assert e.skill_obj("rag-evaluation").slug == "rag-evaluation"
    assert e.prerequisites("nope") == []
    for fn in (e.skill, e.skill_obj):
        with pytest.raises(KeyError, match="unknown skill 'nope'"):
            fn("nope")


def test_concurrent_routing_is_deterministic():
    e = make_engine(mode="sparse")
    e.start()
    queries = ["evaluate my rag answers", "time series forecasting", "explain predictions with shap"]
    expected = [_stable(e.route(q)) for q in queries]
    seen, errors = [], []

    def work():
        try:
            for _ in range(10):
                seen.append([_stable(e.route(q)) for q in queries])
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(seen) == 80 and all(s == expected for s in seen)


def test_composition_dict_has_no_chunk_ids():
    e = make_engine(mode="sparse")
    data = composition_dict(e.compose("Build a RAG pipeline, then evaluate it"))
    text = json.dumps(data)
    assert data["multi_intent"] is True and "chunk_id" not in text
    assert not any(cid in text for cid in e.router._texts)


# -- display_paths: no machine paths in messages that leave the server ------------------------

def test_display_paths_rewrites_repo_paths_repo_relative():
    from sie.engine import display_paths
    skills, overlay = DEFAULT_SKILLS_DIR, ROOT / "docs" / "x.json"
    assert display_paths(f"no skills found under {skills}") == "no skills found under data/skills"
    assert display_paths(f"no skills found under {skills}.") == "no skills found under data/skills."
    assert display_paths(f"cannot load the corpus at {skills}: invalid corpus manifest "
                         f"{skills / 'corpus.toml'}: bad") \
        == "cannot load the corpus at data/skills: invalid corpus manifest data/skills/corpus.toml: bad"
    # OSError messages repr() the file name (doubled backslashes on Windows)
    assert display_paths(f"cannot load edge overlay {overlay}: [Errno 2] No such file: {str(overlay)!r}") \
        == "cannot load edge overlay docs/x.json: [Errno 2] No such file: 'docs/x.json'"
    assert display_paths(f"at {ROOT.as_posix()}/data/chroma, then") == "at data/chroma, then"
    assert display_paths(f"root {ROOT}") == "root ."


def test_display_paths_leaves_other_paths_as_given():
    from sie.engine import display_paths
    for text in (f"sibling {ROOT}-old{os.sep}data", f"under {ROOT.parent / 'elsewhere'}",
                 "no skills found under data/skills", "cross-encoder/ms-marco-MiniLM-L-6-v2", ""):
        assert display_paths(text) == text
    assert display_paths("/srv/sie/data/skills: x; /srv/sie2/a; /mnt/srv/sie/b", root="/srv/sie") \
        == "data/skills: x; /srv/sie2/a; /mnt/srv/sie/b"
    assert display_paths("a / b", root="/") == "a / b"


# -- CLI: python -m sie.router --multi / --overlay --------------------------------------------

def _cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["sie.router", "--skills", str(CORPUS), *args])
    router_mod.main()
    return capsys.readouterr()


def test_cli_multi_prints_the_composition_summary(monkeypatch, capsys):
    out = _cli(monkeypatch, capsys, "Build a RAG pipeline, then evaluate it", "--multi",
               "--mode", "sparse", "--no-rerank").out.splitlines()
    assert out[0].startswith("[compose] 2 intents -> ")
    assert out[1].startswith("  intent 1: 'Build a RAG pipeline' -> ")
    assert out[2].startswith("  intent 2: 'evaluate it' (routed as 'evaluate a RAG pipeline') -> ")
    assert not any(line.startswith("[router] confidence:") for line in out)


def test_cli_multi_json_is_the_composition(monkeypatch, capsys):
    captured = _cli(monkeypatch, capsys, "Build a RAG pipeline, then evaluate it", "--multi", "--json",
                    "--mode", "sparse", "--no-rerank")
    data = json.loads(captured.out)
    assert [it["query"] for it in data["intents"]] == ["Build a RAG pipeline", "evaluate a RAG pipeline"]
    assert data["steps"] and "chunk_id" not in captured.out


def test_cli_multi_errors_exit_cleanly(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, capsys, "build it, then test it", "--multi", "--mode", "sparse",
             "--no-rerank", "--pool", "0")
    assert "pool must be >= 1" in str(exc.value)


def test_cli_path_with_overlay_applies_it(monkeypatch, capsys):
    plain = _cli(monkeypatch, capsys, "--path", "cnn-vision").out.splitlines()
    out = _cli(monkeypatch, capsys, "--path", "cnn-vision", "--overlay", str(OVERLAY)).out.splitlines()
    assert plain[0].startswith("[path] learning path to cnn-vision (1 step,")
    assert out[0].startswith("[path] learning path to cnn-vision (2 steps,")
    assert out[1].split()[:2] == ["1.", "neural-net-fundamentals"]
    assert "  why: neural-net-fundamentals: direct prerequisite of cnn-vision (proposed)" in out


def test_cli_path_via_the_engine_matches_the_plain_renderer(monkeypatch, capsys):
    plain = _cli(monkeypatch, capsys, "--path", "rag-evaluation").out
    via_engine = _cli(monkeypatch, capsys, "--path", "rag-evaluation",
                      "--manifest", str(CORPUS / "corpus.toml")).out
    assert via_engine == plain


def test_cli_path_overlay_errors_exit_cleanly(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, capsys, "--path", "rag-evaluation", "--overlay", str(tmp_path / "x.json"))
    assert "[path] cannot load edge overlay" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, capsys, "--path", "rag-pipelin", "--overlay", str(OVERLAY))
    assert "unknown skill 'rag-pipelin'; did you mean rag-pipeline" in str(exc.value)


# -- packaging --------------------------------------------------------------------------------

def test_imports_stay_light():
    import subprocess
    code = ("import sys, sie; assert 'sie.router' not in sys.modules, 'import sie must stay cheap'; "
            "from sie import Engine; import sie.engine, sie.schemas, sie.api; "
            "assert Engine is sie.engine.Engine; "
            "heavy = {'chromadb', 'sentence_transformers', 'torch', 'onnxruntime', 'streamlit'} & set(sys.modules); "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_pyproject_metadata_matches_the_package():
    project = _pyproject()["project"]
    assert project["version"] == sie.__version__ == "0.2.0"
    assert project["requires-python"] == ">=3.11" and project["license"] == "MIT"
    core = " ".join(project["dependencies"])
    for heavy in ("chromadb", "sentence-transformers", "fastapi", "torch"):
        assert heavy not in core                                # optional extras only
    extras = project["optional-dependencies"]
    assert {"retrieval", "reranker", "api", "evaluation", "llm", "demo", "dev", "all"} <= set(extras)
    assert any(d.startswith("chromadb") for d in extras["retrieval"])
    for name, target in project["scripts"].items():
        module, func = target.split(":")
        assert callable(getattr(__import__(module, fromlist=[func]), func)), name
    assert _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"] == ["sie*"]


def _dist_name(requirement: str) -> str:
    import re
    return re.split(r"[\s<>=!~;\[(]", requirement.strip(), maxsplit=1)[0].lower()


def test_dev_extra_installs_the_testclient_backend_starlette_asks_for():
    # starlette 1.x's testclient imports httpx2 and only falls back to httpx with a
    # deprecation warning; a later release may drop the fallback
    from importlib import metadata
    dev = {_dist_name(d) for d in _pyproject()["project"]["optional-dependencies"]["dev"]}
    assert {"httpx", "httpx2"} <= dev
    try:
        requires = metadata.requires("starlette") or []
    except metadata.PackageNotFoundError:
        return
    wanted = {_dist_name(r) for r in requires if _dist_name(r).startswith("httpx")}
    assert wanted <= dev, sorted(wanted - dev)


def test_env_example_documents_every_variable_the_engine_reads():
    import re
    used = {m for p in (ROOT / "sie").rglob("*.py")
            for m in re.findall(r"\bSIE_[A-Z][A-Z_]*[A-Z]\b", p.read_text(encoding="utf-8"))}
    documented = set(re.findall(r"\bSIE_[A-Z][A-Z_]*[A-Z]\b",
                                (ROOT / ".env.example").read_text(encoding="utf-8")))
    assert set(SIE_VARS) <= used and used <= documented, sorted(used - documented)


def test_requirements_cover_the_core_dependencies():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    reqs = {line.split("#")[0].split(">=")[0].strip().lower() for line in lines if line.split("#")[0].strip()}
    for dep in _pyproject()["project"]["dependencies"]:
        assert dep.split(">=")[0].strip().lower() in reqs, dep


# -- demo -------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeStreamlit(types.ModuleType):
    """Just enough of streamlit to run the demo script headless and record what it shows."""

    def __init__(self, answers):
        super().__init__("streamlit")
        self.answers, self.calls, self.sidebar = answers, [], _Ctx()

    def cache_resource(self, fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)

    def stop(self):
        raise _Stop

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def widget(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            label = args[0] if args else kwargs.get("label")
            if isinstance(label, str) and label in self.answers:
                return self.answers[label]
            if name in ("radio", "selectbox"):
                return list(kwargs["options"])[kwargs.get("index", 0)]
            if name in ("text_input", "checkbox", "toggle", "slider"):
                return kwargs.get("value")
            return _Ctx()
        return widget

    def shown(self, *names):
        return [str(args[0]) for name, args, _ in self.calls if name in names and args]


def test_demo_runs_on_the_engine_headless(monkeypatch, clean_env):
    fake = FakeStreamlit({"Retrievers": "sparse", "Cross-encoder rerank": False,
                          "Describe your task": "Build a RAG pipeline, then evaluate it",
                          "Multi-intent plan": True})
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.setenv("SIE_SKILLS_DIR", str(CORPUS))
    runpy.run_path(str(ROOT / "demo" / "app_streamlit.py"), run_name="__main__")
    assert fake.shown("error") == []
    assert fake.shown("success")[0] == "Ready"
    assert any(s.startswith("Confidence: ") for s in fake.shown("success", "warning", "error"))
    assert any(s.startswith("1. ") for s in fake.shown("expander"))
    assert any(s.startswith("Intent 2: 'evaluate it' (routed as 'evaluate a RAG pipeline')")
               for s in fake.shown("write"))
    assert any("rag-pipeline" in s for s in fake.shown("markdown"))
    at = next(i for i, c in enumerate(fake.calls) if c[0] == "selectbox")
    select = fake.calls[at][2]
    target = select["options"][select["index"]]              # defaults to the top routed skill
    path = Engine(skills_dir=CORPUS, mode="sparse", use_reranker=False).learning_path(target)["path"]
    assert ("write", (" -> ".join(path),), {}) in fake.calls[at:]


@pytest.mark.parametrize("value", ["false", "off", "no", "0"])
def test_demo_honors_every_off_spelling_of_sie_rerank(monkeypatch, clean_env, value):
    fake = FakeStreamlit({"Retrievers": "sparse", "Describe your task": "", "Multi-intent plan": False})
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.setenv("SIE_SKILLS_DIR", str(CORPUS))
    monkeypatch.setenv("SIE_RERANK", value)
    built = []
    real = Engine.from_env.__func__
    monkeypatch.setattr(Engine, "from_env",
                        classmethod(lambda cls, env=None: built.append(env) or real(cls, env)))
    runpy.run_path(str(ROOT / "demo" / "app_streamlit.py"), run_name="__main__")
    (checkbox,) = [kw for name, args, kw in fake.calls if name == "checkbox"]
    assert checkbox["value"] is False and fake.shown("error") == []
    assert built and all(env["SIE_RERANK"] == "0" for env in built)   # the reranker stays off


def test_demo_invalid_sie_rerank_is_an_error_not_a_silent_default(monkeypatch, clean_env):
    fake = FakeStreamlit({})
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.setenv("SIE_RERANK", "maybe")
    with pytest.raises(_Stop):
        runpy.run_path(str(ROOT / "demo" / "app_streamlit.py"), run_name="__main__")
    (error,) = fake.shown("error")
    assert error.startswith("Invalid configuration: SIE_RERANK must be 1 or 0") and "'maybe'" in error
    assert not any(name == "checkbox" for name, _, _ in fake.calls)
