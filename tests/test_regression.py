"""Regression: benchmarks/metrics.json is reproduced with no models, no index and no network.

1. keyword / keyword-shipped / bm25-card / bm25 are recomputed from scratch on every dataset;
2. hybrid and dense-only are recomputed with the dense side replayed from
   tests/fixtures/dense_rankings.json (confidence distributions and compose() included);
3. the fixture and metrics.json belong to the current corpus and query sets.
Values must match within run_eval.TOLERANCE after the same 6-decimal rounding. Any failure
names the metric, dataset, recorded and new value.
"""
import json
import os
import re
import socket
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import sie.compose
from eval import run_eval
from eval.replay import (FIXTURE, RERECORD, ReplayDense, dumps, fixture_queries, load_fixture, record,
                         replay_router)
from sie.models import Chunk, Confidence, Hit, MethodEvidence, RankedSkill, RouteResult
from sie.rerank import LOCAL_CE_DIR, LOCAL_CE_HINT, RerankerUnavailable
from sie.router import corpus_fingerprint
from tests.conftest import NetworkBlocked

ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT / "benchmarks" / "metrics.json"
RERUN = "rerun: python -m eval.run_eval (then python -m eval.run_eval --record-fixture)"


def _rerank_published(recorded) -> bool:
    """metrics.json holds cross-encoder metrics (which no model-free replay can reproduce)."""
    return any("hybrid-rerank" in per for per in recorded["metrics"].values())


# -- the network guard covers module-scoped fixtures too (they do the real work here) ------------

@pytest.fixture(scope="module")
def module_scope_dns():
    try:
        socket.getaddrinfo("pypi.org", 443)
    except NetworkBlocked:
        return "blocked"
    return "resolved"


def test_network_guard_is_active_in_module_scoped_fixtures(module_scope_dns):
    assert module_scope_dns == "blocked"


@pytest.fixture(scope="module")
def repo_cwd():
    """Corpus, query-set and baseline paths are relative to the repo root (as in the CLI)."""
    old = os.getcwd()
    os.chdir(ROOT)
    yield ROOT
    os.chdir(old)


@pytest.fixture(scope="module")
def recorded():
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixture():
    return load_fixture(ROOT / FIXTURE)


@pytest.fixture(scope="module")
def ctx(repo_cwd):
    return run_eval.load_context(ROOT / "eval" / "datasets.toml")


@pytest.fixture(scope="module")
def cfg(recorded):
    config = recorded["config"]
    return run_eval.EvalConfig(pool=config["pool"], rerank_k=config["rerank_k"])


@pytest.fixture(scope="module")
def scored(ctx, cfg, fixture):
    """The smoke systems scored now on every dataset (dataset -> [Scored])."""
    return run_eval.evaluate(run_eval.smoke_systems(fixture), ctx.datasets, cfg)


@pytest.fixture(scope="module")
def current(ctx, cfg, scored):
    """metrics.json-shaped document of the smoke systems, computed now."""
    return run_eval.metrics_document(ctx, scored, cfg)


def _explain(diffs) -> str:
    return "\n".join(f"{metric} on dataset '{ds}' ({system}): metrics.json has {old}, now {new}"
                     for ds, system, metric, old, new in diffs)


# -- 3. the recorded artifacts belong to this corpus --------------------------------------------

def test_fixture_was_recorded_from_the_current_corpus(fixture, ctx):
    now = corpus_fingerprint(ctx.chunks)
    assert fixture["corpus_fingerprint"] == now, (
        f"dense fixture fingerprint {fixture['corpus_fingerprint']} != corpus {now}; {RERECORD}")


def test_fixture_covers_every_query_the_harness_routes(fixture, ctx):
    missing = sorted(set(fixture_queries(ctx.datasets)) - set(fixture["queries"]))
    assert not missing, f"{len(missing)} queries missing from the fixture (e.g. {missing[0]!r}); {RERECORD}"
    assert all(len(r) == min(fixture["depth"], len(ctx.chunks)) for r in fixture["queries"].values())


def test_metrics_json_matches_the_current_corpus_and_query_sets(recorded, current):
    assert recorded["corpus"] == current["corpus"], RERUN
    for name, info in current["datasets"].items():
        assert recorded["datasets"][name]["sha256"] == info["sha256"], f"dataset '{name}' changed; {RERUN}"
    assert set(recorded["datasets"]) == set(current["datasets"])
    assert recorded["datasets"] == current["datasets"], f"dataset path/kind/n/origins changed; {RERUN}"
    assert recorded["config"] == current["config"], f"retrieval/eval config changed in code; {RERUN}"


def test_metrics_json_systems_status_matches_its_metrics(recorded):
    """A system is 'ok' exactly when it has metrics (so a hand-set status can't skip a check)."""
    diffs, n = run_eval.compare_metrics(recorded, recorded, {})
    assert n >= len(recorded["systems"]) and not diffs, _explain(diffs)


def test_compare_metrics_catches_tampered_config_datasets_and_status(recorded):
    current = json.loads(json.dumps(recorded))
    tampered = json.loads(json.dumps(recorded))
    tampered["config"]["rrf_k"] = 30
    tampered["datasets"]["main"]["origins"] = {"human-collected": 76, "scaffold": 12}
    tampered["systems"]["hybrid-rerank"]["status"] = "ok"
    tampered["systems"]["bm25"]["family"] = "keyword"
    diffs, _ = run_eval.compare_metrics(tampered, current, {"bm25": "bm25"})
    assert ("-", "-", "config.rrf_k", 30, current["config"]["rrf_k"]) in diffs
    assert ("main", "-", "origins", {"human-collected": 76, "scaffold": 12},
            current["datasets"]["main"]["origins"]) in diffs
    assert ("-", "hybrid-rerank", "systems.status", "ok", "no metrics") in diffs
    assert ("-", "bm25", "systems.bm25.family", "keyword", "sie") in diffs
    assert len(diffs) == 4


# -- 1. index-free systems, recomputed ---------------------------------------------------------

@pytest.mark.parametrize("system", ["keyword", "keyword-shipped", "bm25-card", "bm25"])
def test_index_free_systems_reproduce_metrics_json(recorded, current, system):
    diffs, n = run_eval.compare_metrics(recorded, current, {system: system})
    assert n > 50
    assert not diffs, _explain(diffs)


# -- 2. hybrid (and dense-only) with the dense side replayed -----------------------------------

@pytest.mark.parametrize("replayed, reference", [("hybrid-replay", "hybrid"), ("dense-replay", "dense")])
def test_replayed_dense_reproduces_metrics_json(recorded, current, replayed, reference):
    diffs, _ = run_eval.compare_metrics(recorded, current, {replayed: reference})
    assert not diffs, _explain(diffs)


def test_hybrid_replay_covers_confidence_and_composition(recorded, current):
    """The comparison above includes the confidence distribution and compose() metrics."""
    for ds in ("main", "source"):
        conf = current["metrics"][ds]["hybrid-replay"]["confidence"]
        assert conf == recorded["metrics"][ds]["hybrid"]["confidence"]
        assert sum(g["n"] for g in conf["by_level"].values()) == current["datasets"][ds]["n"]
    oos = current["metrics"]["out-of-scope"]["hybrid-replay"]
    assert oos["by_action"] == recorded["metrics"]["out-of-scope"]["hybrid"]["by_action"]
    multi = current["metrics"]["multi-intent"]["hybrid-replay"]
    assert set(multi) >= {"single_top1", "top_n_oracle", "compose", "by_n_intents"}
    assert multi["compose"] == recorded["metrics"]["multi-intent"]["hybrid"]["compose"]


def test_pre_existing_headline_numbers(recorded):
    """The routing numbers published before confidence/composition existed have not moved."""
    m = recorded["metrics"]
    expected = {("main", "hybrid", "top-1"): 0.955, ("main", "hybrid", "mrr"): 0.972,
                ("main", "bm25", "top-1"): 0.920, ("main", "dense", "top-1"): 0.886,
                ("main", "keyword", "top-1"): 0.432, ("source", "hybrid", "top-1"): 0.826,
                ("source", "hybrid", "mrr"): 0.895}
    for (ds, system, metric), value in expected.items():
        assert round(m[ds][system][metric], 3) == value, f"{metric} on {ds} ({system})"


def test_metrics_json_is_canonical_and_timing_free(recorded):
    text = METRICS_JSON.read_text(encoding="utf-8")
    assert text == run_eval.dumps_metrics(recorded)            # sorted keys, fixed layout
    assert "timings" not in text and "_ms" not in text and "loaded_at" not in text


# -- the comparison itself ------------------------------------------------------------------------

def _doc(value, sha="a" * 64, fp="f"):
    return {"corpus": {"fingerprint": fp, "chunk_fingerprint": "c"},
            "datasets": {"d": {"sha256": sha}},
            "metrics": {"d": {"s": {"top-1": value, "confidence": {"by_level": {"high": {"n": 3}}},
                                    "none": None, "flag": True}}}}


def test_compare_metrics_tolerance_and_types():
    diffs, n = run_eval.compare_metrics(_doc(0.5), _doc(0.5 + 5e-10), {"s": "s"})
    assert diffs == [] and n == 2 + 1 + 4
    diffs, _ = run_eval.compare_metrics(_doc(0.5), _doc(0.500001), {"s": "s"})
    assert diffs == [("d", "s", "top-1", 0.5, 0.500001)]
    new = _doc(0.5)
    new["metrics"]["d"]["s"]["flag"] = 1                          # bool vs int is a change
    assert run_eval.compare_metrics(_doc(0.5), new, {"s": "s"})[0] == [("d", "s", "flag", True, 1)]


def test_compare_metrics_reports_missing_values_datasets_and_changed_inputs():
    new = _doc(0.5, sha="b" * 64, fp="g")
    del new["metrics"]["d"]["s"]["none"]
    new["metrics"]["d"]["s"]["extra"] = 1.0
    diffs, _ = run_eval.compare_metrics(_doc(0.5), new, {"s": "s"})
    assert ("-", "-", "corpus.fingerprint", "f", "g") in diffs
    assert ("d", "-", "sha256", "a" * 12, "b" * 12) in diffs
    assert ("d", "s", "none", None, "<missing>") in diffs and ("d", "s", "extra", "<missing>", 1.0) in diffs
    only_old = _doc(0.5)
    only_old["datasets"]["gone"] = {"sha256": "x"}
    diffs, _ = run_eval.compare_metrics(only_old, _doc(0.5), {"s": "s"})
    assert diffs == [("gone", "-", "dataset", "scored", "not scored now")]
    diffs, _ = run_eval.compare_metrics(_doc(0.5), _doc(0.5), {"s": "t"})
    assert diffs and "no 't' in metrics.json" in diffs[0][2]


def test_compare_metrics_config_dataset_fields_and_replayed_system_entries():
    old, new = _doc(0.5), _doc(0.5)
    old["config"], new["config"] = {"rrf_k": 60, "pool": 20}, {"rrf_k": 61, "pool": 20}
    old["datasets"]["d"].update(n=3, path="a.jsonl")
    new["datasets"]["d"].update(n=4, path="a.jsonl")
    label = run_eval._LABELS["hybrid"]
    old["systems"] = {"hybrid": {"label": label, "family": "sie", "status": "ok", "confidence": True},
                      "s": {"label": "S", "family": "sie", "status": "ok", "confidence": False}}
    new["systems"] = {"hybrid-replay": {"label": "SIE: hybrid, replayed", "family": "sie",
                                        "status": "ok", "confidence": True}}
    new["metrics"]["d"]["hybrid-replay"] = new["metrics"]["d"].pop("s")
    old["metrics"]["d"]["hybrid"] = old["metrics"]["d"]["s"]
    diffs, n = run_eval.compare_metrics(old, new, {"hybrid-replay": "hybrid"})
    # the replay's own label is not a difference: it stands in for the recorded system
    assert diffs == [("-", "-", "config.rrf_k", 60, 61), ("d", "-", "n", 3, 4)]
    assert n == 2 + 2 + 3 + 4 + 4 + 2          # corpus, config, dataset fields, metrics, system, status
    new["systems"]["hybrid-replay"]["confidence"] = False
    assert ("-", "hybrid-replay", "systems.hybrid.confidence", True, False) in \
        run_eval.compare_metrics(old, new, {"hybrid-replay": "hybrid"})[0]
    del old["metrics"]["d"]["s"]                                   # 's' says ok but has no metrics
    assert ("-", "s", "systems.status", "ok", "no metrics") in \
        run_eval.compare_metrics(old, new, {"hybrid-replay": "hybrid"})[0]


def test_smoke_fails_on_a_tampered_metrics_json(tmp_path, repo_cwd, recorded, scored, ctx, capsys,
                                                monkeypatch):
    tampered = json.loads(json.dumps(recorded))
    tampered["metrics"]["main"]["bm25"]["top-1"] += 0.01
    (tmp_path / "metrics.json").write_text(run_eval.dumps_metrics(tampered), encoding="utf-8")
    # fast: BM25 only, reusing the results scored above instead of scoring again
    keep = [s for s in run_eval.smoke_systems({}) if s.key == "bm25"]
    monkeypatch.setattr(run_eval, "smoke_systems", lambda fixture: keep)
    monkeypatch.setattr(run_eval, "load_fixture",
                        lambda path: {"corpus_fingerprint": corpus_fingerprint(ctx.chunks)})
    monkeypatch.setattr(run_eval, "evaluate", lambda systems, datasets, cfg: {
        name: [s for s in rows if s.system.key == "bm25"] for name, rows in scored.items()})
    assert run_eval.smoke(tmp_path) == 1
    out = capsys.readouterr().out
    assert "main           bm25             bm25             1 MISMATCH" in out
    assert "top-1" in out and "1 mismatch" in out
    (tmp_path / "metrics.json").write_text(run_eval.dumps_metrics(recorded), encoding="utf-8")
    assert run_eval.smoke(tmp_path) == 0
    assert "0 mismatches" in capsys.readouterr().out
    # a config the code no longer uses (e.g. the RRF k) or a hand-set status fails smoke too
    tampered = json.loads(json.dumps(recorded))
    tampered["config"]["rrf_k"] += 1
    tampered["systems"]["hybrid-rerank"]["status"] = "ok"
    (tmp_path / "metrics.json").write_text(run_eval.dumps_metrics(tampered), encoding="utf-8")
    assert run_eval.smoke(tmp_path) == 1
    out = capsys.readouterr().out
    assert "config.rrf_k" in out and "systems.status" in out and "2 mismatches" in out


def test_smoke_without_metrics_json_fails(tmp_path, capsys):
    assert run_eval.smoke(tmp_path) == 1
    assert "not found" in capsys.readouterr().out


# -- the replayed dense index --------------------------------------------------------------------

CHUNKS = [Chunk("a", "Card", "alpha card text " * 20, "a::Card::0"),
          Chunk("a", "Workflow", "alpha workflow", "a::Workflow::0"),
          Chunk("b", "Card", "beta card", "b::Card::0")]


def _fixture(queries, depth=50, fp="fp"):
    return {"corpus_fingerprint": fp, "depth": depth, "embedder": "onnx", "model": "m", "pool": 2,
            "queries": queries}


def test_replay_serves_recorded_order_with_live_hit_fields():
    dense = ReplayDense(_fixture({"q": [["b::Card::0", 0.9], ["a::Card::0", 0.8]]}), CHUNKS)
    hits = dense.search("q", k=5)
    assert [h.chunk_id for h in hits] == ["b::Card::0", "a::Card::0"]       # never re-sorted
    assert hits[1] == Hit(skill_slug="a", score=0.8, section="Card", snippet=CHUNKS[0].text[:200],
                          chunk_id="a::Card::0")
    assert [h.chunk_id for h in dense.search("q", k=1)] == ["b::Card::0"]
    assert dense.fingerprint() == "fp" and dense.built_with() == "onnx" and dense.model_name == "m"
    assert dense.info()["count"] == 3 and dense.generation == 0


def test_replay_unknown_query_names_it():
    dense = ReplayDense(_fixture({}), CHUNKS)
    with pytest.raises(KeyError, match="how do I bake bread"):
        dense.search("how do I bake bread", k=3)


def test_replay_refuses_deeper_pools_and_unknown_chunks():
    dense = ReplayDense(_fixture({"q": [["a::Card::0", 0.9], ["b::Card::0", 0.8]], "r": [["zz::Card::0", 1.0]]},
                                 depth=2), CHUNKS)
    with pytest.raises(ValueError, match="top 2 chunks"):
        dense.search("q", k=3)
    with pytest.raises(RuntimeError, match="not in the corpus"):
        dense.search("r", k=1)


def test_replay_router_rejects_a_fixture_from_another_corpus(repo_cwd):
    with pytest.raises(RuntimeError, match="re-record"):
        replay_router(_fixture({}, fp="0000000000000000"))


def test_record_keeps_the_index_order_and_rounds_scores():
    class LiveDense:
        embedder, model_name = "onnx", "m"

        def search(self, query, k=10):
            return [Hit("b", 0.12345678, chunk_id="b::Card::0"), Hit("a", 0.2, chunk_id="a::Card::0")][:k]

    fx = record(LiveDense(), ["q2", "q1", "q2"], corpus_fingerprint="fp", pool=20, depth=2)
    assert list(fx["queries"]) == ["q1", "q2"]
    assert fx["queries"]["q1"] == [["b::Card::0", 0.123457], ["a::Card::0", 0.2]]
    assert (fx["depth"], fx["pool"], fx["corpus_fingerprint"]) == (2, 20, "fp")


def test_fixture_file_round_trips_one_line_per_query(tmp_path):
    fx = _fixture({"q’2": [["a::Card::0", 0.5]], "q1": [["b::Card::0", 0.25], ["a::Card::0", 0.1]]})
    text = dumps(fx)
    assert text.count("\n") == 5 + 2 + 2 + 2 and '"q’2": [["a::Card::0",0.5]]' in text
    path = tmp_path / "f.json"
    path.write_text(text, encoding="utf-8")
    assert load_fixture(path) == fx
    with pytest.raises(FileNotFoundError, match="re-record"):
        load_fixture(tmp_path / "missing.json")
    path.write_text(json.dumps({"queries": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_fixture(path)


def test_fixture_queries_include_compose_sub_queries(ctx):
    multi = [d for d in ctx.datasets if d.kind == "multi_intent"]
    queries = set(fixture_queries(multi))
    assert {r["query"] for r in multi[0].rows} <= queries
    assert len(queries) > len(multi[0].rows)                   # the split sub-queries too


# -- the generated report sections, rendered from the replayed results ---------------------------

@pytest.fixture(scope="module")
def as_published(scored):
    """The replayed results as a full run lays them out: each replay under the system it
    reproduces, SYSTEMS order, and the cross-encoder row pending (it cannot be replayed)."""
    by_key = {s.key: s for s in run_eval.SYSTEMS}
    order = list(by_key)
    out = {}
    for name, rows in scored.items():
        rows = [replace(s, system=by_key[run_eval.SMOKE_REFERENCE[s.system.key]]) for s in rows]
        rows.append(run_eval.Scored(by_key["hybrid-rerank"], None, note="not replayable", routes=True))
        out[name] = sorted(rows, key=lambda s: order.index(s.system.key))
    return out


def test_committed_results_md_headline_independent_sections(ctx):
    """Sections that do not depend on the headline system are checked on every run (no skip)."""
    text = (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    for part in (run_eval.origin_section(ctx), run_eval.first_measurement_note()):
        assert part in text, f"RESULTS.md differs from the code at: {part.splitlines()[0]!r}; {RERUN}"


def _generated_block(text: str, name: str) -> str | None:
    """The body `eval.report.replace_block` wrote between the `name` markers, or None."""
    m = re.search(rf"<!-- BEGIN:{re.escape(name)} -->\n(.*?)\n<!-- END:{re.escape(name)} -->", text, re.S)
    return m.group(1) if m else None


def test_committed_readme_blocks_are_reproduced_without_models(ctx, as_published, recorded):
    """The README's generated headline and benchmarks blocks, rebuilt from the replay."""
    if _rerank_published(recorded):
        pytest.skip("the committed headline is the cross-encoder run, which cannot be replayed")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, block in (("headline", run_eval.readme_headline(ctx, as_published)),
                        ("benchmarks", run_eval.readme_block(ctx, as_published))):
        assert _generated_block(text, name) == block.strip(), \
            f"README '{name}' block differs from the code; {RERUN}"


def test_generated_block_reads_what_replace_block_writes(tmp_path):
    from eval.report import replace_block
    path = tmp_path / "R.md"
    path.write_text("intro\n", encoding="utf-8")
    replace_block(path, "x", "line 1\n\nline 2\n")
    assert _generated_block(path.read_text(encoding="utf-8"), "x") == "line 1\n\nline 2"
    assert _generated_block("no markers", "x") is None


def test_committed_results_md_is_reproduced_without_models(ctx, as_published, recorded):
    """Every generated table and computed sentence in RESULTS.md, rebuilt from the replay."""
    if _rerank_published(recorded):
        pytest.skip("the committed headline is the cross-encoder run, which cannot be replayed")
    text = (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    routing = ctx.of_kind("routing")
    parts = [run_eval.section(ds.title, as_published[ds.name], ds.rows, "hybrid") for ds in routing]
    parts += ["### Observations (computed)\n\n" + run_eval.observations(ctx, as_published, "hybrid"),
              run_eval.confidence_section(ctx, as_published, "hybrid"),
              run_eval.oos_section(ctx, as_published, "hybrid"),
              run_eval.multi_section(ctx, as_published, "hybrid"),
              run_eval.origin_section(ctx)]
    parts += [run_eval.breakdown(as_published["main"], routing[0].rows, "hybrid", f)
              for f in ("domain", "style", "set")]
    parts += [run_eval.misses(as_published[ds.name], ds.rows, "hybrid") for ds in routing]
    for part in parts:
        assert part in text, f"RESULTS.md differs from the code at: {part.splitlines()[0]!r}; {RERUN}"


def test_confidence_section_matches_metrics_json(ctx, as_published, recorded):
    md = run_eval.confidence_section(ctx, as_published, "hybrid")
    assert md.startswith("### Confidence vs correctness (heuristic levels, uncalibrated)")
    assert "not probabilities" in md and "#### Main set (n=88)" in md
    conf = recorded["metrics"]["main"]["hybrid"]["confidence"]
    routed = conf["by_action"]["route"]
    assert (f"SIE: hybrid RRF (--no-rerank) routes {routed['n']} of 88 queries without asking "
            f"(top-1 {routed['accuracy']:.3f} on those") in md
    high = conf["by_level"]["high"]
    assert f"{high['accuracy']:.3f} <sub>n={high['n']}</sub>" in md
    assert "Keyword router" not in md.split("\n\n", 2)[2]         # no confidence -> not listed
    # the a-priori claim covers only what was not changed; the post-hoc change is disclosed
    assert "fixed before any of these sets were scored" in md
    assert "Those rules and thresholds were fixed" not in md
    assert "the BM25-only rule was changed after the first measurement" in md
    assert md.endswith(run_eval.first_measurement_note())


def test_first_measurement_note_discloses_both_post_hoc_changes():
    note = run_eval.first_measurement_note()
    assert note.startswith("**Changed after the first measurement.**")
    assert "**not held-out measurements**" in note and "first measurement, of the earlier code" in note
    assert "abstained on 55.9% of the in-scope queries (62 of 111, main + source)" in note
    assert "now answers `low` (clarify)" in note
    assert ("intent recall 0.593, precision 0.803, exact 0.280, count match 0.480" in note
            and "SIE: hybrid RRF, mean over requests" in note)
    for change in ("routed verbatim", "routed as written instead", "capped", "noun homographs",
                   '"et al." never ends a sentence', '"etc." ends one unless a lowercase word follows'):
        assert change in note, change
    assert note == run_eval.first_measurement_note()               # constants: same every run


def test_oos_section_marks_systems_without_confidence(ctx, as_published, recorded):
    md = run_eval.oos_section(ctx, as_published, "hybrid")
    assert "### Out-of-scope queries (n=30)" in md and "Keyword router (baseline) †" in md
    assert "SIE: BM25 only †" not in md and "in-scope abstain (n=111)" in md
    share = recorded["metrics"]["out-of-scope"]["hybrid"]["by_action"]["abstain"]["share"]
    assert f"| **SIE: hybrid RRF (--no-rerank)** | {share:.3f} |" in md
    assert md.count("| unrelated |") == 10 + 1                   # 10 listed queries + category row


def test_multi_section_matches_metrics_json(ctx, as_published, recorded):
    md = run_eval.multi_section(ctx, as_published, "hybrid")
    comp = recorded["metrics"]["multi-intent"]["hybrid"]["compose"]
    row = (f"| (c) `compose()`: split into intents, route each | {comp['recall']:.3f} | "
           f"{comp['pooled_recall']:.3f} | {comp['precision']:.3f} | {comp['exact']:.3f} | "
           f"{comp['count_match']:.3f} |")
    assert row in md and "(oracle)" in md and "1 (controls)" in md
    assert md.count("| yes |") == round(comp["exact"] * 25)
    assert "averaged over requests" in md and "all 63 gold intents of the set" in md
    assert "so (c) there is not a held-out measurement" in md and "intent recall 0.593" in md


def test_origin_section_and_readme_claims(ctx, as_published):
    md = run_eval.origin_section(ctx)
    assert "No set here is human-collected production traffic" in md
    assert "| `main` | routing | 88 | scaffold 12, synthetic-llm 76 | `25bfe6665e7e` |" in md
    n_domains = len({s.domain for s in ctx.skills.values()})
    main = ctx.of_kind("routing")[0]
    assert run_eval._coverage_claim(ctx, main) == f"across all {n_domains} domains (every skill covered)"
    beyond = run_eval._beyond_top1(ctx, as_published, "hybrid")
    assert "**Out of scope (30 queries no skill fits):**" in beyond and "`compose()` covers" in beyond
    assert "on average" in beyond and "pooled" in beyond and "an oracle no caller has" in beyond


# -- multi-intent wording: macro mean vs pooled coverage ------------------------------------------

def _multi_ctx():
    """Two requests: 3 gold intents and 1 (a one-intent request weighs as much in the mean)."""
    rows = [{"query": "q1", "gold": [["a"], ["b"], ["c"]]}, {"query": "q2", "gold": [["e"]]}]
    ds = SimpleNamespace(name="m", kind="multi_intent", n=2, rows=rows, path="m.jsonl")
    ctx = SimpleNamespace(of_kind=lambda kind: [ds] if kind == "multi_intent" else [])
    system = next(s for s in run_eval.SYSTEMS if s.key == "hybrid")
    s = run_eval.Scored(system, ranked=[["a", "b", "x"], ["x"]], routes=True,
                        plans=[["a", "b"], ["e"]], parts=[2, 1])
    return ctx, {"m": [s]}, rows


def test_pooled_recall_counts_every_gold_intent_once():
    ctx, results, rows = _multi_ctx()
    per = run_eval._intent_rows(results["m"][0], rows)
    assert [r["recall"] for r in per["compose"]] == [pytest.approx(2 / 3), 1.0]
    assert run_eval.pooled_recall(per["compose"], rows) == 3 / 4          # mean would be 0.833
    assert run_eval.pooled_recall(per["single_top1"], rows) == 1 / 4
    assert run_eval.pooled_recall([], []) == 0.0


def test_beyond_top1_says_average_per_request_and_reports_pooled_and_oracle():
    ctx, results, _ = _multi_ctx()
    text = run_eval._beyond_top1(ctx, results, "hybrid")
    assert ("- **Multi-intent (2 requests, 4 gold intents):** `compose()` covers on average 83% of "
            "each request's gold intents (75% of all 4 pooled), with 50% exact plans; routing the "
            "whole request and taking the top-1 covers on average 17% (25% pooled; 0% exact). The "
            "same ranking's top-n with the true intent count (an oracle no caller has) covers on "
            "average 33% (50% pooled; 0% exact).") in text
    assert "not held-out" in text


# -- tie sensitivity of single-retriever rankings -----------------------------------------------

def _rs(slug, rank, raw, method):
    return RankedSkill(slug=slug, rank=rank, score=1.0 / (60 + rank), score_type="rrf",
                       methods=[method], evidence=[MethodEvidence(method=method, rank=rank, score=raw)])


class _Runner:
    """route() answers b, a, c with strictly decreasing RRF scores; b and a tie on raw score."""

    def __init__(self, mode, method, raw, reranked=False):
        self.mode, self.method, self.raw, self.reranked = mode, method, raw, reranked

    def route(self, query, k=10):
        results = [_rs(slug, i, score, self.method)
                   for i, (slug, score) in enumerate(zip("bac", self.raw), 1)]
        return RouteResult(query=query, results=results, confidence=Confidence("high", "route"),
                           mode=self.mode, reranked=self.reranked)


@pytest.mark.parametrize("mode, method, raw, reranked, tied", [
    ("sparse", "bm25", (7.25, 7.25, 3.0), False, 1),
    ("dense", "dense", (0.5123451, 0.5123449, 0.4), False, 1),   # equal at the fixture's 6 decimals
    ("dense", "dense", (0.51235, 0.51234, 0.4), False, 0),
    ("sparse", "bm25", (7.25, 7.25, 3.0), True, 0),              # cross-encoder scores: its own
    ("hybrid", "bm25", (7.25, 7.25, 3.0), False, 0),             # RRF over two lists: its own
])
def test_single_retriever_ties_are_counted_on_raw_scores(mode, method, raw, reranked, tied):
    system = next(s for s in run_eval.SYSTEMS if s.key == "bm25")
    s = run_eval._score_rows(system, _Runner(mode, method, raw, reranked), [{"query": "q", "gold": "a"}])
    assert s.ranked == [["b", "a", "c"]]                          # the ranking itself is untouched
    assert run_eval.tie_aware_top1(s, [{"gold": "a"}]) == (tied, 0.5 if tied else 0.0)


# -- which cross-encoder produced the numbers --------------------------------------------------------

def test_reranker_identity_is_machine_independent(tmp_path):
    assert run_eval.reranker_identity(None) is None
    hub = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert run_eval.reranker_identity(hub) == {"model": hub}
    assert run_eval.reranker_identity(str(LOCAL_CE_DIR))["model"] == LOCAL_CE_HINT
    assert run_eval.reranker_identity(str(ROOT / "data" / "models" / "other-ce")) == \
        {"model": "data/models/other-ce"}
    weights = tmp_path / "my-ce"
    (weights / "sub").mkdir(parents=True)
    (weights / "config.json").write_text("{}", encoding="utf-8")
    (weights / "sub" / "model.bin").write_bytes(b"\x00\x01")
    ident = run_eval.reranker_identity(str(weights))
    assert ident["model"] == "my-ce" and str(tmp_path) not in json.dumps(ident)
    assert len(ident["model_sha256"]) == 16 and ident == run_eval.reranker_identity(str(weights))
    (weights / "sub" / "model.bin").write_bytes(b"\x00\x02")                # other weights, same name
    assert run_eval.reranker_identity(str(weights))["model_sha256"] != ident["model_sha256"]


class _PendingRerankRouter:
    """A router whose cross-encoder cannot load, quoting the absolute local-weights path."""

    def __init__(self):
        self.reranker = SimpleNamespace(model_name=str(LOCAL_CE_DIR))

    def route(self, query, k=10):
        raise RerankerUnavailable(f"cannot load cross-encoder '{LOCAL_CE_DIR}' (test); run with --no-rerank")


def test_rerank_system_records_its_cross_encoder_in_metrics_and_provenance(ctx):
    system = replace(next(s for s in run_eval.SYSTEMS if s.key == "hybrid-rerank"),
                     make=lambda cfg=None: _PendingRerankRouter())
    results = run_eval.evaluate([system], ctx.datasets, run_eval.EvalConfig())
    s = results[ctx.datasets[0].name][0]
    assert s.ranked is None and s.model["model"] == LOCAL_CE_HINT
    doc = run_eval.metrics_document(ctx, results, run_eval.EvalConfig())
    assert doc["systems"]["hybrid-rerank"]["status"] == "pending"
    assert doc["systems"]["hybrid-rerank"]["model"] == LOCAL_CE_HINT
    text = run_eval.provenance(ctx, results[ctx.datasets[0].name], run_eval.EvalConfig(rerank_k=5))
    assert (f"- cross-encoder (SIE: hybrid + cross-encoder): `{LOCAL_CE_HINT}`") in text
    assert "over the chunks of the top-5 fused skills; not run, see pending below" in text
    assert f"cannot load cross-encoder '{LOCAL_CE_HINT}' (test)" in text
    assert str(LOCAL_CE_DIR) not in text and str(ROOT) not in text
    ran = replace(s, ranked=[[]], model={"model": "org/ce", "model_sha256": "0123456789abcdef"})
    line = run_eval.provenance(ctx, [ran], run_eval.EvalConfig())
    assert ("- cross-encoder (SIE: hybrid + cross-encoder): `org/ce` (files sha256 "
            "`0123456789abcdef`) over the chunks of the top-20 fused skills\n") in line
    assert "cross-encoder (" not in run_eval.provenance(ctx, [], run_eval.EvalConfig())


# -- the fixture records every text compose() may route ----------------------------------------------

def test_fixture_queries_include_every_text_compose_may_route(monkeypatch):
    rows = [{"query": "deploy a RAG pipeline and evaluate it"}]
    multi = SimpleNamespace(kind="multi_intent", rows=rows)
    routing = SimpleNamespace(kind="routing", rows=[{"query": "plain query"}])
    seen = []

    def routed_texts(query, max_intents=5):
        seen.append((query, max_intents))
        return ["evaluate it", "evaluate a RAG pipeline", query]

    monkeypatch.setattr(sie.compose, "routed_texts", routed_texts, raising=False)
    queries = fixture_queries([routing, multi], max_intents=4)
    assert {"evaluate it", "evaluate a RAG pipeline", "plain query", rows[0]["query"]} <= set(queries)
    assert seen == [(rows[0]["query"], 4)] and queries == sorted(set(queries))
    monkeypatch.delattr(sie.compose, "routed_texts", raising=False)
    assert "plain query" in fixture_queries([routing, multi])        # still works without it


def test_smoke_reports_a_stale_fixture_without_a_traceback(tmp_path, repo_cwd, recorded, capsys):
    (tmp_path / "metrics.json").write_text(run_eval.dumps_metrics(recorded), encoding="utf-8")
    stale = tmp_path / "fixture.json"
    stale.write_text(dumps(_fixture({}, fp="0000000000000000")), encoding="utf-8")
    assert run_eval.smoke(tmp_path, fixture_path=stale) == 1
    out = capsys.readouterr().out
    assert "[smoke] FAIL:" in out and "re-record: python -m eval.run_eval --record-fixture" in out


def test_collision_ranking_label_reports_what_actually_ranked():
    from eval.collision import ranking_label
    from sie.models import Confidence, RouteResult
    base = RouteResult(query="q", results=[], confidence=Confidence("none", "abstain"))
    assert ranking_label(replace(base, reranked=True)) == "cross-encoder"
    assert ranking_label(replace(base, rerank_error="no weights")) == "rrf (no reranker weights)"
    assert ranking_label(base) == "rrf"
