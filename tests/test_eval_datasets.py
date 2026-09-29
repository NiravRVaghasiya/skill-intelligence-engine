"""eval/datasets.py: the query-set manifest, row validation, origins, and external sets."""
import json
from pathlib import Path

import pytest

from eval import run_eval
from eval.datasets import (DEFAULT_MANIFEST, KINDS, ORIGINS, DatasetSpec, load_dataset, load_datasets,
                           load_manifest, read_rows)
from sie.ingest import load_corpus

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / DEFAULT_MANIFEST
SLUGS = {s.slug for s in load_corpus(ROOT / "data" / "skills")}


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def _manifest(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _entry(name: str, rows_path: Path, kind: str = "routing", extra: str = 'origin = "synthetic-llm"') -> str:
    return (f'[[dataset]]\nname = "{name}"\npath = "{rows_path.as_posix()}"\nkind = "{kind}"\n'
            f"{extra}\n")


# -- the shipped manifest ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped(request):
    import os
    old = os.getcwd()
    os.chdir(ROOT)                           # manifest paths are relative to the repo root
    try:
        return load_datasets(MANIFEST, SLUGS, "ml-ai-skills")
    finally:
        os.chdir(old)


def test_shipped_manifest_lists_the_four_sets(shipped):
    datasets, skipped = shipped
    assert skipped == []
    assert [(d.name, d.kind, d.n) for d in datasets] == [
        ("main", "routing", 88), ("source", "routing", 23),
        ("out-of-scope", "out_of_scope", 30), ("multi-intent", "multi_intent", 25)]


def test_shipped_origins(shipped):
    by = {d.name: d.origins() for d in shipped[0]}
    assert by == {"main": {"scaffold": 12, "synthetic-llm": 76}, "source": {"source-authored": 23},
                  "out-of-scope": {"synthetic-llm": 30}, "multi-intent": {"synthetic-llm": 25}}
    assert not any("human-collected" in o for o in by.values())


def test_shipped_hashes_match_the_files(shipped):
    import hashlib
    for d in shipped[0]:
        assert d.sha256 == hashlib.sha256((ROOT / d.path).read_bytes()).hexdigest()
    main = shipped[0][0]
    assert main.sha256.startswith("25bfe6665e7e")            # frozen main set (RESULTS.md provenance)


def test_shipped_display_names_keep_the_report_headings(shipped):
    main, source = shipped[0][:2]
    assert main.title == "Main set — `eval/queries.jsonl`" and main.label == "Main set"
    assert main.chart_title == "Main set (n=88, eval/queries.jsonl)"
    assert source.chart_title == "Source-authored set (n=23, the ml-ai-skills repo's own evals/)"


def test_other_corpus_skips_labelled_sets(shipped):
    import os
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        datasets, skipped = load_datasets(MANIFEST, SLUGS, "someone-elses-skills")
    finally:
        os.chdir(old)
    assert datasets == [] and len(skipped) == 4
    assert "labels are for corpus 'ml-ai-skills', but 'someone-elses-skills' is loaded" in skipped[0]


# -- manifest parsing -------------------------------------------------------------------------

def test_defaults_for_display_names(tmp_path):
    rows = _jsonl(tmp_path / "q.jsonl", [{"query": "q", "gold": "rag-pipeline"}])
    spec = load_manifest(_manifest(tmp_path / "d.toml", _entry("mine", rows)))[0]
    ds = load_dataset(spec, SLUGS)
    assert ds.label == "mine" and ds.title == f"mine — `{rows.as_posix()}`"
    assert ds.chart_title == f"mine (n=1, {rows.as_posix()})"


@pytest.mark.parametrize("body, match", [
    ("", "no \\[\\[dataset\\]\\] tables"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\n", "'kind' is required"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\nkind = 'ranking'\n", "unknown kind 'ranking'"),
    ("[[dataset]]\nname = 'a'\nkind = 'routing'\n", "'path' is required"),
    ("[[dataset]]\npath = 'x'\nkind = 'routing'\n", "'name' is required"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\nkind = 'routing'\norigin = 'crowd'\n", "unknown origin 'crowd'"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\nkind = 'routing'\norgin = 'scaffold'\n", "unknown keys orgin"),
    ("[[dataset]]\nname = 'a'\npath = 3\nkind = 'routing'\n", "'path' must be a string"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\nkind = 'routing'\norigin_map = { a = 'scaffold' }\n",
     "'origin_map' needs 'origin_field'"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\nkind = 'routing'\norigin_field = 'set'\n"
     "origin_map = { a = 'nobody' }\n", "unknown origin 'nobody'"),
    ("[[dataset]]\nname = 'a'\npath = 'x'\nkind = 'routing'\n[[dataset]]\nname = 'a'\npath = 'y'\n"
     "kind = 'routing'\n", "duplicate dataset names: a"),
    ("[[dataset]\nname = 'a'\n", "invalid TOML"),
])
def test_invalid_manifests(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_manifest(_manifest(tmp_path / "d.toml", body))


def test_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.toml")


def test_constants():
    assert ORIGINS == ("scaffold", "synthetic-llm", "source-authored", "human-collected")
    assert KINDS == ("routing", "out_of_scope", "multi_intent")


# -- rows ---------------------------------------------------------------------------------------

def _spec(path: Path, kind: str = "routing", **kw) -> DatasetSpec:
    kw.setdefault("origin", "synthetic-llm")
    return DatasetSpec(name="t", path=str(path), kind=kind, **kw)


def test_unknown_gold_slugs_are_all_listed(tmp_path):
    rows = _jsonl(tmp_path / "q.jsonl", [{"query": "a", "gold": "rag-pipeline"},
                                         {"query": "b", "gold": ["nope-2", "rag-evaluation"]},
                                         {"query": "c", "gold": "nope-1"}])
    with pytest.raises(ValueError, match="gold slugs not in the corpus: nope-1, nope-2"):
        load_dataset(_spec(rows), SLUGS)
    assert load_dataset(_spec(rows), None).n == 3          # no corpus given: shape checks only


def test_multi_intent_slugs_are_checked_too(tmp_path):
    rows = _jsonl(tmp_path / "m.jsonl", [{"query": "a", "gold": [["rag-pipeline"], ["ghost"]]}])
    with pytest.raises(ValueError, match="not in the corpus: ghost"):
        load_dataset(_spec(rows, "multi_intent"), SLUGS)


@pytest.mark.parametrize("kind, row, match", [
    ("routing", {"query": "a", "gold": None}, "routing gold must be a slug"),
    ("routing", {"query": "a", "gold": []}, "routing gold must be a slug"),
    ("routing", {"query": "a", "gold": [["rag-pipeline"]]}, "routing gold must be a slug"),
    ("routing", {"query": "a"}, "missing 'gold'"),
    ("routing", {"query": "  ", "gold": "rag-pipeline"}, "'query' must be a non-blank string"),
    ("routing", {"gold": "rag-pipeline"}, "'query' must be a non-blank string"),
    ("out_of_scope", {"query": "a", "gold": "rag-pipeline"}, "out_of_scope gold must be null"),
    ("multi_intent", {"query": "a", "gold": ["rag-pipeline"]}, "multi_intent gold must be"),
    ("multi_intent", {"query": "a", "gold": [[]]}, "multi_intent gold must be"),
    ("multi_intent", {"query": "a", "gold": [["rag-pipeline"]], "n_intents": 2}, "n_intents=2"),
])
def test_row_shape_errors(tmp_path, kind, row, match):
    rows = _jsonl(tmp_path / "q.jsonl", [row])
    with pytest.raises(ValueError, match=match):
        load_dataset(_spec(rows, kind), SLUGS)


def test_bad_json_line_names_the_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"query": "a", "gold": "rag-pipeline"}\n\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"q.jsonl:3: invalid JSON"):
        read_rows(path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        read_rows(path)


def test_empty_set_is_an_error(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no rows"):
        load_dataset(_spec(path), SLUGS)


# -- origins ------------------------------------------------------------------------------------

def test_origin_field_map_and_row_override(tmp_path):
    rows = _jsonl(tmp_path / "q.jsonl", [
        {"query": "a", "gold": "rag-pipeline", "set": "old"},
        {"query": "b", "gold": "rag-pipeline", "set": "new"},
        {"query": "c", "gold": "rag-pipeline", "set": "new", "origin": "human-collected"}])
    spec = _spec(rows, origin="", origin_field="set",
                 origin_map=(("new", "synthetic-llm"), ("old", "scaffold")))
    ds = load_dataset(spec, SLUGS)
    assert ds.row_origins == ["scaffold", "synthetic-llm", "human-collected"]
    assert ds.origins() == {"scaffold": 1, "synthetic-llm": 1, "human-collected": 1}


def test_unmapped_origin_value_falls_back_or_fails(tmp_path):
    rows = _jsonl(tmp_path / "q.jsonl", [{"query": "a", "gold": "rag-pipeline", "set": "other"}])
    mapped = (("new", "synthetic-llm"),)
    with pytest.raises(ValueError, match="set='other' has no origin_map entry"):
        load_dataset(_spec(rows, origin="", origin_field="set", origin_map=mapped), SLUGS)
    ds = load_dataset(_spec(rows, origin="scaffold", origin_field="set", origin_map=mapped), SLUGS)
    assert ds.row_origins == ["scaffold"]


def test_row_level_origin_is_validated(tmp_path):
    rows = _jsonl(tmp_path / "q.jsonl", [{"query": "a", "gold": "rag-pipeline", "origin": "vibes"}])
    with pytest.raises(ValueError, match="unknown origin 'vibes'"):
        load_dataset(_spec(rows), SLUGS)


def test_no_origin_anywhere_is_an_error(tmp_path):
    rows = _jsonl(tmp_path / "q.jsonl", [{"query": "a", "gold": "rag-pipeline"}])
    with pytest.raises(ValueError, match="no origin"):
        load_dataset(_spec(rows, origin=""), SLUGS)


# -- an external set through the harness -------------------------------------------------------

def test_external_manifest_scores_through_the_harness(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)                 # the corpus and eval/baseline resolve from the repo root
    routing = _jsonl(tmp_path / "r.jsonl", [
        {"query": "evaluate the answers of my retrieval-augmented generation pipeline",
         "gold": "rag-evaluation"},
        {"query": "impute missing values and one-hot encode categorical columns",
         "gold": ["data-preprocessing", "feature-engineering"]}])
    oos = _jsonl(tmp_path / "o.jsonl", [{"query": "bake a sourdough loaf", "gold": None}])
    manifest = _manifest(tmp_path / "ext.toml",
                         _entry("ext", routing, extra='origin = "human-collected"\ncorpus = "ml-ai-skills"')
                         + _entry("ext-oos", oos, kind="out_of_scope"))
    ctx = run_eval.load_context(manifest)
    assert [d.name for d in ctx.datasets] == ["ext", "ext-oos"] and ctx.skipped == []
    systems = [s for s in run_eval.SYSTEMS if s.key in ("keyword", "bm25")]
    results = run_eval.evaluate(systems, ctx.datasets)
    doc = run_eval.metrics_document(ctx, results, run_eval.EvalConfig())
    assert doc["datasets"]["ext"]["origins"] == {"human-collected": 2}
    bm25 = doc["metrics"]["ext"]["bm25"]
    assert set(run_eval.METRICS) <= set(bm25) and "confidence" in bm25
    assert "confidence" not in doc["metrics"]["ext"]["keyword"]           # keyword: no confidence
    assert doc["metrics"]["ext-oos"]["keyword"]["by_action"]["route"]["share"] == 1.0
    assert "No set here is human-collected" not in run_eval.origin_section(ctx)


# -- the CLI with an external manifest ---------------------------------------------------------

def _small_manifest(tmp_path: Path) -> Path:
    rows = _jsonl(tmp_path / "r.jsonl", [{"query": "impute missing values before training",
                                          "gold": "data-preprocessing"}])
    return _manifest(tmp_path / "small.toml", _entry("small", rows))


def test_cli_partial_run_writes_only_to_an_explicit_out(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    manifest, out = _small_manifest(tmp_path), tmp_path / "out"
    run_eval.main(["--no-build", "--datasets", str(manifest), "--systems", "bm25,keyword",
                   "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["metrics.json", "per_query.jsonl"]
    doc = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(doc["metrics"]["small"]) == {"keyword", "bm25"}
    assert doc["metrics"]["small"]["bm25"]["top-1"] == 1.0
    row = json.loads((out / "per_query.jsonl").read_text(encoding="utf-8"))
    assert row["bm25"]["rank"] == 1 and row["bm25"]["confidence"] in ("high", "medium", "low", "none")
    assert "[eval] small set (n=1, routing)" in capsys.readouterr().out


def test_cli_partial_run_without_out_prints_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    written = []
    monkeypatch.setattr(run_eval, "_write", lambda *a, **kw: written.append(a))
    run_eval.main(["--no-build", "--datasets", str(_small_manifest(tmp_path)), "--systems", "bm25"])
    assert written == [] and "printing only" in capsys.readouterr().out


def _no_scoring(monkeypatch) -> list:
    """Skip the (model-backed) scoring; record what main() would write."""
    written = []
    monkeypatch.setattr(run_eval, "evaluate", lambda systems, datasets, cfg: {})
    monkeypatch.setattr(run_eval, "print_summary", lambda ctx, results: None)
    monkeypatch.setattr(run_eval, "_write", lambda ctx, results, cfg, out, partial, readme:
                        written.append((Path(out), partial, readme)) or [])
    return written


@pytest.mark.parametrize("extra", [["--pool", "10"], ["--rerank-k", "5"], ["--datasets", "SMALL"]])
def test_cli_non_default_run_never_overwrites_benchmarks(tmp_path, monkeypatch, capsys, extra):
    monkeypatch.chdir(ROOT)
    written = _no_scoring(monkeypatch)
    extra = [str(_small_manifest(tmp_path)) if a == "SMALL" else a for a in extra]
    run_eval.main(["--no-build", *extra])
    assert written == [] and "printing only" in capsys.readouterr().out
    run_eval.main(["--no-build", *extra, "--out", "benchmarks"])            # spelled out: same
    assert written == []
    run_eval.main(["--no-build", *extra, "--out", str(tmp_path / "o")])     # elsewhere: written
    assert written == [(tmp_path / "o", False, False)]                      # never the README


def test_cli_full_default_run_writes_benchmarks_and_readme(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    written = _no_scoring(monkeypatch)
    run_eval.main(["--no-build"])
    assert written == [(Path("benchmarks"), False, True)]
    assert "printing only" not in capsys.readouterr().out
    run_eval.main(["--no-build", "--reranker", "some/other-ce"])    # the model is not a config knob
    assert written[-1] == (Path("benchmarks"), False, True)


@pytest.mark.parametrize("argv, match", [(["--systems", "bm25,nope"], "unknown system"),
                                         (["--pool", "0"], "candidate_pool")])
def test_cli_rejects_bad_arguments(argv, match, capsys):
    with pytest.raises(SystemExit) as e:
        run_eval.main(argv)
    assert e.value.code == 2 and match in capsys.readouterr().err


def test_require_reranker_fails_before_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    written = []
    monkeypatch.setattr(run_eval, "_write", lambda *a, **kw: written.append(a))
    with pytest.raises(SystemExit) as e:
        run_eval.main(["--no-build", "--datasets", str(_small_manifest(tmp_path)), "--systems",
                       "bm25", "--require-reranker", "--out", str(tmp_path / "o")])
    assert "--require-reranker" in str(e.value.code) and written == []


def test_eval_config():
    assert run_eval.EvalConfig().is_default and run_eval.EvalConfig(pool=20).is_default
    assert not run_eval.EvalConfig(pool=40).is_default and not run_eval.EvalConfig(rerank_k=5).is_default
    assert run_eval.EvalConfig(reranker="x").is_default          # the model is not a retrieval knob
    assert run_eval.EvalConfig(pool=40, rerank_k=5).retrieval().candidate_pool == 40
