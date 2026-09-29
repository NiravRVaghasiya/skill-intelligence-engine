# CLAUDE.md — Working Agreement for the Skill Intelligence Engine

This file is read by any AI coding agent (Claude Code, Cursor, etc.) **before** editing this repo.
It encodes the conventions and guardrails. `README.md` gives the overview, headline results and limitations;
`benchmarks/RESULTS.md` says what is measured. Read both, then work.

---

## What this repo is

A **corpus-agnostic intelligence layer** over skill corpora (directories of `SKILL.md` files):
it indexes, retrieves (dense ChromaDB + sparse BM25 → RRF → optional cross-encoder), explains
(per-retriever evidence), grades (heuristic route / clarify / abstain confidence), composes
(multi-intent plans) and evaluates skills, and builds learning paths from typed relationships.
It is **not a skill library**: never add skill content here. The vendored corpus
(`data/skills/`, ml-ai-skills@8328c60, described by `data/skills/corpus.toml`) exists only so the
benchmarks are reproducible; nothing in `sie/` may depend on it. It is a **library + small
FastAPI service**, not a hosted product.

## Golden rules (guardrails)

- **G1 — Don't restructure.** Keep the package layout and module names in `sie/`, `eval/`, `tests/`,
  `demo/`. Fill in and harden; do not rename or move modules without flagging it first.
- **G2 — LLM-optional core.** Retrieval, graph reasoning, and every test must pass with **no API key**.
  `sie/llm.py` defaults to the `noop` provider — keep it that way. The engine never calls it; the only
  LLM use is the optional judge helper `eval/judge.py`.
- **G3 — Lazy heavy imports.** `chromadb`, `sentence-transformers`, and the cross-encoder import
  **inside methods**, never at module top level, so `pytest` runs without them installed. Preserve this.
- **G4 — Tests never download models.** Mock the dense index / reranker in tests. CI must be fast and
  network-free. Test the retrieval/graph *logic*, not the embedding models.
- **G5 — Deterministic retrieval.** Fix seeds. For a fixed corpus + query, results must be reproducible.
- **G6 — No new top-level dependencies** beyond `pyproject.toml` (core deps + optional extras) and
  `requirements.txt` without calling it out in the commit message and updating both. Heavy/optional
  libraries go in an extra, never in core. Prefer stdlib.
- **G7 — Honest evaluation.** No cherry-picking, no hard-coded "gold" answers in the engine. Metrics in
  `benchmarks/RESULTS.md` / `metrics.json` must be reproducible via `python -m eval.run_eval` (no
  timings there; latency lives in `benchmarks/PERFORMANCE.md`). Every query set declares its origin
  in `eval/datasets.toml`; never present LLM-written queries as collected traffic. Confidence
  thresholds and composition rules are not tuned on the eval sets; any change made after seeing
  results is disclosed in RESULTS.md.
- **G8 — Adapt, then note.** If reality contradicts the brief (frontmatter keys differ, the collision
  case moved), adapt the code and record the deviation in the commit message — don't force the plan.

## Conventions

- **Python 3.11**, type hints on all public functions, small single-purpose functions.
- **Dataclasses** (`sie/models.py`) are the shared vocabulary: `Skill`, `CorpusInfo`, `Chunk`, `Hit`,
  `Edge`, `RankedSkill`, `MethodEvidence`, `Confidence`, `RouteResult`, `Intent`, `PlanStep`,
  `Composition`. Reuse them; don't invent parallel dict shapes. API-facing shapes live only in
  `sie/schemas.py` (never return internal objects or chunk ids).
- **Docstrings**: one-line summary + Args/Returns for anything non-trivial.
- **CLI**: every runnable module exposes `python -m sie.<mod>` with `argparse`. Keep `--build` and
  add `--no-rerank` for ablation.
- **Retrieval contract**: `HybridRouter.route(query, k)` returns a `RouteResult` (deduped ranking with
  evidence, confidence, candidate counts, timings); `retrieve(query, k)` returns the same ranking as a
  **deduped** list of `Hit` (one per skill, best section kept), highest score first. Its results
  must not change without a benchmark rerun and an explanation.
- **Graph contract**: `requires` edges point **prereq → skill** and are the only prerequisite
  relation; `learning_path()` returns a topological order ending at the target, with `related` as
  "see also", `conflicts` flagged, reasons per step, and notes when graph information is missing.
  Never fabricate prerequisites.

## Commit discipline

- Work in **small, verifiable increments**; one logical change per commit group.
- Run `pytest` before each commit and paste the output. Green before you move on.
- Conventional-ish messages: `feat(router): add --no-rerank ablation flag`,
  `test(graph): cover cycle detection on real corpus`.

## Definition of done (mirror in every PR description)

- [ ] `python -m sie.ingest --expect 38` loads the vendored corpus cleanly (and any other corpus reports every skip).
- [ ] `python -m sie.router --build` + a query returns a ranked, deduped skill list.
- [ ] `python -m eval.run_eval` prints **SIE and baseline** metrics; `benchmarks/RESULTS.md` has real numbers + chart.
- [ ] `python -m eval.run_eval --smoke` reports 0 mismatches (what CI runs).
- [ ] The documented vocabulary-collision case is reproduced (keyword wrong / SIE correct top-1).
- [ ] `learning_path()` correct for ≥ 3 targets; `find_cycles()` empty on the real corpus.
- [ ] All FastAPI endpoints respond; `pytest` green locally and in CI with no model downloads.
- [ ] `README.md` results blocks are regenerated (not hand-edited) and its limitations match what is
      implemented and measured.

## Where things live

```
CLAUDE.md           this file — conventions & guardrails
README.md           public-facing: positioning, quickstart, generated results blocks, limitations
sie/                core package (corpus, ingest, chunking, index/, rerank, hub, router, confidence,
                    compose, graph/, engine, api, schemas, observability, models, llm)
eval/               datasets.toml + query sets, datasets, metrics, report, run_eval, replay, perf,
                    collision, build_source_queries, judge (optional), baseline/ (vendored keyword router)
benchmarks/         RESULTS.md, metrics.json, per_query.jsonl, collision.json, performance.json,
                    PERFORMANCE.md, charts (all generated)
tests/              pytest — heavy models mocked and blocked; fixtures/; integration/ (opt-in)
docs/               PROPOSED_REQUIRES.md + proposed_requires.edges.json (opt-in edge overlay)
data/skills/        the vendored default corpus (+ corpus.toml); any corpus works via --skills
data/chroma/        built dense index (git-ignored)
demo/               optional Streamlit UI
scripts/            setup_corpus.py (vendor a corpus + build the index)
Makefile, setup.bat dev shortcuts (install, build, eval, smoke, test, serve) / Windows setup
.github/workflows/  tests.yml — CI: ingest --expect 38, pytest, run_eval --smoke
.env.example        every SIE_* setting the API reads
```

## Anti-patterns (do NOT do these)

- ❌ Importing `chromadb` / `sentence_transformers` at module top level (breaks G3/G4).
- ❌ Adding a database, message queue, Docker requirement, or cloud dependency.
- ❌ Making the LLM mandatory for retrieval or tests.
- ❌ Writing metrics by hand into `RESULTS.md` that `run_eval.py` can't reproduce.
- ❌ Silently dropping skills during ingest — always print a load report with any skips.
- ❌ Adding skill content, or engine code that assumes one particular corpus.
- ❌ Logging raw user queries by default (`SIE_LOG_QUERIES=1` is the opt-in).
