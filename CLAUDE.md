# CLAUDE.md — Working Agreement for the Skill Intelligence Engine

This file is read by any AI coding agent (Claude Code, Cursor, etc.) **before** editing this repo.
It encodes the conventions and guardrails. The task list lives in `KICKOFF_PROMPT.md`; the design
rationale lives in the build plan. Read all three, then work.

---

## What this repo is

A hybrid **RAG + GraphRAG** retrieval engine over the `ml-ai-skills` library (38 `SKILL.md` files).
It (a) retrieves the right skill *semantically* (dense ChromaDB + sparse BM25 → RRF → cross-encoder
rerank) and (b) generates dependency-ordered **learning paths** from the `requires`/`related`/
`conflicts` graph. It is a **library + small FastAPI service**, not a hosted product.

## Golden rules (guardrails)

- **G1 — Don't restructure.** Keep the package layout and module names in `sie/`, `eval/`, `tests/`,
  `demo/`. Fill in and harden; do not rename or move modules without flagging it first.
- **G2 — LLM-optional core.** Retrieval, GraphRAG, and every test must pass with **no API key**.
  `sie/llm.py` defaults to the `noop` provider — keep it that way. The LLM is only for the optional
  LLM-as-judge and path narration.
- **G3 — Lazy heavy imports.** `chromadb`, `sentence-transformers`, and the cross-encoder import
  **inside methods**, never at module top level, so `pytest` runs without them installed. Preserve this.
- **G4 — Tests never download models.** Mock the dense index / reranker in tests. CI must be fast and
  network-free. Test the retrieval/graph *logic*, not the embedding models.
- **G5 — Deterministic retrieval.** Fix seeds. For a fixed corpus + query, results must be reproducible.
- **G6 — No new top-level dependencies** beyond `requirements.txt` without calling it out in the commit
  message and updating the file. Prefer stdlib.
- **G7 — Honest evaluation.** No cherry-picking, no hard-coded "gold" answers in the engine. Metrics in
  `benchmarks/RESULTS.md` must be reproducible via `python -m eval.run_eval`.
- **G8 — Adapt, then note.** If reality contradicts the brief (frontmatter keys differ, the collision
  case moved), adapt the code and record the deviation in the commit message — don't force the plan.

## Conventions

- **Python 3.11**, type hints on all public functions, small single-purpose functions.
- **Dataclasses** (`sie/models.py`) are the shared vocabulary: `Skill`, `Chunk`, `Hit`. Reuse them;
  don't invent parallel dict shapes.
- **Docstrings**: one-line summary + Args/Returns for anything non-trivial.
- **CLI**: every runnable module exposes `python -m sie.<mod>` with `argparse`. Keep `--build` and
  add `--no-rerank` for ablation.
- **Retrieval contract**: `HybridRouter.retrieve(query, k)` returns a **deduped** list of `Hit`
  (one per skill, best section kept), highest score first.
- **Graph contract**: `requires` edges point **prereq → skill**; `learning_path()` returns a
  topological order ending at the target, with `related` as "see also" and `conflicts` flagged.

## Commit discipline

- Work in **small, verifiable increments**; one task from `KICKOFF_PROMPT.md` per commit group.
- Run `pytest` before each commit and paste the output. Green before you move on.
- Conventional-ish messages: `feat(router): add --no-rerank ablation flag`,
  `test(graph): cover cycle detection on real corpus`.

## Definition of done (mirror in every PR description)

- [ ] `python -m sie.ingest` loads all 38 skills cleanly.
- [ ] `python -m sie.router --build` + a query returns a ranked, deduped skill list.
- [ ] `python -m eval.run_eval` prints **SIE and baseline** metrics; `benchmarks/RESULTS.md` has real numbers + chart.
- [ ] The documented vocabulary-collision case is reproduced (keyword wrong / SIE correct top-1).
- [ ] `learning_path()` correct for ≥ 3 targets; `find_cycles()` empty on the real corpus.
- [ ] All FastAPI endpoints respond; `pytest` green locally and in CI with no model downloads.
- [ ] `README.md` sells it: architecture diagram, benchmark table, chart, quickstart.

## Where things live

```
KICKOFF_PROMPT.md   the ordered task list — start here
CLAUDE.md           this file — conventions & guardrails
README.md           public-facing: pitch, quickstart, benchmarks
sie/                core package (ingest, chunking, index/, rerank, router, graph/, llm, api)
eval/               queries.jsonl, metrics.py, judge.py, run_eval.py
benchmarks/         RESULTS.md (+ results.png you generate)
tests/              pytest — mock heavy models
data/skills/        vendor the ml-ai-skills corpus here
```

## Anti-patterns (do NOT do these)

- ❌ Importing `chromadb` / `sentence_transformers` at module top level (breaks G3/G4).
- ❌ Adding a database, message queue, Docker requirement, or cloud dependency.
- ❌ Making the LLM mandatory for retrieval or tests.
- ❌ Writing metrics by hand into `RESULTS.md` that `run_eval.py` can't reproduce.
- ❌ Silently dropping skills during ingest — always print a load report with any skips.
