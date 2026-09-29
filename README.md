# Skill Intelligence Engine

**A corpus-agnostic retrieval, routing, evaluation, and graph-reasoning engine for AI agent skills.**

[![tests](https://github.com/NiravRVaghasiya/skill-intelligence-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/NiravRVaghasiya/skill-intelligence-engine/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Agent systems increasingly have access to many *skills*: packaged instructions, usually one
`SKILL.md` file (Markdown with YAML frontmatter) per skill. As a collection grows, finding the
right skill for a natural-language task gets harder, partly because people describe the problem
("our model's precision keeps sliding since launch") while skills are named for the solution
(`ml-monitoring`). The Skill Intelligence Engine (SIE) is a Python library, CLI and small HTTP
service that sits between a skill corpus and an agent and covers this path:

```text
natural-language task
        ↓
skill retrieval             BM25 and dense embeddings over every section of every skill
        ↓
ranking                     reciprocal rank fusion (optional cross-encoder, not benchmarked)
        ↓
confidence / alternatives   route · clarify · abstain, with per-retriever evidence
        ↓
skill composition           multi-part request → ordered skill plan (opt-in)
        ↓
prerequisite reasoning      declared `requires` relationships → learning path
```

**ml-ai-skills provides skills; skill-intelligence-engine provides the intelligence layer for
working with skill collections.** The engine operates *on* a skill corpus; it is not one. Any
directory of `SKILL.md` files can be plugged in: a public repository such as
[ml-ai-skills](https://github.com/NiravRVaghasiya/ml-ai-skills), another source, or a
private/internal corpus. The only skill content in this repository is a pinned copy of
ml-ai-skills under `data/skills/`, kept so that the benchmarks are reproducible.

```text
                   Skill Corpus
              ┌────────┼─────────┐
              │        │         │
         ml-ai-skills  Other   Internal
              │       sources   corpus
              └────────┬─────────┘
                       │
                       ▼
            Skill Intelligence Engine
                       │
          ┌────────────┼────────────┐
          │            │            │
       Retrieve      Rank        Reason
          │            │            │
          └────────────┼────────────┘
                       │
                Compose / Route
                       │
                       ▼
                  AI Agent
```

The three sources are alternatives: one engine instance reads one corpus at a time.

**Contents:** [Repository vs. engine](#skill-repository-vs-skill-intelligence-engine) ·
[Why](#why-this-exists) · [Architecture](#architecture) · [Retrieval](#retrieval) ·
[Graph reasoning](#graph-reasoning) · [Evaluation](#evaluation) ·
[What the results mean](#what-the-results-mean) · [Use cases](#supported-use-cases) ·
[What it is not](#what-this-project-is-not) · [Another corpus](#using-another-corpus) ·
[Quick start](#quick-start) · [HTTP API](#http-api) · [Python API](#python-api) ·
[Structure](#project-structure) · [Testing](#development-and-testing) · [Status](#status) ·
[Limitations](#limitations) · [Roadmap](#roadmap) · [ml-ai-skills](#related-ecosystem)

---

## Skill repository vs. Skill Intelligence Engine

A skill repository and this engine sit at different layers.

| | Skill repository | Skill Intelligence Engine |
|---|---|---|
| Primary purpose | Provide skill content | Find, rank and reason over skills |
| Contains skills | Yes | Not inherently: it reads the corpus it is pointed at |
| Retrieval | Usually the consumer's responsibility (sometimes a lightweight helper ships with it) | Core capability: BM25, dense embeddings, rank fusion |
| Ranking | Usually the consumer's responsibility | Core capability, with per-retriever evidence and a confidence action |
| Evaluation | Optional; typically of the skill content itself | First-class in this repository's harness (`eval/`): retrieval and routing scored over labeled query sets |
| Graph reasoning | Optional; relationships declared in frontmatter | Supported: a typed graph built from those declarations |
| Prerequisite reasoning | Optional; prerequisites declared per skill | Supported: resolved into ordered learning paths |
| Corpus-agnostic | Depends | Yes: any directory of `SKILL.md` files, with an optional `corpus.toml` |
| Can consume external skill repositories | N/A | Yes |

The two columns are different layers, not alternatives: the engine has nothing to work on
without a corpus.

[ml-ai-skills](https://github.com/NiravRVaghasiya/ml-ai-skills) is an example of a skill
corpus. This project can index and reason over such a corpus without becoming the corpus
itself: the corpus supplies the skills and the relationships between them (`requires`,
`related`, …), and the engine consumes them. The pinned copy in `data/skills/` (commit
`8328c60`, 38 skills) makes the benchmarks reproducible and is the default corpus of the CLI,
`Engine()` and the API; no code in the engine package (`sie/`) depends on its content.

## Why this exists

Take a request an agent might receive:

```text
"Help me fine-tune a transformer model and evaluate it."
```

Several skills in the vendored corpus could plausibly apply (`fine-tuning-llms`,
`llm-evaluation`, `model-evaluation`, `nlp-tasks`, `attention-mechanisms`,
`training-deep-models`), and the request contains two tasks. Matching words is not enough:
"model" occurs in 37 of the 38 skills, "transformer" in 8 and "evaluate" in 7. This is what the
engine currently returns for it, with hybrid retrieval and no reranker:

```text
$ python -m sie.router "Help me fine-tune a transformer model and evaluate it." --no-rerank -k 3 --explain
[router] mode=hybrid ranking=rrf
 1. fine-tuning-llms             score=0.031  [Card]
      dense  #7   cosine=0.280  [Overview]
      bm25   #2   bm25=13.472  [Card]  terms: fine, tune, a, model, and
 2. llm-evaluation               score=0.031  [Workflow]
      dense  #4   cosine=0.326  [Workflow]
      bm25   #5   bm25=8.568  [References]  terms: a, and, evaluate
 3. nlp-tasks                    score=0.031  [Workflow]
      dense  #10  cosine=0.250  [Workflow]
      bm25   #1   bm25=13.845  [Workflow]  terms: fine, tune, a, transformer, model, and, it
[router] confidence: low (clarify): dense similarity 0.28 < 0.30; dense ranks model-evaluation first; bm25 ranks nlp-tasks first

$ python -m sie.router "Help me fine-tune a transformer model and evaluate it." --no-rerank --multi
[compose] 2 intents -> 2 skills in the plan
  intent 1: 'Help me fine-tune a transformer model' -> fine-tuning-llms [medium/clarify]
  intent 2: 'evaluate it' -> llm-evaluation [medium/clarify]
  1. fine-tuning-llms           requested    requested by intent 1; ambiguous for intent 1 (vs attention-mechanisms, nlp-tasks)
  2. llm-evaluation             requested    requested by intent 2; ambiguous for intent 2 (vs nlp-tasks, model-evaluation)
  note: intent 1 ('Help me fine-tune a transformer model') is ambiguous: fine-tuning-llms vs attention-mechanisms, nlp-tasks; confirm before acting
  note: intent 2 ('evaluate it') is ambiguous: llm-evaluation vs nlp-tasks, model-evaluation; confirm before acting
```

The two retrievers disagree on this request, and the answer says so: each position comes with
the evidence behind it, and the action is *clarify*, with the competing skills named, rather
than a confident guess.

Keyword matching struggles most when a request describes a symptom instead of naming a
technique. Two queries from the benchmark's main set (rank of the correct skill in each
system's ranking; – means not in its top 10):

| query | correct skill | keyword baseline | BM25 | dense | hybrid RRF (confidence) |
|---|---|:---:|:---:|:---:|:---:|
| "Our fraud model went live six months ago and its precision has been slowly sliding even though nobody touched the code." | `ml-monitoring` | – | 3 | 1 | 1 (medium) |
| "New items added to our catalog never get recommended because nobody has interacted with them yet. How do I get around that?" | `recommender-systems` | – | – | 1 | 5 (low) |

The first is the case the engine is designed for: the query shares no words with the skill's
name (`ML Monitoring`), and the semantic retriever recovers it. The second is a case where
fusion hurts: dense ranks the skill first but BM25 only 14th, and with RRF's constant of 60
that sum (1/61 + 1/74 ≈ 0.0299) falls just below four skills that both retrievers placed in
their top 8, so the skill lands 5th; the confidence level is `low`. Per-query results for every
system are in [`benchmarks/per_query.jsonl`](benchmarks/per_query.jsonl).

## Architecture

```text
                    Skill corpus
(directory of SKILL.md files + optional corpus.toml)
                          │
                  Ingestion / parse ────────────────────┐
                          │                             │
                      Chunking                  Typed skill graph
                          │                  (requires, related, …)
             ┌────────────┴────────────┐                │
             │                         │                │
           BM25                Dense embeddings         │
             │                         │                │
             └────────────┬────────────┘                │
                          │                             │
               Reciprocal Rank Fusion                   │
                          │                             │
              Cross-encoder (optional)                  │
                          │                             │
             Skill ranking + confidence                 │
                          │                             │
             ┌────────────┴────────────┐                │
             │                         │                │
   Routing, composition         Graph reasoning ◄───────┘
             │                         │
             └────────────┬────────────┘
                          │
                 Agent / application
              (CLI · Python · HTTP API)
```

Learning paths use only the graph; they need no index and no model.

| stage | module | what it does |
|---|---|---|
| Corpus loading | `sie/corpus.py`, `sie/ingest.py` | Reads an optional `corpus.toml`, parses YAML frontmatter, records per-skill provenance and a content fingerprint; every skipped file is reported with a reason |
| Chunking | `sie/chunking.py` | One *Card* chunk per skill (display name, description without a trailing "NOT for …" clause, capability tags) plus chunks for each `##` section; sections longer than about 1,200 characters are split, never inside a fenced code block (152 sections → 259 chunks in the vendored corpus). Each section chunk starts with a `<display name> - <section>` line |
| Lexical retrieval | `sie/index/sparse.py` | BM25 over all chunks, built in memory |
| Semantic retrieval | `sie/index/dense.py` | Sentence embeddings in a persisted ChromaDB collection |
| Fusion | `sie/index/fuse.py` | Reciprocal rank fusion, one entry per skill |
| Reranking (optional) | `sie/rerank.py` | Cross-encoder over the retrieved chunks of the top fused skills; on by default when it can load, not benchmarked (see [below](#optional-cross-encoder-reranker)) |
| Routing | `sie/router.py` | `HybridRouter.route()`: ranked skills with evidence, confidence, candidate counts and per-stage timings |
| Confidence | `sie/confidence.py` | Heuristic route / clarify / abstain |
| Composition | `sie/compose.py` | Splits a multi-part request, routes each part, adds declared prerequisites, orders the plan |
| Graph | `sie/graph/` | Typed edges with provenance, overlays, learning paths, cycle detection |
| Service | `sie/engine.py`, `sie/api.py`, `sie/schemas.py` | One `Engine` with an explicit start-up; FastAPI endpoints with response models |
| Observability | `sie/observability.py` | Structured events and hooks; query text hashed by default |

For a fixed corpus, index and query, results are deterministic: seeds are fixed, ties are
broken explicitly (alphabetically by skill id after fusion), and ChromaDB's approximate
nearest-neighbour search (HNSW) is configured to inspect 400 candidates (`ef_search`), more than
the 297 indexed chunks. Where the cross-encoder cannot load, as when the committed benchmark
files were produced, re-running the evaluation reproduces `benchmarks/metrics.json` and
`benchmarks/per_query.jsonl` byte for byte; with loadable weights the run also scores the
reranked system, which adds rows and changes the headline.

Heavy libraries (`chromadb`, `sentence-transformers`, `torch`) are imported lazily. The core
(ingestion, BM25 routing, confidence, the graph, learning paths, composition) needs only
`python-frontmatter`, `networkx`, `rank-bm25` and `numpy`.

## Retrieval

Both retrievers index the same chunks. Each returns its top 20 chunks (`--pool`), which are
collapsed to one entry per skill (its best chunk) before fusion.

### BM25 (lexical)

Exact terms are strong evidence when they occur: library and API names (`XGBoost`,
`SettingWithCopyWarning`), acronyms (`LoRA`, `SHAP`, `PR-AUC`), metric names. BM25 weights rare
terms above common ones and needs no model. The engine uses `rank-bm25` (`BM25Okapi` with the
library's default parameters) over lowercased ASCII alphanumeric runs (`[a-z0-9]+`, so `PR-AUC`
becomes `pr` + `auc` and non-ASCII letters split words), without stop-word removal or
stemming. The index is built in memory the first time the router loads the corpus (at service
start-up or on the first query) and is never persisted.

### Dense embeddings (semantic)

Requests often describe a situation in words no skill uses. Sentence embeddings match on
meaning instead of shared tokens. The engine embeds every chunk with `all-MiniLM-L6-v2`
(384 dimensions), by default through chromadb's ONNX embedding function, which downloads the
model once on first use (`SIE_EMBEDDER=sentence-transformers` switches backends), and stores
the vectors in a persisted ChromaDB collection with cosine distance. The index records a
fingerprint of the chunk texts and the embedder, and the router refuses to serve a stale index
and asks for `python -m sie.router --build`. The model reads at most 256 tokens, so long chunks
are embedded by their beginning only; BM25 sees the whole chunk.

### Reciprocal Rank Fusion

Fusion combines the two rankings instead of choosing one retriever:

```text
score(skill) = Σ over retrievers  1 / (60 + rank of the skill in that retriever's list)
```

RRF uses ranks, not raw scores, so BM25 scores and cosine similarities never have to be put on
one scale. Exact ties are broken alphabetically by skill id. The motivation is empirical, not a
claim that one method is superior. On the included sets BM25 alone is the stronger single
retriever on the main set (MRR 0.949 vs 0.927 for dense) and dense alone on the
source-authored set (0.906 vs 0.749). Hybrid RRF has the best MRR on the main set (0.972) but
not everywhere: on the source-authored set dense alone has a slightly higher MRR (0.906 vs
0.895) and the same top-1, and the `recommender-systems` query above is one that fusion ranks
lower.

### Optional cross-encoder reranker

The router can have a cross-encoder (`sie/rerank.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
re-score the retrieved chunks (those in either retriever's pool) of the top `--rerank-k` fused
skills (default: the pool size) and rank each skill by its best chunk. It is implemented and
unit-tested with stubs, but it has **not been evaluated with real weights**: `benchmarks/`
holds no quality or latency numbers for it (the weights could not be loaded where the
benchmarks ran), its benchmark row reads *pending*, and the headline results use hybrid RRF
**without** it.

It is optional in that it needs an extra dependency and weights, but it is enabled by default in
the CLI, in `Engine(...)` and in the API (`SIE_RERANK=1`). When it cannot load
(`sentence-transformers` not installed; weights neither local nor cached while the Hugging Face
hub is unreachable; a `SIE_RERANKER` path that does not exist; or a load error), the router
prints a warning and falls back to RRF order (`reranked: false`; the API reports `degraded`).
When it can load, the default path uses it (downloading the weights if they are not cached),
which is an unmeasured configuration. Pass `--no-rerank` (CLI) or set `SIE_RERANK=0` (API) for
the benchmarked configuration. The weights come from one source, chosen by precedence:
`$SIE_RERANKER` (a Hugging Face id or a local directory) if set, else
`data/models/ms-marco-MiniLM-L-6-v2/` if it exists, else the Hugging Face hub; if that source
fails, the router falls back to RRF order. With the reranker on, results are limited to the
`rerank_k` skills it scored.

### Evidence and confidence

Every result carries per-retriever evidence: the skill's rank within each retriever, the raw
score (cosine or BM25), the best-matching section and, for BM25, the matched query terms.

```text
$ python -m sie.router "impute missing values and encode categoricals" --no-rerank -k 3 --explain
[router] mode=hybrid ranking=rrf
 1. data-preprocessing           score=0.033  [Card]
      dense  #1   cosine=0.477  [Card]
      bm25   #1   bm25=22.700  [Card]  terms: impute, missing, values, encode
 2. feature-engineering          score=0.032  [Workflow]
      dense  #2   cosine=0.438  [Workflow]
      bm25   #2   bm25=9.642  [Workflow]  terms: values, and, encode, categoricals
 3. ml-problem-framing           score=0.030  [Workflow]
      dense  #9   cosine=0.226  [Workflow]
      bm25   #5   bm25=5.607  [Workflow]  terms: missing, values, and
[router] confidence: high (route): dense and bm25 rank data-preprocessing first
```

The top result is graded by rules over that evidence (`sie/confidence.py`; the first match
wins):

| level | action | when |
|---|---|---|
| `none` | abstain | nothing was retrieved; or the dense retriever ran, the best cosine of *any* candidate is below 0.30 **and** the top result's coverage is below 0.40 |
| `low` | clarify | the top result's own evidence is weak (its cosine is below 0.30; in BM25-only mode, its coverage is below 0.40), or a retriever did not retrieve it |
| `medium` | clarify | another skill is some retriever's #1, or the top two scores tie (exactly; or within 3% in a single-retriever ranking that the cross-encoder did not reorder); `competitors` names them |
| `high` | route | every retriever ranks it first and its evidence is not weak |

*Coverage* is the share of the query's term weight (BM25 idf) that appears in the best of the
top result's retrieved chunks.

```text
$ python -m sie.router "what's a good pizza place in Naples" --no-rerank -k 3
[router] mode=hybrid ranking=rrf
 1. ml-problem-framing           score=0.032  [Card]
 2. cnn-vision                   score=0.030  [Key Concepts]
 3. reinforcement-learning       score=0.030  [Key Concepts]
[router] confidence: none (abstain): no sufficiently good match: best dense similarity 0.12 < 0.30; only 26% of the query's term weight matches the best candidate (< 40%)
```

These levels are **heuristic and uncalibrated**: rules, not probabilities (`POST /route`
responses carry `calibrated: false`). According to [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md),
the thresholds were fixed before the query sets were scored and are not tuned on them; how often
each level is right is measured there, per set. Abstaining does not empty the ranking, so an
agent should act on `confidence.action`. A BM25-only router abstains only when nothing matches
at all, because lexical coverage alone cannot separate "no skill fits" from "a long in-scope
request"; this rule was changed after its first measurement, and RESULTS.md discloses the
earlier numbers.

## Graph reasoning

The graph is a separate layer, built from relationships the corpus declares in frontmatter. It
does **not** influence retrieval: candidate generation, fusion, reranking and confidence never
read it. It is used in four places, none of which changes a ranking:

- direct prerequisites attached to each result of `POST /route` and `/batch-route`;
- prerequisite expansion and ordering in multi-intent composition;
- learning paths: target skill → its prerequisites → an ordered path;
- skill lookups: `requires`, `required_by`, see-also, conflicts and every typed relation in
  `GET /skill/{slug}`.

| relationship (declared on skill X, listing R) | edge | effect |
|---|---|---|
| `requires` | R → X | **hard prerequisite**; the only relation whose skills are pulled into a plan, or into a path by default |
| `recommended_before` | R → X | soft: reorders path members; inserted as a step only with `include_recommended` |
| `related` | X → R | "see also" |
| `conflicts` | X ↔ R | flagged: a learning path lists skills that conflict with any member (and warns about conflicting pairs on it); a plan notes conflicting pairs among its members |
| `alternative_to` | X ↔ R | reported in relations (`/skill/{slug}`) |
| `specializes` | X → R | reported in relations |
| `supersedes` | X → R | noted when R is on a path or in a plan |

Every edge carries a `confidence` (`declared` for frontmatter) and a `provenance` string such as
`ml-ai-skills@8328c60:rag-evaluation/SKILL.md#requires`. *Overlays*, JSON files of extra edges
(`--overlay`, `SIE_EDGE_OVERLAYS`), add edges without editing the corpus; steps that depend on
them are marked with the overlay's confidence, such as `proposed`.

`learning_path()` returns the transitive `requires` closure of the target in a deterministic
topological order that ends at the target, with a reason and provenance for every step, "see
also" skills, conflicts, and notes where the declarations are incomplete. It never invents
prerequisites. A `requires` cycle is an error; `find_cycles()` reports cycles, and the
vendored corpus has none.

```text
$ python -m sie.router --path rag-evaluation
[path] learning path to rag-evaluation (2 steps, hard `requires` edges only)
  1. rag-pipeline               [workflow/llm, intermediate]
  2. rag-evaluation             [workflow/llm, intermediate]  <- target
  see also:  llm-evaluation, model-evaluation, ai-ml-security
  conflicts: none
  why: rag-pipeline: direct prerequisite of rag-evaluation

$ python -m sie.router --path training-deep-models
[path] learning path to training-deep-models (1 step, hard `requires` edges only)
  1. training-deep-models       [workflow/deep-learning, intermediate]  <- target
  see also:  neural-net-fundamentals, pytorch-patterns, hyperparameter-tuning
  conflicts: none
  note: training-deep-models declares no prerequisites

$ python -m sie.router --path training-deep-models --overlay docs/proposed_requires.edges.json
[path] learning path to training-deep-models (2 steps, hard `requires` edges only)
  1. pytorch-patterns           [workflow/deep-learning, intermediate]
  2. training-deep-models       [workflow/deep-learning, intermediate]  <- target
  see also:  neural-net-fundamentals, hyperparameter-tuning
  conflicts: none
  why: pytorch-patterns: direct prerequisite of training-deep-models (proposed)
```

**Graph density is currently low.** The vendored corpus declares 2 `requires` edges
(`agents-and-tools → agent-evaluation`, `rag-pipeline → rag-evaluation`), 138 `related`
references and no edges of the other five types. As a result, 36 of the 38 learning paths are
the target alone, and conflict flags, soft ordering and supersession notes are exercised only
by synthetic test fixtures. [`docs/PROPOSED_REQUIRES.md`](docs/PROPOSED_REQUIRES.md) drafts 9
further `requires` edges for upstream review, each backed by a quotation from the dependent
skill's text; [`docs/proposed_requires.edges.json`](docs/proposed_requires.edges.json) applies
them as an opt-in overlay, as in the last example. The `complete` flag on a learning path
reports gaps in the declarations or in how the path was built (a member without a `requires`
key, a prerequisite outside the corpus, or, with `include_recommended`, an inserted skill whose
own prerequisites were left off), not whether the listed prerequisites are sufficient. Every
vendored skill declares a `requires` key, mostly empty, so the flag is `true` for all 38.

*Terminology:* some code comments and the package keywords say "GraphRAG". In this repository
the term means only what this section describes, a typed graph over declared skill
relationships used after retrieval. There is no graph-based retrieval, entity extraction or text
generation.

## Evaluation

The harness in `eval/` (repository tooling, not part of the installed package) scores every
system on every query set. It writes the generated tables of
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) (with confidence intervals and every query the
headline system misses in its top 3), `metrics.json`, `per_query.jsonl` (every system's rank
for every query) and the charts. The routing, out-of-scope and multi-intent numbers below come
from that run; the collision case comes from `python -m eval.collision` and the latency figures
from `python -m eval.perf`.

**Corpus.** ml-ai-skills at commit `8328c60`, vendored in `data/skills/`: 38 skills in 8
domains, 297 chunks (content fingerprint `7d62d861fc8ce6a7`).

**Systems.**

| system | what it is |
|---|---|
| Keyword router (baseline) | The corpus's own `scripts/router.py`, vendored byte for byte in `eval/baseline/` (hash-checked by the tests) and scored as a full ranking of all 38 skills, without its minimum-score cut or prerequisite expansion. Its `route()` exactly as shipped is an extra row in RESULTS.md. |
| SIE: BM25, Card chunks only | Ablation: BM25 over roughly the fields the keyword router reads |
| SIE: BM25 only | `--mode sparse --no-rerank` |
| SIE: dense only | `--mode dense --no-rerank` |
| **SIE: hybrid RRF (`--no-rerank`)** | The headline configuration |
| SIE: hybrid + cross-encoder | Not run, since no reranker weights were available where the benchmark was produced: *pending* |

The keyword router is the natural reference point because it ships with the corpus and scores
the same skills by keyword overlap with their names, descriptions and capability tags, plus a
bonus when a skill's type (workflow or reference) matches the request's classified intent. It is
dependency-free by design; the corpus's routing notes
([`docs/ROUTING.md`](https://github.com/NiravRVaghasiya/ml-ai-skills/blob/main/docs/ROUTING.md)
in ml-ai-skills) call it "deliberately the simplest thing that could plausibly work". The
comparison measures retrieval techniques on one corpus, not the corpus repository.

**Query sets**, each declared with its origin in [`eval/datasets.toml`](eval/datasets.toml):

| set | n | kind | who wrote the queries |
|---|:---:|---|---|
| main (`eval/queries.jsonl`) | 88 | routing | 76 by LLM agents given only the skill catalog, audited by separate agents (`synthetic-llm`); 12 pre-existing scaffold queries (`scaffold`, process unrecorded) |
| source-authored (`eval/queries_source_evals.jsonl`) | 23 | routing | the corpus repository's own behavioral eval cases: the input of every case in ml-ai-skills' `evals/`, with its skill labels verbatim (`source-authored`; see [below](#the-source-authored-set)) |
| out-of-scope (`eval/queries_out_of_scope.jsonl`) | 30 | no skill fits | an LLM agent working from the catalog, checked by an adversarial auditor (`synthetic-llm`) |
| multi-intent (`eval/queries_multi_intent.jsonl`) | 25 requests, 63 intents | multi-intent | an LLM agent working from the catalog, checked by an auditor (`synthetic-llm`) |

No set is independently collected user traffic. The writing and auditing protocol, and the
statement that each synthetic set was frozen before any system was scored on it, come from the
authoring record in RESULTS.md; the repository pins each file by sha256 but holds no other
record of that process.

**Metrics.** For routing sets: top-1 (= recall@1), recall@3, recall@5, MRR (mean reciprocal
rank: the average of 1/rank) and nDCG@3 / nDCG@5 (normalized discounted cumulative gain, which
gives lower ranks less credit), all from the rank of the first acceptable skill in each system's
top 10; 95% percentile-bootstrap intervals (10,000 resamples, seed 42); paired deltas against
the baseline; tie sensitivity. Beyond ranking: accuracy per confidence level; action shares on
out-of-scope queries (abstain is correct, clarify acceptable, route wrong); intent recall,
precision and exact match on multi-intent requests.

### Current results

On the current 88-query / 38-skill benchmark (the main set), hybrid BM25 + dense retrieval with
RRF and no reranker achieved **95.5% top-1 retrieval** (95% interval 91–99%). On the same
queries the keyword baseline achieved 43.2%, BM25 alone 92.0% and dense alone 88.6%. The
summary below is generated by `python -m eval.run_eval` and checked by the test suite:

<!-- BEGIN:headline -->
**Top-1 routing accuracy goes from 43.2% to 95.5%** on 88 labeled queries (keyword router -> SIE: hybrid RRF (--no-rerank)), and from 34.8% to 82.6% on the 23 queries taken from the source repo's own eval cases. _Generated by `python -m eval.run_eval`._
<!-- END:headline -->

<!-- BEGIN:benchmarks -->
Main set: 88 labeled queries across all 8 domains (every skill covered).

| System | top-1 | recall@3 | recall@5 | mrr | ndcg@3 | ndcg@5 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Keyword router (baseline) | 0.432 | 0.670 | 0.761 | 0.573 | 0.572 | 0.610 |
| SIE: BM25, Card chunks only | 0.727 | 0.864 | 0.898 | 0.806 | 0.810 | 0.825 |
| SIE: BM25 only | 0.920 | 0.989 | 0.989 | 0.949 | 0.959 | 0.959 |
| SIE: dense only | 0.886 | 0.977 | 0.989 | 0.927 | 0.938 | 0.942 |
| **SIE: hybrid RRF (--no-rerank)** | 0.955 | 0.989 | 1.000 | 0.972 | 0.975 | 0.979 |
| SIE: hybrid + cross-encoder | pending | pending | pending | pending | pending | pending |

On the 23 queries taken from the source repo's own `evals/` cases: keyword top-1 **0.348** / MRR 0.536 vs SIE top-1 **0.826** / MRR 0.895. CIs, paired deltas, tie sensitivity, ablations and every miss: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). _Generated by `python -m eval.run_eval`._

**What the ablations say (computed):**

- **Main set, where the lexical gain comes from:** BM25 over the Card chunks alone (about the fields the keyword router reads) takes top-1 from 0.432 to 0.727 (better term weighting on the same text). Indexing the body sections too raises it to 0.920.
- **Source-authored set, where the lexical gain comes from:** BM25 over the Card chunks alone (about the fields the keyword router reads) takes top-1 from 0.348 to 0.696 (better term weighting on the same text). Indexing the body sections too lowers it to 0.652.
- **Main set:** the stronger single retriever is **BM25 only** (MRR 0.949 vs 0.927). SIE: hybrid RRF (--no-rerank) is +0.023 MRR against it and +0.399 against the keyword router.
- **Source-authored set:** the stronger single retriever is **dense only** (MRR 0.906 vs 0.749). SIE: hybrid RRF (--no-rerank) is -0.011 MRR against it and +0.359 against the keyword router.
- The keyword router is strongest on `-` queries (top-1 0.92; `-` marks the scaffold's original 12) and weakest on `symptom` queries (top-1 0.00).

**Beyond top-1 (computed):**

- **Out of scope (30 queries no skill fits):** SIE: hybrid RRF (--no-rerank) abstains on 87%, asks on 13% and routes 0% anyway; the keyword router has no way to abstain, and its `route()` as shipped returned skills for every one. In scope it abstains on 0.9% of queries.
- **Multi-intent (25 requests, 63 gold intents):** `compose()` covers on average 64% of each request's gold intents (60% of all 63 pooled), with 32% exact plans; routing the whole request and taking the top-1 covers on average 48% (38% pooled; 20% exact). The same ranking's top-n with the true intent count (an oracle no caller has) covers on average 77% (75% pooled; 44% exact). `compose()` was revised after its first measurement on this set, so its numbers are not held-out (see RESULTS.md).
- Confidence levels are heuristic and uncalibrated; how often each level is right is measured per set in RESULTS.md.
<!-- END:benchmarks -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results-dark.png">
  <img alt="Bar charts of recall@3, MRR and nDCG@3 for the keyword router and SIE variants on the main and source-authored query sets" src="benchmarks/results.png">
</picture>

Reading notes. Only 1 of the 88 main-set queries falls outside the hybrid system's top 3 (the
`recommender-systems` query above). Two of its main-set top-1 results are exact score ties broken
alphabetically; with random tie-breaking its expected top-1 is 94.3%. On the out-of-scope
line: the shipped keyword `route()` has a minimum-score cut, but every skill whose type matches
the request's classified intent gets a bonus that clears it, so on this corpus it always returns
skills.

### The source-authored set

The 23 source-authored queries are the `input` of every behavioral eval case in ml-ai-skills'
`evals/` directory, with that repository's own skill labels (`python -m
eval.build_source_queries`). They were added to the corpus repository on 2026-09-21, a week
before this repository's first commit, so they were not written against this engine; how they
were drafted is not recorded here. Both repositories have the same maintainer, so the set is
independent of the engine's code, not of its maintainer. At least one case (RAG prompt
injection, labeled `rag-pipeline`) deliberately overlaps another skill (`ai-ml-security`).
Top-1: keyword 34.8%, BM25 65.2%, dense 82.6%, hybrid 82.6%; MRR 0.536, 0.749, 0.906 and 0.895.
With n = 23 the intervals are wide (hybrid top-1: 65–96%).

### Vocabulary-collision case

The corpus's routing notes record a keyword-router failure found during that repository's own
testing: for *"Build a fraud detection model and evaluate it under severe class imbalance"* it
returned `computer-vision` first. In the reproduction, "detection" matches that skill's display
name and description (object detection). [`eval/collision.py`](eval/collision.py) reproduces
the case; `model-evaluation` and `supervised-learning` both fit the task:

| corpus | keyword router top-1 | SIE hybrid top-1 |
|---|---|---|
| ml-ai-skills@`c54f548`, where the failure was observed | `computer-vision` (does not fit) | `model-evaluation` (fits) |
| ml-ai-skills@`8328c60`, after the corpus's capability-tag audit | `supervised-learning` (fits) | `model-evaluation` (fits) |

The same notes name an embedding-based similarity pass over the keyword router's candidates as
what would address this class of failure, at the cost of a model dependency the corpus
repository deliberately avoids. This engine uses embeddings differently, as a first-stage
retriever fused with BM25 rather than a re-ranking of the keyword router's output, and keeps
that dependency in a separate project. Full inputs and outputs:
[RESULTS.md](benchmarks/RESULTS.md#vocabulary-collision-case).

### Latency (machine-dependent)

[`benchmarks/PERFORMANCE.md`](benchmarks/PERFORMANCE.md) records one Windows machine with 12
logical CPUs. Hybrid `route()` without the reranker, at the default pool, measured p50 83–127 ms
(p95 157–212 ms) in the two parts of that run that time it; query embedding alone measured
60–101 ms (p50). On a synthetic 16× replication of the chunk set (4,752 chunks; latency only,
since duplicated skills make quality meaningless), p50 rose from 83 ms to 161 ms within the same
pass, mostly in BM25 (3 → 69 ms). Reranker latency has not been measured. These are not
production latency figures.

### Benchmark limitations

- **One small corpus:** 38 skills and 297 chunks from one subject area (ML/AI).
- **Small query sets:** 88 + 23 routing queries, so intervals are wide.
- **Query provenance:** 76 of the 88 main-set queries, and all out-of-scope and multi-intent
  requests, were written by LLM agents. The 12 remaining main-set queries come from the project
  scaffold with an unrecorded process (the keyword baseline's top-1 is 92% on them, against 36%
  on the other 76). The source-authored queries come from the corpus repository's own eval
  cases, whose drafting is not recorded here. The writing and auditing protocol is described in
  RESULTS.md, and the query files are pinned by sha256.
- **Post-hoc changes:** the BM25-only confidence rule and `compose()` were revised after their
  first measurement on these sets, so their current numbers are not held-out; RESULTS.md lists
  the first-measurement figures.
- **Reranker:** the cross-encoder configuration has not been evaluated.

## What the results mean

The current results show that, on the included benchmark, hybrid retrieval ranks the correct
skill first much more often than the keyword baseline on both routing sets. On the main set its
top-1 is also higher than dense alone (95.5% vs 88.6%) and BM25 alone (92.0%), but the margin
over BM25 is small: hybrid is right where BM25 is wrong on 5 queries and wrong where BM25 is
right on 2 (94.3% vs 92.0% with random tie-breaking), a difference that 88 queries cannot
separate from noise. On the main set, most of the gain over the baseline already comes from
BM25: better term weighting on roughly the same fields, then indexing each skill's full text.

They do **not** establish:

- production-scale accuracy;
- performance on corpora with thousands of skills;
- universal superiority of hybrid retrieval (on the source-authored set, dense alone matches it
  at top-1 and has a higher MRR);
- production-level latency;
- generalization to arbitrary skill corpora or to real user traffic.

## Supported use cases

| use case | what it provides | interface |
|---|---|---|
| Agent skill routing | user task → top skill, an action (route / clarify / abstain) and alternatives | `POST /route`, `Engine.route()`, `python -m sie.router "…"` |
| Skill discovery | "what skill can handle X?" → ranked candidates with evidence | `POST /route` (evidence included by default), `--explain` |
| Skill search | lexical and semantic search over every section of every skill | `GET /search` (mode set per server by `SIE_MODE`), `--mode sparse` / `dense` / `hybrid` |
| Prerequisite reasoning | skill → required skills → ordered path, with reasons | `GET /learning-path`, `python -m sie.router --path SLUG` |
| Multi-part requests (opt-in) | request → intents → dependency-ordered skill plan | `POST /route` with `"multi_intent": true`, `--multi` |
| Skill corpus evaluation | retrieval strategies compared on a fixed corpus with labeled queries | `python -m eval.run_eval` |
| Internal/private skill registries | routing, search, learning paths and composition over a local directory (the evaluation harness still scores `data/skills/` only) | `--skills`, `SIE_SKILLS_DIR` |

The engine does not send queries or skill content anywhere. Its only network use is downloading
models that are not on disk yet: the embedding model (by default chromadb's ONNX export of
all-MiniLM-L6-v2, fetched by chromadb; from Hugging Face with
`SIE_EMBEDDER=sentence-transformers`) and, unless the reranker is disabled, the cross-encoder
weights from Hugging Face. ChromaDB telemetry is turned off.

## What this project is not

- **Not a skill marketplace or registry.** It does not host, publish or distribute skills.
- **Not a replacement for skill repositories.** It needs a corpus as input.
- **Not a collection of AI/ML skills.** The vendored copy of ml-ai-skills is benchmark data.
- **Not an agent execution runtime.** It recommends skills and plans; it does not load skills
  into an agent, run them or call tools.
- **Not an LLM, and it does not call one.** Retrieval, confidence, composition and learning
  paths are deterministic code. `sie/llm.py` is an optional client that defaults to a no-op,
  and the engine does not use it.
- **Not a generic vector-database wrapper.** ChromaDB stores one of the two indexes; fusion,
  evidence, confidence and the skill graph are the engine, with an evaluation harness alongside.
- **Not a hosted service.** The API is a small FastAPI app without authentication, meant for
  local or trusted-network use.

## Using another corpus

The engine reads any directory of `SKILL.md` files; nothing has to be copied into this
repository.

```text
your-skills/
├── corpus.toml          # optional: name, version, source, layout, key mapping
├── skill-a/
│   └── SKILL.md
├── skill-b/
│   └── SKILL.md
└── skill-c/
    └── SKILL.md
```

**What a `SKILL.md` needs.** A YAML frontmatter block with at least one key; no particular key
is required. Files without parseable frontmatter are skipped and listed with a reason.

- The skill id is `id`, else `slug`, else `name` if it looks like a slug (`incident-triage`),
  else the folder name.
- The *Card* chunk is built from `display_name` (else the skill id), `description` and
  `capabilities`. Everything from the first case-sensitive "NOT for" in the description onward is
  left out of it, because that clause names other skills' topics.
- Also read when present: `type` (or `skill_type`; `workflow` / `reference`, default
  `reference`), `domain` (default `unknown`), `level` (`beginner` / `intermediate` /
  `advanced`, default `intermediate`), `risk_level`, `evidence_level`, `version`, `updated_at`
  (or `updated`, `last_updated`, `last_verified`) and the relationship lists below. Other `type`
  or `level` values produce a warning, not a failure. Other keys are kept as `metadata`.
- Each `##` section of the body becomes one or more chunks (sections over about 1,200
  characters are split). Text before the first `##` heading is not indexed; a body with no `##`
  heading is indexed whole. A `## ` line inside a code block also counts as a heading.

| relationship | accepted frontmatter keys |
|---|---|
| `requires` | `requires`, `prerequisites`, `depends_on` |
| `recommended_before` | `recommended_before`, `soft_requires`, `recommended` |
| `related` | `related`, `see_also` |
| `conflicts` | `conflicts`, `conflicts_with` |
| `alternative_to` | `alternative_to`, `alternatives` |
| `specializes` | `specializes`, `specialization_of` |
| `supersedes` | `supersedes`, `replaces` |

A minimal skill:

```markdown
---
name: postmortem-writing
description: Use after an incident is resolved to write a blameless postmortem with timeline, root cause and action items.
type: workflow
domain: operations
requires: [incident-triage]
---
## Overview
Reconstruct the timeline, identify contributing factors and write action items with owners.
```

**`corpus.toml` (optional).** Without it, the corpus is named after its directory and the
defaults shown below apply.

```toml
name = "my-skills"               # default: the directory name
version = "v1.4.0"               # quote it: an unquoted 1.10 is a TOML float and is rejected
source = "https://example.com/my-skills"
license = "Apache-2.0"

[layout]
pattern = "**/SKILL.md"          # default: glob relative to the corpus root
ignore_prefixes = ["_", "."]     # default: path parts starting with these are not skills

# [fields]                       # only for keys that are not built-in aliases, e.g.
# requires = "depends"           # read `depends:` as the requires relationship
```

A `[fields]` entry replaces the built-in aliases for that field (for `slug` it is only read
first). With `requires = "depends"`, the keys `requires`, `prerequisites` and `depends_on` are no
longer read as prerequisites and end up in `metadata`, without a warning; map a field only if
the corpus never uses its built-in keys.

**Commands.** Use one index directory per corpus.

```bash
python -m sie.ingest --skills path/to/your-skills     # load report; --expect N exits 1 unless exactly N skills load with none skipped
python -m sie.router --skills path/to/your-skills --persist-dir path/to/index --build
python -m sie.router --skills path/to/your-skills --persist-dir path/to/index --no-rerank "your task"
python -m sie.router --skills path/to/your-skills --path postmortem-writing   # learning path to a skill id
python -m sie.router --skills path/to/your-skills --mode sparse --no-rerank "your task"   # BM25 only: no index, no model

SIE_SKILLS_DIR=path/to/your-skills SIE_PERSIST_DIR=path/to/index SIE_RERANK=0 uvicorn sie.api:api
```

Each skill keeps its provenance: `source` (the corpus name; the manifest's `source` URL is kept
on the corpus as `source_url`), `source_version`, `path`, `content_hash`, `version`,
`updated_at`, and a string such as `my-skills@v1.4.0:skill-c/SKILL.md`. After the chunk texts or
the embedder change, the router refuses the dense index until it is rebuilt.

For example, the engine can be used directly with a checkout of ml-ai-skills:

```bash
git clone https://github.com/NiravRVaghasiya/ml-ai-skills ../ml-ai-skills
python -m sie.ingest --skills ../ml-ai-skills
# at commit 8328c60:
# [ingest] loaded 38 skills from ../ml-ai-skills (corpus ml-ai-skills, fingerprint 7d62d861fc8ce6a7) (0 skipped, 77 ignored, 0 warnings)
```

The 77 ignored files are its `_TEMPLATE/` and the plugin re-packagings of the same skills under
`plugins/` (duplicate ids; the shallowest copy wins). The fingerprint equals that of the
vendored copy. Without a `corpus.toml` the corpus has no version, so provenance strings read
`ml-ai-skills:<path>`.

**Conventions to know about.** A few defaults come from the corpus the engine was developed
against. None of them decides whether a skill loads, but some shape results: unknown `type` and
`level` values only produce warnings; `level` breaks ties in learning paths and composed plans;
the "NOT for …" rule decides what the Card chunk indexes; and `compose()` splits a request only
before a verb from its list of English task verbs. Lexical retrieval tokenizes ASCII letters and
digits only, and the embedder is the English `all-MiniLM-L6-v2`, so corpora in other languages
load but retrieve worse. The confidence thresholds were chosen for that embedder and may need
adjusting for another corpus or embedder (`ConfidencePolicy`, Python only).

**Evaluating on another corpus.** `python -m eval.run_eval` currently scores the corpus in
`data/skills/` only; it has no `--skills` option. Query sets are JSONL files listed in a TOML
manifest (`--datasets my-sets.toml --out my-results/`), and `python -m eval.datasets --datasets
my-sets.toml --skills path/to/your-skills` validates them against a corpus. A set labeled for a
different corpus (`corpus = "…"` in the manifest) is skipped, with the reason printed. Row
formats (each row also needs an origin: the set's `origin`, an `origin_field` with
`origin_map`, or the row's own `"origin"`; see `eval/datasets.py`):

```text
routing       {"query": "…", "gold": "skill-id"}     or "gold": ["skill-a", "skill-b"]
out_of_scope  {"query": "…", "gold": null}
multi_intent  {"query": "…", "gold": [["skill-a"], ["skill-b", "skill-c"]]}
```

## Quick start

Requires Python 3.11 or newer (CI runs 3.11 and 3.12). Run the commands from the repository
root: the default corpus and index paths are relative to it.

```bash
git clone https://github.com/NiravRVaghasiya/skill-intelligence-engine.git
cd skill-intelligence-engine
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt    # the full local pipeline (sentence-transformers pulls in torch)
```

```bash
# 1. Load the vendored corpus; exits 1 unless exactly 38 skills load with no skips
python -m sie.ingest --skills data/skills --expect 38

# 2. Build the dense index into data/chroma/ (chromadb downloads the embedding model on first use)
python -m sie.router --build

# 3. Route a task with the benchmarked configuration: hybrid RRF, no reranker
python -m sie.router "impute missing values and encode categoricals" --no-rerank --explain

# 4. Learning path from the skill graph (no index or model needed)
python -m sie.router --path rag-evaluation

# 5. Multi-intent plan (opt-in)
python -m sie.router "build a RAG pipeline, evaluate it, then deploy it" --no-rerank --multi

# 6. HTTP API; interactive docs at http://127.0.0.1:8000/docs
SIE_RERANK=0 uvicorn sie.api:api
```

Output of step 5:

```text
[compose] 3 intents -> 3 skills in the plan
  intent 1: 'build a RAG pipeline' -> rag-pipeline [high/route]
  intent 2: 'evaluate it' (routed as 'evaluate a RAG pipeline') -> rag-evaluation [high/route]
  intent 3: 'deploy it' -> model-deployment [high/route]
  1. rag-pipeline               requested    requested by intent 1; also required by rag-evaluation
  2. rag-evaluation             requested    requested by intent 2
  3. model-deployment           requested    requested by intent 3
```

In Windows PowerShell, set the variable separately: `$env:SIE_RERANK = "0"; uvicorn sie.api:api`.

**Without ML dependencies.** The core install supports BM25-only routing, the graph and
composition:

```bash
pip install -e .                  # python-frontmatter, networkx, rank-bm25, numpy
python -m sie.router --mode sparse --no-rerank "tune xgboost hyperparameters"
pip install -e ".[api]"
SIE_MODE=sparse SIE_RERANK=0 uvicorn sie.api:api
```

| install | adds |
|---|---|
| `pip install -e .` | the core: ingestion, BM25 routing (`--mode sparse`), confidence, graph, learning paths, composition; console scripts `sie-ingest` and `sie-route` |
| `.[retrieval]` | `chromadb`: the dense index (`hybrid` and `dense` modes) |
| `.[reranker]` | `sentence-transformers` (with torch): the cross-encoder and the alternative embedder |
| `.[api]` | `fastapi`, `uvicorn`, `pydantic` |
| `.[evaluation]` | `matplotlib`, `pyyaml`: charts and the source-query builder |
| `.[dev]` | `pytest`, `httpx`, `httpx2` and `.[api]` |
| `.[demo]` | `streamlit`, for `demo/app_streamlit.py` |
| `.[llm]` | `openai`, for the optional LLM-judge helper |
| `.[all]` | `retrieval`, `reranker`, `api` and `evaluation` |

`requirements.txt` lists the third-party packages of `.[all,dev]` in one flat file (PyYAML comes
in through `python-frontmatter`). It does not install the `sie` package itself, so run the
commands from the repository root, or use `pip install -e ".[all,dev]"` to also get the
`sie-ingest` and `sie-route` scripts.

## HTTP API

```bash
SIE_RERANK=0 uvicorn sie.api:api      # defaults: the vendored corpus, data/chroma/, hybrid mode
```

Configuration comes from `SIE_*` environment variables, all documented in
[`.env.example`](.env.example): corpus and index paths (`SIE_SKILLS_DIR`, `SIE_PERSIST_DIR`,
`SIE_CORPUS_MANIFEST`), retrieval (`SIE_MODE` = `hybrid` | `dense` | `sparse`, `SIE_EMBEDDER`,
`SIE_RERANK`, `SIE_RERANKER`, `SIE_STRICT_RERANK`, `SIE_TOP_K`, `SIE_POOL`, `SIE_RERANK_K`),
overlays (`SIE_EDGE_OVERLAYS`), start-up (`SIE_EAGER_INIT`) and logging (`SIE_LOG_LEVEL`,
`SIE_LOG_QUERIES`). The file is not loaded automatically: export the variables, or copy it to
`.env` and use `uvicorn --env-file .env`, which needs `python-dotenv`. The CLI takes flags
instead; of the `SIE_*` variables it reads only `SIE_EMBEDDER`, `SIE_RERANKER`,
`SIE_LOG_LEVEL` and `SIE_LOG_QUERIES`.

| method | path | purpose |
|---|---|---|
| GET | `/health` | corpus-level health: status (`ok` / `degraded`), version, skill count, corpus identity; never loads a model |
| GET | `/ready` | readiness, suitable for probes: corpus, graph, BM25, dense-index freshness, reranker state; 503 with `reasons` when not ready |
| GET | `/search?q=&k=&pool=&rerank_k=` | compact ranked list with a brief confidence |
| POST | `/route` | full routing answer: evidence, confidence, alternatives, provenance, direct prerequisites; `"multi_intent": true` adds a composed plan |
| POST | `/batch-route` | up to 32 queries, routed in order |
| GET | `/learning-path?target=&include_recommended=` | ordered path with reasons, notes and a `complete` flag |
| GET | `/skill/{slug}` | metadata, provenance and typed relationships |

Errors: `404` for an unknown skill ("did you mean" suggestions when close ids exist), `409` for a
`requires` cycle, `422` for invalid parameters (`k` 1–50, `pool` and `rerank_k` 1–500, queries
up to 4,000 characters), `503` when the corpus is missing, the index is not built or stale, or a
backend is unavailable. Chunk ids are never returned.

The responses below were captured from the service on the vendored corpus with `SIE_RERANK=0`;
long ones are trimmed as noted.

`GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok", "version": "0.2.0", "skills": 38,
  "corpus": {"name": "ml-ai-skills", "version": "8328c60", "fingerprint": "7d62d861fc8ce6a7"}
}
```

`GET /search`

```bash
curl "http://127.0.0.1:8000/search?q=impute+missing+values+and+encode+categoricals&k=2"
```

```json
{
  "query": "impute missing values and encode categoricals", "reranked": false, "pool": 20,
  "results": [
    {
      "slug": "data-preprocessing", "score": 0.0328, "section": "Card", "rank": 1,
      "name": "Data Preprocessing", "retrieval_methods": ["dense", "bm25"]
    },
    {
      "slug": "feature-engineering", "score": 0.0323, "section": "Workflow", "rank": 2,
      "name": "Feature Engineering", "retrieval_methods": ["dense", "bm25"]
    }
  ],
  "confidence": {"level": "high", "action": "route", "ambiguous": false}
}
```

`POST /route`, a *clarify* case (trimmed: snippets, `timings_ms`, `rerank_error` and part of
`provenance` omitted)

```bash
curl -X POST http://127.0.0.1:8000/route -H "Content-Type: application/json" \
     -d '{"query": "set up retrieval over my company docs and check answers are grounded", "k": 1}'
```

```json
{
  "query": "set up retrieval over my company docs and check answers are grounded",
  "confidence": {
    "level": "medium", "action": "clarify", "ambiguous": true,
    "reasons": ["bm25 ranks llm-evaluation first"], "competitors": ["llm-evaluation"],
    "signals": {
      "methods": ["dense", "bm25"], "leaders": {"dense": "rag-evaluation", "bm25": "llm-evaluation"},
      "agreement": 1, "support": 2, "similarity": 0.3711, "top_similarity": 0.3711,
      "coverage": 0.3451, "margin": 0.016, "reranked": false
    },
    "calibrated": false
  },
  "results": [
    {
      "skill_id": "rag-evaluation", "name": "RAG Evaluation", "rank": 1, "score": 0.0325,
      "score_type": "rrf", "section": "Workflow", "retrieval_methods": ["dense", "bm25"],
      "evidence": [
        {"method": "dense", "rank": 1, "score": 0.3711, "section": "Workflow", "matched_terms": []},
        {
          "method": "bm25", "rank": 2, "score": 14.6522, "section": "Card",
          "matched_terms": ["retrieval", "my", "check", "grounded"]
        }
      ],
      "provenance": {
        "source": "ml-ai-skills", "source_version": "8328c60", "path": "rag-evaluation/SKILL.md",
        "provenance": "ml-ai-skills@8328c60:rag-evaluation/SKILL.md"
      },
      "prerequisites": ["rag-pipeline"]
    }
  ],
  "alternatives": [
    {
      "skill_id": "llm-evaluation", "name": "LLM Evaluation", "rank": null,
      "reason": "ranked first by bm25"
    }
  ],
  "reranked": false, "mode": "hybrid", "candidates": {"dense": 20, "bm25": 20, "fused": 13},
  "corpus": {"name": "ml-ai-skills", "version": "8328c60", "fingerprint": "7d62d861fc8ce6a7"},
  "composition": null
}
```

With `"multi_intent": true`, the response also carries a `composition` object (`multi_intent`,
`intents`, `plan`, `edges`, `conflicts`, `unmatched`, `notes`): each intent with its text, the
query actually routed, the selected skill and its confidence, and the ordered `plan` that the
CLI prints in [Quick start](#quick-start). `results` and `confidence` still describe the whole
request routed once.

`GET /learning-path`

```bash
curl "http://127.0.0.1:8000/learning-path?target=rag-evaluation"
```

```json
{
  "target": "rag-evaluation", "path": ["rag-pipeline", "rag-evaluation"],
  "related": ["llm-evaluation", "model-evaluation", "ai-ml-security"], "conflicts": [],
  "path_conflicts": [], "missing_prerequisites": [],
  "steps": [
    {
      "skill": "rag-pipeline", "position": 1, "relation": "direct", "depth": 1,
      "required_by": ["rag-evaluation"], "reason": "direct prerequisite of rag-evaluation",
      "provenance": ["ml-ai-skills@8328c60:rag-evaluation/SKILL.md#requires"]
    },
    {
      "skill": "rag-evaluation", "position": 2, "relation": "target", "depth": 0, "required_by": [],
      "reason": "target", "provenance": []
    }
  ],
  "direct": ["rag-pipeline"], "transitive": [], "recommended": [], "notes": [], "complete": true
}
```

`GET /skill/{slug}` (trimmed: `skill_id`, `display_name`, `description`, `capabilities`,
`dangling`, `metadata`, most of `provenance` and 8 of 9 `relations` omitted)

```bash
curl http://127.0.0.1:8000/skill/rag-evaluation
```

```json
{
  "slug": "rag-evaluation", "name": "RAG Evaluation", "domain": "llm", "level": "intermediate",
  "skill_type": "workflow", "requires": ["rag-pipeline"], "required_by": [],
  "related": ["rag-pipeline", "llm-evaluation", "model-evaluation", "ai-ml-security"],
  "conflicts": [], "declared": ["requires", "related", "conflicts"],
  "provenance": {"provenance": "ml-ai-skills@8328c60:rag-evaluation/SKILL.md"},
  "relations": [
    {
      "source": "rag-pipeline", "target": "rag-evaluation", "relationship": "requires",
      "confidence": "declared",
      "provenance": "ml-ai-skills@8328c60:rag-evaluation/SKILL.md#requires"
    }
  ]
}
```

`POST /batch-route`

```bash
curl -X POST http://127.0.0.1:8000/batch-route -H "Content-Type: application/json" \
     -d '{"queries": ["tune xgboost hyperparameters", "explain predictions with shap"], "k": 1}'
```

This returns `{"results": [...]}`: one `/route`-shaped answer per query, in request order (top
skills here: `hyperparameter-tuning`, `explainability`). `explain` defaults to `false` for
batches.

**Service behavior.**

- At start-up the service loads the corpus, graph and BM25, verifies the dense index and loads
  the embedding model (and the reranker if enabled), unless `SIE_EAGER_INIT=0`. A failed
  start-up keeps the process running and `/ready` answers 503 with the reason. Start-up is not
  retried: `/ready` keeps reporting the failure until the service is restarted, even though the
  query endpoints re-check on each request and may already succeed once the cause is fixed
  (for example, after building the index). Restart the service after fixing it.
- Requests run concurrently in FastAPI's thread pool; the indexes are read-only after start-up
  and initialization is lock-guarded. Each worker process loads its own engine.
- There is no authentication, CORS configuration or rate limiting.
- With the defaults, start-up may use the network: chromadb downloads the embedding model if it
  is not cached, and with `SIE_RERANK=1` the cross-encoder is downloaded from the Hugging Face
  hub unless it is already in `data/models/ms-marco-MiniLM-L-6-v2/`, a local `SIE_RERANKER`
  directory or the Hugging Face cache. For offline hosts, pre-cache the embedding model and
  either set `SIE_RERANK=0` or copy the cross-encoder weights into that local directory.
- Each call that reaches the router emits a `route` event (mode, top skill, confidence, action,
  candidate counts, per-stage timings) or a `route_error` event; a request that fails while the
  corpus is loading returns 503 without one. Every engine start emits `engine_start` (at
  start-up, or on the first `/ready` when `SIE_EAGER_INIT=0`). Events go to the `sie.events`
  logger as JSON (printed when `SIE_LOG_LEVEL=INFO`) and to hooks registered with
  `sie.observability.add_hook()`. In these events a query appears only as a sha256 prefix and a
  length unless `SIE_LOG_QUERIES=1`. uvicorn's own access log still records each request line,
  including the query string of `GET /search`; run uvicorn with `--no-access-log`, or use
  `POST /route`, to keep query text out of the logs.

## Python API

```python
from sie import Engine

engine = Engine(skills_dir="data/skills", use_reranker=False)   # hybrid; needs the index from --build
result = engine.route("impute missing values and encode categoricals", k=3)
print(result.top.slug, result.confidence.level, result.confidence.action)
# data-preprocessing high route
print(engine.learning_path("rag-evaluation")["path"])
# ['rag-pipeline', 'rag-evaluation']

bm25_only = Engine(skills_dir="path/to/your-skills", mode="sparse", use_reranker=False)   # no index, no model
```

`Engine` also provides `batch()`, `compose()`, `skill()`, `prerequisites()`, `start()` and
`status()`, and `Engine.from_env()` builds one from the `SIE_*` variables. Relative paths
resolve against the working directory.

## Project structure

```text
sie/                     the engine (the installable package)
├── corpus.py            corpus.toml manifests, frontmatter key aliases, fingerprints
├── ingest.py            SKILL.md → Skill objects + load report          (python -m sie.ingest)
├── chunking.py          Card chunk + `##` section chunks (long sections split, code fences kept whole)
├── index/
│   ├── sparse.py        BM25, matched terms, lexical coverage
│   ├── dense.py         ChromaDB + all-MiniLM-L6-v2 (ONNX or sentence-transformers)
│   └── fuse.py          reciprocal rank fusion
├── rerank.py            optional cross-encoder
├── hub.py               model-availability checks (run before importing torch)
├── router.py            HybridRouter: route() / retrieve()               (python -m sie.router)
├── confidence.py        heuristic route / clarify / abstain
├── compose.py           multi-intent composition                        (python -m sie.compose)
├── graph/
│   ├── build.py         typed edges, overlays, cycle detection
│   └── paths.py         learning paths, dependency ordering
├── engine.py            Engine: explicit start-up, status, environment configuration
├── api.py, schemas.py   FastAPI app and response models
├── observability.py     structured events and hooks
├── models.py            shared dataclasses (Skill, Chunk, Hit, Edge, RouteResult, …)
└── llm.py               optional LLM client (no-op by default; not used by the engine)
eval/                    evaluation harness (repository tooling, not installed)
├── datasets.toml        query-set manifest (kind and origin of every set)
├── datasets.py          query-set loader and validator                  (python -m eval.datasets)
├── queries*.jsonl       the four query sets
├── run_eval.py          scores every system on every set → benchmarks/
├── metrics.py, report.py, replay.py, perf.py, collision.py, build_source_queries.py
├── judge.py             optional LLM-judge helper (not called by any command)
└── baseline/            the ml-ai-skills keyword router, vendored verbatim (MIT)
benchmarks/              metrics.json, per_query.jsonl, collision.json, performance.json, PERFORMANCE.md
                         and charts (generated); RESULTS.md (generated between its markers)
tests/                   pytest suite (models mocked, network blocked), fixtures/, integration/ (opt-in)
docs/                    PROPOSED_REQUIRES.md + proposed_requires.edges.json (opt-in overlay)
data/skills/             vendored corpus: ml-ai-skills@8328c60 + corpus.toml (benchmark data)
demo/app_streamlit.py    optional Streamlit UI
scripts/setup_corpus.py  vendors a corpus into data/skills/, writes its manifest, builds the dense
                         index (--no-build to skip)
```

## Development and testing

```bash
pytest -q                                  # unit and regression tests: no model downloads, no network
python -m eval.run_eval --smoke            # what CI runs: recompute the metrics without an index or models; exit 1 on any change
python -m eval.datasets                    # validate the query-set manifest
SIE_INTEGRATION=1 pytest -m integration    # opt-in: a built index and cached models
```

None of these change tracked files.

- **Unit tests** mock the dense index and the reranker. `tests/conftest.py` blocks imports of
  `chromadb`, `sentence_transformers`, `torch`, `onnxruntime`, `transformers` and
  `huggingface_hub`, and blocks network connections other than to localhost, so a test that
  reaches for a model fails instead of downloading one.
- **Regression tests** (`tests/test_regression.py`) recompute the keyword and BM25 systems from
  scratch, and the dense and hybrid systems from recorded dense rankings
  (`tests/fixtures/dense_rankings.json`), and compare every metric with
  `benchmarks/metrics.json`. They also check that the generated README and RESULTS.md sections
  match the code.
- **Integration tests** (`tests/integration/`) run against the live index and cached models;
  they are skipped unless `SIE_INTEGRATION=1`.
- **CI** (GitHub Actions, Python 3.11 and 3.12) installs only the core, `api` and `dev` extras,
  with no model libraries, and runs `python -m sie.ingest --skills data/skills --expect 38`,
  `pytest -q` and `python -m eval.run_eval --smoke`.

**Benchmarks.** These commands write files:

```bash
python -m eval.run_eval                         # rebuilds data/chroma/; rewrites metrics.json, per_query.jsonl, the charts,
                                                # RESULTS.md's metrics block and this README's generated blocks
python -m eval.run_eval --no-build              # the same, reusing the existing index
python -m eval.run_eval --no-build --out DIR    # writes to DIR (other than benchmarks/) instead; leaves the README alone
python -m eval.perf --scale 1,4,16              # latency, pool sweep, synthetic scale → PERFORMANCE.md + performance.json
python -m eval.collision --source ../ml-ai-skills   # → collision.json + RESULTS.md's collision block; needs a git
                                                    #   checkout of ml-ai-skills and a built index
```

With reranker weights available, `python -m eval.run_eval --no-build --require-reranker` fills
the *pending* rows; the reranked system then becomes the headline.

## Status

Nothing in this repository has been validated in a production deployment. Below, *implemented*
means implemented and covered by tests, and *measured* means a number exists in `benchmarks/`.

| status | components |
|---|---|
| Implemented and measured | hybrid BM25 + dense retrieval with RRF, and the BM25-only and dense-only modes (retrieval quality); heuristic confidence with route / clarify / abstain (on in-scope and out-of-scope sets; not calibrated); latency on one machine |
| Implemented, not benchmarked | corpus manifests, provenance and fingerprints; stale-index refusal; per-result evidence; typed skill graph (7 relationship types), overlays, learning paths; the `Engine` facade; the FastAPI service with response models; observability events and hooks; network-free regression tests and CI |
| Experimental (opt-in) | multi-intent composition: rule-based splitting, measured below a simple oracle, revised after its first measurement |
| Optional, not evaluated | cross-encoder reranker (on by default in code, never run with real weights); `sentence-transformers` embedder backend; the proposed-prerequisites overlay; the Streamlit demo (tested headless only) |
| Present but unused | `sie/llm.py` (no-op by default) and `eval/judge.py` (an LLM-judge helper that no command calls; untested) |

The package is at version 0.2.0, and no API stability policy is defined yet.

## Limitations

- **Corpus size.** Retrieval quality has been measured on one 38-skill corpus (297 chunks) from
  one subject area (ML/AI). There is no benchmark on hundreds or thousands of skills; the
  synthetic scale test measures latency only.
- **Benchmark size and provenance.** 88 + 23 routing queries, most of them written by LLM
  agents; none is real user traffic. Intervals are wide.
- **Graph density.** The vendored corpus declares 2 `requires` edges; five of the seven
  relationship types have no instances in it and are tested only on synthetic fixtures.
- **Reranker.** Optional and unevaluated, yet enabled by default in code. Use `--no-rerank`
  (CLI) or `SIE_RERANK=0` (API) for the measured configuration.
- **Confidence.** Heuristic rules, not calibrated probabilities. The thresholds were chosen for
  `all-MiniLM-L6-v2` and can be changed only in Python (`ConfidencePolicy`).
- **Multi-intent composition.** Rule-based and English-centric. On the multi-intent set it
  covers on average 64% of each request's gold intents, against 77% for an oracle that knows the
  number of intents, and those numbers are not held-out.
- **Indexing details.** Text before a skill's first `##` heading is not indexed; long chunks are
  truncated to 256 tokens for embedding; lexical retrieval tokenizes ASCII letters and digits
  only and the embedder is English; BM25 runs in memory, and its query time grows with the
  corpus (3 → 69 ms from 297 to 4,752 chunks on the measured machine).
- **One corpus per engine.** Skill ids are unique within a corpus only, there is no
  multi-corpus federation, and each corpus needs its own index directory.
- **Evaluation harness.** It scores only the corpus in `data/skills/`.
- **API maturity.** Version 0.2.0: no authentication or rate limiting, one engine per process,
  and a failed start-up needs a restart.

## Roadmap

- [ ] Larger human-authored evaluation sets. The harness already accepts
  `origin = "human-collected"`; no such set exists yet.
- [ ] Confidence calibration. It needs independently collected queries; today every
  `POST /route` and `/batch-route` answer says `calibrated: false`.
- [x] Abstention and clarification: heuristic route / clarify / abstain, measured on an
  out-of-scope set.
- [x] Multi-intent skill routing (opt-in, experimental). Still open: closing the gap to the
  top-n oracle, measured on held-out requests.
- [ ] A richer skill relationship graph: more declared relationships in the corpora themselves,
  starting with the 9 `requires` edges drafted in `docs/PROPOSED_REQUIRES.md`.
- [ ] Larger-corpus benchmarks: retrieval quality beyond about 300 chunks, which needs the
  evaluation harness to accept a corpus path.
- [ ] Reranker evaluation: fill the *pending* rows with real weights (`--require-reranker`).
- [x] Retrieval observability: structured events, hooks and per-stage timings.
- [x] Corpus and version provenance: manifests, per-skill provenance, fingerprints, stale-index
  refusal.
- [ ] Multi-tenant corpora. Private corpora work today (any local directory); several corpora
  per engine, with ids namespaced by corpus, are not supported.

## Related ecosystem

[ml-ai-skills](https://github.com/NiravRVaghasiya/ml-ai-skills) is a skill/content repository.
**skill-intelligence-engine** is the intelligence layer that can index, retrieve, rank,
evaluate, and reason over skill repositories. They solve different problems and can be used
together.

```text
ml-ai-skills
    │
    │ skill corpus
    ▼
skill-intelligence-engine
    │
    │ retrieval / reasoning
    ▼
agent or application
```

What this repository uses from ml-ai-skills, each pinned to a commit:

- `data/skills/`: a copy of its 38 skills at `8328c60`, for reproducible benchmarks;
- `eval/baseline/`: its keyword router, vendored verbatim as the benchmark reference (MIT; see
  `eval/baseline/SOURCE.md`);
- `eval/queries_source_evals.jsonl`: the inputs and labels of its `evals/` cases;
- commit `c54f548`, for the vocabulary-collision reproduction.

Neither repository needs the other at runtime: ml-ai-skills works without this engine, and the
engine reads any corpus in the same format (its retrieval quality has been measured on the
vendored corpus only).

## License

MIT (see [LICENSE](LICENSE)). Material vendored from ml-ai-skills, also MIT-licensed: the skill
content in `data/skills/` (license recorded in `data/skills/corpus.toml`) and the code in
`eval/baseline/` (see `eval/baseline/LICENSE`).
