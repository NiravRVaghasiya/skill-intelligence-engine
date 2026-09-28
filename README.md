# Skill Intelligence Engine

> A hybrid **semantic + graph-aware retrieval engine** over the
> [`ml-ai-skills`](https://github.com/NiravRVaghasiya/ml-ai-skills) library.
> Replaces the source repo's keyword router with a **RAG + GraphRAG** system that
> (a) retrieves the right skill *semantically* and (b) generates a dependency-ordered
> **learning path** from the `requires` / `related` / `conflicts` graph.

[![tests](https://img.shields.io/badge/tests-pytest-green)]()
[![python](https://img.shields.io/badge/python-3.11-blue)]()

---

## Why

The `ml-ai-skills` repo ships a **deterministic keyword router** and openly documents its
weakness: *"keyword-based, not a semantic retriever; documents a real vocabulary-collision
failure case."* It also encodes a **dependency graph** in every skill's frontmatter that
nothing currently reasons over.

This project closes both gaps in one system:

| Layer | What it does |
|-------|--------------|
| **Semantic Router** | Hybrid dense (ChromaDB) + sparse (BM25) retrieval, RRF fusion, cross-encoder reranking |
| **GraphRAG** | `requires`-closure + topological sort -> ordered learning paths, `conflicts` flagged |
| **Eval harness** | recall@k / MRR / nDCG vs the keyword baseline + optional LLM-as-judge |

## Quickstart (clone-and-run, LLM optional)

```bash
git clone <this-repo> && cd skill-intelligence-engine
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# 1. point at the ml-ai-skills corpus (vendored or symlinked into data/skills/)
python -m sie.ingest --skills data/skills

# 2. build indexes (dense + sparse) and the dependency graph
python -m sie.router --build

# 3. query
python -m sie.router "impute missing values and encode categoricals"

# 4. serve the API
uvicorn sie.api:api --reload
#   GET /search?q=...            hybrid retrieval
#   GET /learning-path?target=rag-pipeline
#   GET /skill/{slug}
```

## Architecture

```
FastAPI  -->  Hybrid Router --> [BM25 | ChromaDB] -> RRF fuse -> cross-encoder rerank
   |
   +---------> GraphRAG ------> NetworkX DiGraph (requires/related/conflicts) -> topo path
   |
   +---------> Eval harness --> recall@k / MRR / nDCG  +  LLM-as-judge
```

## Benchmarks

See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) — headline: SIE resolves the
vocabulary-collision case the source repo documents, and beats the keyword baseline
on recall@3 / MRR / nDCG.

## Layout

```
sie/       core package (ingest, index, rerank, router, graph, api)
eval/      labeled queries + metrics + LLM-as-judge + baseline comparison
demo/      optional Streamlit UI
tests/     pytest
data/      vendored ml-ai-skills corpus
```

## License
MIT
