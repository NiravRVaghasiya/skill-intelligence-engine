---
name: rag-pipeline
display_name: RAG Pipeline
description: >
  Use when the user wants to build or debug a retrieval-augmented generation
  system — chunking documents, embedding them, indexing into a vector store,
  retrieving relevant passages for a query, and reranking before generation.
  Trigger phrases: "build a RAG pipeline", "chunk these documents for
  retrieval", "set up a vector database for search", "retrieve context for the
  LLM", "why is my RAG retrieving irrelevant chunks", "add a reranker".
  NOT for designing the generation prompt itself once context is retrieved
  (see prompt-engineering), updating model weights (see fine-tuning-llms), or
  the underlying transformer/attention math (see attention-mechanisms).
type: workflow
domain: llm
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - chunking-strategy
  - embedding-retrieval
  - vector-indexing
  - cross-encoder-reranking
  - retrieval-context-assembly
requires:
conflicts:
related:
  - prompt-engineering
  - fine-tuning-llms
  - llm-evaluation
  - attention-mechanisms
  - rag-evaluation
  - ai-ml-security
inputs: A source document corpus and a target query (or query set) to retrieve grounding context for.
outputs: An assembled, source-delimited context block of the top reranked chunks, ready to hand to a generation prompt.
version_constraints:
  - "sentence-transformers: SentenceTransformer.encode(normalize_embeddings=...) and CrossEncoder.predict() APIs stable across 2.x releases (model-knowledge estimate, not live-verified this session)"
  - "faiss: IndexFlatIP construction/add/search API is a long-stable part of the faiss-cpu/faiss-gpu surface (model-knowledge estimate, not live-verified this session)"
---

## Overview
A RAG pipeline grounds an LLM's answers in an external corpus by retrieving
relevant chunks at query time and inserting them into the prompt, instead of
relying on the model's parametric memory. This skill covers the retrieval
*plumbing* — chunking, embedding, indexing, retrieval, and reranking — end to
end with a runnable local stack. It stops at "here is the assembled context";
how you phrase the final generation prompt is `prompt-engineering`'s job.

## Workflow
1. **Chunk the source documents.** Split on structure first (paragraphs/headings),
   then cap chunk size with overlap so no single chunk exceeds the embedding
   model's effective context and no fact gets severed at a boundary.
   ```python
   from langchain_text_splitters import RecursiveCharacterTextSplitter

   splitter = RecursiveCharacterTextSplitter(
       chunk_size=800,       # characters, not tokens — tune per embedding model
       chunk_overlap=120,    # preserves continuity across chunk boundaries
       separators=["\n\n", "\n", ". ", " ", ""],
   )
   documents = ["...full text of doc 1...", "...full text of doc 2..."]
   chunks = [c for doc in documents for c in splitter.split_text(doc)]
   ```
2. **Embed the chunks.** Use a dedicated embedding model (not the generation
   LLM) and normalize vectors so cosine similarity behaves correctly.
   ```python
   from sentence_transformers import SentenceTransformer
   import numpy as np

   embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
   embeddings = embedder.encode(
       chunks, normalize_embeddings=True, show_progress_bar=True
   )
   embeddings = np.asarray(embeddings, dtype="float32")
   ```
3. **Index into a vector store.** For local/prototype work, FAISS with an
   inner-product index over normalized vectors is a fast, dependency-light
   cosine-similarity index.
   ```python
   import faiss

   index = faiss.IndexFlatIP(embeddings.shape[1])
   index.add(embeddings)
   chunk_store = {i: chunks[i] for i in range(len(chunks))}  # id -> text
   ```
4. **Retrieve top-k candidates for a query.** Embed the query with the *same*
   model used for indexing, then search.
   ```python
   def retrieve(query: str, k: int = 10):
       q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
       scores, ids = index.search(q_emb, k)
       return [(chunk_store[i], float(s)) for i, s in zip(ids[0], scores[0])]

   candidates = retrieve("What is the refund window for annual plans?", k=10)
   ```
5. **Rerank candidates with a cross-encoder.** First-stage vector search is fast
   but approximate; a cross-encoder scores (query, chunk) pairs jointly for much
   better precision, so retrieve broad (k=10-50) and rerank down to a few.
   ```python
   from sentence_transformers import CrossEncoder

   reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   pairs = [(query, text) for text, _ in candidates]
   rerank_scores = reranker.predict(pairs)

   reranked = sorted(
       zip([text for text, _ in candidates], rerank_scores),
       key=lambda x: x[1], reverse=True,
   )[:4]   # final context: top 4 chunks after reranking
   ```
6. **Assemble the context and hand off to generation.** Concatenate the final
   chunks with clear delimiters and source markers; the prompt *wording* around
   this context is `prompt-engineering`'s concern, not this skill's.
   ```python
   import anthropic

   context_block = "\n\n".join(
       f"[source {i+1}] {text}" for i, (text, _) in enumerate(reranked)
   )
   llm_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
   answer = llm_client.messages.create(
       model="claude-sonnet-4-5",
       max_tokens=500,
       messages=[{"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"}],
   )
   # For crafting the wording/format of this final prompt itself, see prompt-engineering.
   ```

## Gotchas
- **Retrieved documents are untrusted content, not just topically-relevant text.**
  A chunk that gets embedded into the generation context can carry
  instruction-like text planted by whoever authored/edited the source
  document — see [[ai-ml-security]] "RAG document injection" for the threat
  model and mitigations; this skill only owns the retrieval plumbing, not the
  security review of what that plumbing feeds into the model. Measuring
  whether it actually happened in your outputs is [[rag-evaluation]]'s job.
- **Query-time/index-time embedding mismatch.** Re-embedding with a different
  model version (or a different pooling/normalization setting) than was used to
  build the index silently degrades retrieval — recall the exact model+version
  used at index time and pin it.
- **Forgetting to normalize vectors for cosine similarity.** `IndexFlatIP` (inner
  product) only equals cosine similarity if both indexed and query vectors are
  L2-normalized; skipping this quietly biases retrieval toward longer vectors.
- **Chunking on fixed character/token counts with no overlap** severs sentences
  and tables mid-fact, so the "right" chunk exists but never fully appears in any
  single retrieved passage. Always chunk on structure first, then cap size with
  overlap.
- **Retrieval ≠ relevance.** Top-k by embedding similarity reliably returns
  topically related but not necessarily answer-bearing chunks — this is exactly
  why step 5 (cross-encoder reranking) matters; don't skip it for anything beyond
  a toy demo.
- **Stale indexes.** If source documents change and the vector store isn't
  re-indexed (or incrementally updated with deletions), the pipeline confidently
  retrieves and cites outdated content — treat the index as a build artifact with
  a refresh/versioning strategy.
- **Context window overflow.** Concatenating too many "just in case" chunks can
  exceed the generation model's context or push the real answer into the "lost in
  the middle" zone — keep the final reranked set small and let the reranker do
  the filtering, not raw k.

## References
- [Lewis et al., 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the original RAG paper defining retrieve-then-generate.
- [Sentence-Transformers documentation](https://www.sbert.net/) — embedding and cross-encoder reranker APIs used above.
- [Facebook FAISS wiki](https://github.com/facebookresearch/faiss/wiki) — index types (Flat, IVF, HNSW) and when to move beyond `IndexFlatIP`.
