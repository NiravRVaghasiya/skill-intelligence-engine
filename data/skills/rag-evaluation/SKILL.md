---
name: rag-evaluation
display_name: RAG Evaluation
description: >
  Use when the user wants to measure or debug the quality of a retrieval-augmented
  generation system specifically — retrieval metrics, faithfulness/groundedness,
  citation correctness, or RAG-specific failure modes — as opposed to building the
  pipeline or evaluating a generic LLM output. Trigger phrases: "compute recall@k
  for my retriever", "is my RAG answer grounded in the retrieved context", "why did
  my RAG pipeline hallucinate", "check citation correctness", "evaluate my reranker".
  NOT for building the chunking/embedding/retrieval pipeline itself (see
  rag-pipeline), NOT for generic text-quality LLM evaluation with no retrieval
  component (see llm-evaluation), NOT for agent tool-use trajectories (see
  agent-evaluation).
type: workflow
domain: llm
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - recall-at-k
  - precision-at-k
  - mrr
  - ndcg
  - faithfulness
  - groundedness
  - citation-correctness
  - rag-failure-analysis
  - context-precision
requires:
  - rag-pipeline
conflicts:
related:
  - rag-pipeline
  - llm-evaluation
  - model-evaluation
  - ai-ml-security
inputs: A RAG system (retriever + generator) and an eval set of queries, ideally with gold-relevant document IDs and/or gold answers; unlabeled query sets are usable for a reduced subset of checks (faithfulness, failure-mode scanning) but not for recall/precision/MRR/nDCG, which require relevance labels.
outputs: Separated retrieval-quality metrics, generation-quality (faithfulness/citation) metrics, and a categorized failure-mode report — deliberately not one blended score, per Overview.
---

## Overview
A RAG system fails in retrieval, in generation, or in the interaction between the
two — three different failure classes that a single end-to-end "was the answer
right" score cannot distinguish. This skill evaluates them separately: retrieval
metrics (did the retriever surface the right documents), faithfulness/citation
metrics (did the generator only claim what the retrieved context supports), and a
named taxonomy of RAG-specific failure modes (chunking, reranking, context
overflow, staleness, conflicting sources, injection). Building the pipeline being
evaluated here is covered by [[rag-pipeline]]; this skill assumes that pipeline
already exists.

## Workflow
1. **Build a labeled eval set.** Recall/Precision/MRR/nDCG all require a
   ground-truth relevance judgment per query — without it, skip to step 4 and
   rely on faithfulness/failure-mode checks only, which don't require labels.
   ```python
   eval_set = [
       {"query": "What is our refund window?",
        "gold_doc_ids": {"policy_042"},          # documents that actually answer it
        "gold_answer": "30 days from delivery."},
       {"query": "How do I reset 2FA?",
        "gold_doc_ids": {"help_017", "help_018"},
        "gold_answer": "Contact support to reset 2FA after identity verification."},
   ]
   ```
2. **Compute retrieval metrics against the retriever's ranked output.** Each
   metric answers a different question — don't report just one.
   ```python
   import math

   def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
       if not relevant:
           return float("nan")
       hits = len(set(retrieved[:k]) & relevant)
       return hits / len(relevant)

   def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
       if k == 0:
           return 0.0
       hits = len(set(retrieved[:k]) & relevant)
       return hits / k

   def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
       for i, doc_id in enumerate(retrieved, start=1):
           if doc_id in relevant:
               return 1.0 / i
       return 0.0

   def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
       def dcg(ids: list[str]) -> float:
           return sum((1.0 if d in relevant else 0.0) / math.log2(i + 1) for i, d in enumerate(ids[:k], start=1))
       ideal_order = list(relevant)[:k] + [d for d in retrieved if d not in relevant][:max(0, k - len(relevant))]
       idcg = dcg(ideal_order)
       return dcg(retrieved) / idcg if idcg > 0 else float("nan")

   # example: retriever returned these doc ids, ranked, for query 1
   retrieved = ["policy_017", "policy_042", "policy_003", "policy_099", "policy_042"]
   relevant = eval_set[0]["gold_doc_ids"]
   print("recall@3:", recall_at_k(retrieved, relevant, 3))
   print("precision@3:", precision_at_k(retrieved, relevant, 3))
   print("MRR:", reciprocal_rank(retrieved, relevant))
   print("nDCG@3:", ndcg_at_k(retrieved, relevant, 3))
   ```
   **When each matters:** Recall@k — "did we get the right doc in front of the
   generator at all" (most important when a missed doc means a wrong/unanswerable
   response). Precision@k — "how much noise did we hand the generator" (matters
   more as k grows or context budget shrinks). MRR — "how far down the list is
   the first useful hit" (matters when only the top result gets used, e.g. no
   reranking downstream). nDCG@k — like recall but rewards ranking *multiple*
   relevant docs near the top, so use it when a query can have several
   co-relevant sources and their relative order matters.
3. **Evaluate context precision** (the reranked/final context actually sent to
   the generator, which may differ from raw retrieval if you rerank or truncate).
   ```python
   def context_precision(final_context_ids: list[str], relevant: set[str]) -> float:
       if not final_context_ids:
           return 0.0
       return len(set(final_context_ids) & relevant) / len(final_context_ids)

   final_context_ids = ["policy_042", "policy_003"]  # after reranking/truncation to fit budget
   print("context precision:", context_precision(final_context_ids, relevant))
   ```
   A retrieval step with strong Recall@10 but a reranker/truncation step that
   drops the one relevant doc before generation will still produce a wrong
   answer — measure both stages, not just raw retrieval.
4. **Check faithfulness/groundedness**: does every claim in the generated answer
   follow from the retrieved context, independent of whether it's also true in
   the real world? This is the RAG-specific application of the faithfulness
   check in [[llm-evaluation]] — the reference set here is the *retrieved
   context*, not general world knowledge.
   ```python
   # Illustrative structure only — llm-evaluation/SKILL.md Workflow step 5 has the
   # full runnable NLI-based faithfulness check; re-used here against RAG context
   # instead of a general reference answer.
   def faithfulness_pairs(answer_sentences: list[str], context: str) -> list[tuple[str, str]]:
       """Return (sentence, context) pairs for an NLI entailment model to score;
       see llm-evaluation/SKILL.md for the actual model call."""
       return [(sentence, context) for sentence in answer_sentences]
   ```
5. **Check citation correctness**, if the system emits citations: does the cited
   source ID actually contain support for the claim next to it, not just *some*
   overlap in topic?
   ```python
   def citation_correctness(answer_claims: list[dict], retrieved_by_id: dict[str, str]) -> list[dict]:
       """answer_claims: [{"claim": str, "cited_doc_id": str}, ...]
       Returns claims whose cited doc doesn't exist in what was actually retrieved
       — the cheapest, fully mechanical citation check (existence), which should
       run before any semantic "does it really support the claim" check."""
       broken = []
       for c in answer_claims:
           if c["cited_doc_id"] not in retrieved_by_id:
               broken.append(c)
       return broken
   ```
6. **Scan for RAG-specific failure modes** rather than assuming a low score is
   always a "retrieval problem" or always a "generation problem":
   - **Chunking failure** — the answer requires info split across a chunk
     boundary, so no single retrieved chunk contains the full fact.
   - **Retrieval failure** — the gold document never appears in top-k at all.
   - **Reranking failure** — the gold document was retrieved but the reranker
     pushed it below the final top-k sent to the generator.
   - **Context overflow** — total retrieved tokens exceed the generator's usable
     context budget, silently truncating relevant content.
   - **Stale or duplicated documents** — the index contains an outdated version
     of a doc, or the same content indexed twice with different IDs, either of
     which can make retrieval metrics look fine while the answer is wrong/stale.
   - **Conflicting sources** — two retrieved chunks disagree on the queried
     fact (e.g. an old and a new policy document); a faithful-but-wrong answer
     can result from faithfully repeating the wrong one.
   - **Prompt injection inside retrieved documents** — see [[ai-ml-security]]
     "RAG document injection"; check retrieved chunks for instruction-like text
     as part of failure triage, not just as a security review activity.
   - **Multilingual retrieval mismatch** — query language differs from the
     indexed documents' language and the embedding model doesn't align them
     well, silently degrading recall without any error being raised.
   ```python
   def scan_failure_modes(query: str, retrieved: list[str], relevant: set[str],
                          final_context_ids: list[str], token_budget: int, context_tokens: dict[str, int]) -> list[str]:
       tags = []
       if relevant and not (set(retrieved) & relevant):
           tags.append("retrieval_failure")
       elif relevant and (set(retrieved) & relevant) and not (set(final_context_ids) & relevant):
           tags.append("reranking_failure")
       total_tokens = sum(context_tokens.get(d, 0) for d in final_context_ids)
       if total_tokens > token_budget:
           tags.append("context_overflow")
       return tags
   ```

## Gotchas
- **A blended "RAG score" hides which half is broken.** Always report retrieval
  and generation metrics separately (steps 2–3 vs. 4–5) — a low end-to-end score
  with high Recall@k points at generation/faithfulness; a low end-to-end score
  with low Recall@k points at retrieval, and fixing generation prompting won't
  help the second case.
- **Recall@k needs a real relevant-set label, not "the top result the current
  system returned."** Using the system's own output as its own ground truth
  (common when labels are expensive to collect) will silently validate whatever
  biases the current retriever already has.
- **nDCG with only one relevant document per query degenerates toward MRR** —
  if your labels never have more than one gold doc per query, nDCG isn't adding
  information over MRR; check whether multi-relevant-doc queries actually exist
  in your eval set before reporting nDCG as a headline metric.
- **A reranker can look harmful in aggregate metrics while fixing the cases that
  matter most** (and vice versa) — inspect the per-query deltas, not just the
  averaged Recall@k/nDCG before and after adding a reranking stage.
- **Faithfulness is not the same as correctness.** An answer can be perfectly
  faithful to (i.e., only claims what's in) a retrieved document that is itself
  wrong or outdated — faithfulness catches hallucination relative to context,
  not factual error inherited from a bad source; the "conflicting/stale sources"
  failure mode in step 6 is the check for that.

## References
- [Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)](https://arxiv.org/abs/2005.11401) — the original RAG paper; useful for the retrieval/generation separation this skill is built around.
- [Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)](https://arxiv.org/abs/2309.15217) — proposes faithfulness/context-precision/context-recall metrics specifically for RAG, the basis for steps 3–4 here.
- [Järvelin & Kekäläinen, "Cumulated Gain-Based Evaluation of IR Techniques" (2002)](https://dl.acm.org/doi/10.1145/582415.582418) — the original nDCG formulation used in step 2.
