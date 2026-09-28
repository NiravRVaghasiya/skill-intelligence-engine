---
name: llm-evaluation
display_name: LLM Evaluation
description: >
  Use when the user wants to evaluate an LLM or LLM-based system's output
  quality — building an eval harness, scoring with an LLM-as-judge, or
  checking generated answers for hallucination/faithfulness against source
  context. Trigger phrases: "evaluate my LLM's outputs", "set up an eval
  harness", "check for hallucinations", "LLM-as-judge scoring", "is my RAG
  answer grounded in the retrieved context", "regression test my prompt
  changes". NOT for classical ML metrics like accuracy/precision/recall/ROC-AUC
  on a non-generative model (see model-evaluation), or the retrieval mechanics
  being evaluated (see rag-pipeline).
type: workflow
domain: llm
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - llm-as-judge
  - eval-harness-design
  - hallucination-faithfulness-check
  - rouge-bleu-scoring
  - regression-eval-tracking
requires:
conflicts:
related:
  - rag-pipeline
  - prompt-engineering
  - agents-and-tools
  - model-evaluation
  - agent-evaluation
  - rag-evaluation
inputs: A frozen eval set (prompts with reference answers, grounding context, and/or a rubric) and the LLM system under test.
outputs: A regression report — overlap metrics, LLM-as-judge scores, and hallucination/faithfulness rate — tracked per model/prompt/retrieval version.
version_constraints:
  - "anthropic (Python SDK): messages.create() Messages API shape stable since general availability (model-knowledge estimate, not live-verified this session)"
  - "evaluate (Hugging Face): load('rouge').compute() API stable since early releases (model-knowledge estimate, not live-verified this session)"
---

## Overview
Evaluating an LLM system means scoring open-ended, non-deterministic text output
against a rubric or reference — a fundamentally different problem from scoring a
classifier's labels (that's `model-evaluation`). This skill builds a practical
eval harness: an eval set, automatic overlap metrics where they're meaningful,
LLM-as-judge scoring for quality/instruction-following, and a faithfulness check
to catch hallucination against source context. It covers generic single-output
text quality; scoring an *agent's* multi-step tool-use trajectory is
[[agent-evaluation]]'s job, and RAG-specific retrieval/citation metrics are
[[rag-evaluation]]'s — both build on the faithfulness/judge techniques here but
apply them to a different unit of evaluation. The output is a repeatable
regression suite you rerun every time the prompt, model, or retrieval changes.

## Workflow
1. **Build a fixed eval set.** Curate representative prompts with either a
   reference answer, a grounding context, or a scoring rubric — and freeze it so
   scores are comparable run over run.
   ```python
   import json

   eval_set = [
       {
           "id": "refund-001",
           "prompt": "What is the refund window for annual plans?",
           "context": "Annual plans may be refunded within 30 days of purchase.",
           "reference": "30 days",
       },
       # ... more cases, covering easy, edge-case, and adversarial inputs
   ]
   with open("eval_set.jsonl", "w") as f:
       for row in eval_set:
           f.write(json.dumps(row) + "\n")
   ```
2. **Generate outputs deterministically.** Run the system under test at
   temperature 0 (or fixed seed) so repeated eval runs are comparable and not
   confounded by sampling noise.
   ```python
   import anthropic

   llm_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

   def generate(prompt: str, context: str) -> str:
       # temperature=0 selects the top-probability token at each step, which makes
       # output *close to* deterministic — it is not a full reproducibility
       # guarantee (see Gotchas below and docs/REPRODUCIBILITY.md).
       response = llm_client.messages.create(
           model="claude-sonnet-4-5",
           max_tokens=300,
           temperature=0,
           messages=[{"role": "user", "content": f"Context: {context}\n\nQ: {prompt}"}],
       )
       return response.content[0].text

   outputs = [{**row, "output": generate(row["prompt"], row["context"])} for row in eval_set]
   ```
3. **Score with automatic overlap metrics where they're meaningful.** ROUGE/BLEU
   are weak for open-ended generation but useful as a cheap, fast signal for
   short, reference-constrained answers (extraction-style tasks).
   ```python
   from evaluate import load

   rouge = load("rouge")
   results = rouge.compute(
       predictions=[o["output"] for o in outputs],
       references=[o["reference"] for o in outputs],
   )
   print(results)  # e.g. {'rouge1': 0.82, 'rouge2': 0.71, ...}
   ```
4. **Score quality with an LLM-as-judge.** Use a rubric, ask for a structured
   score, and randomize candidate order if comparing two systems to reduce
   position bias.
   ```python
   JUDGE_PROMPT = """You are grading a model's answer for correctness and
   groundedness given the context. Score 1-5 (5 = fully correct and grounded).
   Return only JSON: {{"score": <int>, "reason": "<one sentence>"}}.

   Context: {context}
   Question: {prompt}
   Reference answer: {reference}
   Model answer: {output}
   """

   def judge(row: dict) -> dict:
       resp = llm_client.messages.create(
           model="claude-sonnet-4-5",
           max_tokens=100,
           temperature=0,
           messages=[{"role": "user", "content": JUDGE_PROMPT.format(**row)}],
       )
       return json.loads(resp.content[0].text)

   judged = [{**row, **judge(row)} for row in outputs]
   avg_score = sum(j["score"] for j in judged) / len(judged)
   ```
5. **Check faithfulness/hallucination against the source context.** For
   RAG-style systems, verify the answer is *entailed* by the retrieved context
   using an NLI model, independent of whether it matches the reference wording.
   ```python
   from transformers import pipeline

   nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

   def is_grounded(context: str, output: str, threshold: float = 0.5) -> bool:
       result = nli(f"{context}", text_pair=output, top_k=None)
       entailment = next(r["score"] for r in result if r["label"] == "ENTAILMENT")
       return entailment >= threshold

   grounding = [is_grounded(row["context"], row["output"]) for row in outputs]
   hallucination_rate = 1 - (sum(grounding) / len(grounding))
   ```
6. **Aggregate into a regression report and gate on it.** Track scores per eval
   run (model version, prompt version, retrieval version) so a prompt or model
   change that regresses quality is caught before shipping.
   ```python
   report = {
       "rouge1": results["rouge1"],
       "avg_judge_score": avg_score,
       "hallucination_rate": hallucination_rate,
       "n_cases": len(eval_set),
   }
   print(report)
   # Store alongside a git/prompt-version identifier for run-over-run comparison.
   ```

## Gotchas
- **LLM-as-judge has position and self-preference bias.** A judge tends to favor
  the first option shown, and favors output from its own model family — always
  randomize candidate order when comparing two systems, and treat judge scores
  as directional signal, not ground truth.
- **ROUGE/BLEU correlate poorly with human judgment for open-ended generation.**
  A fluent, correct paraphrase can score low if it doesn't share n-grams with the
  reference; reserve overlap metrics for extraction/short-answer tasks and lean
  on LLM-as-judge or human review for open-ended quality.
- **Faithfulness ≠ factuality.** An answer can be perfectly grounded in a
  retrieved context that is itself wrong or outdated — the NLI check in step 5
  only catches hallucination *relative to the given context*, not real-world
  correctness. Both checks are needed for a RAG system; conflating them
  under-reports risk.
- **Eval set contamination.** If eval prompts (or close paraphrases) leaked into
  fine-tuning or few-shot examples, scores are inflated and don't predict
  production behavior — keep the eval set held out and periodically refresh it.
- **Non-zero temperature makes eval runs flakier and harder to compare.** Pin
  `temperature=0` (or a fixed seed) for both the system under test and the judge
  when the goal is a comparable regression score, not diversity — but
  `temperature=0` makes output *close to* deterministic, not a bit-for-bit
  reproducibility guarantee: floating-point non-associativity across different
  batch compositions/hardware/kernels, backend routing, and provider-side model
  updates behind a fixed model string can all still change output. Treat
  `temperature=0` evals as approximately, not exactly, reproducible — see
  docs/REPRODUCIBILITY.md.

## References
- [Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — establishes LLM-as-judge methodology and documents its position/bias failure modes referenced above.
- [EleutherAI — lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — widely used open-source harness for running standardized LLM benchmarks.
- [Hugging Face `evaluate` library documentation](https://huggingface.co/docs/evaluate/index) — ROUGE/BLEU and other metric implementations used in step 3.
