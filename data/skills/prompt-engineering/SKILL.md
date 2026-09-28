---
name: prompt-engineering
display_name: Prompt Engineering
description: >
  Use when the user wants to design, structure, or debug prompts for an LLM —
  choosing zero-shot vs few-shot framing, getting reliable structured/JSON
  output, or improving reasoning quality without touching model weights or
  retrieval. Trigger phrases: "write a better prompt", "few-shot examples",
  "get JSON output from the model", "why is the model ignoring my
  instructions", "chain-of-thought prompt", "system prompt design". NOT for
  grounding answers in external documents (see rag-pipeline), changing model
  weights (see fine-tuning-llms), or building multi-step tool-calling loops
  (see agents-and-tools).
type: reference
domain: llm
level: beginner
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - zero-shot-prompting
  - few-shot-prompting
  - chain-of-thought-prompting
  - structured-output-prompting
  - prompt-template-design
requires:
conflicts:
related:
  - rag-pipeline
  - agents-and-tools
  - fine-tuning-llms
  - attention-mechanisms
inputs: A task description and/or a draft prompt (with or without examples) intended for a single LLM call.
outputs: A restructured prompt (zero/few-shot framing, CoT scaffolding, or output-format constraints) plus a diagnosis of why an existing prompt is unreliable.
---

## Overview
Prompt engineering is the practice of shaping the *input* to an LLM — instructions,
examples, formatting, and constraints — to reliably steer its output, without any
retrieval augmentation or weight changes. This card covers the core patterns
(zero/few-shot, chain-of-thought, structured output) and the failure modes that
make prompts brittle. Walk away knowing which pattern to reach for and why a
prompt that works today may silently break on a slightly different input.

## Key Concepts
- **Zero-shot vs. few-shot.** Zero-shot relies purely on the instruction; few-shot
  adds 2–8 input/output examples inside the prompt to demonstrate the exact format
  and reasoning style you want. Few-shot examples act as an implicit spec — the
  model pattern-matches to them more strongly than to prose instructions.
- **Role separation (system vs. user vs. assistant).** The system/developer prompt
  sets durable behavior (persona, constraints, output contract); the user turn
  carries the task; prior assistant turns carry conversational memory. Mixing task
  data into the system prompt makes behavior harder to override per-request.
- **Chain-of-thought (CoT).** Asking the model to "think step by step" before
  giving a final answer measurably improves multi-step reasoning and arithmetic.
  Variants: zero-shot CoT (just ask for reasoning), few-shot CoT (show worked
  examples), and self-consistency (sample multiple CoT paths, take a majority
  vote on the final answer).
- **Structured output.** Getting reliable JSON/XML/function-call output requires
  more than "respond in JSON" — the two robust approaches are (1) provide a
  concrete schema/example in the prompt and validate + retry on parse failure, or
  (2) use the provider's native structured-output / tool-calling mode (e.g.
  Anthropic tool use, OpenAI `response_format`/function calling), which constrains
  decoding rather than hoping the model complies.
- **Prompt templates.** Production prompts are parameterized templates (Jinja-style
  or f-strings) with clearly delimited slots for untrusted content (e.g.
  `<document>{{doc}}</document>`), not ad-hoc string concatenation — this both
  aids maintainability and reduces prompt-injection surface.
- **Instruction placement.** Models weight the beginning and end of a long context
  more heavily than the middle ("lost in the middle" effect) — put critical
  instructions and the actual question near the start or end, not buried in a
  wall of reference text.
- **Decoding parameters as part of the prompt contract.** Temperature, top-p, and
  max tokens are not "prompt engineering" per se, but they interact with it:
  low temperature (near 0) for near-deterministic, structured tasks, higher for
  creative generation — "near", because temperature 0 makes sampling close to
  deterministic but does not guarantee bit-for-bit identical output across runs
  (see docs/REPRODUCIBILITY.md). A great prompt at temperature 1.0 can still look flaky.

## Gotchas
- **Few-shot examples that don't match the real input distribution** actively
  mislead the model — it will copy the *style* of your examples (length, edge
  cases, format) even when it's wrong for the actual query.
- **"Respond only in JSON" without a schema or grammar constraint** still
  produces malformed output often enough to need a parse-validate-retry loop in
  production; prefer native structured-output/tool-calling modes when available.
- **Prompt injection via concatenated untrusted content.** If user-supplied or
  retrieved text is spliced directly into the prompt, it can contain instructions
  that override your system prompt. Always delimit untrusted content clearly and
  treat it as data, never as instructions, in your framing.
- **Overloading one prompt with too many instructions** (tone + format + length +
  five edge cases) degrades adherence to all of them; split into a shorter core
  instruction plus explicit examples, or split into multiple calls.
- **Chain-of-thought increases latency and cost** and can leak reasoning that
  should stay internal (e.g. exposing a policy check) — decide up front whether
  the reasoning trace is shown to the end user.

## References
- [Anthropic — Prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — official guide to Claude-specific prompting techniques and structured output patterns.
- [Wei et al., 2022 — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — the paper establishing the CoT pattern referenced above.
- [Brown et al., 2020 — Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — the foundational few-shot prompting paper.
