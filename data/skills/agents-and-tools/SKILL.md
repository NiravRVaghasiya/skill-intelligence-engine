---
name: agents-and-tools
display_name: Agents and Tools
description: >
  Use when the user wants to understand or design an LLM agent — the
  observe-think-act loop, tool/function-calling schemas, planning vs. reactive
  control, or multi-agent orchestration patterns. Trigger phrases: "how does an
  LLM agent loop work", "design a tool-calling schema", "ReAct pattern",
  "multi-agent orchestration", "when should the agent stop looping", "give the
  model tools to call". NOT for retrieval-specific pipelines (see rag-pipeline),
  prompt wording/few-shot design for a single call (see prompt-engineering), or
  changing model weights (see fine-tuning-llms).
type: reference
domain: llm
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - agent-loop-design
  - tool-calling-schema-design
  - multi-agent-orchestration
  - agent-termination-guardrails
  - planning-vs-reactive-control
requires:
conflicts:
related:
  - prompt-engineering
  - rag-pipeline
  - llm-evaluation
  - agent-evaluation
  - ai-ml-security
inputs: A task an LLM-driven agent should accomplish, plus the set of tools/functions it may call.
outputs: A designed agent loop — tool schema, planning/reactive control choice, termination guardrails, and (if multi-agent) an orchestration pattern.
---

## Overview
An LLM agent extends a single prompt/response call into a loop: the model
decides which tool to call, observes the result, and decides what to do next
until it reaches a final answer or a stopping condition. This card covers the
loop mechanics, tool-calling schema design, and orchestration patterns for
composing multiple agents — the conceptual model to reach for before wiring up
any specific agent framework.

## Key Concepts
- **The agentic loop (ReAct: Reason + Act).** At each step the model produces a
  *thought* (reasoning about what to do next), an *action* (a tool call with
  arguments), then receives an *observation* (the tool's result) that gets
  appended to context for the next step. The loop repeats until the model emits
  a final answer instead of a tool call, or a stop condition fires.
- **Tool/function-calling schemas.** Each tool is exposed to the model as a
  name, a natural-language description, and a typed parameter schema (JSON
  Schema in most APIs, including Anthropic's tool use and OpenAI's function
  calling). The model selects a tool and emits arguments matching that schema;
  the *description* is one of the biggest levers over whether the model picks
  the right tool — treat it like an API doc aimed at the model, not a comment
  for humans. (Tool naming and how mutually-exclusive the tool set's scopes
  are also matter — see the Gotchas below — so a bad description is a strong
  suspect for tool-misselection but not the only possible cause.)
- **Planning vs. reactive control.** Reactive agents decide the next single
  step from current context only (classic ReAct). Planning agents first draft a
  multi-step plan (e.g. plan-and-execute), then execute steps against it,
  optionally replanning on failure — more predictable for long tasks, more
  expensive and rigid for short ones.
- **Memory.** Short-term memory is just the running transcript in context
  (thoughts, actions, observations); it's bounded by the context window and
  grows every step. Long-term memory persists across sessions/tasks (a vector
  store, a scratch file, a database) and is retrieved rather than always
  included — this is where agent design and `rag-pipeline` overlap, but the
  retrieval mechanics themselves belong to that skill.
- **Multi-agent orchestration.** Common patterns: a **supervisor/router** agent
  that delegates subtasks to specialized worker agents and aggregates results;
  a **pipeline** of agents each transforming the previous one's output; and
  **debate/critique** setups where a second agent reviews the first's output
  before it's finalized. Orchestration adds coordination overhead and failure
  modes (miscommunication between agents) in exchange for specialization and
  parallelism — a single well-tooled agent is often simpler and should be the
  default until proven insufficient.
- **Termination and guardrails.** Every agent loop needs an explicit stopping
  condition beyond "the model decided to stop": a max-iteration cap, a max-cost
  or max-tool-call budget, and ideally a way for a human or a validator to
  interrupt. Without this, a confused agent loops indefinitely, burning tokens
  and repeating failed actions.

## Gotchas
- **No max-iteration or budget guard.** The single most common production
  incident with agents is an unbounded loop — a tool keeps returning an error,
  the model keeps retrying the same call. Always cap steps and cost, and fail
  loudly rather than silently retrying forever.
- **Ambiguous or overlapping tool descriptions.** If two tools' descriptions
  could plausibly apply to the same request, the model will pick inconsistently
  between them. Keep tool sets small, names specific, and descriptions
  mutually exclusive in scope.
- **Unvalidated tool output fed straight back into the loop.** A tool that
  returns malformed data, an error string, or attacker-controlled content (e.g.
  scraped web text) becomes part of the model's context and can hijack
  subsequent reasoning (prompt injection via tool results) — validate/sanitize
  before feeding observations back in; see [[ai-ml-security]] "indirect prompt
  injection" for the full threat model, which this card only summarizes.
- **No sandboxing for code-execution or shell tools.** Giving an agent a
  code-execution or file-system tool without a sandboxed, permission-scoped
  environment is a security risk regardless of how well the prompt is written —
  the sandbox is the actual safety boundary, not the instructions; see
  [[ai-ml-security]] "excessive agent permissions" and use [[agent-evaluation]]
  to check whether a given trajectory actually respected that boundary.
- **Over-provisioning tools "just in case."** Every additional tool increases
  the chance of wrong selection and inflates the prompt with schema text on
  every turn; include only what the task genuinely needs, and split large tool
  sets across specialized sub-agents rather than one agent with 30 tools.

## References
- [Yao et al., 2022 — ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the reasoning/acting loop this card is built around.
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — practical guidance on when to use an agent loop vs. a simpler workflow, and orchestration patterns.
- [Anthropic — Tool use with Claude](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — concrete tool-calling schema and loop mechanics.
