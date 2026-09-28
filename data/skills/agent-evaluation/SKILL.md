---
name: agent-evaluation
display_name: Agent Evaluation
description: >
  Use when the user wants to evaluate an agent's *behavior* over a task — tool
  selection, tool arguments, loop/recovery behavior, and safety — rather than the
  quality of a single generated text. Trigger phrases: "did my agent pick the right
  tool", "check this agent trajectory for mistakes", "is my agent stuck in a loop",
  "evaluate whether the agent's actions were safe", "review this tool-call sequence".
  NOT for scoring a single LLM output's text quality (see llm-evaluation), NOT for
  RAG retrieval/generation metrics specifically (see rag-evaluation), NOT for
  designing the agent's tools/loop itself (see agents-and-tools).
type: workflow
domain: llm
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - tool-selection-evaluation
  - tool-argument-validation
  - loop-detection
  - unsafe-action-detection
  - agent-trajectory-review
  - prerequisite-ordering-check
requires:
  - agents-and-tools
conflicts:
related:
  - agents-and-tools
  - llm-evaluation
  - rag-evaluation
  - ai-ml-security
inputs: An agent trajectory (ordered list of thoughts / tool calls / tool results / final answer) plus the task specification and, ideally, a tool registry describing each tool's required arguments and risk level.
outputs: A structured per-dimension report — correct skill/tool selection, unnecessary calls, missing prerequisite calls, invalid arguments, loop/recovery behavior, and unsafe actions — NOT a single collapsed score.
---

## Overview
Generic LLM evaluation (see [[llm-evaluation]]) scores a single piece of generated
text. It says nothing about whether an *agent* — something that plans, calls
tools, observes results, and acts again — chose the right tools, in the right
order, with valid arguments, and recovered sensibly when something failed. This
skill is a workflow for reviewing an agent's full trajectory against those
dimensions separately, because collapsing them into one "was the agent good"
score hides exactly the failure modes that matter for anyone deciding whether to
give the agent more autonomy or fewer tools.

## Workflow
1. **Represent the trajectory as structured data**, not prose. Every downstream
   check operates on this shape.
   ```python
   from dataclasses import dataclass, field

   @dataclass
   class ToolCall:
       name: str
       args: dict
       result: str | None = None
       is_error: bool = False

   @dataclass
   class Trajectory:
       task: str
       calls: list[ToolCall] = field(default_factory=list)
       final_answer: str | None = None

   trajectory = Trajectory(
       task="Refund order #4471 and notify the customer.",
       calls=[
           ToolCall(name="lookup_order", args={"order_id": "4471"}, result="found, status=shipped"),
           ToolCall(name="issue_refund", args={"order_id": "4471"}, result="ERROR: amount required", is_error=True),
           ToolCall(name="issue_refund", args={"order_id": "4471"}, result="ERROR: amount required", is_error=True),
           ToolCall(name="issue_refund", args={"order_id": "4471", "amount": 49.99}, result="refunded"),
           ToolCall(name="send_email", args={"to": "customer", "template": "refund_confirmation"}, result="sent"),
       ],
       final_answer="Refunded order 4471 and emailed the customer.",
   )
   ```
2. **Check tool-argument validity** against a declared schema — this catches the
   most common concrete failure (missing/malformed arguments) before looking at
   anything fuzzier.
   ```python
   TOOL_SCHEMAS = {
       "lookup_order": {"required": ["order_id"]},
       "issue_refund": {"required": ["order_id", "amount"]},
       "send_email": {"required": ["to", "template"]},
   }

   def check_arguments(trajectory: Trajectory) -> list[str]:
       findings = []
       for i, call in enumerate(trajectory.calls):
           schema = TOOL_SCHEMAS.get(call.name)
           if schema is None:
               findings.append(f"step {i}: unknown tool '{call.name}' (not in registry)")
               continue
           missing = [f for f in schema["required"] if f not in call.args]
           if missing:
               findings.append(f"step {i}: '{call.name}' missing required args {missing}")
       return findings

   print(check_arguments(trajectory))
   ```
3. **Detect loops and wasted calls.** A tool call repeated with *identical*
   arguments after an error is a strong loop signal; a tool call repeated with
   *corrected* arguments after an error is legitimate recovery — distinguish them.
   ```python
   def detect_loops(trajectory: Trajectory, max_identical_repeats: int = 1) -> list[str]:
       findings = []
       seen: dict[tuple, int] = {}
       for i, call in enumerate(trajectory.calls):
           key = (call.name, tuple(sorted(call.args.items())))
           seen[key] = seen.get(key, 0) + 1
           if seen[key] > max_identical_repeats + 1:
               findings.append(f"step {i}: '{call.name}' called with identical args "
                                f"{seen[key]} times — likely stuck, not recovering")
       return findings

   print(detect_loops(trajectory))
   # -> [] here: the two failing issue_refund calls have IDENTICAL args and both
   # errored, so a stricter policy (max_identical_repeats=0) would flag them as a
   # loop rather than recovery, because the agent never changed anything between
   # attempts. This is deliberately configurable: some tools are safe to blind-retry
   # (idempotent reads), others are not (state-changing writes like issue_refund).
   ```
4. **Check prerequisite ordering** — did the agent call tools in an order that
   respects known dependencies (e.g. you cannot refund an order you never looked
   up)?
   ```python
   PREREQUISITES = {"issue_refund": ["lookup_order"], "send_email": ["issue_refund"]}

   def check_ordering(trajectory: Trajectory) -> list[str]:
       findings = []
       called_so_far: set[str] = set()
       for i, call in enumerate(trajectory.calls):
           for prereq in PREREQUISITES.get(call.name, []):
               if prereq not in called_so_far:
                   findings.append(f"step {i}: '{call.name}' called before its prerequisite '{prereq}'")
           called_so_far.add(call.name)
       return findings

   print(check_ordering(trajectory))
   ```
5. **Flag unsafe or irreversible actions that lack a confirmation/dry-run step.**
   Maintain an explicit risk registry per tool (this is the agent-evaluation
   counterpart to the `risk_level` field every skill in this repo carries).
   ```python
   TOOL_RISK = {"lookup_order": "low", "send_email": "low", "issue_refund": "high"}

   def check_unsafe_actions(trajectory: Trajectory, require_confirmation_above: str = "medium") -> list[str]:
       order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
       findings = []
       for i, call in enumerate(trajectory.calls):
           risk = TOOL_RISK.get(call.name, "medium")
           if order[risk] >= order[require_confirmation_above] and "confirmed" not in call.args:
               findings.append(f"step {i}: high-risk call '{call.name}' has no confirmation flag/argument")
       return findings

   print(check_unsafe_actions(trajectory))
   ```
6. **Compile a per-dimension report** — resist the urge to collapse it into one
   number; the whole point is that "3/5 tool calls valid" and "1 unsafe action"
   are different failures needing different fixes.
   ```python
   report = {
       "argument_errors": check_arguments(trajectory),
       "loop_findings": detect_loops(trajectory, max_identical_repeats=0),
       "ordering_findings": check_ordering(trajectory),
       "unsafe_action_findings": check_unsafe_actions(trajectory),
   }
   for dimension, findings in report.items():
       print(dimension, "->", findings or "OK")
   ```

## Gotchas
- **A single trajectory is not a reliable sample.** LLM agents are stochastic;
  one run passing tells you little about the failure rate. Run the same task
  multiple times (or across a small held-out set of task variants) before
  concluding a behavior is fixed or absent — see [[llm-evaluation]] Gotchas on
  non-zero-temperature flakiness, which applies here too.
- **"Missing tool call" failures are invisible if you only inspect what WAS
  called.** Always check the trajectory against what the task *required*
  (e.g. did it skip a mandatory verification step?), not just validate the
  calls it happened to make.
- **LLM-as-judge grading of agent trajectories inherits judge bias** (position
  bias, self-preference if the judge is the same model family as the agent) —
  the same caveat as in [[llm-evaluation]]; prefer the structured/rule-based
  checks in this workflow for anything checkable mechanically, and reserve
  judge-based grading for genuinely subjective dimensions (e.g. "was the final
  explanation clear").
- **Blind retry-until-success is not the same as recovery.** A tool call
  repeated with the exact same arguments after an error is a loop, not
  resilience; only count it as recovery if the arguments or approach changed
  in response to the error (see step 3 above).
- **Correct tool, wrong skill (or vice versa) look identical in a trajectory
  diff** if you only check "was a valid tool called" — you also need to check
  that the tool called was the *right* one for the step of the task it was
  meant to accomplish, which requires a task-specific rubric, not a generic one.

## References
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)](https://arxiv.org/abs/2210.03629) — the reasoning-then-acting loop most tool-using agents are built on, and the vocabulary ("thought", "action", "observation") this evaluation workflow assumes.
- [Liu et al., "AgentBench: Evaluating LLMs as Agents" (2023)](https://arxiv.org/abs/2308.03688) — a benchmark methodology for scoring agents across multiple environments; useful for how it separates task success from process quality.
- [Anthropic, "Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents) — practitioner guidance on agent/tool design whose failure modes (unbounded loops, over-broad tools) this skill is built to detect.
