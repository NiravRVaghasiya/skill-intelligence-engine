# Proposed `requires` edges for ml-ai-skills (not applied)

The corpus declares only 2 hard prerequisites (`agents-and-tools -> agent-evaluation`,
`rag-pipeline -> rag-evaluation`), so most learning paths are a single node. This page
proposes additions **for the upstream maintainer to review**. The vendored corpus in
`data/skills/` is deliberately left unchanged, and every learning path and benchmark in this repo
uses the corpus as published.

The 9 accepted edges are also a machine-readable **edge overlay**,
[`proposed_requires.edges.json`](proposed_requires.edges.json) (confidence `proposed`). It is
applied only on request: `python -m sie.router --path <slug> --overlay docs/proposed_requires.edges.json`,
`--multi --overlay ...`, or `SIE_EDGE_OVERLAYS` for the API. Steps that depend on it are marked
`(proposed)` and keep this page as their provenance.

**How these were produced.** Four proposer agents (one per domain group) read the
SKILL.md bodies and proposed an edge only when the skill's own text *uses* something the
prerequisite teaches and does not explain it, quoting that text as evidence. One
adversarial verifier per group re-read both files and tried to refute every edge, defaulting
to reject when unsure. Scope redirects ("NOT for ... see X"), "related" and "useful
background" were grounds for rejection. Of 23 proposed edges, **9 survived** and 14 were
rejected. Adding the 9 survivors creates no `requires` cycle, and no prerequisite sits at a
higher level than the skill that depends on it.

## Accepted (9)

| prerequisite -> skill | evidence in the skill's body | verifier's reasoning |
| --- | --- | --- |
| `neural-net-fundamentals` -> `cnn-vision` | Key Concepts, 'Classic architecture lineage': "AlexNet (2012): deeper, ReLU instead of tanh, dropout"; "VGG ... hits diminishing (then negative) returns past ~20 layers due to vanishing gradients"; "ResNet ... The identity shortcut gives gradients a direct … | Accepted, but this is weaker than the rnn-sequence edge. The convolution, pooling and receptive-field arithmetic in cnn-vision stands on its own. However, the resnet-skip-connections capability is explained entirely through gradient flow: VGG fails "due to … |
| `pytorch-patterns` -> `computer-vision` | Workflow step 4 "Run the training loop.": `loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))`, `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. Workflow step 6: `model.eval()` … | The evidence is real. Step 4 uses DataLoader(train_dataset, ..., collate_fn=lambda b: tuple(zip(*b))), but train_dataset is never defined. To run it, a learner has to write a custom Dataset that returns (image, target-dict with boxes/labels), and the skill … |
| `supervised-learning` -> `explainability` | Frontmatter inputs: "A fitted, sklearn-compatible model plus held-out feature data". Workflow step 1: "# Already have a fitted sklearn-compatible estimator `model` and\n # held-out data (X_test, y_test) for the rest of this workflow." and "# Model-specific + … | Confirmed, medium-high confidence. explainability's required input is 'A fitted, sklearn-compatible model plus held-out feature data', and step 1 says 'Already have a fitted sklearn-compatible estimator `model`'. Its code works directly on that model: … |
| `data-preprocessing` -> `feature-engineering` | Overview: "Assumes the input is already cleaned/imputed/scaled (see `data-preprocessing`); the output is a reusable `Pipeline` step plus a ranked list of the features that actually helped." Frontmatter inputs: "A cleaned, imputed/scaled train/test split (see … | The quoted evidence is real. feature-engineering's frontmatter says its input is 'A cleaned, imputed/scaled train/test split (see data-preprocessing)', and the Overview says it 'Assumes the input is already cleaned/imputed/scaled (see `data-preprocessing`)'. … |
| `model-evaluation` -> `hyperparameter-tuning` | Frontmatter inputs: "a decided optimization metric (see model-evaluation)". Overview: "Assumes the metric to optimize is already decided (see `model-evaluation`)". Workflow step 4: "Get an unbiased performance estimate with nested cross-validation... Nest an … | The evidence is real. hyperparameter-tuning's inputs say 'a decided optimization metric (see model-evaluation)', and the Overview says 'Assumes the metric to optimize is already decided (see `model-evaluation`)'. The whole body is built on cross-validation: … |
| `pytorch-patterns` -> `model-optimization` | Workflow step 1: "model.eval()\n with torch.no_grad():"; step 2: "quantize_dynamic(\n model,\n {torch.nn.Linear}," and "torch.save(quantized_model.state_dict(), \"model_int8.pt\")"; step 5: "teacher.eval()\n for x, y in train_loader:\n with … | Confirmed. All quoted code is in model-optimization: model.eval() with torch.no_grad() in benchmark(), quantize_dynamic targeting torch.nn.Linear, torch.save(quantized_model.state_dict(), ...), and a distillation loop that does teacher.eval(), iterates … |
| `llm-evaluation` -> `rag-evaluation` | Workflow step 4: "This is the RAG-specific application of the faithfulness check in [[llm-evaluation]]"; code comment: "# Illustrative structure only — llm-evaluation/SKILL.md Workflow step 5 has the full runnable NLI-based faithfulness check; re-used here … | The quoted evidence is in the file word for word. In rag-evaluation, Workflow step 4 (faithfulness/groundedness) only builds (sentence, context) pairs, and its code comment calls this 'Illustrative structure only' and says llm-evaluation step 5 'has the full … |
| `neural-net-fundamentals` -> `rnn-sequence` | Key Concepts, 'Backpropagation through time (BPTT)': "This is structurally identical to backprop through a very deep feedforward network — the same product-of-derivatives that causes vanishing/exploding gradients in deep nets (see neural-net-fundamentals) … | Accepted. The BPTT bullet in rnn-sequence stops mid-explanation to hand the mechanism to another card: "the same product-of-derivatives that causes vanishing/exploding gradients in deep nets (see neural-net-fundamentals) causes it here too". It never … |
| `pytorch-patterns` -> `training-deep-models` | Frontmatter inputs: "An existing PyTorch training loop and a goal". Description: "NOT for ... idiomatic PyTorch loop/Dataset/hook structure itself (see pytorch-patterns)". Workflow step 3 uses `len(train_loader)`, step 4 … | Accepted because this is an artifact dependency. training-deep-models' `inputs` field requires "An existing PyTorch training loop". That matches the `outputs` of pytorch-patterns: "Idiomatic PyTorch Dataset/DataLoader, train/eval loop, hook, and checkpoint … |

### Frontmatter patch

```yaml
# cnn-vision/SKILL.md
requires:
  - neural-net-fundamentals
# computer-vision/SKILL.md
requires:
  - pytorch-patterns
# explainability/SKILL.md
requires:
  - supervised-learning
# feature-engineering/SKILL.md
requires:
  - data-preprocessing
# hyperparameter-tuning/SKILL.md
requires:
  - model-evaluation
# model-optimization/SKILL.md
requires:
  - pytorch-patterns
# rag-evaluation/SKILL.md
requires:
  - llm-evaluation
  - rag-pipeline
# rnn-sequence/SKILL.md
requires:
  - neural-net-fundamentals
# training-deep-models/SKILL.md
requires:
  - pytorch-patterns
```

### Learning paths that would change

| target | today | with the proposed edges |
| --- | --- | --- |
| `cnn-vision` | cnn-vision | neural-net-fundamentals -> cnn-vision |
| `computer-vision` | computer-vision | pytorch-patterns -> computer-vision |
| `explainability` | explainability | supervised-learning -> explainability |
| `feature-engineering` | feature-engineering | data-preprocessing -> feature-engineering |
| `hyperparameter-tuning` | hyperparameter-tuning | model-evaluation -> hyperparameter-tuning |
| `model-optimization` | model-optimization | pytorch-patterns -> model-optimization |
| `rag-evaluation` | rag-pipeline -> rag-evaluation | llm-evaluation -> rag-pipeline -> rag-evaluation |
| `rnn-sequence` | rnn-sequence | neural-net-fundamentals -> rnn-sequence |
| `training-deep-models` | training-deep-models | pytorch-patterns -> training-deep-models |

<details><summary>Rejected (14) and why</summary>

| prerequisite -> skill | verifier's reason for rejecting |
| --- | --- |
| `model-evaluation` -> `ai-ethics-fairness` | Refuted. First, the concept gap is not filled by the prereq. model-evaluation never defines TPR or FPR. It only uses classification_report, a confusion-matrix plot and ROC-AUC, and assumes the reader already knows … |
| `neural-net-fundamentals` -> `attention-mechanisms` | Rejected. The core mechanism is fully explained inside the card: the Q/K/V soft lookup, the full scaled dot-product formula with each term annotated, self-attention, multi-head, positional encoding and O(n²) cost. The … |
| `experiment-tracking` -> `ci-cd-for-ml` | Refuted. ci-cd-for-ml never calls any MLflow API. 'mlflow==2.16.2' is only one line in an example requirements.txt that illustrates exact version pinning. MIN_ACCEPTABLE_F1 = 0.80 is a hard-coded constant, and '# … |
| `attention-mechanisms` -> `fine-tuning-llms` | Rejected. The frontmatter link is exactly the kind the rules exclude: a 'NOT for ... the transformer/attention internals being adapted (see attention-mechanisms)' redirect. attention-mechanisms also redirects back … |
| `training-deep-models` -> `fine-tuning-llms` | Rejected. The quotes exist, but they don't show a blocking dependency. fine-tuning-llms trains through TRL's SFTTrainer, which runs the whole training loop. The learner sets config flags (bf16=True, learning_rate=2e-4, … |
| `supervised-learning` -> `hyperparameter-tuning` | This is workflow ordering, not a knowledge dependency. hyperparameter-tuning builds its own estimators (RandomForestClassifier(random_state=42), GradientBoostingClassifier(**params)). It does not take a fitted artifact … |
| `model-deployment` -> `ml-monitoring` | Refuted. 'A model already serving live traffic' is an operational precondition (the lifecycle order), not a knowledge dependency. ml-monitoring's body uses nothing that model-deployment teaches: no joblib or state_dict … |
| `supervised-learning` -> `model-evaluation` | A learner is not blocked. model-evaluation only needs 'a fitted model'. Any sklearn estimator after .fit() works, and nothing supervised-learning teaches is used in its body: no baseline comparison, candidate … |
| `neural-net-fundamentals` -> `pytorch-patterns` | Rejected. The main evidence is the description's "NOT for ... optimizer/backprop theory (see neural-net-fundamentals)", which is a scope redirect. "Debugging dead ReLUs" is only mentioned in passing, in a list of … |
| `ml-math-essentials` -> `reinforcement-learning` | This is the 'useful background' case the criteria exclude. reinforcement-learning is a conceptual reference card. Expectation ('expected discounted return'), conditional probability P(s'/s,a), variance and gradient … |
| `ml-math-essentials` -> `statistics-for-ml` | Rejected. The MAP bullet writes out the needed Bayes relation inline, `P(θ/data) ∝ P(data/θ)P(θ)`, together with its meaning ("adds a prior"). A reader does not need a separate card to apply it. The link from L1/L2 … |
| `feature-engineering` -> `supervised-learning` | The proposer admits that a clean numeric matrix from data-preprocessing alone would unblock the learner, so this is not a hard prerequisite. supervised-learning's body never uses anything feature-engineering teaches: … |
| `statistics-for-ml` -> `time-series` | The ADF usage is incidental, and the skill explicitly plays it down. Step 3 calls the p < 0.05 cutoff 'a starting guess' and tells the learner to 'let auto_arima's own order search (step 4) confirm or override it'. So … |
| `neural-net-fundamentals` -> `training-deep-models` | Rejected. The card's description explicitly places this theory out of scope ("NOT for the optimizer/backprop theory behind these choices (see neural-net-fundamentals)"), which is a redirect. training-deep-models is a … |

</details>
