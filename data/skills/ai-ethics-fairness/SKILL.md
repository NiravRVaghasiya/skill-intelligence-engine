---
name: ai-ethics-fairness
display_name: AI Ethics & Fairness
description: >
  Use when the user wants to reason about or measure bias and fairness in a
  dataset or model. Trigger phrases: "check my model for bias", "is this
  model fair across groups", "compute demographic parity", "run a fairness
  audit", "disparate impact analysis", "which fairness metric should I use".
  NOT for explaining individual predictions (see explainability) or
  detecting/anonymizing PII (see data-privacy).
type: reference
domain: responsible-ai
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: primary
last_verified: 2026-09-21
capabilities:
  - demographic-parity
  - equalized-odds
  - disparate-impact-ratio
  - fairness-metric-tradeoffs
  - bias-mitigation-strategies
requires:
conflicts:
related:
  - explainability
  - data-privacy
  - model-evaluation
  - ai-ml-security
inputs: A trained model's predictions plus group/protected-attribute labels for the population being evaluated.
outputs: A set of group-fairness metric values, an explanation of their mutual tradeoffs, and a recommended mitigation approach.
---

## Overview
Covers how bias enters ML systems and how to quantify fairness once a model
exists — which group-fairness metrics to compute, how they trade off against
each other, and the standard mitigation strategies (pre-, in-, and
post-processing). The user walks away knowing which metric answers their
actual question and why no single metric is a silver bullet.

## Key Concepts
- **Where bias comes from.** *Historical/label bias* — the training labels
  encode past discriminatory decisions (e.g. past loan approvals). *Sampling
  bias* — some groups are under-represented in the collected data.
  *Measurement bias* — features are measured with different fidelity or
  meaning across groups (e.g. a proxy like "zip code" behaves differently by
  region). Fixing the metric downstream never fixes bias baked in upstream.
- **Group fairness metrics** — the main ones, and the question each answers:
  - *Demographic (statistical) parity*: \(P(\hat{y}=1 \mid A=a)\) equal across
    groups \(a\). "Does the model select each group at the same rate?"
  - *Equalized odds*: TPR and FPR equal across groups. "Is the model equally
    accurate for each group, conditioned on the true label?"
  - *Equal opportunity*: TPR equal across groups only (relaxation of
    equalized odds) — common in loan/hiring contexts where false negatives
    (denying a qualified candidate) matter most.
  - *Predictive parity*: precision equal across groups — "when the model says
    yes, is it equally trustworthy for every group?"
  - *Disparate impact ratio* ("80% rule"): \(\frac{P(\hat{y}=1|A=\text{minority})}{P(\hat{y}=1|A=\text{majority})}\)
    below 0.8 is a common (US EEOC-derived) legal red flag threshold.
- **Individual vs. group fairness.** Group fairness balances aggregate rates
  across protected groups; individual fairness demands "similar individuals
  get similar predictions." The two can conflict — satisfying one does not
  guarantee the other.
- **The impossibility result.** Chouldechova (2017) and Kleinberg, Mullainathan
  & Raghavan (2016) showed that when base rates differ across groups,
  demographic parity, equalized odds, and predictive parity **cannot all hold
  simultaneously** except in degenerate cases. Picking a fairness metric is a
  value judgment about which error the deployment context cares about most,
  not a purely technical choice.
- **Mitigation strategies, by stage:**
  - *Pre-processing* — reweight or resample training examples so groups have
    balanced influence (e.g. Kamiran & Calders reweighing).
  - *In-processing* — add a fairness constraint or regularizer directly to
    the training objective (e.g. Fairlearn's `ExponentiatedGradient`).
  - *Post-processing* — adjust decision thresholds per group after training
    (e.g. Hardt et al.'s equalized-odds post-processing). Cheapest to apply
    but requires access to the protected attribute at inference time, which
    is not always legal or available.

## Gotchas
- **Metrics conflict by design.** Don't chase "fix demographic parity AND
  equalized odds" — per the impossibility result, that's only possible if
  base rates are equal across groups. Pick the metric that matches the
  real-world cost of the error you're trying to avoid.
- **Removing the protected attribute isn't enough.** Correlated proxy
  features (zip code, name, school) let the model reconstruct group
  membership indirectly ("fairness through unawareness" fails). You typically
  need the attribute *during auditing/mitigation*, just not as a model input.
- **Small subgroup counts produce noisy metrics.** A fairness gap computed on
  40 samples of a minority group can be sampling noise, not a real effect —
  always report confidence intervals or minimum group sizes alongside the
  metric.
- **Aggregation can hide or fabricate disparities (Simpson's paradox).**
  A model can look fair overall while being unfair within every subgroup
  once you condition on a hidden confounder (e.g. department, region) —
  always slice by intersections of relevant attributes, not just top-level
  groups.
- **Accuracy–fairness tradeoffs are real but not fixed.** Constrained
  training often costs some aggregate accuracy; report both numbers rather
  than presenting a "fixed" model as free of tradeoffs.

## References
- [Fairlearn documentation](https://fairlearn.org/) — open-source toolkit with implementations of the metrics and mitigations above.
- [Hardt, Price & Srebro (2016), "Equality of Opportunity in Supervised Learning"](https://arxiv.org/abs/1610.02413) — defines equalized odds/opportunity and the post-processing fix.
- [Barocas, Hardt & Narayanan, *Fairness and Machine Learning*](https://fairmlbook.org/) — the standard free textbook covering the impossibility results and mitigation taxonomy in depth.
- [AIF360 (IBM AI Fairness 360) toolkit](https://aif360.res.ibm.com/) — reference implementations of bias metrics and pre/in/post-processing mitigation algorithms.
