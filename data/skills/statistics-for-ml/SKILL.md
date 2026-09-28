---
name: statistics-for-ml
display_name: Statistics for ML
description: >
  Use when the user wants to reason about distributions, run or interpret a
  hypothesis test, size a confidence interval, or understand MLE vs. MAP
  estimation. Trigger phrases: "is this difference statistically
  significant", "which distribution fits this data", "explain this p-value",
  "MLE vs MAP", "how do I A/B test this". NOT for the underlying calculus or
  linear algebra (see ml-math-essentials), and NOT for model performance
  metrics like AUC/F1/calibration (see model-evaluation).
type: reference
domain: foundations
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - hypothesis-testing
  - p-value-interpretation
  - confidence-interval-construction
  - mle-map-estimation
  - ab-test-design
  - multiple-comparison-correction
requires:
conflicts:
related:
  - ml-math-essentials
  - model-evaluation
  - supervised-learning
inputs: A question about which distribution/test/estimator applies to a dataset or experiment result, or a p-value/confidence interval to interpret.
outputs: Guidance on which distribution, statistical test, or estimator (MLE/MAP) fits the situation, plus correct interpretation of resulting p-values/CIs and pitfalls to check for.
---

## Overview
Covers the statistical toolkit ML practitioners lean on outside of the
model-fitting step itself: knowing which distribution describes your data,
estimating parameters from samples (MLE/MAP), and deciding whether an
observed difference (a new model, a new feature, an A/B test) is real or
noise. This is the "is it true" layer that sits underneath model evaluation
metrics — a metric can look better and still be a coin flip.

## Key Concepts
- **Random variables & common distributions.** Bernoulli/Binomial (binary
  outcomes, e.g. click/no-click), Poisson (event counts in a fixed window,
  e.g. arrivals per minute), Normal/Gaussian (sums of many small effects, by
  the CLT), Exponential (time between Poisson events). Picking the right
  distribution shapes both the model likelihood and which tests are valid.
- **The Central Limit Theorem (CLT).** The sampling distribution of a sample
  mean approaches Normal as sample size grows, *regardless of the underlying
  distribution* — this is why t-tests and z-intervals on averages are
  broadly applicable even when raw data isn't Gaussian, provided n is large
  enough (rule of thumb: n ≳ 30, more for heavy-tailed data).
- **Maximum Likelihood Estimation (MLE).** Choose the parameter \(\theta\)
  that maximizes \(P(\text{data} \mid \theta)\) — i.e., makes the observed
  data most probable. Minimizing cross-entropy loss *is* MLE for a
  categorical/Bernoulli likelihood; minimizing squared error *is* MLE under a
  Gaussian-noise assumption. Most standard ML losses are MLE in disguise.
- **Maximum A Posteriori (MAP).** Adds a prior: maximize
  \(P(\theta\mid\text{data}) \propto P(\text{data}\mid\theta)P(\theta)\).
  L2 regularization is MAP with a Gaussian prior on weights; L1 is MAP with a
  Laplace prior. MAP == MLE when the prior is flat (uninformative).
- **Hypothesis testing.** State a null hypothesis \(H_0\) (e.g. "no
  difference"), compute a test statistic, and get a **p-value** — the
  probability of seeing data this extreme *if \(H_0\) were true*. A small
  p-value is evidence against \(H_0\), not proof of the alternative, and not
  the probability that \(H_0\) is true.
- **Type I/II errors and power.** Type I = false positive (rejecting a true
  \(H_0\), rate \(\alpha\)); Type II = false negative (rate \(\beta\));
  power = \(1-\beta\) = probability of detecting a real effect. Underpowered
  tests (small samples, small expected effect) routinely produce misleading
  "no significant difference" results.
- **Confidence intervals.** A 95% CI is a range constructed by a procedure
  that captures the true parameter 95% of the time *across repeated
  sampling* — a single interval either does or doesn't contain the truth; it
  is not "95% probability the parameter is in this range" (that's a Bayesian
  credible interval, a related but distinct concept).
- **A/B testing pitfalls.** Peeking at results before the pre-committed
  sample size inflates false positives; testing many metrics/segments without
  correction (Bonferroni, Benjamini-Hochberg) does too. Both are extremely
  common in practice.

## Gotchas
- **p-value misinterpretation.** `p = 0.03` does *not* mean "3% chance the
  null hypothesis is true" — it means "if the null were true, data this
  extreme would occur 3% of the time." Conflating the two is the single most
  common statistics error in applied ML/analytics writeups.
- **Multiple comparisons without correction.** Running 20 hypothesis tests at
  \(\alpha=0.05\) with no correction gives roughly a 64% chance of at least
  one false positive by chance alone. Apply Bonferroni or Benjamini-Hochberg
  when testing many metrics or segments.
- **Assuming normality without checking.** t-tests and z-intervals rely on
  CLT kicking in; with small samples and skewed/heavy-tailed data (e.g.
  revenue, latency), the normal approximation can be badly wrong — check with
  a Q-Q plot or use a non-parametric test (Mann-Whitney U) or bootstrap
  instead.
- **Peeking / early stopping in A/B tests.** Checking results daily and
  stopping as soon as `p < 0.05` appears inflates the true false-positive
  rate far above 5% — use a pre-registered sample size or a sequential
  testing method designed for repeated looks.
- **Correlation vs. causation.** A statistically significant correlation
  between a feature and the target says nothing about whether intervening on
  the feature changes the outcome — confounders can produce strong,
  significant, spurious correlations.

## References
- [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) — implementations of every distribution and test mentioned here.
- [Wasserman, *All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9) — the standard concise graduate-level reference bridging classical stats and ML.
- [Evan Miller — How Not To Run An A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) — the canonical explanation of the peeking problem.
