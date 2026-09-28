---
name: explainability
display_name: Model Explainability
description: >
  Use when the user wants to explain what a trained model is doing — globally
  (which features matter overall) or locally (why it made a specific
  prediction). Trigger phrases: "explain this prediction", "why did the model
  predict this", "compute SHAP values", "use LIME to explain my model",
  "feature importance for this model". NOT for measuring bias/fairness across
  groups (see ai-ethics-fairness) or removing/anonymizing PII (see
  data-privacy).
type: workflow
domain: responsible-ai
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - shap-value-computation
  - lime-local-explanation
  - permutation-importance
  - global-feature-importance
  - explanation-cross-checking
requires:
conflicts:
related:
  - ai-ethics-fairness
  - data-privacy
  - model-evaluation
inputs: A fitted, sklearn-compatible model plus held-out feature data (and optionally training data as a background sample) to explain.
outputs: Global feature-importance rankings and per-instance attributions (SHAP/LIME) with a cross-method agreement check.
version_constraints:
  - "scikit-learn: sklearn.inspection.permutation_importance available since 0.22 (model-knowledge estimate, not live-verified this session)"
  - "shap: the Explanation-object plotting API (shap.plots.bar, shap.plots.waterfall) was introduced in the 0.3x release series, superseding older shap.summary_plot-style calls (model-knowledge estimate, not live-verified this session)"
---

## Overview
Turns an opaque trained model into explanations a human can act on: global
feature-importance rankings and per-instance attributions. Covers the three
workhorse techniques — permutation importance, SHAP, and LIME — and how to
sanity-check that the explanations agree with each other before trusting
them. Output is a reusable explanation pipeline, not a one-off plot.

## Workflow
1. **Pick the explanation type before picking a tool.** Decide model-specific
   vs. model-agnostic, and global vs. local — this determines which method
   below to reach for.
   ```python
   # Model-specific + fast (tree models only)   -> SHAP TreeExplainer
   # Model-agnostic + global                     -> permutation importance
   # Model-agnostic + local (single prediction)  -> SHAP KernelExplainer or LIME
   # Already have a fitted sklearn-compatible estimator `model` and
   # held-out data (X_test, y_test) for the rest of this workflow.
   ```
2. **Compute global importance with permutation importance.** Model-agnostic,
   uses only predict/score, and directly measures the metric drop the model
   actually cares about.
   ```python
   from sklearn.inspection import permutation_importance
   import numpy as np

   result = permutation_importance(
       model, X_test, y_test,
       n_repeats=30, random_state=42, scoring="roc_auc", n_jobs=-1
   )
   importances = (
       pd.Series(result.importances_mean, index=X_test.columns)
       .sort_values(ascending=False)
   )
   print(importances.head(10))
   ```
3. **Compute SHAP values for consistent local + global attribution.** For
   tree ensembles, `TreeExplainer` is exact and fast; for anything else, fall
   back to `KernelExplainer` on a background sample.
   ```python
   import shap

   # Tree models (XGBoost, LightGBM, RandomForest, ...): exact and fast.
   explainer = shap.TreeExplainer(model)
   shap_values = explainer(X_test)

   # Global view: mean |SHAP value| per feature
   shap.plots.bar(shap_values, show=False)

   # Local view: explain one prediction
   shap.plots.waterfall(shap_values[0], show=False)
   ```
   ```python
   # Non-tree / black-box models: model-agnostic but much slower.
   background = shap.sample(X_train, 100, random_state=42)
   explainer = shap.KernelExplainer(model.predict_proba, background)
   shap_values = explainer.shap_values(X_test.iloc[:50], nsamples=200)
   ```
4. **Explain a single prediction with LIME when you need a lightweight,
   surrogate-model explanation instead of SHAP's game-theoretic one.**
   ```python
   from lime.lime_tabular import LimeTabularExplainer

   lime_explainer = LimeTabularExplainer(
       X_train.values,
       feature_names=X_train.columns.tolist(),
       class_names=["negative", "positive"],
       mode="classification",
       discretize_continuous=True,
       random_state=42,
   )
   instance = X_test.iloc[0].values
   explanation = lime_explainer.explain_instance(
       instance, model.predict_proba, num_features=8
   )
   explanation.show_in_notebook(show_table=True)   # or .as_list() for raw output
   ```
5. **Cross-check explanations before trusting them.** Different methods
   should broadly agree on the top drivers; large disagreement is a signal to
   dig deeper (correlated features, unstable local surrogate, or a
   mis-specified background dataset), not to average blindly.
   ```python
   shap_rank = (
       pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_test.columns)
       .sort_values(ascending=False)
   )
   agreement = pd.concat(
       [importances.rename("permutation"), shap_rank.rename("shap")], axis=1
   ).dropna()
   print(agreement.head(10))   # eyeball rank agreement on the top features
   ```

## Gotchas
- **`KernelExplainer` is slow and approximate.** It's model-agnostic but
  samples feature coalitions, so results are noisy and expensive at scale —
  always prefer `TreeExplainer` (tree models) or `DeepExplainer`/`GradientExplainer`
  (neural nets) when applicable.
- **Correlated features break SHAP's additivity assumption.** SHAP assumes
  features can be independently "turned off"; with strongly correlated
  features (e.g. `height_cm` and `height_in`), importance gets arbitrarily
  split between them rather than attributed to the underlying signal.
- **LIME explanations are unstable.** Because it fits a local linear surrogate
  on randomly perturbed samples, re-running `explain_instance` on the same row
  can produce visibly different top features — always set a `random_state`
  and consider averaging over multiple runs before reporting a result.
- **Permutation importance is biased downward for correlated features.**
  Shuffling one of two correlated features barely hurts the model (the other
  still carries the signal), understating both features' true importance.
- **High feature importance is not causation.** All three methods describe
  what the model relies on statistically, not what actually drives the
  real-world outcome — don't present SHAP/LIME/permutation results as causal
  claims without a causal design to back them.

## References
- [SHAP documentation](https://shap.readthedocs.io/en/latest/) — API reference and explainer selection guide (Tree/Kernel/Deep/Linear).
- [Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874) — the SHAP paper; defines the Shapley-value formulation.
- [Ribeiro, Singh & Guestrin (2016), "Why Should I Trust You?" (LIME)](https://arxiv.org/abs/1602.04938) — the original LIME paper.
- [scikit-learn: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) — covers the correlated-feature bias gotcha in detail.
