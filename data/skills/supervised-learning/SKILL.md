---
name: supervised-learning
display_name: Supervised Learning
description: >
  Use when the user wants to pick and fit a regression or classification model.
  Trigger phrases: "which model should I use", "train a classifier", "fit a
  regression model", "baseline vs. a real model", "compare a few algorithms",
  "predict a numeric/categorical target". NOT for building the input features
  (see feature-engineering), scoring/comparing models rigorously (see
  model-evaluation), or tuning the winning model's hyperparameters (see
  hyperparameter-tuning).
type: workflow
domain: classical-ml
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - baseline-model-comparison
  - classifier-model-selection
  - regression-model-selection
  - cross-validated-model-comparison
  - class-imbalance-handling
requires:
conflicts:
related:
  - feature-engineering
  - model-evaluation
  - hyperparameter-tuning
  - data-preprocessing
inputs: A feature-engineered train/test split (see feature-engineering) with a numeric or categorical target.
outputs: A fitted candidate model, or a ranked shortlist of candidates with cross-validated scores, ready for tuning or rigorous evaluation.
---

## Overview
Goes from a clean, feature-engineered matrix to a fitted model, via a disciplined
spot-check: always start with a trivial baseline, then compare a linear model
against a tree ensemble under the same cross-validation splits, and pick a winner
by a defensible margin — not by eyeballing one fold. Model *scoring* mechanics
(metrics, calibration, CV design) live in `model-evaluation`; this skill is about
which estimator to fit and how to compare candidates fairly.

## Workflow
1. **Always fit a dumb baseline first.** Without one, a 0.75 R² or 80% accuracy
   means nothing — you need a floor to beat.
   ```python
   from sklearn.dummy import DummyClassifier, DummyRegressor
   from sklearn.model_selection import cross_val_score

   baseline = DummyClassifier(strategy="most_frequent")
   baseline_scores = cross_val_score(baseline, X_train, y_train, cv=5, scoring="roc_auc")
   print("baseline:", baseline_scores.mean())
   ```
2. **Fit a linear model as the second baseline.** Fast, interpretable, and a strong
   floor for anything roughly linearly separable — it also tells you whether feature
   scaling/engineering is doing its job.
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   linear = Pipeline([
       ("scale", StandardScaler()),
       ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
   ])
   linear_scores = cross_val_score(linear, X_train, y_train, cv=5, scoring="roc_auc")
   print("linear:", linear_scores.mean())
   ```
3. **Fit a tree ensemble.** Handles non-linearities and interactions without manual
   feature engineering, is scale-invariant, and is a strong default for tabular data.
   ```python
   from sklearn.ensemble import HistGradientBoostingClassifier

   trees = HistGradientBoostingClassifier(random_state=42)
   tree_scores = cross_val_score(trees, X_train, y_train, cv=5, scoring="roc_auc")
   print("trees:", tree_scores.mean())
   ```
4. **Compare candidates under identical folds, not separate ones.** Reuse a single
   `cv` splitter object across every `cross_val_score` call so the exact same rows
   are held out for every model — otherwise differences may just be fold luck.
   ```python
   from sklearn.model_selection import StratifiedKFold
   from sklearn.svm import SVC

   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   candidates = {
       "baseline": baseline,
       "logistic": linear,
       "hist_gbm": trees,
       "svm_rbf": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True))]),
   }
   results = {
       name: cross_val_score(est, X_train, y_train, cv=cv, scoring="roc_auc").mean()
       for name, est in candidates.items()
   }
   print(sorted(results.items(), key=lambda kv: -kv[1]))
   ```
5. **Handle class imbalance at the model level, not just via metrics.** Reweight
   or resample before assuming a fancier model is needed.
   ```python
   from sklearn.ensemble import RandomForestClassifier

   balanced_rf = RandomForestClassifier(
       n_estimators=300, class_weight="balanced_subsample", random_state=42
   )
   ```
6. **Fit the chosen model on the full training set for handoff.** Once a winner is
   picked (by a margin that survives `model-evaluation`'s scrutiny — see that skill
   for how to judge "better"), fit it once on all of train before tuning or shipping.
   ```python
   final_model = trees.fit(X_train, y_train)
   ```

## Gotchas
- **Comparing models on different CV folds.** Calling `cross_val_score` separately
  with `cv=5` (an int) reshuffles/reselects folds differently per call in some
  configurations — pass one shared `KFold`/`StratifiedKFold` *object* so every
  candidate sees the same splits.
- **Skipping the baseline.** A model that "achieves 92% accuracy" is meaningless if
  the majority class is 91% of the data — always compare against `DummyClassifier`/
  `DummyRegressor` first.
- **Scaling trees, or not scaling linear/distance models.** `StandardScaler` is
  required for `LogisticRegression`, `SVC`, and kNN but wasted effort (and harmless)
  for tree ensembles — know which family you're fitting.
- **Ignoring class imbalance until the model already looks "good."** A model with no
  `class_weight` adjustment on a 95/5 split will default to predicting the majority
  class almost everywhere; fix this at fit time, not after the fact with threshold
  hacks.
- **Picking a winner from a single train/test split.** A model that wins by 0.3
  points on one split can lose on another — only trust a difference that's larger
  than the cross-validation fold-to-fold standard deviation.

## References
- [scikit-learn: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html) — full catalog of estimators by family.
- [scikit-learn: Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html) — flowchart for narrowing model families fast.
- [scikit-learn: HistGradientBoosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) — the strong tabular-data default used in step 3.
