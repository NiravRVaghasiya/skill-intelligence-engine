---
name: feature-engineering
display_name: Feature Engineering
description: >
  Use when the user wants to create, extract, or select predictive features from
  an already-clean dataset. Trigger phrases: "engineer new features", "create
  interaction terms", "select the best features", "reduce feature leakage",
  "encode a high-cardinality categorical", "which features matter most". NOT for
  cleaning/imputing raw data (see data-preprocessing) or reducing dimensionality
  for visualization/clustering (see unsupervised-learning).
type: workflow
domain: classical-ml
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - feature-interaction-creation
  - target-encoding
  - feature-selection
  - leakage-safe-feature-pipelines
  - recursive-feature-elimination
requires:
conflicts:
related:
  - data-preprocessing
  - supervised-learning
  - unsupervised-learning
  - model-evaluation
inputs: A cleaned, imputed/scaled train/test split (see data-preprocessing) with a defined target column.
outputs: A ranked, leakage-checked feature set plus a reusable Pipeline step for engineering and selection.
version_constraints:
  - "scikit-learn: preprocessing.TargetEncoder (cross-fitted target encoding) available since 1.3 — not live-verified this pass"
---

## Overview
Turns a clean, numeric-ready dataset into a set of *predictive* features and then
trims that set down to the ones worth keeping. Covers feature creation (interactions,
domain transforms, target encoding for high-cardinality categoricals) and feature
selection (filter, wrapper, embedded methods) — always with an eye on leakage, since
feature engineering is where leakage most often sneaks in unnoticed. Assumes the
input is already cleaned/imputed/scaled (see `data-preprocessing`); the output is a
reusable `Pipeline` step plus a ranked list of the features that actually helped.

## Workflow
1. **Split before engineering anything.** Any feature that "learns" from data (target
   encoding, feature selection, PCA loadings) must be fit on train only.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42)
   ```
2. **Create interaction and domain features.** Let the model see combinations it
   can't easily discover on its own — ratios, products, and datetime decompositions
   are the highest-leverage, lowest-risk additions.
   ```python
   import numpy as np
   from sklearn.preprocessing import PolynomialFeatures

   # Pairwise interactions/ratios for a handful of numeric columns (not the whole matrix —
   # this explodes combinatorially).
   poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
   X_train_inter = poly.fit_transform(X_train[["price", "sqft"]])

   # Domain feature: a ratio that's meaningful to a human.
   X_train["price_per_sqft"] = X_train["price"] / X_train["sqft"].replace(0, np.nan)

   # Datetime decomposition — raw timestamps are useless to most models.
   X_train["order_dow"] = X_train["order_ts"].dt.dayofweek
   X_train["order_hour"] = X_train["order_ts"].dt.hour
   X_train["order_is_weekend"] = X_train["order_dow"].isin([5, 6]).astype(int)
   ```
3. **Encode high-cardinality categoricals safely.** As a rule of thumb — not a hard
   threshold — once a categorical column's one-hot expansion starts rivaling or
   exceeding your row count or feature budget (often somewhere in the dozens-to-
   hundreds of unique values, depending on dataset size and downstream model),
   target encoding compresses it to one leakage-aware column instead. Use sklearn's
   built-in encoder — it internally cross-fits to avoid leaking the target into its
   own encoding.
   ```python
   from sklearn.preprocessing import TargetEncoder

   te = TargetEncoder(target_type="continuous", cv=5, random_state=42)
   zip_encoded_train = te.fit_transform(X_train[["zip_code"]], y_train)
   zip_encoded_test = te.transform(X_test[["zip_code"]])   # transform only — never re-fit
   ```
4. **Filter features by univariate relevance.** Cheap first pass to drop obvious
   noise before running anything expensive.
   ```python
   from sklearn.feature_selection import SelectKBest, mutual_info_classif

   selector = SelectKBest(score_func=mutual_info_classif, k=20)
   X_train_filtered = selector.fit_transform(X_train_numeric, y_train)
   kept_cols = X_train_numeric.columns[selector.get_support()]
   ```
5. **Rank by embedded importance and prune.** Let a model that computes feature
   importance natively (L1-regularized linear model or tree ensemble) do the
   selection — cheaper than a full wrapper search.
   ```python
   from sklearn.feature_selection import SelectFromModel
   from sklearn.ensemble import RandomForestClassifier

   embedded = SelectFromModel(
       RandomForestClassifier(n_estimators=300, random_state=42),
       threshold="median",   # keep the top half by importance
   )
   embedded.fit(X_train_filtered, y_train)
   final_cols = kept_cols[embedded.get_support()]
   ```
6. **(Optional, small feature sets only) Wrapper search with cross-validated RFE.**
   Most rigorous, most expensive — reserve for a shortlist of candidate features,
   not hundreds.
   ```python
   from sklearn.feature_selection import RFECV
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import StratifiedKFold

   rfecv = RFECV(
       estimator=LogisticRegression(max_iter=1000),
       cv=StratifiedKFold(5), scoring="roc_auc", min_features_to_select=5,
   )
   rfecv.fit(X_train[final_cols], y_train)
   selected = final_cols[rfecv.support_]
   ```
7. **Wrap engineering + selection in one `Pipeline` and validate honestly.** Never
   report a score computed on the same fold used to pick features — that number is
   inflated by selection leakage.
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.model_selection import cross_val_score

   full_pipe = Pipeline([
       ("select", SelectFromModel(RandomForestClassifier(n_estimators=300, random_state=42))),
       ("model", LogisticRegression(max_iter=1000)),
   ])
   scores = cross_val_score(full_pipe, X_train_numeric, y_train, cv=5, scoring="roc_auc")
   print(scores.mean(), scores.std())
   ```

## Gotchas
- **Target/mean encoding fit on the full dataset leaks the label into itself.**
  Always use a cross-fitted encoder (`sklearn.preprocessing.TargetEncoder`) or fit
  strictly on train and `transform` (never `fit_transform`) on test.
- **Selecting features outside cross-validation inflates the reported score.**
  Selection must happen *inside* each CV fold (via `Pipeline` + `cross_val_score`),
  not once on the whole training set before CV.
- **Look-ahead leakage in time-ordered data.** A rolling mean, lag feature, or
  "days since X" column computed using future rows (e.g., a centered rolling window,
  or a global aggregate that includes future dates) silently leaks the future into
  the past. Use `shift()`/expanding windows anchored strictly before the prediction
  time.
- **Polynomial/interaction features explode dimensionality fast.** `degree=2` on 50
  numeric columns produces >1,200 features, most of them noise that hurts linear
  models and slows tree models. Apply interactions to a hand-picked subset, not the
  whole matrix.
- **Correlated features destabilize importance rankings.** Tree/L1 importance
  splits credit arbitrarily among collinear features, making "top features" look
  different across random seeds — drop near-duplicates before ranking, don't trust
  a single run's order.

## References
- [scikit-learn: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) — filter/wrapper/embedded methods with API details.
- [scikit-learn: TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) — the built-in, leakage-aware encoder used in step 3.
- [scikit-learn: Common pitfalls — data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) — canonical explanation of why selection must live inside CV.
