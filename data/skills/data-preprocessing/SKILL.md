---
name: data-preprocessing
display_name: Data Preprocessing
description: >
  Use when the user wants to clean, encode, scale, or impute a tabular dataset
  before modeling. Trigger phrases: "clean this data", "handle missing values",
  "encode categorical features", "scale my features", "prep this dataset for
  training", "build a preprocessing pipeline". NOT for feature *creation* (see
  feature-engineering) or model training (see supervised-learning).
type: workflow
domain: classical-ml
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - missing-value-imputation
  - categorical-encoding
  - feature-scaling
  - column-transformer-pipelines
  - train-test-leakage-prevention
requires:
conflicts:
related:
  - feature-engineering
  - model-evaluation
  - supervised-learning
inputs: A raw tabular dataset (pandas DataFrame) with a defined target column, not yet split or transformed.
outputs: A fitted scikit-learn Pipeline/ColumnTransformer that turns raw rows into a clean numeric matrix, plus a persisted joblib artifact.
version_constraints:
  - "scikit-learn: ColumnTransformer + OneHotEncoder(handle_unknown='ignore') API stable since 0.20+ — not live-verified this pass"
  - "scikit-learn: SimpleImputer stable since 0.20+ — not live-verified this pass"
---

## Overview
Turns a raw tabular dataset into a clean, numeric, leakage-free matrix ready for
modeling. The output is a fitted, reusable `ColumnTransformer`/`Pipeline` — never
hand-transformed arrays — so the exact same steps apply to train, validation, and
production data. The golden rule: **fit on train only, transform everywhere.**

## Workflow
1. **Profile the data first.** Understand dtypes, missingness, cardinality, and
   target balance before touching anything.
   ```python
   import pandas as pd
   df.info()
   print(df.isna().mean().sort_values(ascending=False))   # missingness per column
   print(df.select_dtypes("object").nunique())            # categorical cardinality
   ```
2. **Split BEFORE preprocessing.** Fitting any transformer on the full dataset
   leaks test information. Split first, then fit only on train.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42)
   ```
3. **Declare column groups.** Separate numeric vs. categorical so each gets the
   right treatment.
   ```python
   num_cols = X_train.select_dtypes("number").columns.tolist()
   cat_cols = X_train.select_dtypes(["object", "category"]).columns.tolist()
   ```
4. **Build per-type pipelines.** Impute → scale for numeric; impute → encode for
   categorical. Bundle in a `ColumnTransformer`.
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import StandardScaler, OneHotEncoder

   numeric = Pipeline([
       ("impute", SimpleImputer(strategy="median")),
       ("scale", StandardScaler()),
   ])
   categorical = Pipeline([
       ("impute", SimpleImputer(strategy="most_frequent")),
       ("encode", OneHotEncoder(handle_unknown="ignore")),
   ])
   preprocess = ColumnTransformer([
       ("num", numeric, num_cols),
       ("cat", categorical, cat_cols),
   ])
   ```
5. **Chain into the full model pipeline & fit once.** Preprocessing and the
   estimator live in one object, so cross-validation refits preprocessing on each
   fold (no leakage).
   ```python
   from sklearn.ensemble import RandomForestClassifier
   clf = Pipeline([("prep", preprocess),
                   ("model", RandomForestClassifier(random_state=42))])
   clf.fit(X_train, y_train)
   ```
6. **Persist the fitted pipeline.** Ship the whole object — it carries the learned
   medians, scales, and encoder categories.
   ```python
   import joblib
   joblib.dump(clf, "model.joblib")
   ```

## Gotchas
- **Leakage from early fitting** is the #1 error: scaling/imputing before the split,
  or fitting the encoder on train+test combined. Always fit on train only.
- **Unseen categories at inference** crash `OneHotEncoder` unless you set
  `handle_unknown="ignore"`.
- **Scaling tree models is wasteful** — random forests/gradient boosting are
  scale-invariant. Scaling only matters for distance/gradient-based models
  (SVM, kNN, linear, neural nets).
- **Imputing the target** — never impute or scale `y`. Handle target issues separately.
- **High-cardinality categoricals** blow up with one-hot. As a rule of thumb —
  not a hard threshold — once a column has on the order of dozens of unique
  values, start evaluating target/ordinal encoding (see feature-engineering)
  instead; the right cutoff depends on row count and downstream model, so treat
  ~50 as a prompt to check, not a rule to apply blindly.

## References
- [scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — canonical transformer reference.
- [scikit-learn: ColumnTransformer guide](https://scikit-learn.org/stable/modules/compose.html) — mixed-type pipelines done right.
