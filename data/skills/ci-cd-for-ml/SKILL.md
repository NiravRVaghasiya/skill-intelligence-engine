---
name: ci-cd-for-ml
display_name: CI/CD for ML
description: >
  Use when the user wants to automate testing, reproducibility, or release
  gating for an ML codebase or model — CI pipelines that must pass before a
  model or pipeline change ships. Trigger phrases: "set up CI for my ML repo",
  "add tests for my training pipeline", "automate model retraining and
  release", "gate deployment on model quality", "make my training run
  reproducible". NOT for the runtime serving infrastructure itself (see
  model-deployment) or logging individual training runs (see
  experiment-tracking).
type: workflow
domain: mlops
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - ml-ci-pipeline-setup
  - data-contract-testing
  - model-quality-gating
  - reproducible-environment-pinning
  - release-promotion-automation
requires:
conflicts:
related:
  - experiment-tracking
  - model-deployment
  - data-pipelines
inputs: An ML codebase (training pipeline and model code) in a version-controlled repo with a CI runner (e.g. GitHub Actions) available.
outputs: A CI workflow that blocks merges/releases on data-contract and model-quality test failures, plus a gated release job that promotes only passing models.
---

## Overview
Applies standard software CI/CD discipline to an ML codebase, plus the ML-specific
pieces vanilla CI/CD doesn't cover: pinned/reproducible environments, data-contract
tests, and a model-quality gate that can actually fail a build. Covers pytest-based
tests for data and model quality, a GitHub Actions workflow that runs them on every PR,
and a release job that only promotes a model when the quality gate is green. The output
is a pipeline where a bad data change or a regressed model can't merge, not just a repo
that has tests somewhere.

## Workflow
1. **Pin the environment.** CI reproducibility starts with exact versions, and fixing
   every source of randomness so tests aren't flaky.
   ```txt
   # requirements.txt — exact versions, not ranges
   scikit-learn==1.5.2
   pandas==2.2.3
   mlflow==2.16.2
   pytest==8.3.3
   ```
   ```python
   # conftest.py
   import random
   import numpy as np
   import pytest

   @pytest.fixture(autouse=True)
   def _seed_everything():
       random.seed(42)
       np.random.seed(42)
   ```
2. **Write data-contract tests**, not just code tests — schema, nulls, and leakage.
   ```python
   # tests/test_data_contract.py
   from src.pipeline import load_training_data

   def test_schema_matches_contract():
       df = load_training_data()
       expected = {"customer_id", "orders_30d", "spend_30d", "churned"}
       assert expected.issubset(df.columns)

   def test_no_target_leakage_into_features():
       df = load_training_data()
       feature_cols = [c for c in df.columns if c != "churned"]
       assert "churned" not in feature_cols

   def test_no_nulls_in_required_columns():
       df = load_training_data()
       assert df[["customer_id", "churned"]].isna().sum().sum() == 0
   ```
3. **Write a model-quality gate test.** This is what makes CI able to reject a model
   regression, not just a syntax error.
   ```python
   # tests/test_model_quality.py
   from sklearn.metrics import f1_score
   from src.pipeline import load_training_data, train_model, split

   MIN_ACCEPTABLE_F1 = 0.80   # baseline captured via experiment-tracking

   def test_model_beats_quality_bar():
       df = load_training_data()
       X_train, X_test, y_train, y_test = split(df)
       model = train_model(X_train, y_train)
       preds = model.predict(X_test)
       score = f1_score(y_test, preds, average="macro")
       assert score >= MIN_ACCEPTABLE_F1, f"F1 {score:.3f} below bar {MIN_ACCEPTABLE_F1}"
   ```
4. **Wire it into CI**, running on every pull request.
   ```yaml
   # .github/workflows/ci.yml
   name: ml-ci
   on:
     pull_request:
     push:
       branches: [main]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.11"
         - run: pip install -r requirements.txt
         - run: pytest tests/ -v --maxfail=1
         - run: python -m src.validate_pipeline_config   # fail fast on config errors
   ```
5. **Gate the release stage on the same quality test.** Promotion only happens if the
   gate job passes.
   ```yaml
   # .github/workflows/release.yml
   name: ml-release
   on:
     push:
       branches: [main]

   jobs:
     quality-gate:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.11"
         - run: pip install -r requirements.txt
         - run: pytest tests/test_model_quality.py -v   # the gate

     promote:
       needs: quality-gate
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Register model version as Production
           run: python -m src.promote_model --stage Production
   ```

## Gotchas
- **Unpinned dependencies** (`scikit-learn>=1.5`) make CI green today and red in three
  months when a transitive dependency ships a breaking change on an unrelated PR — pin
  exact versions and bump them deliberately, in their own PR.
- **A "model quality" test that retrains on the full dataset in CI** can exceed CI
  timeouts and burns compute on every PR — train on a small, fixed, checked-in sample for
  the gate, and run full retraining as a separate scheduled job (see data-pipelines).
- **Not seeding every source of randomness** (Python's `random`, NumPy, and
  framework-specific seeds like `torch.manual_seed`) makes the quality-gate test flaky —
  it passes on some runs and fails on others with no code change, and teams learn to
  ignore it.
- **Testing only code, never the data contract**, lets a broken upstream data change ship
  silently — the pipeline runs fine and all code tests pass; it just trains on garbage.
- **Gating release on a single aggregate metric** without a slice-level regression check
  against the current production model can let a model that's meaningfully worse on a
  minority class or critical segment pass, because the aggregate score still clears the
  bar.

## References
- [GitHub Actions docs](https://docs.github.com/en/actions) — workflow syntax, jobs, and triggers.
- [pytest docs](https://docs.pytest.org/en/stable/) — fixtures, parametrization, and test organization.
- [Google: Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) — practical guidance that informs the testing/gating practices above.
