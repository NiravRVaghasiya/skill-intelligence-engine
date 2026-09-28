---
name: experiment-tracking
display_name: Experiment Tracking
description: >
  Use when the user wants to log, compare, or reproduce ML experiments —
  tracking hyperparameters, metrics, and artifacts across training runs.
  Trigger phrases: "track my experiments", "log metrics with MLflow",
  "compare training runs", "set up W&B logging", "which run had the best
  accuracy". NOT for searching the hyperparameter space itself (see
  hyperparameter-tuning) or watching a deployed model's live performance
  (see ml-monitoring).
type: workflow
domain: mlops
level: beginner
lifecycle: stable
risk_level: low
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - experiment-run-logging
  - hyperparameter-metric-tracking
  - run-comparison-querying
  - model-registry-staging
  - run-reproducibility
requires:
conflicts:
related:
  - hyperparameter-tuning
  - model-evaluation
  - ci-cd-for-ml
inputs: A training script or notebook whose hyperparameters, metrics, and artifacts need to be logged, plus a tracking backend (local mlruns/ dir or a shared MLflow/W&B server).
outputs: Logged, queryable runs grouped by experiment, a comparison table across runs, and a registered model version with a stage label.
version_constraints:
  - "mlflow: Model Registry stage-based transitions (`transition_model_version_stage`, used in step 4) were marked deprecated in favor of aliases/tags starting around MLflow 2.9 per the project's own changelog — this is a model-knowledge recollection, not live-verified this session; check the installed version's docs before relying on stages in new code."
---

## Overview
Turns ad-hoc training scripts into a searchable history of runs: every hyperparameter,
metric, and artifact gets logged against a run ID, so "what changed between the good
run and the bad run" is a query instead of a guess. This skill covers MLflow (with a W&B
equivalent) end to end — instrument a run, compare runs, and promote the best one to a
registry. The output is a reproducible, comparable record, not just a training script
that prints to stdout.

## Workflow
1. **Point at a tracking store and name the experiment.** Everything logged after this
   groups under one experiment so runs are comparable.
   ```python
   # pip install mlflow
   import mlflow

   mlflow.set_tracking_uri("http://localhost:5000")  # or a shared server URI in a team setting
   mlflow.set_experiment("churn-prediction")
   ```
2. **Instrument the run.** Use autologging for the framework's own metrics, plus manual
   `log_params`/`log_metrics`/`log_artifact` for anything autolog can't see (custom
   preprocessing, business metrics, plots).
   ```python
   import mlflow
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, f1_score

   mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

   with mlflow.start_run(run_name="rf-baseline") as run:
       mlflow.set_tags({"git_commit": "a1b2c3d", "dataset_version": "v3"})
       params = {"n_estimators": 200, "max_depth": 8}
       mlflow.log_params(params)

       clf = RandomForestClassifier(random_state=42, **params)
       clf.fit(X_train, y_train)
       preds = clf.predict(X_test)

       mlflow.log_metrics({
           "accuracy": accuracy_score(y_test, preds),
           "f1_macro": f1_score(y_test, preds, average="macro"),
       })
       mlflow.log_artifact("confusion_matrix.png")
   ```
3. **Compare runs to find the best one.** Query programmatically instead of eyeballing
   the UI once you have more than a handful of runs.
   ```python
   import mlflow

   runs = mlflow.search_runs(
       experiment_names=["churn-prediction"],
       order_by=["metrics.f1_macro DESC"],
       max_results=5,
   )
   print(runs[["run_id", "params.n_estimators", "params.max_depth", "metrics.f1_macro"]])
   best_run_id = runs.iloc[0]["run_id"]
   ```
4. **Register the best model.** The registry gives the model a stable name and version
   history independent of the run that produced it.
   ```python
   from mlflow.tracking import MlflowClient

   client = MlflowClient()
   result = mlflow.register_model(
       model_uri=f"runs:/{best_run_id}/model",
       name="churn-rf-classifier",
   )
   client.transition_model_version_stage(
       name="churn-rf-classifier", version=result.version, stage="Staging",
   )
   ```
5. **(Alternative) Same idea in Weights & Biases**, if the team standardizes on W&B
   instead of MLflow.
   ```python
   # pip install wandb
   import wandb

   run = wandb.init(project="churn-prediction", config={"n_estimators": 200, "max_depth": 8})
   wandb.log({"accuracy": accuracy_score(y_test, preds), "f1_macro": f1_score(y_test, preds, average="macro")})
   wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
   run.finish()
   ```
6. **Reproduce a past run.** Pull the exact params, environment, and artifacts back down
   instead of trying to remember what you ran.
   ```bash
   mlflow runs describe --run-id <best_run_id>
   mlflow artifacts download --run-id <best_run_id> -d ./reproduced
   ```

## Gotchas
- **Autolog doesn't see everything.** Custom preprocessing, feature engineering, or data
  versioning that happens outside the estimator's `fit()` call won't be captured — log
  those params/tags manually or the "best run" isn't actually reproducible.
- **The default tracking URI is a local `mlruns/` folder.** If you forget to point at a
  shared server or artifact store, your teammates simply can't see your runs — they don't
  error, they just look empty on their machine.
- **Logging raw datasets or every checkpoint as an artifact bloats the backend store**
  fast. Log data *references* (path + version hash) and representative samples, not the
  full dataset, on every run.
- **Unnamed, untagged runs are indistinguishable** six weeks later. Always set `run_name`
  and tags like git commit and dataset version — "Run 47" tells you nothing.
- **Registry stage labels (Staging/Production) are just metadata** — transitioning a
  version to "Production" in MLflow does not deploy anything or gate anything by itself;
  that enforcement is the job of ci-cd-for-ml and model-deployment. Note that stage-based
  transitions are also being superseded by alias/tag-based model versioning in newer MLflow
  releases — check the docs for the installed version before building new automation on
  `transition_model_version_stage`.

## References
- [MLflow Tracking docs](https://mlflow.org/docs/latest/tracking.html) — canonical guide to runs, experiments, and autologging.
- [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html) — versioning and stage transitions for registered models.
- [Weights & Biases docs — Experiments](https://docs.wandb.ai/guides/track/) — the W&B equivalent of steps 2–3.
