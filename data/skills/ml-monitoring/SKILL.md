---
name: ml-monitoring
display_name: ML Monitoring
description: >
  Use when the user wants to detect data drift, model performance decay, or
  set up alerting for a model that is already running in production.
  Trigger phrases: "detect data drift", "is my model degrading in
  production", "monitor model performance over time", "set up alerts for
  prediction drift", "check for feature skew". NOT for logging training-time
  experiments (see experiment-tracking) or running pre-deploy test suites
  (see ci-cd-for-ml).
type: workflow
domain: mlops
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - data-drift-detection
  - concept-drift-detection
  - prediction-drift-detection
  - performance-degradation-tracking
  - drift-alerting-thresholds
  - scheduled-monitoring-jobs
requires:
conflicts:
related:
  - model-deployment
  - experiment-tracking
  - data-pipelines
inputs: A model already serving live traffic, a frozen training-time reference dataset, and (eventually) delayed ground-truth labels for served predictions.
outputs: A scheduled drift/performance check plus alerts that fire before customers notice a degraded model.
version_constraints:
  - "evidently: the `Report`/`metric_preset` API (used in step 2) reflects the evidently 0.4.x+ redesign; earlier versions used a different `ColumnMapping`+dashboard-based API — this is a model-knowledge recollection, not live-verified this session, per the existing Gotcha on Evidently API churn."
---

## Overview
Watches a model *after* it's serving live traffic. This skill treats four distinct
failure modes as distinct, because they have different causes and different fixes:
**data drift** (the distribution of input features shifts away from training time),
**prediction drift** (the distribution of the model's *outputs* shifts — which can be
caused by data drift, by concept drift, or by a plain serving bug), **concept drift**
(the input-output relationship `P(y|x)` itself changes, so the same inputs now warrant a
different answer), and **performance degradation** (the ground-truth-measured metric
drops, for *any* reason — including causes that have nothing to do with drift, like a
broken feature pipeline). Covers logging production predictions, computing data drift and
prediction drift against a training-time reference, measuring delayed-label performance
decay, and wiring both into alerting. The output is a running check (scheduled job) that
tells you a model needs attention before customers notice, rather than a training-time
metric that goes stale the day the model ships.

## Workflow
1. **Log every prediction with its inputs.** Without this, there's nothing to compare
   against later.
   ```python
   import os
   import pandas as pd
   from datetime import datetime, timezone

   LOG_PATH = "prediction_log.csv"

   def log_prediction(features: dict, prediction: float, model_version: str) -> None:
       row = {**features, "prediction": prediction, "model_version": model_version,
              "logged_at": datetime.now(timezone.utc).isoformat()}
       pd.DataFrame([row]).to_csv(
           LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False,
       )
   ```
2. **Compute data drift against the training-time reference.** `evidently` compares
   *input feature* distributions and reports a share of drifted columns. This tells you
   the world the model sees has changed — it does not, by itself, tell you whether the
   model is still accurate.
   ```python
   # pip install evidently
   import pandas as pd
   from evidently.report import Report
   from evidently.metric_preset import DataDriftPreset

   reference = pd.read_csv("train_reference.csv")          # snapshot frozen at training time
   current = pd.read_csv("prediction_log.csv").tail(5000)   # recent production window
   feature_cols = [c for c in reference.columns if c != "prediction"]

   report = Report(metrics=[DataDriftPreset()])
   report.run(reference_data=reference[feature_cols], current_data=current[feature_cols])
   report.save_html("data_drift_report.html")

   drift_share = report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"]
   print(f"Share of drifted feature columns: {drift_share:.2%}")
   ```
3. **Separately, compute prediction drift on the model's output column.** Run the same
   drift test against the *prediction* distribution, not the features. Prediction drift
   can fire even when data drift doesn't — e.g. a serving-side bug (wrong feature order,
   a stale encoder) can shift outputs with no visible change in the raw input features.
   ```python
   report = Report(metrics=[DataDriftPreset()])
   report.run(
       reference_data=reference[["prediction"]],
       current_data=current[["prediction"]],
       column_mapping=None,
   )
   report.save_html("prediction_drift_report.html")
   pred_drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
   print(f"Prediction distribution drifted: {pred_drift_detected}")
   ```
4. **Measure performance decay once ground truth catches up.** This is the lagging,
   ground-truth-confirmed check, and it is the only one of the four signals here that can
   directly reveal **concept drift** — a real shift in `P(y|x)` where the same feature
   values now warrant a different label. If this check shows decay *without* meaningful
   data or prediction drift, that pattern (stable inputs and outputs, but worse ground-truth
   accuracy) is itself evidence of concept drift rather than a data problem. If it shows
   decay *with* no drift signal at all and no plausible real-world explanation, suspect a
   pipeline bug (label join, feature computation, or a serving defect) before assuming the
   model itself needs retraining — drift detection is not a substitute for checking the
   pipeline.
   ```python
   from sklearn.metrics import f1_score

   def compute_delayed_performance(pred_log: pd.DataFrame, labels: pd.DataFrame) -> float:
       joined = pred_log.merge(labels, on="request_id", how="inner")
       return f1_score(joined["y_true"], joined["prediction"], average="macro")

   current_f1 = compute_delayed_performance(pred_log, ground_truth)
   baseline_f1 = 0.87  # captured at model registration time via experiment-tracking
   if current_f1 < baseline_f1 - 0.05:
       print(f"Performance decay detected: {current_f1:.3f} vs baseline {baseline_f1:.3f}")
   ```
5. **Set a threshold and page someone, don't just log a number.**
   ```python
   import requests

   def alert_if_drifted(drift_share: float, threshold: float = 0.3, webhook_url: str = "") -> None:
       if drift_share > threshold:
           requests.post(webhook_url, json={
               "text": f":rotating_light: Data drift alert: {drift_share:.0%} of "
                       f"features drifted (threshold {threshold:.0%})."
           })
   ```
6. **Run it on a schedule** instead of remembering to check manually.
   ```python
   # monitoring_dag.py — Airflow DAG, runs the checks above daily
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta

   def run_drift_check():
       # calls the drift + alert_if_drifted functions defined above
       ...

   with DAG(
       dag_id="model_drift_monitoring",
       schedule_interval="@daily",
       start_date=datetime(2026, 1, 1),
       catchup=False,
       default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
   ) as dag:
       PythonOperator(task_id="check_drift", python_callable=run_drift_check)
   ```

## Gotchas
- **Data drift, prediction drift, concept drift, and performance degradation are four
  different things** and treating any pair as synonyms leads to the wrong fix: data drift
  (input features shifted) and prediction drift (outputs shifted) are both detectable
  *without* labels but neither proves the model got worse; concept drift (`P(y|x)` itself
  changed) can only be inferred from ground-truth performance decay that shows up *without*
  a matching data/prediction drift signal; and performance degradation is the catch-all
  ground-truth symptom that can also be caused by something that isn't drift at all — a
  broken label join, a feature-pipeline bug, a schema change — so a degraded metric with no
  drift signal should send you to check the pipeline before you retrain.
- **A reference set frozen forever at training time** eventually flags nearly everything
  as "drifted" as the world naturally shifts — refresh the reference window periodically
  and re-baseline after every retrain, or you'll get alert fatigue and start ignoring it.
- **Evidently's API has changed across major versions** (legacy `ColumnMapping`-based API
  vs. the 0.4+ `Report`/`metric_preset` interface vs. later releases) — pin the version
  and check the changelog before upgrading, existing pipelines can break silently.
- **Ground truth often arrives late** (weeks, for churn/fraud/credit models) — you cannot
  wait on labels alone to detect a problem. Feature/prediction drift is your leading
  signal in the interim; treat label-based decay as confirmation, not the first alarm.
- **Alerting on raw statistical-test p-values at production data volumes** triggers
  constantly, since huge sample sizes make even trivial shifts "significant" — alert on
  effect-size thresholds (share of drifted columns, PSI > 0.2) instead of p < 0.05.
- **Logging raw, unredacted PII into the prediction log** for drift analysis creates a
  compliance problem of its own — hash or strip identifying fields before persisting.

## References
- [Evidently AI docs](https://docs.evidentlyai.com/) — drift and data-quality report reference.
- [Google: Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) — practical guidance on monitoring models in production.
- [Apache Airflow docs](https://airflow.apache.org/docs/apache-airflow/stable/index.html) — scheduling recurring monitoring jobs.
