---
name: data-pipelines
display_name: Data Pipelines
description: >
  Use when the user wants to orchestrate a recurring ML data workflow — DAGs
  that extract/transform data on a schedule, compute and serve features
  consistently, or move data reliably between training and serving. Trigger
  phrases: "build a data pipeline", "orchestrate this ETL with Airflow", "set
  up a feature store", "schedule my feature computation job", "fix train/serve
  skew in my features". NOT for one-off cleaning/encoding of a single training
  dataset (see data-preprocessing) or creating new feature columns (see
  feature-engineering).
type: workflow
domain: mlops
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - dag-orchestration
  - data-quality-gating
  - feature-store-materialization
  - train-serve-skew-prevention
  - pipeline-backfill-recovery
requires:
conflicts:
related:
  - data-preprocessing
  - feature-engineering
  - ml-monitoring
inputs: A recurring data source (table/feed) and a feature-computation definition that needs to run on a schedule and be served consistently offline and online.
outputs: A scheduled, validated, idempotent pipeline plus an offline/online feature store serving identical values to training and inference.
version_constraints:
  - "feast: `FeatureView(schema=[Field(name=..., dtype=...)])` (used in step 3) reflects the Feast 0.2x+ API; earlier Feast versions used a `features=[Feature(name=..., dtype=...)]` argument instead — this is a model-knowledge recollection, not live-verified this session."
---

## Overview
Covers the recurring, scheduled infrastructure that keeps ML features fresh and
consistent — not the one-time cleaning of a training set. Builds an orchestrated DAG
(Airflow) with a data-quality gate, then shows how a feature store (Feast) guarantees
the same feature-computation logic is used offline (training) and online (serving),
which is the single most common cause of silent production accuracy loss. The output is
a scheduled, validated, idempotent pipeline — not a notebook that someone re-runs by hand.

## Workflow
1. **Define the DAG: extract → transform → load.** Each task is independently retryable.
   ```python
   # feature_pipeline_dag.py
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta
   import pandas as pd

   def extract(**context):
       df = pd.read_sql(
           "SELECT * FROM orders WHERE order_date = %(ds)s",
           con=get_db_connection(), params={"ds": context["ds"]},
       )
       df.to_parquet(f"/tmp/orders_{context['ds']}.parquet")

   def transform(**context):
       df = pd.read_parquet(f"/tmp/orders_{context['ds']}.parquet")
       features = df.groupby("customer_id").agg(
           orders_30d=("order_id", "count"),
           spend_30d=("amount", "sum"),
       ).reset_index()
       features.to_parquet(f"/tmp/features_{context['ds']}.parquet")

   def load(**context):
       features = pd.read_parquet(f"/tmp/features_{context['ds']}.parquet")
       # delete-then-insert by partition keeps retries idempotent (see Gotchas)
       con = get_db_connection()
       con.execute("DELETE FROM customer_features WHERE ds = %s", (context["ds"],))
       features.assign(ds=context["ds"]).to_sql(
           "customer_features", con=con, if_exists="append", index=False)

   default_args = {"retries": 3, "retry_delay": timedelta(minutes=5)}

   with DAG(
       dag_id="customer_feature_pipeline",
       schedule_interval="@daily",
       start_date=datetime(2026, 1, 1),
       catchup=False,
       default_args=default_args,
   ) as dag:
       t1 = PythonOperator(task_id="extract", python_callable=extract)
       t2 = PythonOperator(task_id="transform", python_callable=transform)
       t3 = PythonOperator(task_id="load", python_callable=load)
       t1 >> t2 >> t3
   ```
2. **Add a data-quality gate between transform and load.** Fail loudly on a schema
   violation instead of loading garbage.
   ```python
   # pip install pandera
   import pandera as pa
   from pandera import Column, Check

   feature_schema = pa.DataFrameSchema({
       "customer_id": Column(int, nullable=False),
       "orders_30d": Column(int, Check.ge(0)),
       "spend_30d": Column(float, Check.ge(0)),
   })

   def validate(**context):
       df = pd.read_parquet(f"/tmp/features_{context['ds']}.parquet")
       feature_schema.validate(df, lazy=True)   # raises SchemaErrors with a full report

   validate_task = PythonOperator(task_id="validate", python_callable=validate)
   t2 >> validate_task >> t3
   ```
3. **Define features once, in a feature store, instead of twice.** This is what prevents
   train/serve skew — the same `FeatureView` backs both batch training and online lookup.
   ```python
   # feature_repo/features.py
   from feast import Entity, FeatureView, Field, FileSource
   from feast.types import Int64, Float32
   from datetime import timedelta

   customer = Entity(name="customer_id", join_keys=["customer_id"])

   customer_source = FileSource(
       path="/tmp/customer_features.parquet",
       timestamp_field="event_timestamp",
   )

   customer_features = FeatureView(
       name="customer_features",
       entities=[customer],
       ttl=timedelta(days=30),
       schema=[
           Field(name="orders_30d", dtype=Int64),
           Field(name="spend_30d", dtype=Float32),
       ],
       source=customer_source,
   )
   ```
4. **Materialize offline features into the online store.**
   ```bash
   # pip install feast
   feast apply                                                     # register definitions
   feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"    # offline -> online
   ```
5. **Read the same features at inference time.** No re-implementing the aggregation logic
   in the serving path.
   ```python
   from feast import FeatureStore

   store = FeatureStore(repo_path="feature_repo/")
   features = store.get_online_features(
       features=["customer_features:orders_30d", "customer_features:spend_30d"],
       entity_rows=[{"customer_id": 42}],
   ).to_dict()
   ```
6. **Backfill deliberately, and route failures to a human.**
   ```bash
   airflow dags backfill customer_feature_pipeline --start-date 2026-08-01 --end-date 2026-08-15
   ```
   ```python
   default_args = {
       "retries": 3,
       "retry_delay": timedelta(minutes=5),
       "on_failure_callback": notify_slack_on_failure,
   }
   ```

## Gotchas
- **Train/serve skew** — computing a feature one way in the offline batch job and a
  subtly different way in the online request path (different window, different null
  handling) silently degrades the model. A feature store exists specifically to make the
  transformation logic identical in both places; hand-rolled duplicate code paths reliably
  drift apart.
- **Non-idempotent load tasks.** A `load` task that blindly appends on every retry (no
  delete-by-partition or upsert) duplicates rows the moment Airflow retries a failed task
  — and retries are the default behavior, not the exception.
- **No data-quality gate between extract and load** means a silent upstream schema
  change (renamed column, sudden null spike, a partner feed dropping a field) flows
  straight into the model instead of failing the pipeline where it's cheap to catch.
- **`catchup=True` on a DAG with a long `start_date` history** triggers a burst of
  backfill runs the instant it's turned on or unpaused — default to `catchup=False` and
  backfill explicitly and intentionally when you actually need historical runs.
- **Feature TTLs set carelessly** — too long, and inactive entities keep serving stale
  features indefinitely; too short, and legitimately sparse entities get nulls at serving
  time. Tune `ttl` to the feature's real staleness tolerance, not a default.

## References
- [Apache Airflow docs](https://airflow.apache.org/docs/apache-airflow/stable/index.html) — DAGs, scheduling, retries, backfills.
- [Feast docs](https://docs.feast.dev/) — feature store concepts, offline/online stores, materialization.
- [Pandera docs](https://pandera.readthedocs.io/en/stable/) — dataframe schema validation for the quality gate.
