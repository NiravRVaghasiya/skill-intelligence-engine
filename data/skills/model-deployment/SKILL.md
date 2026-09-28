---
name: model-deployment
display_name: Model Deployment
description: >
  Use when the user wants to package a trained model and serve it behind an
  endpoint — wrapping it in an API, containerizing it, or standing up a
  serving process. Trigger phrases: "deploy this model", "serve my model as
  an API", "containerize this model", "set up a model endpoint", "put this
  model behind FastAPI". NOT for shrinking or quantizing the model itself
  (see model-optimization) or watching it after it's live (see ml-monitoring).
type: workflow
domain: mlops
level: intermediate
lifecycle: stable
risk_level: high
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - model-serialization
  - rest-api-wrapping
  - container-image-build
  - readiness-liveness-probes
  - high-throughput-model-serving
requires:
conflicts:
related:
  - model-optimization
  - ml-monitoring
  - ci-cd-for-ml
inputs: A trained model artifact (weights or a fitted estimator) and the inference code/input schema needed to run it.
outputs: A containerized, health-checked serving endpoint that returns predictions for real HTTP requests.
---

## Overview
Takes a trained model artifact from a notebook or training job to a running network
endpoint that answers real requests. Covers serialization, wrapping it in a serving API,
containerizing, and the readiness/health-check plumbing an orchestrator needs to run it
safely. The end state is a container that starts, reports when it's actually ready to
serve, and returns predictions for a real HTTP request — not just a model that "works"
in a script.

## Workflow
1. **Serialize the model in a serving-friendly format.** Prefer state/weights over a
   full pickled object when the framework supports it — pickling the whole estimator
   ties the artifact to the exact library version that produced it.
   ```python
   import joblib
   joblib.dump(clf, "model.joblib")            # scikit-learn / generic estimators

   import torch
   torch.save(model.state_dict(), "model_state.pt")   # PyTorch — weights, not the object
   ```
2. **Wrap it in a serving API.** A thin FastAPI app is enough for most low/medium-QPS
   use cases and gives you request validation for free via Pydantic.
   ```python
   # app.py
   from fastapi import FastAPI
   from pydantic import BaseModel
   import joblib
   import numpy as np

   app = FastAPI()
   model = joblib.load("model.joblib")

   class PredictRequest(BaseModel):
       features: list[float]

   @app.get("/health")
   def health():
       return {"status": "ok", "model_loaded": model is not None}

   @app.post("/predict")
   def predict(req: PredictRequest):
       x = np.array(req.features, dtype=float).reshape(1, -1)
       pred = model.predict(x)[0]
       return {"prediction": float(pred)}
   ```
3. **Containerize it.** Pin the base image and dependency versions so the image behaves
   the same in beta and prod.
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY app.py model.joblib ./
   EXPOSE 8080
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
   ```
   ```bash
   docker build -t churn-model:1.0 .
   docker run -p 8080:8080 --memory=1g --cpus=1 churn-model:1.0
   ```
4. **Give the orchestrator a real readiness signal.** Distinguish "process is up" from
   "model is loaded and warm" so traffic isn't routed to a container that will fail
   every request.
   ```yaml
   # kubernetes deployment snippet
   readinessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 5
     periodSeconds: 10
   livenessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 15
     periodSeconds: 20
   ```
5. **For higher-throughput deep learning models, consider a dedicated model server**
   instead of hand-rolling batching/threading in FastAPI.
   ```bash
   # pip install torchserve torch-model-archiver
   torch-model-archiver --model-name churn-net --version 1.0 \
     --serialized-file model_state.pt --handler image_classifier \
     --export-path model_store

   torchserve --start --ncs --model-store model_store --models churn-net=churn-net.mar
   ```
6. **Smoke test the live endpoint** before calling it deployed.
   ```bash
   curl -s -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 5.2, 3.3, 0.0]}'
   ```

## Gotchas
- **Pickling the full estimator/model object** (rather than weights/state) couples the
  artifact to the exact library version and Python version that trained it — a routine
  dependency bump on the serving side can make the artifact unpicklable in prod.
- **No health check distinct from a liveness check** means an orchestrator can report
  "healthy" for a container whose model failed to load — traffic gets routed and every
  request 500s until someone notices.
- **No input validation on the API surface** (accepting a raw, unchecked list of floats)
  produces two different failure modes, and neither is a clean 400: a client sending
  features in the wrong *order* (same count) gets a silent wrong prediction with no error
  at all, while a client sending the wrong feature *count* usually surfaces as an
  unhandled 500 from the estimator rather than a validated error. Check shape, dtype, and
  — where feasible — feature identity explicitly so callers get an actionable 400 instead
  of either failure mode.
- **Baking environment-specific config or secrets into the image** instead of injecting
  them via env vars at runtime breaks "build once, promote everywhere" and often leaks
  credentials into image layers.
- **Unpinned base images/dependencies** (`FROM python:3.11` instead of a pinned digest,
  or unpinned `requirements.txt`) mean the "same" image can behave differently on rebuild
  weeks later.

## References
- [FastAPI docs](https://fastapi.tiangolo.com/) — request/response models, validation, async endpoints.
- [TorchServe docs](https://pytorch.org/serve/) — production model serving for PyTorch at higher throughput.
- [Docker: Building best practices](https://docs.docker.com/build/building/best-practices/) — image layering, pinning, and size guidance.
