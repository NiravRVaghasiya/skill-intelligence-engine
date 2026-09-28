---
name: model-optimization
display_name: Model Optimization
description: >
  Use when the user wants to make a trained model smaller or faster —
  quantization, knowledge distillation, or converting to ONNX for portable,
  optimized inference. Trigger phrases: "quantize this model", "distill this
  model into a smaller one", "convert my model to ONNX", "speed up inference
  latency", "shrink this model for edge deployment". NOT for standing up the
  serving endpoint itself (see model-deployment) or searching hyperparameters
  during training (see hyperparameter-tuning).
type: workflow
domain: mlops
level: advanced
lifecycle: stable
risk_level: medium
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - post-training-quantization
  - onnx-export-and-runtime-inference
  - static-quantization-calibration
  - knowledge-distillation
  - latency-accuracy-benchmarking
requires:
conflicts:
related:
  - model-deployment
  - training-deep-models
  - hyperparameter-tuning
inputs: A trained model (typically PyTorch) plus a representative sample input and, for static quantization, a calibration dataset.
outputs: A smaller/faster model artifact (quantized weights, an ONNX file, or a distilled student model) with a benchmarked latency/size/accuracy comparison against the fp32 baseline.
version_constraints:
  - "pytorch: `torch.quantization` (used in step 2) was reorganized under `torch.ao.quantization` starting around PyTorch 1.13/2.0, with the old import path kept as a deprecated alias for some releases — this is a model-knowledge recollection, not live-verified this session; check the installed PyTorch version's docs before relying on the old path."
---

## Overview
Takes a trained model and reduces its size/latency footprint before it goes behind an
endpoint — quantization, ONNX export, and distillation, in roughly increasing order of
effort and payoff. Every technique here trades some accuracy for speed/size, so the
workflow always benchmarks a baseline first and validates the tradeoff last: the goal is
a model that is measurably faster or smaller *and* still within an acceptable accuracy
band, not just a model that technically ran through a quantization API.

## Workflow
1. **Benchmark the unoptimized baseline first.** Without this you can't tell whether an
   "optimized" model actually improved anything on your target hardware.
   ```python
   import time
   import torch

   def benchmark(model, sample_input, n=50):
       model.eval()
       with torch.no_grad():
           for _ in range(5):          # warmup
               model(sample_input)
           start = time.perf_counter()
           for _ in range(n):
               model(sample_input)
           latency_ms = (time.perf_counter() - start) / n * 1000
       size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
       return {"latency_ms": round(latency_ms, 2), "size_mb": round(size_mb, 2)}

   baseline = benchmark(model, sample_input)
   print(baseline)
   ```
2. **Try post-training dynamic quantization first.** No calibration data needed — it's
   the cheapest thing to try.
   ```python
   import torch
   from torch.quantization import quantize_dynamic

   quantized_model = quantize_dynamic(
       model,
       {torch.nn.Linear},          # layer types to quantize
       dtype=torch.qint8,
   )
   torch.save(quantized_model.state_dict(), "model_int8.pt")
   print(benchmark(quantized_model, sample_input))
   ```
3. **Export to ONNX for a portable, runtime-agnostic speedup.** ONNX Runtime often beats
   native framework inference even before quantization.
   ```python
   # pip install onnx onnxruntime
   import torch

   torch.onnx.export(
       model, sample_input, "model.onnx",
       input_names=["input"], output_names=["output"],
       dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
       opset_version=17,
   )
   ```
   ```python
   import onnxruntime as ort
   import numpy as np

   session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
   outputs = session.run(None, {"input": sample_input.numpy().astype(np.float32)})
   ```
4. **Use static quantization with calibration data when dynamic quantization's accuracy
   loss is too high.** Bigger speedup than dynamic, but needs representative data.
   ```python
   from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

   class CalibReader(CalibrationDataReader):
       def __init__(self, calib_batches):
           self._iter = iter(calib_batches)
       def get_next(self):
           batch = next(self._iter, None)
           return {"input": batch} if batch is not None else None

   quantize_static(
       model_input="model.onnx",
       model_output="model_int8_static.onnx",
       calibration_data_reader=CalibReader(calibration_batches),
       weight_type=QuantType.QInt8,
   )
   ```
5. **Distill into a smaller student model when quantization alone isn't enough** (e.g.
   an architecture change is needed, not just lower precision).
   ```python
   import torch.nn.functional as F

   def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
       soft_loss = F.kl_div(
           F.log_softmax(student_logits / T, dim=-1),
           F.softmax(teacher_logits / T, dim=-1),
           reduction="batchmean",
       ) * (T ** 2)
       hard_loss = F.cross_entropy(student_logits, labels)
       return alpha * soft_loss + (1 - alpha) * hard_loss

   teacher.eval()
   for x, y in train_loader:
       with torch.no_grad():
           teacher_logits = teacher(x)
       student_logits = student(x)
       loss = distillation_loss(student_logits, teacher_logits, y)
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
   ```
6. **Validate the accuracy/latency tradeoff before shipping.** An optimization that
   fails this check goes back to step 2/4 with different settings, not straight to prod.
   ```python
   results = {
       "fp32_baseline": {**baseline, "accuracy": eval_accuracy(model, test_loader)},
       "int8_dynamic": {**benchmark(quantized_model, sample_input),
                         "accuracy": eval_accuracy(quantized_model, test_loader)},
   }
   for name, r in results.items():
       print(f"{name}: {r['latency_ms']}ms  {r['size_mb']}MB  acc={r['accuracy']:.3f}")
   assert results["int8_dynamic"]["accuracy"] >= results["fp32_baseline"]["accuracy"] - 0.01
   ```

## Gotchas
- **Dynamic quantization only speeds up the layers you target** (typically `nn.Linear`)
  — it barely helps convolution-heavy vision models, where static quantization or a
  hardware-specific compiler (TensorRT) is usually needed for a real win.
- **Skipping the baseline benchmark** means you can't tell whether a "faster" model is
  actually faster on the *target* hardware — int8 kernels can be slower than fp32 on
  CPUs/GPUs that lack efficient int8 support.
- **Calibrating static quantization on a narrow or unrepresentative data sample** silently
  tanks accuracy on inputs the calibration set didn't cover — the calibration data must
  mirror the real production input distribution, not just a convenient dev batch.
- **ONNX export can silently drop data-dependent control flow** (an `if`/loop whose
  branch depends on tensor values) that doesn't trace cleanly through `torch.onnx.export`
  — always run a numerical parity check (`np.allclose` between original and exported
  outputs on the same input), not just a shape/sanity check.
- **Distillation quality is capped by the teacher and the temperature `T`.** Distilling
  from a mediocre teacher just produces a faster mediocre model; a poorly tuned `T` (too
  high smooths away useful signal, too low approaches plain cross-entropy) wastes the
  whole exercise.

## References
- [PyTorch quantization docs](https://pytorch.org/docs/stable/quantization.html) — dynamic vs. static vs. QAT quantization APIs.
- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — static/dynamic quantization for ONNX models.
- [Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean, 2015)](https://arxiv.org/abs/1503.02531) — the original knowledge-distillation paper.
