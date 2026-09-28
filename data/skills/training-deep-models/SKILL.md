---
name: training-deep-models
display_name: Training Deep Models
description: >
  Use when the user wants to make a deep learning training run faster and more
  robust — mixed precision, learning-rate schedulers with warmup, gradient
  clipping, weight decay, or early stopping/checkpointing to fight overfitting.
  Trigger phrases: "set up mixed precision training", "add a learning rate
  scheduler", "my model is overfitting, what do I add", "add early stopping to
  my training loop", "speed up training on GPU". NOT for the optimizer/backprop
  theory behind these choices (see neural-net-fundamentals) or for idiomatic
  PyTorch loop/Dataset/hook structure itself (see pytorch-patterns).
type: workflow
domain: deep-learning
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - mixed-precision-training
  - learning-rate-scheduling
  - gradient-clipping
  - weight-decay-decoupling
  - early-stopping-checkpointing
requires:
conflicts:
related:
  - neural-net-fundamentals
  - pytorch-patterns
  - hyperparameter-tuning
inputs: An existing PyTorch training loop and a goal (faster training, less overfitting, more stability).
outputs: A hardened training loop with AMP, LR scheduling, gradient clipping, decoupled weight decay, and early stopping wired in.
version_constraints:
  - "PyTorch: torch.cuda.amp.autocast/GradScaler (and the newer torch.autocast(device_type=...) entrypoint used here) stable since 1.6+ — model-knowledge estimate, not live-verified this session."
---

## Overview
Getting a model architecture right is only half of training a deep network well
— the other half is the training *recipe*: precision, learning-rate schedule,
regularization, and stopping criteria. This skill walks through hardening a
PyTorch training loop with automatic mixed precision, a warmup+decay LR
scheduler, gradient clipping, decoupled weight decay, and early
stopping/checkpointing — the standard toolkit for training that is both fast and
generalizes.

## Workflow
1. **Start from a plain, reproducible baseline loop.** Fix seeds and put the
   model/data on the right device before touching any training tricks — every
   optimization below should be measured against this baseline.
   ```python
   import torch, random, numpy as np

   def set_seed(seed=42):
       random.seed(seed); np.random.seed(seed)
       torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

   set_seed(42)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ```

2. **Split parameters into decay / no-decay groups, then use AdamW.** Weight
   decay should not be applied to biases or normalization parameters (LayerNorm/
   BatchNorm weight & bias) — folding it in uniformly measurably hurts
   generalization on larger models.
   ```python
   decay, no_decay = [], []
   for name, p in model.named_parameters():
       if not p.requires_grad:
           continue
       if p.ndim == 1 or name.endswith(".bias"):   # norms & biases
           no_decay.append(p)
       else:
           decay.append(p)

   optimizer = torch.optim.AdamW(
       [{"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0}],
       lr=3e-4,
   )
   ```

3. **Add a warmup + cosine-decay learning-rate schedule.** A short linear warmup
   avoids destabilizing the model with large updates while Adam's second-moment
   estimate is still uninitialized; cosine decay anneals smoothly to a low LR
   instead of dropping in hard steps.
   ```python
   from torch.optim.lr_scheduler import LambdaLR
   import math

   num_epochs, steps_per_epoch = 20, len(train_loader)
   total_steps = num_epochs * steps_per_epoch
   warmup_steps = int(0.05 * total_steps)

   def lr_lambda(step):
       if step < warmup_steps:
           return step / max(1, warmup_steps)
       progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
       return 0.5 * (1 + math.cos(math.pi * progress))

   scheduler = LambdaLR(optimizer, lr_lambda)
   ```

4. **Enable automatic mixed precision (AMP).** `autocast` runs eligible ops in
   fp16/bf16 for speed and memory savings while keeping numerically sensitive ops
   (e.g., reductions) in fp32; `GradScaler` prevents fp16 gradients from
   underflowing to zero.
   ```python
   scaler = torch.cuda.amp.GradScaler()

   def train_step(batch, targets):
       optimizer.zero_grad(set_to_none=True)
       with torch.autocast(device_type="cuda", dtype=torch.float16):
           outputs = model(batch)
           loss = criterion(outputs, targets)

       scaler.scale(loss).backward()
       scaler.unscale_(optimizer)                      # required before clipping
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       scaler.step(optimizer)
       scaler.update()
       scheduler.step()          # per-step schedule: call every batch, not every epoch
       return loss.item()
   ```

5. **Add early stopping with checkpointing on validation loss.** Track the best
   validation score, save a checkpoint whenever it improves, and stop once
   improvement stalls for `patience` epochs — then reload the best checkpoint
   rather than using the final (possibly overfit) weights.
   ```python
   best_val_loss, patience, bad_epochs = float("inf"), 5, 0

   for epoch in range(num_epochs):
       model.train()
       for batch, targets in train_loader:
           train_step(batch.to(device), targets.to(device))

       model.eval()
       val_loss = 0.0
       with torch.no_grad():
           for batch, targets in val_loader:
               val_loss += criterion(model(batch.to(device)), targets.to(device)).item()
       val_loss /= len(val_loader)

       if val_loss < best_val_loss:
           best_val_loss, bad_epochs = val_loss, 0
           torch.save({"model": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "epoch": epoch}, "best.pt")
       else:
           bad_epochs += 1
           if bad_epochs >= patience:
               print(f"Early stopping at epoch {epoch}")
               break

   model.load_state_dict(torch.load("best.pt")["model"])   # restore best, not final, weights
   ```

## Gotchas
- **Scheduler step granularity must match how it was built.** A per-step schedule
  (like the warmup+cosine one above, built over `total_steps`) must be stepped
  once per *batch*; stepping it once per *epoch* instead silently compresses the
  entire decay into the first few epochs. `OneCycleLR` and similar schedulers
  have the same requirement — check the granularity the scheduler expects.
- **Gradient clipping order matters under AMP.** You must call
  `scaler.unscale_(optimizer)` before `clip_grad_norm_`; clipping the still-scaled
  (fp16-scale) gradients clips against the wrong magnitude and makes the clip
  threshold meaningless.
- **AMP dtype choice affects stability, not just speed.** fp16 needs
  `GradScaler` because its dynamic range is narrow enough to underflow small
  gradients to zero; `bfloat16` (on Ampere+ GPUs) has fp32's exponent range and
  usually needs no scaler at all — mixing up the two configs is a common source
  of "AMP made my loss NaN" reports.
- **Early stopping on the wrong signal.** Stopping on training loss instead of
  validation loss defeats the purpose entirely (training loss keeps improving
  even as the model overfits). Also make sure the *final* model returned is the
  restored best checkpoint, not whatever the loop happened to end on.
- **Uniform weight decay on norm/bias parameters** regularizes parameters that
  aren't causing overfitting in the way weight matrices are, and measurably hurts
  accuracy on transformer-scale models — always split param groups as in step 2.

## References
- [PyTorch: Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html) — official `autocast`/`GradScaler` API and correct-usage examples.
- [torch.optim.lr_scheduler documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — canonical reference for scheduler step semantics (per-batch vs per-epoch).
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101) — the AdamW paper this recipe's weight-decay handling is based on.
- [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)](https://arxiv.org/abs/1608.03983) — the cosine-annealing schedule used in step 3.
