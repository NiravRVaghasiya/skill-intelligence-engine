---
name: neural-net-fundamentals
display_name: Neural Network Fundamentals
description: >
  Use when the user wants to understand how neural networks actually learn —
  forward/backward propagation, gradient descent, optimizers, weight
  initialization, or activation functions — before writing or debugging deep
  learning code. Trigger phrases: "explain backpropagation", "how does
  gradient descent work", "which optimizer should I use, Adam or SGD",
  "why is my network not training", "what is Xavier/He initialization".
  NOT for convolutional architectures (see cnn-vision), recurrent/sequence
  models (see rnn-sequence), or hands-on training-loop code (see
  pytorch-patterns).
type: reference
domain: deep-learning
level: beginner
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - backpropagation
  - gradient-descent-variants
  - adam-optimizer-mechanics
  - weight-initialization-schemes
  - activation-function-selection
  - vanishing-exploding-gradients
requires:
conflicts:
related:
  - attention-mechanisms
  - cnn-vision
  - rnn-sequence
  - training-deep-models
inputs: A question about how neural networks learn — backprop, optimizers, initialization, or activations — needing conceptual grounding.
outputs: A conceptual explanation of the forward/backward/optimizer learning loop to ground further discussion or code.
---

## Overview
Every neural network, regardless of architecture, learns the same way: a forward
pass computes a prediction and a loss, backpropagation uses the chain rule to turn
that loss into a gradient for every parameter, and an optimizer uses those
gradients to update the weights. This card covers that learning loop — forward/
backward propagation, gradient descent variants, the major optimizers, weight
initialization, and activation functions — the shared vocabulary underneath every
other deep learning skill in this library.

## Key Concepts
- **Forward pass.** Input flows layer by layer: `z = Wx + b`, then a nonlinear
  activation `a = f(z)`. The final layer's output is compared to the target via a
  loss function (cross-entropy for classification, MSE for regression). The whole
  computation forms a directed acyclic graph that frameworks like PyTorch build
  automatically (`autograd`).
- **Backpropagation — the chain rule, applied systematically.** To update a weight
  `W` in an early layer, you need `∂L/∂W`. Backprop computes this by propagating
  the loss gradient backward through the graph, layer by layer:
  \[ \frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(L)}}
     \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots
     \frac{\partial a^{(l+1)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}} \]
  Each layer only needs to know its *local* derivative and the gradient handed to
  it from the layer above — this is what makes it tractable for networks with
  millions of parameters.
- **Gradient descent variants.** Batch GD (full dataset per step) is accurate but
  slow and memory-heavy; SGD (one sample per step) is noisy but cheap; **mini-batch
  SGD** (32–512 samples) is the practical default — noisy enough to escape sharp
  local minima, stable enough to converge, and maps cleanly onto GPU parallelism.
- **Optimizers.**
  - **SGD + momentum** accumulates a velocity vector so gradients from consecutive
    steps reinforce each other along consistent directions and cancel out along
    oscillating ones: `v = βv - lr·∇L`, `w += v`.
  - **RMSprop** divides the learning rate by a running average of recent squared
    gradients — per-parameter adaptive step sizes, good for non-stationary
    objectives.
  - **Adam** combines momentum (1st moment) with RMSprop-style scaling (2nd
    moment) plus bias correction — the default starting point for most deep
    learning tasks because it's robust to learning-rate choice.
  - **AdamW** decouples weight decay from the gradient-based update (see
    References) instead of folding it into the gradient like plain Adam+L2 does —
    it is the modern default over Adam for transformer-style models.
- **Weight initialization.** Initialization sets the *scale* of activations and
  gradients before any learning happens. **Xavier/Glorot** init (`Var(W) =
  1/n_in`, or the averaged `2/(n_in+n_out)` form) keeps variance stable through
  layers with symmetric activations (tanh, sigmoid). **He init** (`Var(W) =
  2/n_in`) accounts for ReLU zeroing out half its inputs and is the standard for
  ReLU-family networks. Both exist to prevent activations/gradients from shrinking
  to zero or blowing up as depth increases.
- **Activation functions.** Sigmoid/tanh saturate for large `|z|`, driving their
  derivative toward 0 — the root cause of vanishing gradients in deep sigmoid/tanh
  networks. **ReLU** (`max(0, z)`) has a constant gradient of 1 for `z > 0`,
  which is why it enabled much deeper networks; its failure mode is "dying ReLUs"
  (a unit that's permanently in the `z < 0` region and never updates again).
  **Leaky ReLU / GELU / SiLU** patch this by keeping a small or smooth gradient
  on the negative side.
- **Vanishing / exploding gradients.** Because backprop multiplies many local
  derivatives together, a deep network's gradient is a long product — if the
  terms are consistently `<1` (saturating activations, poor init) it vanishes; if
  consistently `>1` it explodes. This is why depth, initialization, activation
  choice, normalization layers, and residual connections are all really the same
  underlying problem viewed from different angles.

## Gotchas
- **Zero (or any symmetric) initialization** makes every neuron in a layer
  compute the identical gradient, so they update identically forever — the layer
  collapses to a single effective neuron. Weights must be initialized with
  independent randomness; only biases are safe to zero-init.
- **Adam "fixes" a bad learning rate less than people assume.** Adam adapts
  *per-parameter* step sizes, not the global scale — an LR that's off by 10x will
  still diverge or stall. Always sweep LR; don't treat Adam as LR-free.
- **L2 regularization ≠ weight decay under Adam.** Adding an L2 term to the loss
  before computing Adam's adaptive moments produces a different (weaker) effect
  than true weight decay — this is exactly the bug AdamW was designed to fix. If a
  model "needs" AdamW to match a paper's reported numbers, that's why.
  Regularization mechanics (dropout, weight decay in practice, schedulers) live in
  training-deep-models — this card only covers why decoupling matters.
- **Mismatched initialization and activation** (e.g., Xavier init with ReLU, or
  He init with tanh) reintroduces the vanishing/exploding problem the init
  scheme was meant to solve — the variance formulas assume a specific activation.

## References
- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) — the canonical textbook treatment of backprop, optimization, and initialization.
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) — the original Adam paper.
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101) — the AdamW paper, explains why L2-in-the-loss ≠ weight decay under adaptive optimizers.
- [Delving Deep into Rectifiers (He et al., 2015)](https://arxiv.org/abs/1502.01852) — introduces He initialization and analyzes why ReLU needs it.
