---
name: attention-mechanisms
display_name: Attention Mechanisms
description: >
  Use when the user wants to understand attention, self-attention, multi-head
  attention, or the core math behind transformers. Trigger phrases: "explain
  attention", "how does self-attention work", "what is Q K V", "why multi-head",
  "transformer intuition". NOT for building/fine-tuning models (see
  fine-tuning-llms) or architecture-level training (see training-deep-models).
type: reference
domain: deep-learning
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: primary
last_verified: 2026-09-21
capabilities:
  - scaled-dot-product-attention
  - multi-head-attention
  - self-attention-mechanics
  - positional-encoding-necessity
  - attention-complexity-scaling
requires:
conflicts:
related:
  - neural-net-fundamentals
  - rnn-sequence
  - prompt-engineering
inputs: A question about how attention, self-attention, or multi-head attention works.
outputs: A conceptual explanation of the attention mechanism grounded in the scaled dot-product formula.
---

## Overview
Attention lets a model decide, for each token, which other tokens matter most —
replacing the fixed, sequential bottleneck of RNNs with direct, parallel,
content-based lookups. It is the single mechanism that makes transformers work.
This card builds intuition from the scaled dot-product formula up to multi-head
self-attention.

## Key Concepts
- **The core idea — soft lookup.** Each token emits a **Query** ("what am I looking
  for?"). Every token also exposes a **Key** ("what do I offer?") and a **Value**
  ("what I'll contribute if chosen"). Attention scores each query against all keys,
  softmaxes them into weights, and returns a weighted blend of values.
- **Scaled dot-product attention** — the whole thing in one formula:
  \[ \text{Attention}(Q,K,V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)V \]
  - `QKᵀ` = similarity of every query to every key.
  - `√dₖ` scaling keeps the dot products from growing large and saturating the
    softmax into near-one-hot (which kills gradients).
  - `softmax` → attention weights that sum to 1. `× V` → the blended output.
- **Self-attention.** Q, K, and V all come from the *same* sequence, so every token
  attends to every other token — capturing long-range dependencies in one step,
  regardless of distance. This is what RNNs struggled with.
- **Multi-head attention.** Run several attention operations in parallel, each with
  its own learned Q/K/V projections into a lower dimension, then concatenate. Each
  head can specialize (syntax, coreference, positional patterns), giving the model
  multiple "representation subspaces" instead of one averaged view.
- **Why it beat RNNs.** Fully parallel (no sequential unrolling), constant path
  length between any two tokens (better gradient flow), and content-based rather
  than position-based mixing.

## Gotchas
- **Quadratic cost.** Attention is O(n²) in sequence length — the reason long-context
  models need tricks (sparse/flash/linear attention). This is a scaling wall, not a
  detail.
- **Positional information is not built in.** Self-attention is permutation-invariant;
  without positional encodings the model can't tell word order. Always paired with
  positional embeddings (sinusoidal, learned, or RoPE).
- **Attention weights ≠ explanations.** High attention weight is not a reliable
  causal explanation of a prediction — a common misinterpretation.
- **Masking matters.** Decoder (causal) attention masks future tokens; padding masks
  ignore pad positions. Forgetting a mask silently corrupts training.

## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper.
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — best visual intuition builder.
