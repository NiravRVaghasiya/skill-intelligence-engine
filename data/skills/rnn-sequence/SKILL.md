---
name: rnn-sequence
display_name: RNNs & Sequence Modeling
description: >
  Use when the user wants to understand recurrent neural networks — vanilla
  RNNs, LSTMs, GRUs, or how sequence models process ordered data over time.
  Trigger phrases: "explain how LSTMs work", "LSTM vs GRU, which one",
  "what causes vanishing gradients in RNNs", "how does a recurrent network
  process a sequence", "when should I use an RNN instead of a transformer".
  NOT for attention/transformer internals (see attention-mechanisms) or
  time-series forecasting workflows like ARIMA/Prophet (see time-series).
type: reference
domain: deep-learning
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - rnn-recurrence
  - backpropagation-through-time
  - lstm-gating
  - gru-gating
  - rnn-vs-attention-tradeoffs
requires:
conflicts:
related:
  - neural-net-fundamentals
  - attention-mechanisms
  - time-series
inputs: A question about how RNNs/LSTMs/GRUs process sequences or why vanilla RNNs struggle with long-range dependencies.
outputs: A conceptual explanation of recurrent architectures and their gating mechanisms, contrasted with attention.
---

## Overview
Recurrent networks process a sequence one element at a time, carrying a hidden
state forward as memory of everything seen so far — the natural fit for text,
audio, or time series where order matters. This card covers the vanilla RNN
recurrence, why it fails on long sequences, and how LSTMs and GRUs fix that with
learned gating. It's the mechanistic foundation for understanding why attention
(see attention-mechanisms) eventually replaced RNNs for most sequence tasks.

## Key Concepts
- **The recurrence.** At each timestep `t`, a vanilla RNN combines the current
  input with the previous hidden state using the *same* shared weights:
  \[ h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b) \]
  `h_t` is both the output at that step and the memory carried to the next one —
  weight sharing across time is what lets the network handle sequences of any
  length with a fixed parameter count.
- **Backpropagation through time (BPTT).** Training unrolls the recurrence into a
  chain as long as the sequence and backpropagates through the whole chain. This
  is structurally identical to backprop through a very deep feedforward network —
  the same product-of-derivatives that causes vanishing/exploding gradients in
  deep nets (see neural-net-fundamentals) causes it here too, except the "depth"
  is the sequence length, so long sequences make it worse, not just deep stacking.
- **Why vanilla RNNs forget.** Because `h_{t-1}` is squashed through `tanh` and
  remixed with new input at every step, information from many steps back gets
  diluted and its gradient shrinks geometrically — a vanilla RNN's *effective*
  memory is much shorter than its nominal sequence length.
- **LSTM — gating solves the memory problem.** An LSTM keeps a separate **cell
  state** `c_t` that flows across time with only *additive*, gated updates
  (no repeated squashing), controlled by three sigmoid gates:
  - **Forget gate** `f_t`: how much of the old cell state to keep.
  - **Input gate** `i_t`: how much of the new candidate value to write in.
  - **Output gate** `o_t`: how much of the cell state to expose as the hidden
    state.
  \[ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t) \]
  Because the forget gate can sit near 1, gradients can flow through many
  timesteps largely unchanged — this additive path is the actual fix for
  vanishing gradients, not just "more parameters."
- **GRU — a lighter-weight alternative.** Merges the forget/input gates into a
  single **update gate** and drops the separate cell state, using only the
  hidden state itself. Fewer parameters and matrix multiplies per step than an
  LSTM, and in practice performs comparably on many tasks — a reasonable default
  when training speed or parameter budget matters more than squeezing out the
  last bit of accuracy.
- **Bidirectional & stacked RNNs.** A bidirectional RNN runs one RNN
  forward and one backward over the sequence and concatenates their hidden
  states — useful whenever the *entire* sequence is available at once (e.g.,
  offline text classification), not for autoregressive generation where future
  tokens don't exist yet. Stacking multiple RNN layers lets later layers operate
  on the sequence of hidden states from earlier layers, building higher-level
  temporal features.
- **Why attention displaced RNNs.** Two structural limits: (1) recurrence is
  inherently sequential — step `t` needs step `t-1`'s output, so it can't be
  parallelized across the sequence during training the way attention can; (2)
  even with gating, the path length between two distant tokens is still
  proportional to their distance, versus the constant path length self-attention
  gives every pair of tokens. See attention-mechanisms for the mechanism that
  replaced this.

## Gotchas
- **Exploding gradients need explicit clipping.** Unlike vanishing gradients,
  which gating architecturally mitigates, exploding gradients in RNNs are common
  and are handled operationally with gradient norm clipping
  (`torch.nn.utils.clip_grad_norm_`) — skipping this is a frequent cause of loss
  suddenly spiking to `NaN` partway through training.
- **LSTMs/GRUs reduce the vanishing-gradient problem, they don't eliminate it.**
  Sequences of many hundreds of steps still degrade in practice; truncated BPTT
  (splitting long sequences into chunks and carrying hidden state across chunk
  boundaries) introduces its own bias by cutting gradient flow at chunk edges.
- **Padding variable-length sequences without masking corrupts training.**
  Feeding zero-padded timesteps through an RNN without a length-aware pack/mask
  (e.g., `pack_padded_sequence` in PyTorch) lets the model compute loss and
  gradients on padding positions, subtly skewing the learned weights — this is
  one of the most common silent bugs in RNN code.
- **Output shape confusion with bidirectional/stacked RNNs.** The hidden-state
  tensor's shape scales with `num_layers * num_directions`, and it's easy to
  index into it incorrectly (e.g., assuming the last hidden state is at index 0)
  when wiring the final hidden state into a classifier head.

## References
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) — the original LSTM paper.
- [Learning Phrase Representations using RNN Encoder-Decoder (Cho et al., 2014)](https://arxiv.org/abs/1406.1078) — introduces the GRU.
- [Understanding LSTM Networks (Christopher Olah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — the best visual walkthrough of the gating math.
