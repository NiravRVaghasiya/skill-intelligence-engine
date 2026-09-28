---
name: cnn-vision
display_name: CNNs for Vision
description: >
  Use when the user wants to understand convolutional neural networks — the
  convolution operation, kernels/filters, pooling, receptive fields, or classic
  CNN architectures (LeNet, VGG, ResNet, etc.). Trigger phrases: "explain how
  convolution works", "what is a receptive field", "max pooling vs average
  pooling", "why do ResNets use skip connections", "how do CNNs see images".
  NOT for the idiomatic PyTorch code to build/train one (see pytorch-patterns
  and training-deep-models) or task-level computer vision workflows like
  detection/segmentation (see computer-vision).
type: reference
domain: deep-learning
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: primary
last_verified: 2026-09-21
capabilities:
  - convolution-arithmetic
  - receptive-field-calculation
  - pooling-strategies
  - resnet-skip-connections
  - cnn-architecture-lineage
requires:
conflicts:
related:
  - neural-net-fundamentals
  - training-deep-models
  - pytorch-patterns
inputs: A question about convolution, pooling, receptive fields, or classic CNN architectures (LeNet/AlexNet/VGG/ResNet).
outputs: A conceptual explanation of CNN mechanics and the architectural reasoning behind VGG/ResNet-style designs.
---

## Overview
Convolutional neural networks exploit the structure of images — nearby pixels are
related, and a pattern (an edge, a texture) is meaningful wherever it appears in
the frame. Instead of a fully-connected layer's one-weight-per-pixel-pair, a CNN
slides a small, shared filter across the whole image. This card covers the
convolution and pooling operations, how receptive fields grow with depth, and the
architectural ideas (VGG-style stacking, ResNet skip connections) that let CNNs
go deep.

## Key Concepts
- **The convolution operation.** A kernel (e.g., 3×3×`C_in`) slides over the input
  with a given **stride**, computing a dot product at each position to produce one
  value in the output **feature map**. With `C_out` kernels you get `C_out`
  feature maps. Output spatial size:
  \[ H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1 \]
  where `P` = padding, `K` = kernel size, `S` = stride. `padding="same"` picks `P`
  so `H_out == H_in`; `padding="valid"` means `P=0` and the output shrinks.
- **Parameter sharing & translation equivariance.** The same kernel is applied at
  every spatial location, so a detector learned for "vertical edge" fires
  wherever a vertical edge appears — this is the core inductive bias that makes
  CNNs vastly more parameter-efficient than a fully-connected layer on images, and
  why they generalize better from limited data.
- **Channels & 1×1 convolutions.** Each layer's output has `C_out` channels, each
  a different learned feature detector. A 1×1 convolution doesn't look at
  neighboring pixels at all — it only mixes channels — making it a cheap way to
  change channel depth (dimensionality reduction/expansion) without touching
  spatial size, heavily used in bottleneck blocks (ResNet, Inception).
- **Pooling.** Downsamples feature maps to reduce computation and build a small
  amount of translation *invariance* (not just equivariance). **Max pooling**
  keeps the strongest activation in each window (good for "does this feature
  exist here" tasks like classification). **Average pooling** smooths instead of
  picking a peak (used e.g. in global average pooling before the final
  classifier, which replaces a large flatten+FC layer and cuts parameters
  drastically).
- **Receptive field.** The region of the *original input* that a given output
  unit is sensitive to. It grows with depth — stacking 3×3 convolutions increases
  the receptive field roughly linearly with layer count, which is why VGG-style
  networks favor many small 3×3 kernels stacked deep over a few large kernels
  (cheaper, and more nonlinearities for the same field of view).
- **Classic architecture lineage.**
  - **LeNet (1998):** conv → pool stacks → FC, proof of concept on digits.
  - **AlexNet (2012):** deeper, ReLU instead of tanh, dropout — the ImageNet
    breakthrough that revived deep learning.
  - **VGG (2014):** uniform 3×3 conv stacks + pooling, showed depth alone helps —
    but purely sequential stacking hits diminishing (then negative) returns past
    ~20 layers due to vanishing gradients.
  - **ResNet (2015):** introduces the **residual/skip connection** — each block
    learns a residual `F(x)` added to its input, `y = F(x) + x`, instead of
    learning the full mapping. The identity shortcut gives gradients a direct
    path back to early layers, which is what actually enabled networks with
    50–150+ layers to train at all.
- **Feature hierarchy.** Early layers learn low-level features (edges,
  color blobs, oriented gradients); middle layers combine these into textures and
  parts; late layers represent whole objects/categories. This is why transfer
  learning works — early-layer filters transfer across almost any vision task.

## Gotchas
- **Padding math mismatches** are the most common CNN shape bug: `"same"` padding
  in one framework's convention can differ by one pixel from another's when
  stride > 1. Always print `.shape` after each layer when debugging a new
  architecture rather than trusting arithmetic by hand.
- **Aggressive early pooling throws away spatial information you can't get back**
  — fine for pure classification, but a mistake for detection/segmentation
  backbones where fine spatial detail late in the network matters (see
  computer-vision for architectures like U-Net/FPN that address this).
- **Transfer learning with mismatched normalization statistics silently hurts
  accuracy.** A pretrained backbone (e.g., ImageNet ResNet) expects inputs
  normalized with the *same* per-channel mean/std used during its pretraining —
  using your own dataset's statistics instead is a subtle, easy-to-miss bug.
- **Global average pooling vs. flatten+FC is not a stylistic choice.** Flattening
  a large feature map into an FC layer creates a huge, overfitting-prone weight
  matrix tied to a fixed input resolution; global average pooling is far cheaper
  and resolution-agnostic, but it discards positional information the FC head
  would have retained.

## References
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385) — the ResNet paper; motivates skip connections via the degradation problem in very deep nets.
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556) — the VGG paper.
- [CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)](https://cs231n.github.io/convolutional-networks/) — the best worked-through explanation of the convolution/pooling arithmetic and intuition.
