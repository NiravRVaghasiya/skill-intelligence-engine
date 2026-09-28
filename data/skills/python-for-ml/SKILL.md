---
name: python-for-ml
display_name: Python for ML (NumPy & Pandas)
description: >
  Use when the user wants to write idiomatic numpy/pandas code, vectorize a
  slow Python loop, or debug unexpected array/DataFrame behavior.
  Trigger phrases: "vectorize this loop", "numpy broadcasting rules", "speed
  up my pandas code", "why is my pandas apply so slow", "how do I reshape
  this array". NOT for statistical analysis of the resulting data (see
  statistics-for-ml) and NOT for building a modeling-ready preprocessing
  pipeline (see data-preprocessing).
type: reference
domain: foundations
level: beginner
lifecycle: stable
risk_level: low
evidence_level: official-documentation
last_verified: 2026-09-21
capabilities:
  - numpy-broadcasting-rules
  - vectorization-vs-loops
  - view-vs-copy-semantics
  - pandas-chained-indexing-fixes
  - dtype-overflow-debugging
  - pandas-vectorized-groupby
requires:
conflicts:
related:
  - data-preprocessing
  - feature-engineering
  - ml-math-essentials
inputs: A numpy/pandas code snippet or performance problem (slow loop, unexpected shape, chained-indexing warning) to vectorize or debug.
outputs: An idiomatic vectorized numpy/pandas rewrite, or an explanation of the broadcasting/view/dtype mechanism causing the observed bug.
version_constraints:
  - "numpy: np.random.default_rng() Generator API estimated available since 1.17 (model-knowledge estimate, not live-verified this session)"
---

## Overview
The idioms and gotchas of numpy and pandas that separate fast, correct
numerical Python from code that silently does the wrong thing or falls back
to a slow Python-level loop. This is the implementation substrate underneath
almost every other skill in this library — read it when the question is
"how do I express this computation efficiently in numpy/pandas," not "what
should this computation be."

## Key Concepts
- **`ndarray` and dtype.** A numpy array is a fixed-dtype, contiguous block
  of memory — this is what makes vectorized ops fast (SIMD, no per-element
  Python object overhead). Mixed types force `dtype=object`, which loses all
  of that speed; check `arr.dtype` when something is unexpectedly slow.
- **Broadcasting.** Numpy aligns array shapes from the *trailing* dimension:
  shapes are compatible if, for each dimension (right to left), they're equal
  or one of them is 1. `(n, d) + (d,)` broadcasts the vector across every
  row; `(n, 1) + (1, d)` produces an `(n, d)` outer-product-like result. Most
  "why is my output huge/wrong shape" bugs are a broadcasting mismatch.
- **Vectorization vs. Python loops.** `for` loops over array elements run at
  Python speed (~10-100x slower); pushing the loop into numpy/pandas C
  internals (`arr * 2`, `df.col.sum()`, boolean masks) is the single biggest
  performance lever available without leaving pure Python.
- **Views vs. copies.** Basic slicing (`arr[1:5]`) returns a *view* — mutating
  it mutates the original. Fancy indexing (`arr[[1,3,5]]`) and boolean masks
  return *copies*. Use `.copy()` explicitly whenever you need to guarantee
  independence, and `arr.base is not None` to check if something is a view.
- **pandas Series/DataFrame vectorized ops.** Prefer `df["a"] + df["b"]`,
  `df.groupby("key")["val"].sum()`, and boolean masks (`df[df["x"] > 0]`) over
  `.apply()`/`.iterrows()` — the former call optimized C/Cython code, the
  latter are Python-level loops in disguise.
- **`np.where` / `np.select` / `pd.merge`.** `np.where(cond, a, b)` is a
  vectorized ternary; `np.select([cond1, cond2], [val1, val2], default)`
  generalizes it to multiple conditions; `pd.merge` is the vectorized
  equivalent of a manual row-by-row lookup join.

```python
import numpy as np
import pandas as pd

# Broadcasting: normalize every row of a (n, d) matrix by its own mean/std
X = np.random.default_rng(0).normal(size=(1000, 5))
X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# Vectorized instead of a Python loop over rows
df = pd.DataFrame({"category": ["a", "b", "a", "c"], "value": [10, 20, 30, 40]})
totals = df.groupby("category")["value"].sum()          # fast, C-level
tier = np.select(
    [df["value"] < 15, df["value"] < 35],
    ["low", "mid"],
    default="high",
)
df["tier"] = tier
```

## Gotchas
- **`SettingWithCopyWarning` from chained indexing.** `df[df.x > 0]["y"] = 1`
  may silently fail to modify `df` because the intermediate `df[df.x > 0]` can
  be a copy. Use `df.loc[df.x > 0, "y"] = 1` instead — it's unambiguous and
  guaranteed to mutate in place.
- **`.apply()` is not vectorization.** `df["col"].apply(some_func)` still
  calls `some_func` once per row in a Python-level loop — it's more readable
  than `.iterrows()` but not fundamentally faster. Reach for a truly vectorized
  expression, `np.where`/`np.select`, or `.map()` with a dict for lookups first.
- **Broadcasting shape mismatches fail silently, not loudly.** `(n,)` vs.
  `(n, 1)` vs. `(1, n)` combine into different (often unintended) shapes
  without an error. Always check `.shape` after a broadcasted op you didn't
  expect to be tricky.
- **Default integer dtype overflow.** `np.array([1, 2, 3], dtype=np.int32).sum()`
  can silently overflow on large sums; on Windows, numpy's default int dtype
  is `int32` (vs. `int64` on Linux/Mac) for the same code, causing
  platform-dependent bugs. Be explicit about `dtype` in numeric-heavy code.
- **Mutating a view mutates the source.** `sub = df.iloc[:5]; sub["x"] = 0`
  can raise the same chained-assignment warning and/or unexpectedly alter
  `df`, because `.iloc` slices can return views. Use `.copy()` when you intend
  an independent subset.

## References
- [NumPy — Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — the authoritative rules, worth memorizing.
- [pandas — Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html) — covers `.loc`/`.iloc`, views vs. copies, and chained-indexing pitfalls.
- [pandas — Enhancing performance](https://pandas.pydata.org/docs/user_guide/enhancingperf.html) — official guidance on replacing `.apply()`/loops with vectorized code.
