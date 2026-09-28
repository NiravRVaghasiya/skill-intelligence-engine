"""Retrieval metrics: recall@k, MRR, nDCG, with paired bootstrap confidence intervals.

`gold` is one slug or a list of acceptable slugs (a query where two skills are equally
correct). A query is satisfied by its first acceptable hit, so every metric is a function
of that rank alone and the single-gold case is unchanged.
"""
from __future__ import annotations
import math
import random

Gold = str | list[str]


def first_relevant_rank(ranked: list[str], gold: Gold) -> int | None:
    """1-based rank of the first acceptable slug, or None if none was retrieved."""
    golds = {gold} if isinstance(gold, str) else set(gold)
    for i, slug in enumerate(ranked, 1):
        if slug in golds:
            return i
    return None


def recall_at_k(ranked: list[str], gold: Gold, k: int) -> float:
    rank = first_relevant_rank(ranked, gold)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(ranked: list[str], gold: Gold) -> float:
    rank = first_relevant_rank(ranked, gold)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked: list[str], gold: Gold, k: int) -> float:
    rank = first_relevant_rank(ranked, gold)
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def per_query(rows: list[dict], k: int = 3) -> dict[str, list[float]]:
    """Metric name -> one value per row (rows carry "ranked" and "gold")."""
    return {
        f"recall@{k}": [recall_at_k(r["ranked"], r["gold"], k) for r in rows],
        "mrr": [reciprocal_rank(r["ranked"], r["gold"]) for r in rows],
        f"ndcg@{k}": [ndcg_at_k(r["ranked"], r["gold"], k) for r in rows],
    }


def aggregate(rows: list[dict], k: int = 3) -> dict:
    n = len(rows) or 1
    return {name: sum(vals) / n for name, vals in per_query(rows, k).items()}


def bootstrap_ci(values: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values` (fixed seed -> reproducible)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_delta_ci(a: list[float], b: list[float], **kw) -> tuple[float, float, float]:
    """Mean of (a - b) over paired queries with its bootstrap CI: (delta, lo, hi)."""
    if len(a) != len(b):
        raise ValueError("paired metrics need equal-length inputs")
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, **kw)
    return (sum(diffs) / (len(diffs) or 1), lo, hi)
