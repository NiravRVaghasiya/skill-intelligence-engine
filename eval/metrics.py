"""Retrieval metrics: recall@k, MRR, nDCG (single-gold setting)."""
from __future__ import annotations
import math


def recall_at_k(ranked: list[str], gold: str, k: int) -> float:
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked: list[str], gold: str) -> float:
    for i, slug in enumerate(ranked, 1):
        if slug == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: str, k: int) -> float:
    for i, slug in enumerate(ranked[:k], 1):
        if slug == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


def aggregate(rows: list[dict], k: int = 3) -> dict:
    n = len(rows) or 1
    return {
        f"recall@{k}": sum(recall_at_k(r["ranked"], r["gold"], k) for r in rows) / n,
        "mrr": sum(reciprocal_rank(r["ranked"], r["gold"]) for r in rows) / n,
        f"ndcg@{k}": sum(ndcg_at_k(r["ranked"], r["gold"], k) for r in rows) / n,
    }
