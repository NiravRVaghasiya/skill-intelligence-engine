"""Reciprocal Rank Fusion (RRF) of multiple ranked hit lists."""
from __future__ import annotations
from collections import defaultdict

from ..models import Hit


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Fuse ranked lists into one, scoring each doc by sum(1 / (k + rank))."""
    scores: dict[str, float] = defaultdict(float)
    best_snippet: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.skill_slug] += 1.0 / (k + rank + 1)
            best_snippet.setdefault(hit.skill_slug, hit)
    fused = [Hit(skill_slug=slug, score=score,
                 section=best_snippet[slug].section, snippet=best_snippet[slug].snippet)
             for slug, score in scores.items()]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused
