"""Reciprocal Rank Fusion (RRF) of multiple ranked hit lists."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace

from ..models import Hit


def best_per_skill(ranking: list[Hit]) -> list[Hit]:
    """Collapse a chunk-level ranking to one hit per skill (its best-ranked chunk), order kept."""
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in ranking:
        if hit.skill_slug not in seen:
            seen.add(hit.skill_slug)
            out.append(hit)
    return out


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Fuse ranked lists into one skill-level ranking, scoring each skill by sum(1 / (k + rank)).

    Each list is first collapsed to one entry per skill, so a skill with many matching
    chunks is not over-counted. A fused hit keeps the section/snippet of the skill's
    best-ranked chunk (earlier lists win ties); equal scores break by slug.
    """
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, tuple[int, Hit]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(best_per_skill(ranking), 1):
            scores[hit.skill_slug] += 1.0 / (k + rank)
            if hit.skill_slug not in best or rank < best[hit.skill_slug][0]:
                best[hit.skill_slug] = (rank, hit)
    fused = [replace(best[slug][1], score=score) for slug, score in scores.items()]
    fused.sort(key=lambda h: (-round(h.score, 12), h.skill_slug))
    return fused
