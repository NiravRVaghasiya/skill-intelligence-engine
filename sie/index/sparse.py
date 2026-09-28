"""Sparse BM25 index over the same chunks (pure-python rank_bm25)."""
from __future__ import annotations
import re

from ..models import Chunk, Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    def __init__(self):
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(skill_slug=self._chunks[i].skill_slug, score=float(scores[i]),
                    section=self._chunks[i].section, snippet=self._chunks[i].text[:200])
                for i in ranked]
