"""Sparse BM25 index over the same chunks (pure-python rank_bm25).

BM25 is rebuilt in memory from the corpus on first use (the corpus is ~300 chunks),
so it needs no persistence alongside the ChromaDB dense index.
"""
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

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build a BM25 index over zero chunks")
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top-k chunks by BM25; chunks sharing no query term are never returned.

        Ties break by corpus order, so results are deterministic.
        """
        if self._bm25 is None:
            raise RuntimeError("sparse index not built; call build() first")
        tokens = _tok(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(skill_slug=self._chunks[i].skill_slug, score=float(scores[i]),
                    section=self._chunks[i].section, snippet=self._chunks[i].text[:200],
                    chunk_id=self._chunks[i].chunk_id)
                for i in ranked if scores[i] > 0]
