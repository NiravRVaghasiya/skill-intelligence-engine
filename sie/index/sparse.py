"""Sparse BM25 index over the same chunks (pure-python rank_bm25).

BM25 is rebuilt in memory from the corpus on first use (the corpus is ~300 chunks),
so it needs no persistence alongside the ChromaDB dense index. Besides search, the index
explains lexical evidence with its own idf: which query terms a chunk contains
(`matched_terms`) and what idf-weighted share of the query they make up (`coverage`).
"""
from __future__ import annotations
import re

from ..models import Chunk, Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric runs; the one tokenizer used for indexing, queries and evidence."""
    return _TOKEN_RE.findall(text.lower())


_tok = tokenize        # backward-compatible alias


class SparseIndex:
    def __init__(self):
        self._bm25 = None
        self._chunks: list[Chunk] = []
        self._max_idf = 0.0

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build a BM25 index over zero chunks")
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._max_idf = max(bm25.idf.values(), default=0.0)
        self._chunks = chunks
        self._bm25 = bm25              # published last: is_built implies everything else is set

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top-k chunks by BM25; chunks sharing no query term are never returned.

        Ties break by corpus order, so results are deterministic.
        """
        if self._bm25 is None:
            raise RuntimeError("sparse index not built; call build() first")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(skill_slug=self._chunks[i].skill_slug, score=float(scores[i]),
                    section=self._chunks[i].section, snippet=self._chunks[i].text[:200],
                    chunk_id=self._chunks[i].chunk_id)
                for i in ranked if scores[i] > 0]

    @staticmethod
    def terms(text: str) -> list[str]:
        """Unique tokens of `text` in first-seen order."""
        return list(dict.fromkeys(tokenize(text)))

    def idf(self, term: str) -> float:
        """BM25 idf of `term`; a term the corpus never uses gets the index's maximum idf.

        Raises:
            RuntimeError: the index has not been built.
        """
        if self._bm25 is None:
            raise RuntimeError("sparse index not built; call build() first")
        return float(self._bm25.idf.get(term, self._max_idf))

    def coverage(self, query: str, text: str) -> float:
        """Idf-weighted share of the query's unique terms that occur in `text`, in [0, 1].

        Weights are the index's idf floored at 0 (rank_bm25 can assign negative idf to terms
        in most chunks). If every query term weighs 0, the unweighted share is returned.
        A query with no tokens has coverage 0.0.
        """
        query_terms = self.terms(query)
        if not query_terms:
            return 0.0
        present = set(tokenize(text))
        weights = {t: max(self.idf(t), 0.0) for t in query_terms}
        total = sum(weights.values())
        if total <= 0.0:
            return sum(t in present for t in query_terms) / len(query_terms)
        return min(1.0, sum(w for t, w in weights.items() if t in present) / total)

    def matched_terms(self, query: str, text: str) -> list[str]:
        """Unique query terms that occur in `text`, in query order."""
        present = set(tokenize(text))
        return [t for t in self.terms(query) if t in present]
