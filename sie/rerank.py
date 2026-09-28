"""Cross-encoder reranking of fused candidates."""
from __future__ import annotations

from .models import Hit

_DEFAULT_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = _DEFAULT_CE):
        self.model_name = model_name
        self._model = None

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []
        self._lazy()
        pairs = [(query, h.snippet or h.skill_slug) for h in hits]
        scores = self._model.predict(pairs)
        for h, s in zip(hits, scores):
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
