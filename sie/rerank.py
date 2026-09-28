"""Cross-encoder reranking of fused candidates.

Weights resolve in order: SIE_RERANKER (HF id or local dir) > data/models/ms-marco-MiniLM-L-6-v2
(if present) > the Hugging Face hub id. Where huggingface.co is blocked, copy the model
files into that local directory.
"""
from __future__ import annotations
import os
from dataclasses import replace
from pathlib import Path

from .hub import cached_locally, unavailable_reason
from .models import Hit

_DEFAULT_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_CE_DIR = Path("data/models/ms-marco-MiniLM-L-6-v2")


class RerankerUnavailable(RuntimeError):
    """The cross-encoder weights could not be loaded (e.g. huggingface.co unreachable)."""


def default_model() -> str:
    """Reranker model id/path honoring SIE_RERANKER and the local weights directory."""
    return os.getenv("SIE_RERANKER") or (str(LOCAL_CE_DIR) if LOCAL_CE_DIR.is_dir() else _DEFAULT_CE)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or default_model()
        self._model = None

    def _unavailable(self, why: str) -> RerankerUnavailable:
        return RerankerUnavailable(
            f"cannot load cross-encoder '{self.model_name}' ({why}); copy the weights into "
            f"{LOCAL_CE_DIR.as_posix()}/ or set SIE_RERANKER, or run with --no-rerank")

    def _lazy(self):
        if self._model is not None:
            return
        why = unavailable_reason(self.model_name)   # decided before importing torch
        if why:
            raise self._unavailable(why)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, local_files_only=cached_locally(self.model_name))
        except ImportError:
            raise self._unavailable("sentence-transformers not installed") from None
        except Exception as e:
            raise self._unavailable(type(e).__name__) from e

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None,
               texts: dict[str, str] | None = None) -> list[Hit]:
        """Score (query, chunk text) pairs and return hits sorted by cross-encoder score.

        Args:
            query: the user task description.
            hits: chunk-level candidates (inputs are not mutated).
            top_k: truncate after sorting; None keeps all (the router dedups by skill after).
            texts: chunk_id -> full chunk text; falls back to the hit's 200-char snippet.
        """
        if not hits:
            return []
        self._lazy()
        texts = texts or {}
        pairs = [(query, texts.get(h.chunk_id) or h.snippet or h.skill_slug) for h in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)
        out = [replace(h, score=float(s)) for h, s in zip(hits, scores)]
        out.sort(key=lambda h: (-h.score, h.skill_slug, h.chunk_id))
        return out if top_k is None else out[:top_k]
