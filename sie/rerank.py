"""Cross-encoder reranking of fused candidates.

Weights resolve in order: SIE_RERANKER (HF id or local dir) > data/models/ms-marco-MiniLM-L-6-v2
(if present) > the Hugging Face hub id. Where huggingface.co is blocked, copy the model
files into that local directory.
"""
from __future__ import annotations
import importlib.util
import os
import re
import sys
import threading
from dataclasses import replace
from pathlib import Path

from .hub import cached_locally, looks_like_path, unavailable_reason
from .models import Hit

_DEFAULT_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# <repo>/data/models/..., independent of the cwd (like the API's corpus/index defaults);
# messages show the repo-relative spelling so generated reports carry no machine paths.
LOCAL_CE_HINT = "data/models/ms-marco-MiniLM-L-6-v2"
LOCAL_CE_DIR = Path(__file__).resolve().parents[1] / LOCAL_CE_HINT
_LIBRARY_BATCH_SIZE = 32          # CrossEncoder.predict's own default


class RerankerUnavailable(RuntimeError):
    """The cross-encoder weights could not be loaded (e.g. huggingface.co unreachable)."""


def default_model() -> str:
    """Reranker model id/path honoring SIE_RERANKER and the local weights directory."""
    return os.getenv("SIE_RERANKER") or (str(LOCAL_CE_DIR) if LOCAL_CE_DIR.is_dir() else _DEFAULT_CE)


def _is_local_ce_dir(name: str) -> bool:
    if name == str(LOCAL_CE_DIR):
        return True
    try:
        resolved = Path(name).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return os.path.normcase(str(resolved)) == os.path.normcase(str(LOCAL_CE_DIR))


def display_model(name: str) -> str:
    """A report-safe spelling of a cross-encoder id/path: no machine paths in messages or status.

    Args:
        name: HF hub id or local model directory.

    Returns:
        LOCAL_CE_HINT for the repo's default local weights directory, the last path component
        for any other local path, and hub ids unchanged.
    """
    if _is_local_ce_dir(name):
        return LOCAL_CE_HINT
    if looks_like_path(name):
        tail = re.split(r"[\\/]", name.rstrip("\\/"))[-1]   # either separator, on any OS
        return tail or name
    return name


def _st_installed() -> bool:
    """Whether sentence-transformers is importable, checked without importing it (or torch)."""
    if sys.modules.get("sentence_transformers") is not None:
        return True
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):      # meta-path blockers (tests/conftest.py) raise
        return False


def _positive_int(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


class Reranker:
    def __init__(self, model_name: str | None = None, batch_size: int = _LIBRARY_BATCH_SIZE,
                 max_length: int | None = None):
        """
        Args:
            model_name: HF id or local directory; default `default_model()`.
            batch_size: (query, chunk) pairs scored per forward pass.
            max_length: truncate each pair to this many tokens; None keeps the model's limit.
        """
        _positive_int("batch_size", batch_size)
        if max_length is not None:
            _positive_int("max_length", max_length)
        self.model_name = model_name or default_model()
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """True once the cross-encoder weights are in memory."""
        return self._model is not None

    def _unavailable(self, why: str) -> RerankerUnavailable:
        shown = display_model(self.model_name)
        why = why.replace(self.model_name, shown)       # e.g. "local model path not found: <path>"
        return RerankerUnavailable(
            f"cannot load cross-encoder '{shown}' ({why}); copy the weights into "
            f"{LOCAL_CE_HINT}/ or set SIE_RERANKER, or run with --no-rerank")

    def _not_installed(self) -> RerankerUnavailable:
        return RerankerUnavailable(
            f"cannot load cross-encoder '{display_model(self.model_name)}' (sentence-transformers "
            "not installed); install the 'reranker' extra (pip install -e '.[reranker]') "
            "or run with --no-rerank")

    def load(self) -> None:
        """Load the weights once (thread-safe); a no-op when already loaded.

        Raises:
            RerankerUnavailable: no local weights and an unreachable hub, a missing local
                path, sentence-transformers not installed, or the model failed to load.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not _st_installed():                     # no hub probe for a library that can't load
                raise self._not_installed()
            why = unavailable_reason(self.model_name)   # decided before importing torch
            if why:
                raise self._unavailable(why)
            kwargs = {"local_files_only": cached_locally(self.model_name)}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, **kwargs)
            except ImportError:
                raise self._not_installed() from None
            except Exception as e:
                raise self._unavailable(type(e).__name__) from e

    def _lazy(self) -> None:
        """Backward-compatible alias of `load()`."""
        self.load()

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
        self.load()
        texts = texts or {}
        pairs = [(query, texts.get(h.chunk_id) or h.snippet or h.skill_slug) for h in hits]
        kwargs = {"show_progress_bar": False}
        if self.batch_size != _LIBRARY_BATCH_SIZE:     # the default is predict()'s own
            kwargs["batch_size"] = self.batch_size
        scores = self._model.predict(pairs, **kwargs)
        out = [replace(h, score=float(s)) for h, s in zip(hits, scores)]
        out.sort(key=lambda h: (-h.score, h.skill_slug, h.chunk_id))
        return out if top_k is None else out[:top_k]
