"""Dense vector index backed by ChromaDB (persisted to data/chroma/).

Embeddings are all-MiniLM-L6-v2 from one of two interchangeable backends, chosen with
SIE_EMBEDDER (or the `embedder` argument):
    onnx (default)         chromadb's bundled ONNX export; fetched from chroma's S3 bucket,
                           so it works where huggingface.co is unreachable
    sentence-transformers  the PyTorch model from the Hugging Face hub
"""
from __future__ import annotations
import os
from pathlib import Path

from ..hub import cached_locally, unavailable_reason
from ..models import Chunk, Hit

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION = "skills"
_STAGING = "skills__staging"
EMBEDDERS = ("onnx", "sentence-transformers")
# ef_search >= corpus size makes HNSW effectively exact (and so reproducible) at this scale.
_HNSW = {"space": "cosine", "ef_construction": 200, "ef_search": 400}


class DenseIndex:
    def __init__(self, persist_dir: str = "data/chroma", model_name: str = _DEFAULT_MODEL,
                 embedder: str | None = None):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embedder = embedder or os.getenv("SIE_EMBEDDER", "onnx")
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"unknown embedder {self.embedder!r}; choose one of {EMBEDDERS}")
        self._client = None
        self._coll = None
        self._embed_fn = None
        self.generation = 0          # bumps whenever the collection handle is (re)bound

    def _lazy_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        return self._client

    def _lazy_embedder(self):
        if self._embed_fn is None:
            if self.embedder == "onnx":
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                self._embed_fn = DefaultEmbeddingFunction()
            else:
                self._embed_fn = self._load_sentence_transformer()
        return self._embed_fn

    def _load_sentence_transformer(self):
        why = unavailable_reason(self.model_name)   # decided before importing torch
        if why:
            raise RuntimeError(f"cannot load embedder '{self.model_name}' ({why}); "
                               f"use SIE_EMBEDDER=onnx")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name, local_files_only=cached_locally(self.model_name))
        return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._lazy_embedder()(texts)
        except RuntimeError:
            raise
        except Exception as e:     # e.g. the ONNX model can't be fetched
            raise RuntimeError(f"embedding with the {self.embedder} backend failed: {e}") from e
        return [[float(x) for x in vec] for vec in vectors]

    def _collection(self):
        """The persisted collection, or None if it has never been built."""
        if self._coll is None:
            from chromadb.errors import NotFoundError
            try:
                self._coll = self._lazy_client().get_collection(_COLLECTION, embedding_function=None)
            except NotFoundError:
                return None
        return self._coll

    def build(self, chunks: list[Chunk], fingerprint: str = "") -> None:
        """Rebuild from scratch (removed chunks never linger), swapping in atomically.

        Embeddings are computed and written to a staging collection first; the live
        collection is replaced only once that succeeded, so a failed build leaves the
        previous index intact.
        """
        ids = [c.chunk_id for c in chunks]
        if not chunks or len(set(ids)) != len(ids):
            raise RuntimeError("cannot build: chunk list is empty or has duplicate chunk ids")
        embeddings = self._embed([c.text for c in chunks])
        client = self._lazy_client()
        self._drop(_STAGING)
        staging = client.create_collection(
            _STAGING, configuration={"hnsw": _HNSW}, embedding_function=None,
            metadata={"embedder": self.embedder, "fingerprint": fingerprint or "-"})
        staging.add(ids=ids, embeddings=embeddings, documents=[c.text for c in chunks],
                    metadatas=[{"slug": c.skill_slug, "section": c.section} for c in chunks])
        self._drop(_COLLECTION)
        staging.modify(name=_COLLECTION)
        self._coll = staging
        self.generation += 1

    def _drop(self, name: str) -> None:
        from chromadb.errors import NotFoundError
        try:
            self._lazy_client().delete_collection(name)
        except NotFoundError:
            pass

    def fingerprint(self) -> str | None:
        """Corpus fingerprint recorded at build time, or None if the index was never built."""
        coll = self._collection()
        return None if coll is None else (coll.metadata or {}).get("fingerprint")

    def built_with(self) -> str | None:
        coll = self._collection()
        return None if coll is None else (coll.metadata or {}).get("embedder")

    def _query(self, embedding: list[list[float]], k: int) -> dict:
        coll = self._collection()
        if coll is None or coll.count() == 0:
            raise RuntimeError("dense index is empty; run `python -m sie.router --build`")
        return coll.query(query_embeddings=embedding, n_results=min(k, coll.count()),
                          include=["documents", "metadatas", "distances"])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top-k chunks by cosine similarity (score = 1 - cosine distance)."""
        from chromadb.errors import NotFoundError
        embedding = self._embed([query])
        try:
            res = self._query(embedding, k)
        except NotFoundError:      # rebuilt by another process: rebind the handle once
            self._coll = None
            self.generation += 1
            try:
                res = self._query(embedding, k)
            except NotFoundError as e:
                raise RuntimeError("dense index vanished mid-query; rebuild it") from e
        hits = [Hit(skill_slug=meta["slug"], score=1.0 - dist, section=meta.get("section", ""),
                    snippet=doc[:200], chunk_id=cid)
                for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]
        hits.sort(key=lambda h: (-round(h.score, 9), h.chunk_id))
        return hits
