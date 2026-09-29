"""Dense vector index backed by ChromaDB (persisted to data/chroma/).

Embeddings are all-MiniLM-L6-v2 from one of two interchangeable backends, chosen with
SIE_EMBEDDER (or the `embedder` argument):
    onnx (default)         chromadb's bundled ONNX export; fetched from chroma's S3 bucket,
                           so it works where huggingface.co is unreachable
    sentence-transformers  the PyTorch model from the Hugging Face hub

Search uses an HNSW graph. `ef_search` (default 400) is the candidate-list size per query;
at or above the corpus size (~300 chunks here) the search is effectively exact, and so
reproducible. For much larger corpora HNSW is approximate and ef_search trades recall for
latency; that trade-off has not been measured in this repo.
"""
from __future__ import annotations
import os
import threading
from pathlib import Path

from ..hub import cached_locally, unavailable_reason
from ..models import Chunk, Hit

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION = "skills"
_STAGING = "skills__staging"
_BACKUP = "skills__backup"
EMBEDDERS = ("onnx", "sentence-transformers")
# ef_search >= corpus size makes HNSW effectively exact (and so reproducible) at this scale.
_HNSW = {"space": "cosine", "ef_construction": 200, "ef_search": 400}
_CORE_METADATA = ("embedder", "fingerprint")     # freshness keys; build() metadata can't override


def _metadata_value(value) -> str | int | float | bool:
    """Chroma collection metadata holds scalars only: keep str/int/float/bool, stringify the rest."""
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _max_batch(client, default: int = 5000) -> int:
    """Rows one `collection.add()` accepts on this client (its own limit when it reports one)."""
    get = getattr(client, "get_max_batch_size", None)
    try:
        size = int(get()) if callable(get) else default
    except Exception:
        size = default
    return max(1, size)


class DenseIndex:
    def __init__(self, persist_dir: str = "data/chroma", model_name: str = _DEFAULT_MODEL,
                 embedder: str | None = None, ef_search: int = 400):
        """
        Args:
            persist_dir: ChromaDB directory.
            model_name: sentence-transformers model id (the onnx backend is the same model).
            embedder: "onnx" or "sentence-transformers"; default SIE_EMBEDDER, else "onnx".
            ef_search: HNSW query-time candidate list size, fixed when the index is built
                (see the module docstring).
        """
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.embedder = embedder or os.getenv("SIE_EMBEDDER", "onnx")
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"unknown embedder {self.embedder!r}; choose one of {EMBEDDERS}")
        if isinstance(ef_search, bool) or not isinstance(ef_search, int) or ef_search < 1:
            raise ValueError(f"ef_search must be an integer >= 1, got {ef_search!r}")
        self.ef_search = ef_search
        self._client = None
        self._coll = None
        self._embed_fn = None
        self._embed_lock = threading.Lock()   # one embedding-model load, even for concurrent cold calls
        self._client_lock = threading.Lock()  # one chroma client, even for concurrent cold calls
        self.generation = 0          # bumps whenever the collection handle is (re)bound

    def _lazy_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import chromadb
                    from chromadb.config import Settings
                    Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        return self._client

    def rebind(self) -> None:
        """Forget the cached collection handle, so the next read sees the index as persisted now.

        A handle's metadata is a snapshot: after another process rebuilds the index, a freshness
        check through the old handle would keep reporting the old (e.g. stale) fingerprint.
        """
        self._coll = None

    def _lazy_embedder(self):
        """The embedding function, loaded once (thread-safe, double-checked).

        The first embedding call runs inside the lock too: the ONNX backend downloads the
        model and builds its session on first use, not in its constructor. The function is
        published only after that call succeeded, so a failed load is retried next time.
        """
        if self._embed_fn is None:
            with self._embed_lock:
                if self._embed_fn is None:
                    if self.embedder == "onnx":
                        # The model DefaultEmbeddingFunction delegates to, held once: that
                        # wrapper builds a fresh ONNX session per call (~0.5 s per query on
                        # CPU). Same weights, tokenizer and pooling: identical embeddings.
                        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
                        fn = ONNXMiniLM_L6_V2()
                    else:
                        fn = self._load_sentence_transformer()
                    fn(["warm up"])          # onnx: download + InferenceSession, exactly once
                    self._embed_fn = fn
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

    def build(self, chunks: list[Chunk], fingerprint: str = "",
              metadata: dict | None = None) -> None:
        """Rebuild from scratch (removed chunks never linger) without losing the old index.

        Embeddings are computed and written to a staging collection first; the live
        collection is swapped out (renamed to a backup, restored on failure) only once that
        succeeded, so a failed build leaves the previous index intact.

        Args:
            chunks: every chunk of the corpus.
            fingerprint: chunk-level corpus fingerprint, checked for index freshness.
            metadata: extra facts recorded on the collection (see `info()`); values that are
                not str/int/float/bool are stored as str, None values are dropped, and the
                `embedder` / `fingerprint` keys always reflect this build.
        """
        ids = [c.chunk_id for c in chunks]
        if not chunks or len(set(ids)) != len(ids):
            raise RuntimeError("cannot build: chunk list is empty or has duplicate chunk ids")
        embeddings = self._embed([c.text for c in chunks])
        client = self._lazy_client()
        self._drop(_STAGING)
        extra = {str(key): _metadata_value(value) for key, value in (metadata or {}).items()
                 if value is not None and str(key) not in _CORE_METADATA}
        staging = client.create_collection(
            _STAGING, configuration={"hnsw": {**_HNSW, "ef_search": self.ef_search}},
            embedding_function=None,
            metadata={**extra, "embedder": self.embedder, "fingerprint": fingerprint or "-"})
        step = _max_batch(client)
        for i in range(0, len(chunks), step):   # one add() is capped (5461 rows on chromadb 1.5)
            part = chunks[i:i + step]
            staging.add(ids=ids[i:i + step], embeddings=embeddings[i:i + step],
                        documents=[c.text for c in part],
                        metadatas=[{"slug": c.skill_slug, "section": c.section} for c in part])
        self._swap_in(staging)
        self._coll = staging
        self.generation += 1

    def _swap_in(self, staging) -> None:
        """live -> backup, staging -> live, drop backup; restore the backup if the rename fails.

        Concurrent readers may briefly find no live collection; search() rebinds once.
        """
        self._coll = None
        live = self._collection()
        self._drop(_BACKUP)
        if live is not None:
            live.modify(name=_BACKUP)
        try:
            staging.modify(name=_COLLECTION)
        except Exception:
            if live is not None:
                live.modify(name=_COLLECTION)
            raise
        self._drop(_BACKUP)

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

    def info(self) -> dict | None:
        """What the persisted index records (build metadata + `count` of chunks), None if never built.

        Indexes built before metadata was recorded only carry `embedder` and `fingerprint`.
        A missing persist_dir means "never built": it is not created just to say so.
        """
        if self._coll is None and not Path(self.persist_dir).is_dir():
            return None
        coll = self._collection()
        if coll is None:
            return None
        return {**(coll.metadata or {}), "count": coll.count()}

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
