"""Dense vector index backed by ChromaDB + sentence-transformers."""
from __future__ import annotations
from pathlib import Path

from ..models import Chunk, Hit

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseIndex:
    def __init__(self, persist_dir: str = "data/chroma", model_name: str = _DEFAULT_MODEL):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self._client = None
        self._coll = None
        self._model = None

    def _lazy(self):
        if self._coll is None:
            import chromadb
            from sentence_transformers import SentenceTransformer
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._coll = self._client.get_or_create_collection("skills")
            self._model = SentenceTransformer(self.model_name)

    def build(self, chunks: list[Chunk]) -> None:
        self._lazy()
        embeddings = self._model.encode([c.text for c in chunks], show_progress_bar=False).tolist()
        self._coll.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"slug": c.skill_slug, "section": c.section} for c in chunks],
        )

    def search(self, query: str, k: int = 10) -> list[Hit]:
        self._lazy()
        q = self._model.encode([query]).tolist()
        res = self._coll.query(query_embeddings=q, n_results=k)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(skill_slug=meta["slug"], score=1.0 - dist,
                            section=meta.get("section", ""), snippet=doc[:200]))
        return hits
