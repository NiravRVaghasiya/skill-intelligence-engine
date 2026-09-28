"""Top-level hybrid retrieval: dense + sparse -> RRF -> rerank -> dedup by skill.

Usage:
    python -m sie.router --build                        # build indexes from data/skills
    python -m sie.router "your task description"        # query (hybrid + rerank)
    python -m sie.router "your task" --no-rerank        # ablation: RRF only
    python -m sie.router "your task" --mode dense       # ablation: one retriever only
"""
from __future__ import annotations
import argparse
import hashlib
import random
import sys

from .ingest import load_corpus
from .chunking import chunk_skill
from .index.dense import DenseIndex
from .index.sparse import SparseIndex
from .index.fuse import reciprocal_rank_fusion
from .rerank import Reranker, RerankerUnavailable
from .models import Chunk, Hit

MODES = ("hybrid", "dense", "sparse")
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed python/numpy, and torch if it is already loaded (never imports it)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Stable hash of chunk ids + texts; detects a dense index built from another corpus."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8") + b"\0" + c.text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def dedup_by_skill(hits: list[Hit], k: int) -> list[Hit]:
    """Keep the first (best-scored) hit per skill, up to k."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.skill_slug in seen:
            continue
        seen.add(h.skill_slug)
        out.append(h)
        if len(out) >= k:
            break
    return out


class HybridRouter:
    def __init__(self, skills_dir: str = "data/skills", use_reranker: bool = True,
                 mode: str = "hybrid", persist_dir: str = "data/chroma",
                 strict_rerank: bool = False):
        """
        Args:
            skills_dir: corpus root.
            use_reranker: cross-encode the fused candidates (False == `--no-rerank`).
            mode: "hybrid" (dense + BM25), or "dense" / "sparse" alone for ablations.
            persist_dir: ChromaDB directory.
            strict_rerank: raise if the cross-encoder can't load, instead of warning and
                falling back to RRF order.
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
        self.skills_dir = skills_dir
        self.mode = mode
        self.dense = DenseIndex(persist_dir=persist_dir)
        self.sparse = SparseIndex()
        self.reranker = Reranker() if use_reranker else None
        self.strict_rerank = strict_rerank
        self.rerank_error: str | None = None
        self._chunks: list[Chunk] | None = None
        self._texts: dict[str, str] = {}
        self._dense_gen: int | None = None   # dense.generation last verified fresh
        set_seeds()

    def _load_chunks(self) -> list[Chunk]:
        if self._chunks is None:
            corpus = load_corpus(self.skills_dir)
            if not corpus:
                raise RuntimeError(f"no skills found under {self.skills_dir}")
            chunks = [c for s in corpus for c in chunk_skill(s)]
            texts = {c.chunk_id: c.text for c in chunks}
            if len(texts) != len(chunks):
                raise RuntimeError("chunking produced duplicate chunk ids")
            self._chunks, self._texts = chunks, texts
        return self._chunks

    def build(self) -> int:
        """Rebuild the persisted dense index and the in-memory BM25 index; returns #chunks."""
        chunks = self._load_chunks()
        self.dense.build(chunks, fingerprint=corpus_fingerprint(chunks))
        self.sparse.build(chunks)
        self._dense_gen = self.dense.generation
        return len(chunks)

    def _ensure_ready(self) -> None:
        chunks = self._load_chunks()
        if not self.sparse.is_built:
            self.sparse.build(chunks)
        if self.mode == "sparse" or self._dense_gen == self.dense.generation:
            return
        built = self.dense.fingerprint()
        if built is None:
            raise RuntimeError("dense index not built; run `python -m sie.router --build`")
        if built != corpus_fingerprint(chunks) or self.dense.built_with() != self.dense.embedder:
            raise RuntimeError("dense index is stale (corpus or embedder changed); "
                               "run `python -m sie.router --build`")
        self._dense_gen = self.dense.generation

    def _rankings(self, query: str, pool: int) -> list[list[Hit]]:
        out = []
        if self.mode in ("hybrid", "dense"):
            out.append(self.dense.search(query, k=pool))
        if self.mode in ("hybrid", "sparse"):
            out.append(self.sparse.search(query, k=pool))
        return out

    def _rerank(self, query: str, rankings: list[list[Hit]], top: list[Hit]) -> list[Hit] | None:
        """Cross-encode every retrieved chunk of the top fused skills; None if unavailable."""
        keep = {h.skill_slug for h in top}
        candidates: dict[str, Hit] = {}
        for ranking in rankings:
            for h in ranking:
                if h.skill_slug in keep:
                    candidates.setdefault(h.chunk_id, h)
        try:
            return self.reranker.rerank(query, list(candidates.values()), texts=self._texts)
        except RerankerUnavailable as e:
            if self.strict_rerank:
                raise
            self.rerank_error, self.reranker = str(e), None
            print(f"[router] WARNING: {e}; falling back to RRF order", file=sys.stderr)
            return None

    def retrieve(self, query: str, k: int = 5, pool: int = 20) -> list[Hit]:
        """Return up to k hits, one per skill (best section kept), highest score first.

        Args:
            query: free-text task description.
            k: number of skills to return.
            pool: chunks taken from each retriever, and fused skills passed to the reranker.

        The pool is fixed rather than scaled with k, so retrieve(q, 3) is always a prefix of
        retrieve(q, 10). Fewer than k hits come back when the pool covers fewer skills
        (~16 at pool=20 on this corpus); raise `pool` for exhaustive listings.
        """
        if pool < 1:
            raise ValueError("pool must be >= 1")
        if k <= 0 or not query.strip():
            return []
        self._ensure_ready()
        rankings = self._rankings(query, pool)
        fused = reciprocal_rank_fusion(rankings)
        if self.reranker is not None:
            reranked = self._rerank(query, rankings, fused[:pool])
            if reranked is not None:
                return dedup_by_skill(reranked, k)
        return dedup_by_skill(fused, k)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid skill retrieval: dense + BM25 -> RRF -> rerank.")
    ap.add_argument("query", nargs="?", help="task description to route")
    ap.add_argument("--build", action="store_true", help="(re)build indexes")
    ap.add_argument("--skills", default="data/skills")
    ap.add_argument("-k", type=int, default=5, help="number of skills to return")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (ablation)")
    ap.add_argument("--mode", choices=MODES, default="hybrid", help="retrievers to use")
    ap.add_argument("--pool", type=int, default=20, help="candidate chunks per retriever")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # never crash on a cp1252 console
    r = HybridRouter(skills_dir=args.skills, use_reranker=not args.no_rerank, mode=args.mode)
    try:
        if args.build:
            print(f"[router] indexed {r.build()} chunks -> {r.dense.persist_dir}/")
            if not args.query:
                return
        if not args.query:
            ap.error("provide a query, or use --build")
        hits = r.retrieve(args.query, k=args.k, pool=args.pool)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"[router] {e}")
    print(f"[router] mode={r.mode} ranking={'cross-encoder' if r.reranker else 'rrf'}")
    for i, h in enumerate(hits, 1):
        print(f"{i:2d}. {h.skill_slug:28s} score={h.score:.3f}  [{h.section}]")


if __name__ == "__main__":
    main()
