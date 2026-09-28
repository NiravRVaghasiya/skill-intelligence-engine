"""Top-level hybrid retrieval: dense + sparse -> RRF -> rerank -> dedup by skill.

Usage:
    python -m sie.router --build                  # build indexes from data/skills
    python -m sie.router "your task description"   # query
"""
from __future__ import annotations
import argparse

from .ingest import load_corpus
from .chunking import chunk_skill
from .index.dense import DenseIndex
from .index.sparse import SparseIndex
from .index.fuse import reciprocal_rank_fusion
from .rerank import Reranker
from .models import Hit


class HybridRouter:
    def __init__(self, skills_dir: str = "data/skills", use_reranker: bool = True):
        self.skills_dir = skills_dir
        self.dense = DenseIndex()
        self.sparse = SparseIndex()
        self.reranker = Reranker() if use_reranker else None

    def build(self) -> int:
        corpus = load_corpus(self.skills_dir)
        chunks = [c for s in corpus for c in chunk_skill(s)]
        self.dense.build(chunks)
        self.sparse.build(chunks)
        return len(chunks)

    def retrieve(self, query: str, k: int = 5, pool: int = 20) -> list[Hit]:
        dense_hits = self.dense.search(query, k=pool)
        sparse_hits = self.sparse.search(query, k=pool)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
        if self.reranker:
            fused = self.reranker.rerank(query, fused[:pool], top_k=k)
        seen, out = set(), []
        for h in fused:
            if h.skill_slug in seen:
                continue
            seen.add(h.skill_slug)
            out.append(h)
            if len(out) >= k:
                break
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="task description to route")
    ap.add_argument("--build", action="store_true", help="(re)build indexes")
    ap.add_argument("--skills", default="data/skills")
    args = ap.parse_args()
    r = HybridRouter(skills_dir=args.skills)
    if args.build:
        n = r.build()
        print(f"[router] indexed {n} chunks")
        return
    if not args.query:
        ap.error("provide a query, or use --build")
    for i, h in enumerate(r.retrieve(args.query), 1):
        print(f"{i:2d}. {h.skill_slug:28s} score={h.score:.3f}  [{h.section}]")


if __name__ == "__main__":
    main()
