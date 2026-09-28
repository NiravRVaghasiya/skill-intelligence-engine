"""FastAPI surface: /health, /search, /learning-path, /skill/{slug}.

    uvicorn sie.api:api --reload
Env: SIE_SKILLS_DIR (default data/skills), SIE_RERANK=0 to serve RRF order without a reranker.
"""
from __future__ import annotations
import os
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .router import HybridRouter
from .ingest import load_corpus
from .graph.build import build_graph
from .graph.paths import learning_path

SKILLS_DIR = os.getenv("SIE_SKILLS_DIR", "data/skills")
api = FastAPI(title="Skill Intelligence Engine", version=__version__)
_search_lock = threading.Lock()   # the router lazy-loads its indexes/models on first use


@lru_cache
def _router() -> HybridRouter:
    return HybridRouter(skills_dir=SKILLS_DIR, use_reranker=os.getenv("SIE_RERANK", "1") != "0")


@lru_cache
def _graph():
    return build_graph(load_corpus(SKILLS_DIR))


@api.get("/health")
def health() -> dict:
    """Liveness + corpus size; never loads an embedding model."""
    return {"status": "ok", "version": __version__, "skills": _graph().number_of_nodes()}


@api.get("/search")
def search(q: str = Query(..., min_length=1, description="task description"),
           k: int = Query(5, ge=1, le=50)) -> dict:
    router = _router()
    try:
        with _search_lock:
            hits = router.retrieve(q, k=k)
    except RuntimeError as e:          # index not built / stale
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"query": q, "reranked": router.reranker is not None, "results": [
        {"slug": h.skill_slug, "score": round(h.score, 4), "section": h.section} for h in hits]}


@api.get("/learning-path")
def path(target: str = Query(..., description="skill slug to reach")) -> dict:
    try:
        return learning_path(_graph(), target)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown skill: {target}") from None
    except ValueError as e:            # requires cycle in the corpus
        raise HTTPException(status_code=409, detail=str(e)) from e


@api.get("/skill/{slug}")
def skill(slug: str) -> dict:
    g = _graph()
    if slug not in g:
        raise HTTPException(status_code=404, detail=f"unknown skill: {slug}")
    kinds = lambda edges: sorted(n for n, d in edges if d.get("kind") == "requires")
    return {"slug": slug, **g.nodes[slug],
            "requires": kinds((u, d) for u, _, d in g.in_edges(slug, data=True)),
            "required_by": kinds((v, d) for _, v, d in g.out_edges(slug, data=True))}
