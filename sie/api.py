"""FastAPI surface: /health, /search, /learning-path, /skill/{slug}.

    uvicorn sie.api:api --reload
Env: SIE_SKILLS_DIR / SIE_PERSIST_DIR (default <repo>/data/skills and <repo>/data/chroma,
independent of the cwd), SIE_RERANK=0 to serve RRF order without a reranker.
"""
from __future__ import annotations
import os
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import __version__
from .router import HybridRouter
from .ingest import load_corpus
from .graph.build import build_graph
from .graph.paths import conflicts_of, learning_path, see_also

_ROOT = Path(__file__).resolve().parents[1]          # defaults work from any cwd
SKILLS_DIR = os.getenv("SIE_SKILLS_DIR", str(_ROOT / "data" / "skills"))
PERSIST_DIR = os.getenv("SIE_PERSIST_DIR", str(_ROOT / "data" / "chroma"))
api = FastAPI(title="Skill Intelligence Engine", version=__version__)
_search_lock = threading.Lock()   # the router lazy-loads its indexes/models on first use


@lru_cache
def _router() -> HybridRouter:
    return HybridRouter(skills_dir=SKILLS_DIR, persist_dir=PERSIST_DIR,
                        use_reranker=os.getenv("SIE_RERANK", "1") != "0")


@lru_cache
def _graph_cached():
    skills = load_corpus(SKILLS_DIR)
    if not skills:                 # raising keeps lru_cache from pinning an empty graph
        raise RuntimeError(f"no skills found under {os.path.abspath(SKILLS_DIR)} (set SIE_SKILLS_DIR)")
    return build_graph(skills)


def _graph():
    try:
        return _graph_cached()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@api.get("/health")
def health():
    """Liveness + corpus size; never loads an embedding model. 503 if the corpus is missing."""
    try:
        n = _graph_cached().number_of_nodes()
    except RuntimeError as e:
        return JSONResponse(status_code=503,
                            content={"status": "degraded", "version": __version__, "skills": 0,
                                     "detail": str(e)})
    return {"status": "ok", "version": __version__, "skills": n}


@api.get("/search")
def search(q: str = Query(..., min_length=1, description="task description"),
           k: int = Query(5, ge=1, le=50),
           pool: int = Query(20, ge=1, le=500, description=(
               "candidate chunks per retriever; fewer than k skills come back when the pool "
               "covers fewer (about 10-16 at the default)"))) -> dict:
    router = _router()
    try:
        with _search_lock:
            hits = router.retrieve(q, k=k, pool=pool)
    except RuntimeError as e:          # corpus missing / index not built / stale
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"query": q, "reranked": router.last_reranked, "pool": pool, "results": [
        {"slug": h.skill_slug, "score": round(h.score, 4), "section": h.section} for h in hits]}


@api.get("/learning-path")
def path(target: str = Query(..., description="skill slug to reach")) -> dict:
    g = _graph()
    try:
        return learning_path(g, target)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown skill: {target}") from None
    except ValueError as e:            # requires cycle in the corpus
        raise HTTPException(status_code=409, detail=str(e)) from e


@api.get("/skill/{slug}")
def skill(slug: str) -> dict:
    """Skill metadata; relationships resolved the same way /learning-path resolves them."""
    g = _graph()
    if slug not in g:
        raise HTTPException(status_code=404, detail=f"unknown skill: {slug}")
    kinds = lambda edges: sorted(n for n, d in edges if d.get("kind") == "requires")
    conflicts = conflicts_of(g, slug)
    return {"slug": slug, **g.nodes[slug],
            "related": see_also(g, slug, exclude=conflicts),
            "conflicts": sorted(conflicts),
            "requires": kinds((u, d) for u, _, d in g.in_edges(slug, data=True)),
            "required_by": kinds((v, d) for _, v, d in g.out_edges(slug, data=True)),
            "dangling": sorted([kind, ref] for s, kind, ref in g.graph.get("dangling", []) if s == slug)}
