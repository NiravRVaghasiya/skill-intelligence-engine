"""FastAPI surface: /search, /learning-path, /skill/{slug}."""
from __future__ import annotations
from functools import lru_cache

from fastapi import FastAPI, Query

from .router import HybridRouter
from .ingest import load_corpus
from .graph.build import build_graph
from .graph.paths import learning_path

api = FastAPI(title="Skill Intelligence Engine", version="0.1.0")


@lru_cache
def _router() -> HybridRouter:
    return HybridRouter()


@lru_cache
def _graph():
    return build_graph(load_corpus("data/skills"))


@api.get("/search")
def search(q: str = Query(..., description="task description"), k: int = 5):
    hits = _router().retrieve(q, k=k)
    return {"query": q, "results": [
        {"slug": h.skill_slug, "score": round(h.score, 4), "section": h.section} for h in hits]}


@api.get("/learning-path")
def path(target: str = Query(..., description="skill slug to reach")):
    return learning_path(_graph(), target)


@api.get("/skill/{slug}")
def skill(slug: str):
    g = _graph()
    if slug not in g:
        return {"error": f"unknown skill: {slug}"}
    return {"slug": slug, **g.nodes[slug],
            "requires": [u for u, v, d in g.in_edges(slug, data=True) if d.get("kind") == "requires"]}
