"""FastAPI service over one `sie.engine.Engine`: routing, composition, learning paths, skills.

    uvicorn sie.api:api                     # pip install -e ".[api]" (+ "retrieval" for dense)

Endpoints: GET /health (liveness), GET /ready (readiness), GET /search (backward-compatible
routing), POST /route (explained routing, optional multi-intent composition), POST
/batch-route, GET /learning-path, GET /skill/{slug}. Response shapes: sie/schemas.py.

Configuration comes from SIE_* environment variables (`Engine.from_env`, see .env.example);
defaults are `<repo>/data/skills` and `<repo>/data/chroma`, independent of the cwd.

Startup: the lifespan builds the engine and, unless SIE_EAGER_INIT is off (0/false/no/off),
runs `engine.start()` in a worker thread (corpus -> graph -> BM25 -> dense check + embedder ->
reranker). A failure never crashes the app: /ready reports it (503 with reasons). An invalid
SIE_EAGER_INIT is logged, the default (eager) applies, and /ready reports it as a reason. With
SIE_EAGER_INIT off, or when the lifespan did not run (e.g. a TestClient used without `with`),
the first /ready call performs that start once; other endpoints load lazily on first use.

Concurrency model: FastAPI runs these sync endpoints in its threadpool. The engine's corpus/
graph loading and the router's index initialization are each guarded by a lock (double-
checked, so warm requests take none), and routing only reads immutable indexes, so requests
run concurrently without a global lock. Warm-up at startup keeps first-use model loading off
the request path.

Errors: corpus missing / index not built or stale / backend unavailable -> 503; invalid
routing parameters -> 422; unknown skill -> 404; a `requires` cycle -> 409. Paths under the
repo root (the default corpus, index and model directories) appear repo-relative in response
bodies (`sie.engine.display_paths`); the server log keeps them in full.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__
from .engine import CycleError, Engine, UnknownSkillError, display_paths, env_bool
from .observability import configure_logging
from .schemas import (BatchRouteRequest, BatchRouteResponse, ErrorResponse, HealthResponse,
                      LearningPathResponse, MAX_QUERY_CHARS, ReadyResponse, RouteRequest,
                      RouteResponse, SearchResponse, SkillDetail, corpus_ref, learning_path_response,
                      ready_response, route_response, search_response, skill_detail)

log = logging.getLogger("sie.api")
T = TypeVar("T")

_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    """The process-wide engine, created from the environment on first use (thread-safe).

    Raises:
        RuntimeError: the SIE_* configuration is invalid (not cached: every call re-reads it).
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    _engine = Engine.from_env()
                except ValueError as e:
                    raise RuntimeError(f"invalid engine configuration: {e}") from e
    return _engine


def _configure_logging() -> None:
    try:
        configure_logging()
    except ValueError as e:                     # a bad SIE_LOG_LEVEL must not stop the service
        configure_logging("WARNING")
        log.warning("SIE_LOG_LEVEL ignored: %s", e)


def _eager_init() -> tuple[bool, str | None]:
    """SIE_EAGER_INIT, parsed like every SIE_* boolean (default on).

    Returns:
        (eager, problem): an invalid value gives the default (True) and says why.
    """
    try:
        return env_bool(os.environ, "SIE_EAGER_INIT", True), None
    except ValueError as e:
        return True, f"invalid configuration: {e}; the default (eager start) applies"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the engine and (unless SIE_EAGER_INIT is off) start it before serving requests."""
    _configure_logging()
    eager, problem = _eager_init()
    if problem:
        log.error("%s", problem)                # /ready reports it too
    try:
        engine = get_engine()
    except RuntimeError as e:
        log.error("engine not created: %s", e)  # every endpoint reports it as a 503
    else:
        if eager:
            status = await run_in_threadpool(engine.start, True)
            if status["ready"]:
                log.info("engine ready (mode=%s)", status["router"]["mode"])
            else:
                log.warning("engine not ready: %s", "; ".join(status["reasons"]))
    yield


api = FastAPI(
    title="Skill Intelligence Engine",
    version=__version__,
    description="Corpus-agnostic retrieval, ranking, explanation and composition over a "
                "SKILL.md skill library. Confidence levels are heuristic and uncalibrated.",
    lifespan=lifespan,
)

_E503 = {"model": ErrorResponse, "description": "corpus missing, index not built/stale, or a "
                                                "retrieval backend is unavailable"}
_E404 = {"model": ErrorResponse, "description": "unknown skill"}
_E409 = {"model": ErrorResponse, "description": "a `requires` cycle blocks ordering"}
_E422 = {"description": "invalid request parameters"}


def _call(fn: Callable[..., T], *args: Any, lookups: bool = False, **kwargs: Any) -> T:
    """Run an engine call, mapping its errors to HTTP status codes.

    Args:
        lookups: the call looks up a skill by slug, so an unknown slug is a 404 (a KeyError
            anywhere else is a bug and stays a 500).
    """
    try:
        return fn(*args, **kwargs)
    except CycleError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except UnknownSkillError as e:
        if not lookups:
            raise
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:          # corpus missing / index not built or stale / no backend
        raise HTTPException(status_code=503, detail=display_paths(str(e))) from e
    except ValueError as e:            # routing parameters
        raise HTTPException(status_code=422, detail=display_paths(str(e))) from e


def _engine_or_503() -> Engine:
    return _call(get_engine)


@api.get("/health", response_model=HealthResponse, response_model_exclude_none=True,
         summary="Liveness", responses={503: {"model": HealthResponse,
                                              "description": "corpus missing (status degraded)"}})
def health() -> Any:
    """Liveness plus corpus identity. Loads the corpus metadata if needed, never a model.

    `status` is "degraded" (still 200) when the cross-encoder failed to load and routing fell
    back to RRF order; 503 with status "degraded" when the corpus can't be loaded.
    """
    try:
        engine = get_engine()
        engine.load()
    except RuntimeError as e:
        body = HealthResponse(status="degraded", version=__version__, skills=0,
                              detail=display_paths(str(e)))
        return JSONResponse(status_code=503, content=body.model_dump(exclude_none=True))
    corpus = engine.corpus
    return HealthResponse(status="degraded" if engine.degraded else "ok", version=__version__,
                          skills=corpus.n_skills if corpus else 0, corpus=corpus_ref(corpus))


@api.get("/ready", response_model=ReadyResponse, summary="Readiness",
         responses={503: {"model": ReadyResponse, "description": "not ready; `reasons` says why"}})
def ready() -> Any:
    """200 with the engine status when ready to serve, else 503 with the same body.

    Never loads a model itself, except once: if the engine was never started (SIE_EAGER_INIT
    off, or no lifespan), this call runs `engine.start(eager=True)` and reports its outcome. An
    invalid SIE_EAGER_INIT is a reason (not ready), though the default was applied.
    """
    try:
        engine = get_engine()
    except RuntimeError as e:
        detail = display_paths(str(e))
        body = ReadyResponse(ready=False, reasons=[detail], errors=[detail], version=__version__)
        return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
    data = engine.ensure_started(eager=True)
    _, problem = _eager_init()
    if problem:
        data = {**data, "ready": False, "reasons": [*data["reasons"], problem]}
    status = ready_response(data)
    if not status.ready:
        return JSONResponse(status_code=503, content=status.model_dump(mode="json"))
    return status


@api.get("/search", response_model=SearchResponse, summary="Route a query (compact, v0.1 shape)",
         responses={503: _E503, 422: _E422})
def search(q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS, description="task description"),
           k: int | None = Query(None, ge=1, le=50,
                                 description="skills to return (default: engine top_k, i.e. 5)"),
           pool: int | None = Query(None, ge=1, le=500, description=(
               "candidate chunks per retriever (default: engine config, i.e. 20); fewer than k "
               "skills come back when the pool covers fewer")),
           rerank_k: int | None = Query(None, ge=1, le=500,
                                        description="top fused skills cross-encoded (default: pool)"),
           ) -> Any:
    """Ranked skills for a task description: slug, score, section, rank, name, methods, plus a
    brief confidence. Use POST /route for evidence, alternatives and provenance."""
    engine = _engine_or_503()
    result = _call(engine.route, q, k=k, pool=pool, rerank_k=rerank_k)
    return search_response(engine, result, pool=pool if pool is not None else engine.config.candidate_pool)


@api.post("/route", response_model=RouteResponse, summary="Route a query, explained",
          responses={503: _E503, 409: _E409, 422: _E422})
def route(req: RouteRequest) -> Any:
    """Rank skills for a query with per-method evidence, a heuristic confidence (route /
    clarify / abstain), alternatives and provenance. With `multi_intent`, the query is also
    split into intents and composed into a dependency-ordered plan (`composition`); `results`
    and `confidence` still describe the whole query routed as one."""
    engine = _engine_or_503()
    result = _call(engine.route, req.query, k=req.k, pool=req.pool, rerank_k=req.rerank_k)
    composition = None
    if req.multi_intent:
        kw = {"k": req.k} if req.k is not None else {}
        composition = _call(engine.compose, req.query, max_intents=req.max_intents,
                            pool=req.pool, rerank_k=req.rerank_k, **kw)
    return route_response(engine, result, explain=req.explain, composition=composition)


@api.post("/batch-route", response_model=BatchRouteResponse, summary="Route up to 32 queries",
          responses={503: _E503, 422: _E422})
def batch_route(req: BatchRouteRequest) -> Any:
    """Route each query in order (sequentially: deterministic, one worker thread)."""
    engine = _engine_or_503()
    results = _call(engine.batch, req.queries, k=req.k, pool=req.pool, rerank_k=req.rerank_k)
    return BatchRouteResponse(results=[route_response(engine, r, explain=req.explain) for r in results])


@api.get("/learning-path", response_model=LearningPathResponse, summary="Learning path to a skill",
         responses={404: _E404, 409: _E409, 503: _E503})
def path(target: str = Query(..., description="skill slug to reach"),
         include_recommended: bool = Query(False, description=(
             "also insert soft prerequisites (recommended_before), without their own prerequisites"))
         ) -> Any:
    """Dependency-ordered path ending at `target`, with the reason for every step, see-also,
    conflicts, missing prerequisites and notes on declaration gaps."""
    engine = _engine_or_503()
    lp = _call(engine.learning_path, target, include_recommended=include_recommended, lookups=True)
    return learning_path_response(lp)


@api.get("/skill/{slug}", response_model=SkillDetail, summary="Skill metadata and relationships",
         responses={404: _E404, 503: _E503})
def skill(slug: str = PathParam(..., description="skill slug")) -> Any:
    """Metadata, provenance and typed relationships; resolved the same way /learning-path
    resolves them."""
    engine = _engine_or_503()
    return skill_detail(_call(engine.skill, slug, lookups=True))
