"""One object that owns a corpus, its skill graph and the router, for services and scripts.

    startup -> load corpus metadata -> build graph -> BM25 -> dense index check/warm-up
            -> optional reranker -> ready

`Engine.load()` reads the corpus once (through the router, so it is parsed a single time) and
builds the typed skill graph, applying any edge overlays. `Engine.start()` also warms the
indexes and models, recording failures instead of raising so a service can come up and report
*why* it is not ready (`status()`). Every query method (`route`, `batch`, `compose`,
`learning_path`, `skill`) loads lazily, so a script can skip `start()` entirely.

    from sie.engine import Engine
    engine = Engine.from_env()            # SIE_* variables; see .env.example
    engine.start()                        # optional: warm up at service start
    engine.route("evaluate my RAG answers").top.slug

Concurrency: `load()` and `start()` are guarded by locks (double-checked, so the fast path
takes no lock), the router guards its own lazy index initialization, and routing only reads
immutable indexes. One Engine can therefore serve concurrent callers.

A failed `start()` is not retried automatically: fix the corpus / index and call `start()`
again (or restart the service). A failed `load()` is never cached, so query methods retry.
"""
from __future__ import annotations

import dataclasses
import difflib
import functools
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .compose import compose as _compose
from .confidence import ConfidencePolicy
from .graph.build import build_graph, find_cycles, load_overlay, relations, requires_graph
from .graph.paths import conflicts_of, learning_path as _learning_path, render_learning_path, see_also
from .index.dense import EMBEDDERS
from .models import RELATIONSHIPS, Composition, CorpusInfo, Edge, RouteResult, Skill
from .observability import emit
from .rerank import LOCAL_CE_DIR
from .router import MODES, HybridRouter, RetrievalConfig

if TYPE_CHECKING:                                       # annotations only; imported lazily
    import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = REPO_ROOT / "data" / "skills"      # defaults work from any cwd
DEFAULT_PERSIST_DIR = REPO_ROOT / "data" / "chroma"
_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")


class UnknownSkillError(KeyError):
    """A slug that is not in the corpus; the message suggests close matches."""

    def __str__(self) -> str:                   # KeyError would repr() the message
        return str(self.args[0]) if self.args else "unknown skill"


class CycleError(ValueError):
    """A `requires` cycle blocks ordering (a learning path or a composed plan)."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _missing_dependency(error: ImportError) -> RuntimeError:
    """An ImportError raised while routing, as a service-level error with an install hint."""
    if (error.name or "").split(".")[0] == "chromadb":
        hint = "install the 'retrieval' extra, or use mode 'sparse' (SIE_MODE=sparse)"
    else:
        hint = "install the missing package (see pyproject.toml extras)"
    return RuntimeError(f"missing dependency: {error}; {hint}")


# -- paths in messages that leave the server --------------------------------------------------

# One path component: no separators, quotes, whitespace or list punctuation, no trailing dot
# (so "under <root>/data/skills." keeps the sentence's full stop outside the path).
_PART = r"[^\s\\/'\"`<>|:;,()\[\]{}*?]*[^\s\\/'\"`<>|:;,()\[\]{}*?.]"
_SEP = r"[\\/]{1,2}"                     # "\" or "/", or "\\" as repr() writes it in OSErrors


@functools.lru_cache(maxsize=8)
def _root_pattern(root: str) -> re.Pattern[str] | None:
    """Matches `root`, plus any components below it, as an absolute path inside free text."""
    parts = [re.escape(p) for p in re.split(r"[\\/]+", root) if p]
    if not parts:                        # a filesystem root: nothing sensible to rewrite
        return None
    lead = len(root) - len(root.lstrip("\\/"))
    head = rf"[\\/]{{{lead},{2 * lead}}}" if lead else ""
    return re.compile(rf"(?<![\w.~-]){head}{_SEP.join(parts)}(?![\w-]|\.[\w-])"
                      rf"((?:{_SEP}{_PART})*)", re.IGNORECASE if os.name == "nt" else 0)


def display_paths(text: str, root: str | Path | None = None) -> str:
    """`text` with every absolute path under the repo root rewritten repo-relative (posix).

    For messages that leave the server (API bodies): the default corpus / index / model
    directories are absolute paths under the source checkout, which would otherwise put a
    machine path into responses and reports. Paths outside the root are left as given.

    Args:
        text: a message, e.g. "no skills found under /srv/sie/data/skills".
        root: the root to strip (default: the repo root).

    Returns:
        e.g. "no skills found under data/skills" ("." for the root itself).
    """
    pattern = _root_pattern(str(REPO_ROOT if root is None else root))
    if pattern is None or not text:
        return text
    return pattern.sub(lambda m: "/".join(re.split(r"[\\/]+", m.group(1))[1:]) or ".", text)


# -- environment parsing ----------------------------------------------------------------------

def _env_text(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    """A boolean SIE_* variable: 1/0, true/false, yes/no, on/off (any case); unset/empty = default.

    Raises:
        ValueError: any other value; the message names the variable.
    """
    raw = _env_text(env, name).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{name} must be 1 or 0 (or true/false, yes/no, on/off), got {env.get(name)!r}")


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    raw = _env_text(env, name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {env.get(name)!r}")
    return value


def _default_reranker() -> str | None:
    """The repo's local cross-encoder weights (an absolute, cwd-independent path), if present."""
    return str(LOCAL_CE_DIR) if LOCAL_CE_DIR.is_dir() else None


def _cycles(graph: nx.DiGraph) -> list[list[str]]:
    """`find_cycles`, with a linear-time acyclicity check first (enumerating cycles is costly)."""
    import networkx as nx
    if nx.is_directed_acyclic_graph(requires_graph(graph)):
        return []
    return find_cycles(graph)


# -- serialization helpers --------------------------------------------------------------------

def edge_dict(edge: Edge) -> dict[str, str]:
    """An Edge as plain data."""
    return {"source": edge.source, "target": edge.target, "relationship": edge.relationship,
            "confidence": edge.confidence, "provenance": edge.provenance}


def provenance_dict(skill: Skill) -> dict[str, str]:
    """Where a skill came from: corpus, version, path, content hash, dates."""
    return {"source": skill.source, "source_version": skill.source_version, "path": skill.path,
            "content_hash": skill.content_hash, "updated_at": skill.updated_at,
            "version": skill.version, "provenance": skill.provenance}


def composition_dict(c: Composition) -> dict[str, Any]:
    """A Composition as JSON-ready data, without internal chunk ids."""
    data = dataclasses.asdict(c)
    for intent in data["intents"]:
        for item in intent["result"]["results"]:
            item.pop("chunk_id", None)
    data["multi_intent"] = c.multi_intent
    return data


class _PinnedRouter:
    """`route(query, k)` for `compose`, with a fixed pool / rerank_k passed through."""

    def __init__(self, router: HybridRouter, pool: int | None, rerank_k: int | None):
        self._router, self._pool, self._rerank_k = router, pool, rerank_k

    def route(self, query: str, k: int | None = None) -> RouteResult:
        return self._router.route(query, k=k, pool=self._pool, rerank_k=self._rerank_k)


class Engine:
    """Corpus + skill graph + router behind one explicit lifecycle (see the module docstring)."""

    def __init__(self, skills_dir: str | Path = "data/skills", persist_dir: str | Path = "data/chroma",
                 mode: str = "hybrid", use_reranker: bool = True, reranker_model: str | None = None,
                 strict_rerank: bool = False, manifest: str | Path | None = None,
                 overlays: Sequence[str | Path] = (), config: RetrievalConfig | None = None,
                 policy: ConfidencePolicy | None = None):
        """
        Args:
            skills_dir: corpus root (a directory of SKILL.md files, optionally with corpus.toml).
            persist_dir: ChromaDB directory of the dense index.
            mode: "hybrid", "dense" or "sparse" (BM25 only: no index or model needed).
            use_reranker: cross-encode the fused candidates.
            reranker_model: cross-encoder id or local dir (default `rerank.default_model()`).
            strict_rerank: fail (and report not ready) instead of falling back to RRF order
                when the cross-encoder can't load.
            manifest: explicit corpus.toml (default `<skills_dir>/corpus.toml` if present).
            overlays: edge-overlay JSON files (`sie.graph.load_overlay`) applied to the graph.
            config: retrieval knobs (default `RetrievalConfig()`).
            policy: routing-confidence thresholds (default `confidence.DEFAULT_POLICY`).

        Raises:
            ValueError: unknown mode or invalid config.
        """
        if isinstance(overlays, (str, Path)):          # a lone path, not a sequence of them
            overlays = (overlays,)
        self.overlays: tuple[str, ...] = tuple(str(p) for p in overlays)
        self.strict_rerank = strict_rerank
        self.router = HybridRouter(skills_dir=str(skills_dir), use_reranker=use_reranker, mode=mode,
                                   persist_dir=str(persist_dir), strict_rerank=strict_rerank,
                                   config=config, policy=policy, reranker_model=reranker_model,
                                   manifest=None if manifest is None else str(manifest))
        self.errors: list[str] = []          # hard failures of the latest start()
        self.started_at: str | None = None   # UTC time of the latest start(); None = never
        self._graph: nx.DiGraph | None = None
        self._cycles: list[list[str]] = []
        self._load_lock = threading.Lock()
        self._start_lock = threading.Lock()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Engine:
        """Build an Engine from SIE_* variables (documented in .env.example).

        SIE_SKILLS_DIR / SIE_PERSIST_DIR default to `<repo>/data/skills` and `<repo>/data/chroma`
        (independent of the cwd; these defaults assume a source checkout, so set both when the
        package is installed elsewhere); SIE_CORPUS_MANIFEST; SIE_MODE (hybrid|dense|sparse, default
        hybrid); SIE_RERANK (0 disables, default 1); SIE_RERANKER (model id/path; default the
        repo's `data/models/ms-marco-MiniLM-L-6-v2` if present); SIE_STRICT_RERANK (1 enables);
        SIE_EDGE_OVERLAYS (os.pathsep-separated paths); SIE_TOP_K, SIE_POOL, SIE_RERANK_K
        (integers >= 1); SIE_EMBEDDER (onnx|sentence-transformers). Booleans are parsed by
        `env_bool` (1/0, true/false, yes/no, on/off).

        Args:
            environ: variables to read (default `os.environ`).

        Raises:
            ValueError: an invalid value; the message names the variable.
        """
        env = os.environ if environ is None else environ
        mode = _env_text(env, "SIE_MODE").lower() or "hybrid"
        if mode not in MODES:
            raise ValueError(f"SIE_MODE must be one of {', '.join(MODES)}, got {env.get('SIE_MODE')!r}")
        embedder = _env_text(env, "SIE_EMBEDDER")
        if embedder and embedder not in EMBEDDERS:
            raise ValueError(f"SIE_EMBEDDER must be one of {', '.join(EMBEDDERS)}, got {embedder!r}")
        knobs = {"top_k": _env_int(env, "SIE_TOP_K"), "candidate_pool": _env_int(env, "SIE_POOL"),
                 "rerank_k": _env_int(env, "SIE_RERANK_K")}
        overlays = tuple(p.strip() for p in _env_text(env, "SIE_EDGE_OVERLAYS").split(os.pathsep)
                         if p.strip())
        engine = cls(skills_dir=_env_text(env, "SIE_SKILLS_DIR") or DEFAULT_SKILLS_DIR,
                     persist_dir=_env_text(env, "SIE_PERSIST_DIR") or DEFAULT_PERSIST_DIR,
                     mode=mode,
                     use_reranker=env_bool(env, "SIE_RERANK", True),
                     reranker_model=_env_text(env, "SIE_RERANKER") or _default_reranker(),
                     strict_rerank=env_bool(env, "SIE_STRICT_RERANK", False),
                     manifest=_env_text(env, "SIE_CORPUS_MANIFEST") or None,
                     overlays=overlays,
                     config=RetrievalConfig(**{k: v for k, v in knobs.items() if v is not None}))
        if embedder:                         # the dense index reads os.environ; honor `environ`
            engine.router.dense.embedder = embedder
        return engine

    # -- lifecycle ----------------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.router.mode

    @property
    def config(self) -> RetrievalConfig:
        return self.router.config

    @property
    def corpus(self) -> CorpusInfo | None:
        """Identity of the loaded corpus; None until `load()` succeeded."""
        return self.router.corpus if self._graph is not None else None

    @property
    def loaded(self) -> bool:
        return self._graph is not None

    @property
    def graph(self) -> nx.DiGraph:
        """The typed skill graph (loads the corpus on first use)."""
        self.load()
        assert self._graph is not None
        return self._graph

    def load(self) -> None:
        """Load the corpus (once, through the router) and build the skill graph with overlays.

        Idempotent and thread-safe; a failure is not cached, so the next call retries.

        Raises:
            RuntimeError: the corpus is missing, empty or its manifest invalid, or an edge
                overlay can't be read.
        """
        if self._graph is not None:
            return
        with self._load_lock:
            if self._graph is not None:
                return
            self.router._load_chunks()               # parses the corpus once; raises RuntimeError
            edges = self._overlay_edges()
            graph = build_graph(list(self.router.skills.values()), overlays=edges)
            self._cycles = _cycles(graph)
            self._graph = graph                      # published last: the fast path keys on it

    def _overlay_edges(self) -> list[Edge]:
        edges: list[Edge] = []
        for path in self.overlays:
            try:
                edges += load_overlay(path)
            except (OSError, ValueError) as e:
                raise RuntimeError(f"cannot load edge overlay {path}: {e}") from e
        return edges

    def start(self, eager: bool = True) -> dict[str, Any]:
        """Load the corpus and graph, then (if eager) warm BM25, the dense index and models.

        Never raises for a load/warm-up failure: it is recorded in `errors` (which this call
        resets) and reported by `status()`. Stamps `started_at`.

        Args:
            eager: also run `router.warm_up()` (BM25, dense freshness + embedder, reranker).

        Returns:
            `status()`.
        """
        with self._start_lock:
            return self._start_locked(eager)

    def ensure_started(self, eager: bool = True) -> dict[str, Any]:
        """`start(eager)` if this engine was never started, else `status()`; thread-safe."""
        with self._start_lock:
            if self.started_at is None:
                return self._start_locked(eager)
        return self.status()

    def _start_locked(self, eager: bool) -> dict[str, Any]:
        began = time.perf_counter()
        errors: list[str] = []
        try:
            self.load()
        except Exception as e:
            errors.append(f"load failed: {_describe(e)}")
        else:
            if eager:
                try:
                    self.router.warm_up()
                except ImportError as e:
                    errors.append(f"warm-up failed: {_describe(_missing_dependency(e))}")
                except Exception as e:
                    errors.append(f"warm-up failed: {_describe(e)}")
        self.errors = errors
        self.started_at = _utc_now()
        status = self.status()
        corpus = status["corpus"] or {}
        emit("engine_start", ready=status["ready"], degraded=status["degraded"],
             reasons=list(status["reasons"]), mode=self.mode, eager=eager,
             corpus=corpus.get("name"), corpus_version=corpus.get("version"),
             fingerprint=corpus.get("fingerprint"),
             duration_ms=round((time.perf_counter() - began) * 1000.0, 3))
        return status

    @property
    def degraded(self) -> bool:
        """The cross-encoder failed to load and routing fell back to RRF order (non-strict)."""
        return self.router.rerank_error is not None and not self.strict_rerank

    def _revalidate(self) -> str | None:
        """Re-check dense freshness if the index handle was rebound since it was verified.

        An external rebuild (`python -m sie.router --build`) makes the next dense search rebind
        its handle; without this, readiness would stay "not verified" until another routing
        request arrived. Cheap: reads the index metadata, loads no model and no corpus.

        Returns:
            Why the index is not usable (e.g. "dense index is stale ..."), or None.
        """
        revalidate = getattr(self.router, "revalidate", None)
        if not callable(revalidate):
            return None
        try:
            revalidate()
        except ImportError as e:
            return str(_missing_dependency(e))
        except RuntimeError as e:
            return str(e)
        except Exception as e:               # status() must never raise
            return f"dense index check failed: {_describe(e)}"
        return None

    def status(self) -> dict[str, Any]:
        """Readiness and what is loaded; never loads the corpus or a model.

        Ready iff the corpus and graph are loaded, BM25 is built, the dense index was verified
        fresh (dense modes), a strict reranker is loaded, and the latest `start()` recorded no
        error. A reranker that failed to load and fell back to RRF is not a readiness failure
        (unless strict); it sets `degraded` and shows under `router.reranker.error`. A dense
        index rebound since it was verified (rebuilt externally) is re-verified here
        (`HybridRouter.revalidate()`: metadata only).

        Returns:
            {"ready", "degraded", "reasons", "errors", "version", "started_at", "corpus"
            (CorpusInfo fields except `root`; `manifest` is the file name only), "graph"
            ({"nodes", "edges_by_relationship", "overlays", "cycles", "dangling"}), "router"
            (`HybridRouter.status()`)}. "corpus" and "graph" are None until loaded (the corpus
            can be loaded while the graph is not, e.g. when an overlay is invalid).
        """
        dense_problem = self._revalidate()
        router = self.router.status()
        graph, corpus = self._graph, self.router.corpus
        reasons = list(self.errors)
        if corpus is None:
            reasons.append("corpus not loaded")
        elif graph is None:
            reasons.append("skill graph not built")
        if not router["sparse"]:
            reasons.append("BM25 index not built")
        if router["dense"]["required"] and not router["dense"]["ready"]:
            reasons.append(dense_problem or "dense index not verified (missing, stale, or not warmed up)")
        reranker = router["reranker"]
        if self.strict_rerank and reranker["enabled"] and not reranker["loaded"]:
            reasons.append("cross-encoder required (strict_rerank) but not loaded")
        return {
            "ready": not reasons,
            "degraded": self.degraded,
            "reasons": reasons,
            "errors": list(self.errors),
            "version": __version__,
            "started_at": self.started_at,
            "corpus": self._corpus_status(corpus) if corpus is not None else None,
            "graph": self._graph_status(graph) if graph is not None else None,
            "router": router,
        }

    @staticmethod
    def _corpus_status(corpus: CorpusInfo) -> dict[str, Any]:
        info = dataclasses.asdict(corpus)
        info.pop("root", None)                       # no server filesystem paths in status
        info["manifest"] = Path(info["manifest"]).name if info.get("manifest") else ""
        return info

    def _graph_status(self, graph: nx.DiGraph) -> dict[str, Any]:
        counts = Counter(e.relationship for e in graph.graph.get("edges", []))
        return {"nodes": graph.number_of_nodes(),
                "edges_by_relationship": {r: counts.get(r, 0) for r in RELATIONSHIPS},
                "overlays": list(graph.graph.get("overlays", [])),
                "cycles": [list(c) for c in self._cycles],
                "dangling": len(graph.graph.get("dangling", []))}

    # -- queries ------------------------------------------------------------------------------

    def route(self, query: str, k: int | None = None, pool: int | None = None,
              rerank_k: int | None = None) -> RouteResult:
        """Route one query (`HybridRouter.route`); None arguments use the engine's config.

        Raises:
            ValueError: pool / rerank_k < 1.
            RuntimeError: corpus missing, dense index missing/stale or its backend not
                installed, or (strict) no reranker.
        """
        self.load()
        try:
            return self.router.route(query, k=k, pool=pool, rerank_k=rerank_k)
        except ImportError as e:
            raise _missing_dependency(e) from e

    def batch(self, queries: Sequence[str], k: int | None = None, pool: int | None = None,
              rerank_k: int | None = None) -> list[RouteResult]:
        """Route each query in order (sequential, deterministic); same arguments as `route`."""
        if isinstance(queries, str):
            raise TypeError("batch() takes a sequence of queries, not one string")
        return [self.route(q, k=k, pool=pool, rerank_k=rerank_k) for q in queries]

    def compose(self, query: str, k: int = 3, max_intents: int = 5,
                include_prerequisites: bool = True, pool: int | None = None,
                rerank_k: int | None = None) -> Composition:
        """Split a multi-part request into intents and plan the skills (`sie.compose.compose`).

        Args:
            query: free text, possibly several tasks.
            k: skills each intent's routing returns (only the top one is selected).
            max_intents: route at most this many intents.
            include_prerequisites: add the `requires` closure of every selected skill.
            pool, rerank_k: per-intent routing knobs (default: the engine's config).

        Raises:
            ValueError: k / max_intents / pool / rerank_k < 1.
            CycleError: a `requires` cycle among the plan's skills.
            RuntimeError: as in `route`.
        """
        graph = self.graph
        try:
            return _compose(_PinnedRouter(self.router, pool, rerank_k), graph, query, k=k,
                            max_intents=max_intents, include_prerequisites=include_prerequisites)
        except ImportError as e:
            raise _missing_dependency(e) from e
        except ValueError as e:
            if str(e).startswith("requires cycle"):   # sie.graph.paths.order_skills' message
                raise CycleError(str(e)) from e
            raise

    def _unknown(self, slug: str) -> UnknownSkillError:
        close = difflib.get_close_matches(slug, list(self.graph.nodes), n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        return UnknownSkillError(f"unknown skill '{slug}'{hint}")

    def learning_path(self, target: str, include_recommended: bool = False) -> dict[str, Any]:
        """Explained, dependency-ordered path to `target` (`sie.graph.paths.learning_path`).

        Raises:
            UnknownSkillError (a KeyError): unknown target. CycleError (a ValueError): a
                requires cycle blocks ordering. RuntimeError: corpus missing.
        """
        graph = self.graph
        if target not in graph:
            raise self._unknown(target)
        try:
            return _learning_path(graph, target, include_recommended=include_recommended)
        except ValueError as e:
            raise CycleError(str(e)) from e

    def render_learning_path(self, target: str, include_recommended: bool = False) -> str:
        """The `python -m sie.router --path` text for `target` on this engine's graph."""
        return render_learning_path(self.graph, self.learning_path(target, include_recommended))

    def skill_obj(self, slug: str) -> Skill:
        """The parsed Skill.

        Raises:
            UnknownSkillError (a KeyError): unknown slug. RuntimeError: corpus missing.
        """
        self.load()
        skill = self.router.skills.get(slug)
        if skill is None:
            raise self._unknown(slug)
        return skill

    def prerequisites(self, slug: str) -> list[str]:
        """Direct `requires` of `slug` in the graph (declarations + applied overlays), sorted.

        [] for a slug outside the graph.
        """
        graph = self.graph
        if slug not in graph:
            return []
        return sorted(u for u, _, d in graph.in_edges(slug, data=True) if d.get("kind") == "requires")

    def skill(self, slug: str) -> dict[str, Any]:
        """Skill metadata, provenance and relationships, resolved like `learning_path` does.

        Returns:
            {"slug", "skill_id", "name", "domain", "level", "skill_type", "display_name",
            "description", "capabilities", "related" (in-corpus see-also minus conflicts),
            "conflicts" (either direction), "requires", "required_by", "dangling"
            ([relationship, ref] not in the corpus), "declared", "provenance", "relations"
            (every typed edge touching the skill), "metadata" (other frontmatter keys)}.

        Raises:
            UnknownSkillError (a KeyError): unknown slug. RuntimeError: corpus missing.
        """
        graph = self.graph
        if slug not in graph:
            raise self._unknown(slug)
        node = graph.nodes[slug]
        skill = self.router.skills.get(slug)
        conflicts = conflicts_of(graph, slug)
        required_by = sorted(v for _, v, d in graph.out_edges(slug, data=True)
                             if d.get("kind") == "requires")
        return {
            "slug": slug, "skill_id": slug, "name": node.get("name") or slug,
            "domain": node["domain"], "level": node["level"], "skill_type": node["skill_type"],
            "display_name": node["display_name"], "description": node["description"],
            "capabilities": list(node["capabilities"]),
            "related": see_also(graph, slug, exclude=conflicts),
            "conflicts": sorted(conflicts),
            "requires": self.prerequisites(slug),
            "required_by": required_by,
            "dangling": sorted([rel, ref] for s, rel, ref in graph.graph.get("dangling", [])
                               if s == slug),
            "declared": list(node.get("declared") or []),
            "provenance": provenance_dict(skill) if skill is not None else None,
            "relations": [edge_dict(e) for e in relations(graph, slug)],
            "metadata": dict(skill.metadata) if skill is not None else {},
        }
