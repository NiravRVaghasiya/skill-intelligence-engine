"""Recorded dense rankings: hybrid regression checks with no embedding model and no index.

`python -m eval.run_eval --record-fixture` asks the persisted dense index (never rebuilt for
this) for the top `DEPTH` chunks of every evaluation query, and of every text `compose` may
route for the multi-intent sets (`fixture_queries`), and stores them in
`tests/fixtures/dense_rankings.json`:

    {"corpus_fingerprint": <chunk-level fingerprint the index was built from>,
     "embedder": ..., "model": ..., "pool": <benchmark pool>, "depth": 50,
     "queries": {query: [[chunk_id, cosine rounded to 6 decimals], ...]}}

Each list is in exactly the order `DenseIndex.search` returned it. `ReplayDense` serves those
rankings through the dense-index interface the router uses, so `HybridRouter` over the real
corpus BM25 + the replayed dense side reproduces the live hybrid ranking (RRF only reads
ranks; confidence reads the cosines) — the basis of `run_eval --smoke` and
tests/test_regression.py. A different corpus makes the router reject the fixture as stale.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import sie.compose as _compose
from sie.compose import split_intents
from sie.models import Chunk, Hit

FIXTURE = Path("tests/fixtures/dense_rankings.json")
DEPTH = 50                     # chunks stored per query: replays any pool up to 50
SCORE_DECIMALS = 6
SNIPPET_CHARS = 200            # DenseIndex.search's snippet length
RERECORD = "re-record: python -m eval.run_eval --record-fixture"
_HEADER = ("corpus_fingerprint", "depth", "embedder", "model", "pool")


def fixture_queries(datasets: Iterable, max_intents: int = 5) -> list[str]:
    """Every query the dense side sees when the harness scores `datasets`, sorted.

    The query of every row, plus, for multi-intent sets, each routed sub-query
    `split_intents` yields and every text `sie.compose.routed_texts` says `compose` may route
    (the verbatim single-intent query, resolved and as-written parts), when that exists.

    Args:
        max_intents: `compose`'s max_intents in the harness (bounds `routed_texts`).
    """
    routed_texts = getattr(_compose, "routed_texts", None)
    queries: set[str] = set()
    for ds in datasets:
        for row in ds.rows:
            queries.add(row["query"])
            if ds.kind == "multi_intent":
                queries.update(routed for _, routed in split_intents(row["query"]))
                if callable(routed_texts):
                    queries.update(routed_texts(row["query"], max_intents=max_intents))
    return sorted(queries)


def record(dense, queries: Iterable[str], corpus_fingerprint: str, pool: int,
           depth: int = DEPTH) -> dict:
    """Rankings of a live dense index for `queries`, in its own order (never re-sorted).

    Args:
        dense: a built `DenseIndex` (or anything with search/embedder/model_name).
        queries: query strings.
        corpus_fingerprint: chunk-level fingerprint of the corpus the index was built from.
        pool: the benchmark candidate pool (recorded for reference).
        depth: chunks stored per query.
    """
    rankings = {q: [[h.chunk_id, round(float(h.score), SCORE_DECIMALS)]
                    for h in dense.search(q, k=depth)] for q in sorted(set(queries))}
    return {"corpus_fingerprint": corpus_fingerprint, "depth": depth,
            "embedder": dense.embedder, "model": dense.model_name, "pool": pool,
            "queries": rankings}


def dumps(fixture: dict) -> str:
    """Compact, deterministic JSON: header keys one per line, then one line per query."""
    lines = ["{"]
    lines += [f"{json.dumps(key)}: {json.dumps(fixture[key])}," for key in _HEADER]
    lines.append('"queries": {')
    items = sorted(fixture["queries"].items())
    lines += [f"{json.dumps(q, ensure_ascii=False)}: "
              f"{json.dumps(r, ensure_ascii=False, separators=(',', ':'))}"
              + ("," if i < len(items) - 1 else "") for i, (q, r) in enumerate(items)]
    lines += ["}", "}"]
    return "\n".join(lines) + "\n"


def write_fixture(fixture: dict, path: str | Path = FIXTURE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(fixture), encoding="utf-8", newline="\n")


def load_fixture(path: str | Path = FIXTURE) -> dict:
    """Read a fixture written by `write_fixture`.

    Raises:
        FileNotFoundError: no fixture (the message says how to record one).
        ValueError: missing keys or malformed rankings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no dense fixture at {path}; {RERECORD}")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in _HEADER + ("queries",) if key not in data]
    if missing:
        raise ValueError(f"{path}: missing keys {', '.join(missing)}; {RERECORD}")
    for q, ranking in data["queries"].items():
        if not all(isinstance(e, list) and len(e) == 2 for e in ranking):
            raise ValueError(f"{path}: malformed ranking for query {q!r}; {RERECORD}")
    return data


class ReplayDense:
    """Dense-index stand-in answering from a recorded fixture (no model, no chromadb).

    Implements what `HybridRouter` reads: `fingerprint()`, `built_with()`, `info()`,
    `search(query, k)`, and the `generation` / `embedder` / `model_name` / `persist_dir`
    attributes. Hits carry the chunk's slug, section and first 200 characters, as live.
    """

    def __init__(self, fixture: dict, chunks: list[Chunk], source: str = "fixture"):
        """
        Args:
            fixture: `load_fixture` / `record` output.
            chunks: the corpus's chunks (maps recorded chunk ids back to slugs and text).
            source: where the fixture came from, for messages.
        """
        self._fixture = fixture
        self._queries: dict[str, list] = fixture["queries"]
        self._chunks = {c.chunk_id: c for c in chunks}
        self.depth = int(fixture["depth"])
        self.embedder = fixture["embedder"]
        self.model_name = fixture["model"]
        self.persist_dir = f"replay:{source}"
        self.generation = 0

    def fingerprint(self) -> str:
        return self._fixture["corpus_fingerprint"]

    def built_with(self) -> str:
        return self.embedder

    def info(self) -> dict:
        return {"embedder": self.embedder, "fingerprint": self.fingerprint(),
                "model_name": self.model_name, "count": len(self._chunks), "replay": True}

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The recorded top-k chunks for `query`.

        Raises:
            KeyError: the query was not recorded (names it). ValueError: k exceeds the
            recorded depth. RuntimeError: a recorded chunk is not in the corpus.
        """
        ranking = self._queries.get(query)
        if ranking is None:
            raise KeyError(f"query not in the dense fixture: {query!r}; {RERECORD}")
        if k > self.depth and len(ranking) >= self.depth:
            raise ValueError(f"the fixture holds the top {self.depth} chunks per query; "
                             f"k={k} needs a deeper recording")
        hits = []
        for chunk_id, score in ranking[:k]:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                raise RuntimeError(f"fixture chunk {chunk_id!r} is not in the corpus; {RERECORD}")
            hits.append(Hit(skill_slug=chunk.skill_slug, score=float(score), section=chunk.section,
                            snippet=chunk.text[:SNIPPET_CHARS], chunk_id=chunk_id))
        return hits


def replay_router(fixture: dict, mode: str = "hybrid", **router_kw):
    """A `HybridRouter` (no reranker) whose dense side replays `fixture`.

    Raises:
        RuntimeError: the corpus is missing, or the fixture was recorded from another corpus.
    """
    from sie.router import HybridRouter, corpus_fingerprint
    router = HybridRouter(mode=mode, use_reranker=False, **router_kw)
    chunks = router._load_chunks()
    current = corpus_fingerprint(chunks)
    if fixture["corpus_fingerprint"] != current:
        raise RuntimeError(f"the dense fixture was recorded from another corpus (fixture "
                           f"{fixture['corpus_fingerprint']}, corpus {current}); {RERECORD}")
    router.dense = ReplayDense(fixture, chunks)
    return router
