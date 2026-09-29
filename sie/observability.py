"""Structured routing events: in-process hooks plus JSON lines on the `sie.events` logger.

Stdlib only, and silent by default: the "sie" logger carries a NullHandler, so nothing is
printed until an application configures logging (or calls `configure_logging`).

    from sie import observability
    observability.add_hook(lambda event: metrics.record(event))   # e.g. export to your stack
    observability.configure_logging("INFO")                       # JSON events on stderr

Privacy: events identify a query by `query_sha` (first 12 hex of its sha256) and
`query_chars`; the raw text is included only when SIE_LOG_QUERIES=1.

Events emitted by `HybridRouter.route`:
    route        mode, reranked, rerank_error, top_skill, confidence (level), action,
                 ambiguous, n_results, candidates, timings_ms, query_sha, query_chars[, query]
    route_error  error_type, error, mode, query_sha, query_chars[, query]
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import sys
import threading
from typing import Any, Callable

Hook = Callable[[dict[str, Any]], None]

_TRUTHY = {"1", "true", "yes", "on"}      # same spellings as sie.engine.env_bool
logger = logging.getLogger("sie.events")
_root = logging.getLogger("sie")
_root.addHandler(logging.NullHandler())

_hooks: list[Hook] = []
_hooks_lock = threading.Lock()


def add_hook(fn: Hook) -> None:
    """Call `fn(event_dict)` for every event; adding the same hook twice registers it once."""
    with _hooks_lock:
        if fn not in _hooks:
            _hooks.append(fn)


def remove_hook(fn: Hook) -> None:
    """Unregister `fn`; unknown hooks are ignored."""
    with _hooks_lock:
        if fn in _hooks:
            _hooks.remove(fn)


def clear_hooks() -> None:
    """Unregister every hook."""
    with _hooks_lock:
        _hooks.clear()


def emit(event: str, **fields: Any) -> None:
    """Deliver `{"event": event, **fields}` to every hook, then log it as one JSON line (INFO).

    Each hook gets its own shallow copy; a failing hook is logged at DEBUG and never
    propagates, so observability can't break the call that emitted the event.
    """
    record: dict[str, Any] = {"event": event, **fields}
    with _hooks_lock:
        hooks = list(_hooks)
    for fn in hooks:
        try:
            fn(dict(record))
        except Exception:
            logger.debug("observability hook %r failed on %s", fn, event, exc_info=True)
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps(record, sort_keys=True, default=str))


def query_fields(query: str) -> dict[str, Any]:
    """Privacy-preserving query identity: sha256 prefix + length; raw text only if SIE_LOG_QUERIES is 1/true/yes/on.

    The hash encodes with `surrogatepass`, so it is deterministic and never raises for any
    str (e.g. a lone surrogate from surrogateescape-decoded argv); valid text hashes exactly
    as plain UTF-8.
    """
    text = query if isinstance(query, str) else str(query)
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    out: dict[str, Any] = {"query_sha": digest, "query_chars": len(text)}
    if os.getenv("SIE_LOG_QUERIES", "").strip().lower() in _TRUTHY:   # anything else: off (private)
        out["query"] = text
    return out


def _level(level: str | int) -> int:
    if isinstance(level, int) and not isinstance(level, bool):
        return level
    name = str(level).strip().upper()
    if name.isdigit():
        return int(name)
    value = logging.getLevelNamesMapping().get(name)
    if value is None:
        raise ValueError(f"unknown log level {level!r}; use DEBUG, INFO, WARNING, ERROR or CRITICAL")
    return value


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Send SIE logs (routing events at INFO) to stderr.

    Args:
        level: log level name or number; default env SIE_LOG_LEVEL, else WARNING.

    Returns:
        The "sie" logger. A stderr handler is added only if it has no real handler yet,
        so calling this twice never duplicates output.
    """
    _root.setLevel(_level(level if level is not None else os.getenv("SIE_LOG_LEVEL") or "WARNING"))
    if not any(not isinstance(h, logging.NullHandler) for h in _root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
        _root.addHandler(handler)
    return _root
