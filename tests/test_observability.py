"""Routing events: hooks, privacy of queries, failure isolation, JSON logging (stdlib only)."""
import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest

from sie import observability as obs
from sie.router import HybridRouter

CORPUS = Path(__file__).resolve().parents[1] / "data" / "skills"
QUERY = "time series forecasting seasonality"
ROUTE_KEYS = {"event", "mode", "reranked", "rerank_error", "top_skill", "confidence", "action",
              "ambiguous", "n_results", "candidates", "timings_ms", "query_sha", "query_chars"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No hooks leak between tests; the "sie" logger's handlers and level are restored."""
    monkeypatch.delenv("SIE_LOG_QUERIES", raising=False)
    monkeypatch.delenv("SIE_LOG_LEVEL", raising=False)
    root = logging.getLogger("sie")
    handlers, level = list(root.handlers), root.level
    obs.clear_hooks()
    yield
    obs.clear_hooks()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def events():
    seen = []
    obs.add_hook(seen.append)
    return seen


def _sparse_router():
    return HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode="sparse")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def test_route_emits_one_event_matching_the_result(events):
    result = _sparse_router().route(QUERY, k=3)
    assert len(events) == 1
    e = events[0]
    assert set(e) == ROUTE_KEYS and e["event"] == "route"
    assert (e["mode"], e["reranked"], e["rerank_error"]) == ("sparse", False, None)
    assert e["top_skill"] == result.top.slug == "time-series"
    assert (e["confidence"], e["action"], e["ambiguous"]) == (
        result.confidence.level, result.confidence.action, result.confidence.ambiguous)
    assert e["n_results"] == 3 and e["candidates"] == result.candidates
    assert e["timings_ms"] == result.timings_ms


def test_raw_query_is_never_included_by_default(events):
    _sparse_router().route(QUERY)
    e = events[0]
    assert "query" not in e and QUERY not in json.dumps(e)
    assert (e["query_sha"], e["query_chars"]) == (_sha(QUERY), len(QUERY))


def test_raw_query_only_with_the_env_opt_in(events, monkeypatch):
    monkeypatch.setenv("SIE_LOG_QUERIES", "0")
    assert "query" not in obs.query_fields(QUERY)
    monkeypatch.setenv("SIE_LOG_QUERIES", "1")
    _sparse_router().route(QUERY)
    assert events[0]["query"] == QUERY and events[0]["query_sha"] == _sha(QUERY)


def test_empty_query_still_emits_an_abstaining_route_event(events):
    _sparse_router().route("   ")
    e = events[0]
    assert (e["event"], e["top_skill"], e["n_results"], e["confidence"], e["action"]) == (
        "route", None, 0, "none", "abstain")


def test_a_failing_hook_does_not_break_route_or_other_hooks(events):
    def boom(event):
        raise RuntimeError("hook exploded")

    obs.clear_hooks()
    later = []
    obs.add_hook(boom)
    obs.add_hook(later.append)
    result = _sparse_router().route(QUERY)
    assert result.top.slug == "time-series" and len(later) == 1


def test_hooks_cannot_corrupt_the_result_or_each_other():
    def vandal(event):
        event["top_skill"] = "vandalized"
        event["candidates"]["bm25"] = -1
        event["timings_ms"].clear()

    seen = []
    obs.add_hook(vandal)
    obs.add_hook(seen.append)
    result = _sparse_router().route(QUERY)
    assert seen[0]["top_skill"] == "time-series"               # each hook gets its own dict
    assert result.candidates["bm25"] > 0 and "total" in result.timings_ms


def test_route_error_is_emitted_then_the_exception_re_raised(events):
    r = HybridRouter(skills_dir=str(CORPUS), use_reranker=False, mode="hybrid")

    class StaleDense:
        embedder, generation = "onnx", 0

        def fingerprint(self):
            return "built-from-another-corpus"

        def built_with(self):
            return "onnx"

    r.dense = StaleDense()
    with pytest.raises(RuntimeError, match="stale"):
        r.route(QUERY)
    e = events[-1]
    assert set(e) == {"event", "error_type", "error", "mode", "query_sha", "query_chars"}
    assert (e["event"], e["error_type"], e["mode"]) == ("route_error", "RuntimeError", "hybrid")
    assert "stale" in e["error"] and e["query_sha"] == _sha(QUERY)
    with pytest.raises(ValueError):
        r.route(QUERY, pool=0)
    assert (events[-1]["event"], events[-1]["error_type"]) == ("route_error", "ValueError")
    assert all(ev["event"] == "route_error" for ev in events)


LONE_SURROGATE = "time series \udcff forecasting"      # e.g. surrogateescape-decoded argv


def test_query_fields_never_raise_and_hash_valid_text_as_utf8(monkeypatch):
    fields = obs.query_fields(LONE_SURROGATE)
    expected = hashlib.sha256(LONE_SURROGATE.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    assert fields == {"query_sha": expected, "query_chars": len(LONE_SURROGATE)}
    assert obs.query_fields(LONE_SURROGATE) == fields                     # deterministic
    assert obs.query_fields(QUERY)["query_sha"] == _sha(QUERY)            # unchanged for valid text
    monkeypatch.setenv("SIE_LOG_QUERIES", "1")
    assert obs.query_fields(LONE_SURROGATE)["query"] == LONE_SURROGATE


def test_a_lone_surrogate_query_keeps_its_result_and_event(events):
    result = _sparse_router().route(LONE_SURROGATE, k=3)
    assert result.top.slug == "time-series"
    (e,) = events
    assert e["event"] == "route" and e["query_sha"] == obs.query_fields(LONE_SURROGATE)["query_sha"]


def test_failing_telemetry_never_discards_a_result_or_masks_an_error(events, monkeypatch):
    import sie.router as router_mod

    def broken(query):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "telemetry failed")

    monkeypatch.setattr(router_mod, "query_fields", broken)
    r = _sparse_router()
    assert r.route(QUERY).top.slug == "time-series"                       # result kept
    assert events[-1]["event"] == "route" and "query_sha" not in events[-1]
    with pytest.raises(ValueError, match="pool must be >= 1"):            # the original error
        r.route(QUERY, pool=0)
    assert events[-1]["event"] == "route_error" and events[-1]["error_type"] == "ValueError"

    def vandal_emit(event, **fields):
        raise RuntimeError("emit exploded")

    monkeypatch.setattr(router_mod, "emit", vandal_emit)
    assert r.route(QUERY).top.slug == "time-series"
    with pytest.raises(ValueError, match="pool must be >= 1"):
        r.route(QUERY, pool=0)


def test_hook_registry():
    seen = []
    obs.add_hook(seen.append)
    obs.add_hook(seen.append)                                   # registered once
    obs.emit("custom", n=1)
    assert seen == [{"event": "custom", "n": 1}]
    obs.remove_hook(seen.append)
    obs.remove_hook(seen.append)                                # unknown: ignored
    obs.emit("custom", n=2)
    assert len(seen) == 1
    obs.add_hook(seen.append)
    obs.clear_hooks()
    obs.emit("custom", n=3)
    assert len(seen) == 1


def test_events_are_logged_as_sorted_json_at_info(caplog):
    with caplog.at_level(logging.WARNING, logger="sie.events"):
        obs.emit("quiet", a=1)
    assert not caplog.records                                   # silent below INFO
    with caplog.at_level(logging.INFO, logger="sie.events"):
        obs.emit("route", zeta=1, alpha=Path("x"))
    (record,) = [r for r in caplog.records if r.name == "sie.events"]
    assert record.levelno == logging.INFO
    assert record.getMessage() == '{"alpha": "x", "event": "route", "zeta": 1}'


def test_library_is_silent_by_default():
    assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("sie").handlers)


def test_configure_logging_levels_and_single_handler(monkeypatch):
    root = logging.getLogger("sie")
    root.handlers[:] = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
    assert obs.configure_logging() is root and root.level == logging.WARNING
    monkeypatch.setenv("SIE_LOG_LEVEL", "debug")
    obs.configure_logging()
    assert root.level == logging.DEBUG
    obs.configure_logging("INFO")                               # the argument beats the env
    assert root.level == logging.INFO
    streams = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]
    assert len(streams) == 1 and isinstance(streams[0], logging.StreamHandler)
    obs.configure_logging(logging.ERROR)
    assert root.level == logging.ERROR
    with pytest.raises(ValueError, match="unknown log level"):
        obs.configure_logging("LOUD")


def test_configure_logging_prints_route_events_to_stderr(capsys):
    root = logging.getLogger("sie")
    root.handlers[:] = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
    obs.configure_logging("INFO")
    _sparse_router().route(QUERY)
    err = capsys.readouterr().err
    line = next(l for l in err.splitlines() if l.startswith("[sie.events] INFO "))
    event = json.loads(line.split("INFO ", 1)[1])
    assert event["event"] == "route" and event["top_skill"] == "time-series" and "query" not in event


def test_observability_is_stdlib_only():
    import subprocess
    code = ("import sys, sie.observability, sie.confidence; "
            "heavy = {'chromadb', 'sentence_transformers', 'torch', 'onnxruntime', 'numpy'} & set(sys.modules); "
            "assert not heavy, heavy")
    subprocess.run([sys.executable, "-c", code], cwd=CORPUS.parents[1], check=True)


@pytest.mark.parametrize("value,raw", [("1", True), ("true", True), ("YES", True), ("on", True),
                                       ("0", False), ("false", False), ("maybe", False), ("", False)])
def test_raw_query_logging_accepts_the_documented_boolean_spellings(monkeypatch, value, raw):
    from sie.observability import query_fields
    monkeypatch.setenv("SIE_LOG_QUERIES", value)
    assert ("query" in query_fields("secret task")) is raw
