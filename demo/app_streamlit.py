"""Optional one-screen demo over `sie.engine.Engine`: routing with confidence and per-method
evidence, an opt-in multi-intent plan, and learning paths.

    pip install -e ".[demo]"              # streamlit; add ".[retrieval]" for dense / hybrid
    streamlit run demo/app_streamlit.py   # or: make demo

Configuration: the SIE_* variables (see .env.example); the sidebar overrides the retrievers
and reranking. Sparse mode (BM25 only) needs no index and no model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # `streamlit run` puts demo/ on sys.path, not the repo
    sys.path.insert(0, str(ROOT))

from sie.engine import CycleError, Engine, env_bool  # noqa: E402
from sie.models import Composition, RouteResult  # noqa: E402
from sie.router import MODES  # noqa: E402

DEFAULT_QUERY = "impute missing values and encode categoricals"


@st.cache_resource(show_spinner="Loading the corpus and warming up...")
def _engine(mode: str, rerank: bool) -> Engine:
    """One started engine per (mode, rerank); start() records failures instead of raising."""
    env = {**os.environ, "SIE_MODE": mode, "SIE_RERANK": "1" if rerank else "0"}
    engine = Engine.from_env(env)
    engine.start(eager=True)
    return engine


def _status_panel(status: dict) -> None:
    corpus = status["corpus"]
    if corpus:
        version = f"@{corpus['version']}" if corpus["version"] else ""
        st.caption(f"Corpus {corpus['name']}{version}: {corpus['n_skills']} skills, "
                   f"fingerprint {corpus['fingerprint']}")
    if status["ready"]:
        st.success("Ready" + (" (degraded: reranker unavailable, RRF order)" if status["degraded"] else ""))
    else:
        st.error("Not ready: " + "; ".join(status["reasons"]))


def _confidence(result: RouteResult) -> None:
    conf = result.confidence
    box = {"high": st.success, "medium": st.warning, "low": st.warning}.get(conf.level, st.error)
    ambiguous = f" (ambiguous: vs {', '.join(conf.competitors)})" if conf.ambiguous else ""
    box(f"Confidence: {conf.level} -> {conf.action}{ambiguous}")
    st.caption("Why: " + ("; ".join(conf.reasons) or "-") + ". Levels are heuristic, not calibrated.")
    ranking = "cross-encoder" if result.reranked else "RRF"
    fallback = f"; reranker unavailable: {result.rerank_error}" if result.rerank_error else ""
    st.caption(f"Ranking: {ranking}, mode {result.mode}{fallback}")


def _results(engine: Engine, result: RouteResult) -> None:
    for s in result.results:
        skill = engine.skill_obj(s.slug)
        with st.expander(f"{s.rank}. {skill.name} ({s.slug}), score {s.score:.3f} [{s.section}]",
                         expanded=s.rank == 1):
            st.write(skill.description)
            rows = [{"method": e.method, "rank": e.rank, "score": round(float(e.score), 4),
                     "section": e.section, "matched terms": ", ".join(e.matched_terms) or "-"}
                    for e in s.evidence]
            if rows:
                st.table(rows)
            prerequisites = engine.prerequisites(s.slug)
            if prerequisites:
                st.write("Prerequisites: " + ", ".join(prerequisites))
            st.caption(f"Source: {skill.provenance}")


def _composition(c: Composition) -> None:
    st.subheader("Plan")
    for it in c.intents:
        routed = f" (routed as '{it.query}')" if it.query != it.text else ""
        conf = it.result.confidence
        st.write(f"Intent {it.index}: '{it.text}'{routed} -> **{it.selected or 'no match'}** "
                 f"[{conf.level}/{conf.action}]")
    for step in c.steps:
        st.markdown(f"{step.position}. **{step.slug}** ({step.role}): {step.reason}")
    if c.conflicts:
        st.warning("Conflicts: " + "; ".join(" <-> ".join(p) for p in c.conflicts))
    for note in c.notes:
        st.caption(f"Note: {note}")


def _learning_path(engine: Engine, default: str | None) -> None:
    st.subheader("Learning path")
    slugs = sorted(engine.graph.nodes)
    target = st.selectbox("Learning path to reach", options=slugs,
                          index=slugs.index(default) if default in slugs else 0)
    try:
        lp = engine.learning_path(target)
    except CycleError as e:
        st.error(str(e))
        return
    st.write(" -> ".join(lp["path"]))
    for step in lp["steps"]:
        if step["relation"] != "target":
            st.write(f"- {step['skill']}: {step['reason']}")
    if lp["related"]:
        st.caption("See also: " + ", ".join(lp["related"]))
    if lp["conflicts"]:
        st.warning("Conflicts: " + ", ".join(lp["conflicts"]))
    for note in lp["notes"]:
        st.caption(f"Note: {note}")


def main() -> None:
    st.set_page_config(page_title="Skill Intelligence Engine", layout="wide")
    st.title("Skill Intelligence Engine")
    st.caption("Routes a task to skills from an external SKILL.md corpus, explains the ranking, "
               "and plans multi-part requests from the corpus's declared dependencies.")
    env_mode = (os.getenv("SIE_MODE") or "hybrid").strip().lower()
    try:
        rerank_default = env_bool(os.environ, "SIE_RERANK", True)   # same parser as the engine
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
        st.stop()
        return
    with st.sidebar:
        mode = st.radio("Retrievers", options=list(MODES),
                        index=MODES.index(env_mode) if env_mode in MODES else 0,
                        help="sparse (BM25 only) needs no dense index and no model")
        rerank = st.checkbox("Cross-encoder rerank", value=rerank_default)
        k = st.slider("Skills to show", min_value=1, max_value=10, value=5)
    try:
        engine = _engine(mode, rerank)
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
        st.stop()
        return
    status = engine.status()
    with st.sidebar:
        _status_panel(status)
    if not engine.loaded:
        st.error("The corpus could not be loaded; set SIE_SKILLS_DIR (see .env.example).")
        st.stop()
        return

    query = st.text_input("Describe your task", value=DEFAULT_QUERY)
    multi = st.toggle("Multi-intent plan", value=False,
                      help="split the request into tasks, route each, add declared prerequisites")
    top = None
    if query.strip():
        try:
            result = engine.route(query, k=k)
            composition = engine.compose(query, k=k) if multi else None
        except (RuntimeError, ValueError) as e:
            st.error(f"{e}. Build the index (`python -m sie.router --build`) or choose sparse.")
        else:
            _confidence(result)
            _results(engine, result)
            if composition is not None:
                _composition(composition)
            top = result.top.slug if result.top else None
    _learning_path(engine, top)


main()
