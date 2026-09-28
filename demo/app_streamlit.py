"""Optional one-screen demo: search box + learning-path viewer."""
from __future__ import annotations

import streamlit as st

from sie.router import HybridRouter
from sie.ingest import load_corpus
from sie.graph.build import build_graph
from sie.graph.paths import learning_path


@st.cache_resource
def _engine():
    r = HybridRouter()
    g = build_graph(load_corpus("data/skills"))
    return r, g


st.title("Skill Intelligence Engine")
r, g = _engine()

q = st.text_input("Describe your task", "impute missing values and encode categoricals")
if q:
    st.subheader("Top skills")
    for h in r.retrieve(q, k=5):
        st.write(f"**{h.skill_slug}** — score {h.score:.3f}  ·  _{h.section}_")

target = st.selectbox("Learning path to reach:", sorted(g.nodes()))
if target:
    lp = learning_path(g, target)
    st.subheader("Ordered learning path")
    st.write(" -> ".join(lp["path"]) or target)
    if lp["conflicts"]:
        st.warning("Conflicts: " + ", ".join(lp["conflicts"]))
