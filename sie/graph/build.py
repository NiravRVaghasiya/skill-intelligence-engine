"""Build a NetworkX DiGraph from skill frontmatter edges."""
from __future__ import annotations

from ..models import Skill


def build_graph(skills: list[Skill]):
    """Return a DiGraph with requires (directed), related & conflicts (as edge attrs)."""
    import networkx as nx
    g = nx.DiGraph()
    for s in skills:
        g.add_node(s.slug, domain=s.domain, level=s.level, skill_type=s.skill_type)
    for s in skills:
        for req in s.requires:
            g.add_edge(req, s.slug, kind="requires")   # prereq -> skill
        for rel in s.related:
            if not g.has_edge(s.slug, rel):
                g.add_edge(s.slug, rel, kind="related")
        for conf in s.conflicts:
            g.add_edge(s.slug, conf, kind="conflicts")
    return g


def find_cycles(g) -> list[list[str]]:
    """Detect requires-cycles (should be none in a valid corpus)."""
    import networkx as nx
    req = g.edge_subgraph([(u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "requires"])
    return list(nx.simple_cycles(req))
