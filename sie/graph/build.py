"""Build a NetworkX DiGraph from skill frontmatter edges.

`requires` edges point prereq -> skill and are never overwritten by a `related`/`conflicts`
edge on the same node pair. Each node also keeps its declared `related` and `conflicts`
lists as attributes (the source of truth for see-also/conflict lookups), so no declaration
is lost to DiGraph's one-edge-per-pair limit. References to slugs outside the corpus never
become phantom nodes; they are recorded in `g.graph["dangling"]`.
"""
from __future__ import annotations

from ..models import Skill

EDGE_KINDS = ("requires", "related", "conflicts")


def build_graph(skills: list[Skill]):
    """Return a DiGraph with requires (directed), related & conflicts (as edge attrs)."""
    import networkx as nx
    g = nx.DiGraph(dangling=[])
    for s in skills:
        g.add_node(s.slug, domain=s.domain, level=s.level, skill_type=s.skill_type,
                   display_name=s.display_name or s.slug, description=s.description,
                   capabilities=list(s.capabilities),
                   related=list(s.related), conflicts=list(s.conflicts))
    for s in skills:
        for kind in EDGE_KINDS:
            for ref in getattr(s, kind):
                if ref not in g:
                    g.graph["dangling"].append((s.slug, kind, ref))
                elif kind == "requires":
                    g.add_edge(ref, s.slug, kind="requires")      # prereq -> skill
                elif not g.has_edge(s.slug, ref) or kind == "conflicts" and \
                        g.edges[s.slug, ref]["kind"] == "related":
                    g.add_edge(s.slug, ref, kind=kind)
    return g


def requires_graph(g):
    """Every node, but only the `requires` edges."""
    import networkx as nx
    req = nx.DiGraph()
    req.add_nodes_from(g.nodes(data=True))
    req.add_edges_from((u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "requires")
    return req


def find_cycles(g) -> list[list[str]]:
    """Detect requires-cycles (should be none in a valid corpus), in a stable order."""
    import networkx as nx
    return sorted(sorted(c) for c in nx.simple_cycles(requires_graph(g)))
