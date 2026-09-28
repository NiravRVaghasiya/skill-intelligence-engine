"""Learning-path generation: requires-closure + topological ordering."""
from __future__ import annotations


def prerequisite_closure(g, target: str) -> set[str]:
    """All skills that must be learned before `target` (transitive requires)."""
    import networkx as nx
    req_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "requires"]
    req = g.edge_subgraph(req_edges) if req_edges else g.subgraph([])
    if target not in req:
        return set()
    return set(nx.ancestors(req, target))


def learning_path(g, target: str) -> dict:
    """Return an ordered learning path to reach `target`, plus see-also / conflicts."""
    import networkx as nx
    prereqs = prerequisite_closure(g, target)
    nodes = prereqs | {target}
    req_edges = [(u, v) for u, v, d in g.edges(data=True)
                 if d.get("kind") == "requires" and u in nodes and v in nodes]
    sub = g.edge_subgraph(req_edges) if req_edges else g.subgraph(nodes)
    try:
        ordered = [n for n in nx.topological_sort(sub) if n in nodes]
    except nx.NetworkXUnfeasible:
        ordered = list(nodes)
    for n in nodes:                       # include isolated prereqs
        if n not in ordered:
            ordered.insert(0, n)
    related = [v for u, v, d in g.edges(target, data=True) if d.get("kind") == "related"]
    conflicts = [v for u, v, d in g.edges(target, data=True) if d.get("kind") == "conflicts"]
    return {"target": target, "path": ordered, "related": related, "conflicts": conflicts}
