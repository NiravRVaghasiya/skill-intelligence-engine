"""Learning-path generation: requires-closure + topological ordering."""
from __future__ import annotations

from .build import requires_graph

_LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}


def prerequisite_closure(g, target: str) -> set[str]:
    """All skills that must be learned before `target` (transitive requires)."""
    import networkx as nx
    if target not in g:
        raise KeyError(target)
    return set(nx.ancestors(requires_graph(g), target))


def _order(g, nodes: set[str]) -> list[str]:
    """Topological order of `nodes` under requires; ties by (level, slug) for determinism.

    Raises:
        ValueError: the requires edges among `nodes` contain a cycle.
    """
    import networkx as nx
    sub = requires_graph(g).subgraph(nodes)
    key = lambda n: (_LEVEL_ORDER.get(g.nodes[n].get("level"), 1), n)
    try:
        return list(nx.lexicographical_topological_sort(sub, key=key))
    except nx.NetworkXUnfeasible:
        cycle = " -> ".join(u for u, _ in nx.find_cycle(sub))
        raise ValueError(f"requires cycle: {cycle}") from None


def conflicts_of(g, slug: str) -> set[str]:
    """Skills that conflict with `slug`, declared in either direction (in-corpus only)."""
    declared = set(g.nodes[slug].get("conflicts", []))
    declared |= {n for n in g if slug in g.nodes[n].get("conflicts", [])}
    return {c for c in declared if c in g and c != slug}


def see_also(g, slug: str, exclude: set[str] = frozenset()) -> list[str]:
    """Declared `related` skills in declared order: in-corpus, deduplicated, minus `exclude`."""
    return list(dict.fromkeys(r for r in g.nodes[slug].get("related", [])
                              if r in g and r not in exclude and r != slug))


def learning_path(g, target: str) -> dict:
    """Return an ordered learning path to reach `target`, plus see-also / conflicts.

    Returns:
        {"target", "path" (prereqs in topological order, ending at target), "related"
        (target's see-also, minus path members and conflicting skills), "conflicts" (skills
        that conflict with any skill on the path), "path_conflicts" (conflicting pairs
        *within* the path), "missing_prerequisites" (declared `requires` of path members
        that are not in the corpus, e.g. typos: [skill, missing_slug])}.

    Raises:
        KeyError: unknown target. ValueError: a requires cycle blocks ordering.
    """
    nodes = prerequisite_closure(g, target) | {target}
    ordered = _order(g, nodes)
    conflicts: set[str] = set()
    path_conflicts: set[tuple[str, str]] = set()
    for n in ordered:
        for c in conflicts_of(g, n):
            if c in nodes:
                path_conflicts.add(tuple(sorted((n, c))))
            else:
                conflicts.add(c)
    missing = sorted([s, ref] for s, kind, ref in g.graph.get("dangling", [])
                     if kind == "requires" and s in nodes)
    return {"target": target, "path": ordered, "related": see_also(g, target, nodes | conflicts),
            "conflicts": sorted(conflicts), "path_conflicts": [list(p) for p in sorted(path_conflicts)],
            "missing_prerequisites": missing}
