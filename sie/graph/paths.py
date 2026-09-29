"""Learning-path generation: requires-closure + deterministic topological ordering.

Only `requires` edges decide *membership* of a path. `recommended_before` (soft) edges may
reorder skills already on it and are listed separately; `include_recommended=True` inserts
them too, without pulling in their own prerequisites. Every explanation (step reasons, notes)
is derived from edges in the graph; nothing is inferred.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..models import Edge
from .build import ordering_graph, requires_graph

if TYPE_CHECKING:                                       # annotations only; imported lazily below
    import networkx as nx

_LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}
_UNRANKED = float("inf")          # priority of nodes the caller did not rank


def prerequisite_closure(g, target: str) -> set[str]:
    """All skills that must be learned before `target` (transitive requires)."""
    import networkx as nx
    if target not in g:
        raise KeyError(target)
    return set(nx.ancestors(requires_graph(g), target))


def _cycle_message(sub) -> str:
    """`requires cycle: a -> b -> a` for the first hard cycle in an ordering graph."""
    import networkx as nx
    hard = nx.DiGraph()
    hard.add_nodes_from(sub.nodes)
    hard.add_edges_from((u, v) for u, v, d in sub.edges(data=True) if d.get("kind") == "requires")
    cycle = nx.find_cycle(hard)
    return "requires cycle: " + " -> ".join([u for u, _ in cycle] + [cycle[-1][1]])


def _ordered(g: nx.DiGraph, nodes: Iterable[str], priority: dict[str, int] | None,
             soft: bool) -> tuple[list[str], list[Edge]]:
    """`order_skills`, plus the soft edges it had to drop to stay acyclic."""
    import networkx as nx
    sub, dropped = ordering_graph(g, nodes, soft=soft)
    prio = priority or {}

    def key(n: str):
        level = g.nodes[n].get("level") if n in g else None
        return prio.get(n, _UNRANKED), _LEVEL_ORDER.get(level, 1), n

    try:
        return list(nx.lexicographical_topological_sort(sub, key=key)), dropped
    except nx.NetworkXUnfeasible:
        raise ValueError(_cycle_message(sub)) from None


def order_skills(g: nx.DiGraph, nodes: Iterable[str], priority: dict[str, int] | None = None,
                 soft: bool = True) -> list[str]:
    """Deterministic topological order of `nodes` under the graph's ordering edges.

    Args:
        g: graph from `build_graph`.
        nodes: slugs to order (any iterable). Slugs not in `g` are ordered as isolated nodes.
        priority: lower comes first among skills whose prerequisites are already placed
            (e.g. intent index); unranked skills sort after ranked ones.
        soft: honor acyclic `recommended_before` edges among `nodes` (they never add nodes).

    Returns:
        Every node exactly once. Hard `requires` edges always win; soft edges next; remaining
        ties break by (priority, level beginner->advanced, slug).

    Raises:
        ValueError: `requires cycle: a -> b -> a` among `nodes`.
    """
    return _ordered(g, nodes, priority, soft)[0]


def _order(g, nodes: set[str]) -> list[str]:
    """Topological order of `nodes` under requires; ties by (level, slug) for determinism.

    Raises:
        ValueError: the requires edges among `nodes` contain a cycle.
    """
    return order_skills(g, nodes, soft=False)


def _typed_edges(g, relationship: str) -> list[Edge]:
    return [e for e in g.graph.get("edges", []) if e.relationship == relationship]


def conflicts_of(g, slug: str) -> set[str]:
    """Skills that conflict with `slug`, declared in either direction (in-corpus only)."""
    declared = set(g.nodes[slug].get("conflicts", []))
    declared |= {n for n in g if slug in g.nodes[n].get("conflicts", [])}
    for e in _typed_edges(g, "conflicts"):                  # overlay conflicts, if any
        if slug in (e.source, e.target):
            declared.add(e.target if e.source == slug else e.source)
    return {c for c in declared if c in g and c != slug}


def see_also(g, slug: str, exclude: set[str] = frozenset()) -> list[str]:
    """Declared `related` skills in declared order (then overlay ones): in-corpus, deduplicated,
    minus `exclude`."""
    refs = list(g.nodes[slug].get("related", []))
    refs += [e.target for e in _typed_edges(g, "related") if e.source == slug]
    return list(dict.fromkeys(r for r in refs if r in g and r not in exclude and r != slug))


def superseded_by(g: nx.DiGraph, slug: str) -> list[str]:
    """Skills that declare they supersede (replace) `slug`, sorted."""
    return sorted({e.source for e in _typed_edges(g, "supersedes")
                   if e.target == slug and e.source != slug})


def _declares_requires(g, slug: str) -> bool:
    return "requires" in (g.nodes[slug].get("declared") or [])


def _edge_label(e: Edge) -> str:
    """Provenance of an edge; non-declared edges carry their confidence, e.g. `... [proposed]`."""
    label = e.provenance or f"{e.source} -> {e.target} ({e.relationship})"
    return label if e.confidence == "declared" else f"{label} [{e.confidence}]"


def _marker(edges: Iterable[Edge]) -> str:
    """` (proposed)` (etc.) for edges that are not corpus declarations; "" when all are declared."""
    labels = sorted({e.confidence for e in edges if e.confidence != "declared"})
    return f" ({', '.join(labels)})" if labels else ""


def _steps(g, target: str, ordered: list[str], hard: set[str], soft_in: list[Edge]) -> list[dict]:
    """One explanation per path entry (see `learning_path`).

    `soft_in` must hold only the soft edges the final order honors, so no step claims an
    ordering that was dropped.
    """
    import networkx as nx
    req = requires_graph(g)
    depth = nx.single_source_shortest_path_length(req.reverse(copy=False), target)
    index = {(e.source, e.target, e.relationship): e for e in g.graph.get("edges", [])}
    pos = {s: i for i, s in enumerate(ordered, 1)}
    steps = []
    for slug in ordered:
        required_by = [m for m in ordered if m != slug and req.has_edge(slug, m)]
        support = [index[(slug, m, "requires")] for m in required_by if (slug, m, "requires") in index]
        marked = support                                # edges whose confidence the reason reports
        if slug == target:
            relation, reason = "target", "target"
        elif slug not in hard:
            before = sorted((e for e in soft_in if e.source == slug), key=lambda e: pos[e.target])
            support += before
            relation = "recommended"
            reason = f"recommended before {', '.join(e.target for e in before)}; not a prerequisite"
        elif req.has_edge(slug, target):
            relation = "direct"
            others = [m for m in required_by if m != target]
            reason = f"direct prerequisite of {target}" + (
                f" (also required by {', '.join(others)})" if others else "")
        else:
            # only hard-path dependants lead to the target; an included recommended skill that
            # requires this one is on the path but has no requires route to the target
            relation = "transitive"
            hard_by = [m for m in required_by if m in hard]
            soft_by = [m for m in required_by if m not in hard]
            verb = "leads" if len(hard_by) == 1 else "lead"
            reason = f"prerequisite of {', '.join(hard_by)}, which {verb} to {target} (transitive)"
            marked = [e for e in support if e.target in hard]
            if soft_by:
                reason += _marker(marked) + f"; also required by recommended {', '.join(soft_by)}"
                marked = [e for e in support if e.target not in hard]
        reason += _marker(marked)
        steps.append({"skill": slug, "position": pos[slug], "relation": relation,
                      "depth": depth.get(slug) if slug in hard else None,
                      "required_by": required_by, "reason": reason,
                      "provenance": [_edge_label(e) for e in support]})
    return steps


def learning_path(g: nx.DiGraph, target: str, include_recommended: bool = False) -> dict:
    """Return an ordered learning path to reach `target`, plus see-also / conflicts / why.

    Args:
        g: graph from `build_graph`.
        target: slug to reach.
        include_recommended: also insert soft prerequisites (`recommended_before`) of path
            members, ordered with the soft edges. Their own prerequisites are NOT pulled in
            (a note says so); by default they are only listed under "recommended". One whose
            every soft edge onto the path had to be dropped (it contradicts other ordering) is
            left off the path (still listed under "recommended", with a note), so the path
            always ends at the target.

    Returns:
        {"target", "path" (prereqs in topological order, ending at target), "related"
        (target's see-also, minus path members and conflicting skills), "conflicts" (skills
        that conflict with any skill on the path), "path_conflicts" (conflicting pairs
        *within* the path), "missing_prerequisites" (declared `requires` of path members
        that are not in the corpus, e.g. typos: [skill, missing_slug]),
        "steps" (per path entry: skill, position, relation target|direct|transitive|recommended,
        depth = fewest requires hops to target (None for recommended), required_by, reason,
        provenance), "direct" / "transitive" (prerequisites by relation, path order),
        "recommended" ([{skill, before, provenance}] soft prerequisites not on the hard path),
        "notes" (declaration gaps, omissions, superseded skills, dropped soft edges,
        recommended skills left off the path),
        "complete" (no missing prerequisite, every path member declares `requires`, and no
        included recommended skill has prerequisites left off the path)}.
        Only the new keys depend on `recommended_before`/overlay edges being present; for a
        corpus without them the original six keys are unchanged.

    Raises:
        KeyError: unknown target. ValueError: a requires cycle blocks ordering.
    """
    hard = prerequisite_closure(g, target) | {target}
    soft_in = [e for e in _typed_edges(g, "recommended_before")
               if e.target in hard and e.source not in hard]
    members = (hard | {e.source for e in soft_in}) if include_recommended else hard
    ordered, dropped = _ordered(g, members, None, soft=True)
    left_off: set[str] = set()
    while include_recommended:
        # A recommended skill whose every soft edge onto the path was dropped (it contradicts
        # other ordering) would be placed anywhere, even after the target: leave it off. Removing
        # nodes only removes constraints; loop anyway in case a freed edge drops another's.
        on_path = {e.source for e in soft_in if e.source in members}
        kept = {e.source for e in soft_in if e.source in members and e not in dropped}
        if not (lost := on_path - kept):
            break
        left_off |= lost
        members = members - lost
        ordered, dropped = _ordered(g, members, None, soft=True)
    kept_soft = [e for e in soft_in if e.source in members and e not in dropped]
    pos = {s: i for i, s in enumerate(ordered)}
    conflicts: set[str] = set()
    path_conflicts: set[tuple[str, str]] = set()
    for n in ordered:
        for c in conflicts_of(g, n):
            if c in members:
                path_conflicts.add(tuple(sorted((n, c))))
            else:
                conflicts.add(c)
    missing = sorted([s, ref] for s, kind, ref in g.graph.get("dangling", [])
                     if kind == "requires" and s in members)
    steps = _steps(g, target, ordered, hard, kept_soft if include_recommended else [])
    recommended = [{"skill": e.source, "before": e.target, "provenance": _edge_label(e)}
                   for e in sorted(soft_in, key=lambda e: (pos[e.target], e.source))]
    left_out = {r: sorted(prerequisite_closure(g, r) - members) for r in ordered if r not in hard}
    off_path = {r: sorted({e.target for e in soft_in if e.source == r}, key=pos.__getitem__)
                for r in sorted(left_off)}
    notes = _notes(g, target, ordered, missing, dropped, left_out, off_path)
    complete = (not missing and all(_declares_requires(g, s) for s in ordered)
                and not any(left_out.values()))
    return {"target": target, "path": ordered, "related": see_also(g, target, members | conflicts),
            "conflicts": sorted(conflicts), "path_conflicts": [list(p) for p in sorted(path_conflicts)],
            "missing_prerequisites": missing,
            "steps": steps,
            "direct": [s["skill"] for s in steps if s["relation"] == "direct"],
            "transitive": [s["skill"] for s in steps if s["relation"] == "transitive"],
            "recommended": recommended, "notes": notes, "complete": complete}


def _notes(g, target: str, ordered: list[str], missing: list[list[str]], dropped: list[Edge],
           left_out: dict[str, list[str]], off_path: dict[str, list[str]]) -> list[str]:
    """Deterministic caveats for a learning path, most important first."""
    notes = []
    has_prereqs = any(u != target for u, _ in g.in_edges(target)
                      if g.edges[u, target].get("kind") == "requires")
    if not _declares_requires(g, target):
        notes.append(f"{target} does not declare prerequisites; the path may be incomplete")
    elif not has_prereqs and not any(s == target for s, _ in missing):
        notes.append(f"{target} declares no prerequisites")
    notes += [f"{s} does not declare prerequisites; the path may be incomplete"
              for s in ordered if s != target and not _declares_requires(g, s)]
    notes += [f"{s} requires '{ref}', which is not in the corpus; it is omitted"
              for s, ref in dict.fromkeys(tuple(m) for m in missing)]
    for s in ordered:
        if sup := superseded_by(g, s):
            notes.append(f"{s} is superseded by {', '.join(sup)}")
    notes += [f"ignored recommended_before {e.source} -> {e.target}: it would create an "
              f"ordering cycle" for e in dropped]
    notes += [f"{r} is recommended before {', '.join(before)}, but that conflicts with other "
              f"ordering; left off the path" for r, before in off_path.items()]
    notes += [f"{r} is recommended, but its own prerequisites are not included: {', '.join(pre)}"
              for r, pre in left_out.items() if pre]
    return notes


def render_learning_path(g: nx.DiGraph, lp: dict) -> str:
    """Human-readable learning path (the `python -m sie.router --path` output), then the why.

    Args:
        g: the graph `lp` was computed on (for skill type/domain/level).
        lp: a `learning_path` result.

    Returns:
        Header, numbered steps, see also, conflicts, WARNING lines; then one `why:` line per
        non-target step, one `note:` line per note, one `recommended before` line per
        soft prerequisite.
    """
    target, path = lp["target"], lp["path"]
    steps = lp.get("steps") or []
    relation = {s["skill"]: s["relation"] for s in steps}
    edges = ("`requires` + `recommended_before` edges" if "recommended" in relation.values()
             else "hard `requires` edges only")
    n_steps = len(path)
    lines = [f"[path] learning path to {target} ({n_steps} step{'s' if n_steps != 1 else ''}, {edges})"]
    for i, slug in enumerate(path, 1):
        n = g.nodes[slug]
        mark = ("  <- target" if slug == target
                else "  (recommended)" if relation.get(slug) == "recommended" else "")
        lines.append(f"  {i}. {slug:26s} [{n['skill_type']}/{n['domain']}, {n['level']}]{mark}")
    lines.append(f"  see also:  {', '.join(lp['related']) or 'none'}")
    lines.append(f"  conflicts: {', '.join(lp['conflicts']) or 'none'}")
    if lp["path_conflicts"]:
        lines.append("  WARNING conflicting skills on the path: "
                     + "; ".join(" <-> ".join(p) for p in lp["path_conflicts"]))
    if lp["missing_prerequisites"]:
        lines.append("  WARNING prerequisites not in the corpus (typo?): "
                     + "; ".join(f"{s} requires '{ref}'" for s, ref in lp["missing_prerequisites"]))
    lines += [f"  why: {s['skill']}: {s['reason']}" for s in steps if s["relation"] != "target"]
    lines += [f"  note: {note}" for note in lp.get("notes", [])]
    lines += [f"  recommended before {r['before']}: {r['skill']}" for r in lp.get("recommended", [])]
    return "\n".join(lines)
