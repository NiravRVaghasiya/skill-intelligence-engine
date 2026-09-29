"""Build a typed NetworkX DiGraph from skill frontmatter relationships (+ optional overlays).

Every relationship is declared in one skill X's frontmatter as a list of refs R. Direction:

| relationship       | declared on X listing R means  | Edge(source, target) | ordering |
| ------------------ | ------------------------------ | -------------------- | -------- |
| requires           | R must be learned before X     | (R, X)               | hard     |
| recommended_before | R is recommended before X      | (R, X)               | soft     |
| related            | see also R                     | (X, R)               | none     |
| conflicts          | X and R conflict (symmetric)   | (X, R)               | none     |
| alternative_to     | X is an alternative to R (sym) | (X, R)               | none     |
| specializes        | X is a narrower form of R      | (X, R)               | none     |
| supersedes         | X replaces R (R is outdated)   | (X, R)               | none     |

Only `requires` is a prerequisite. Soft (`recommended_before`) edges may order skills that are
already on a path, never add members; `ordering_graph` drops any soft edge that would create a
cycle. Symmetric relations are stored once, as declared; lookups read both directions.

Two views of the same declarations:
- DiGraph edges (`kind` attr) for `requires` / `related` / `conflicts`. `requires` edges point
  prereq -> skill and are never overwritten by a `related`/`conflicts` edge on the same node pair.
  Each node also keeps its declared relationship lists as attributes (the source of truth for
  see-also/conflict lookups), so no declaration is lost to DiGraph's one-edge-per-pair limit.
- `g.graph["edges"]`: every typed `Edge` (all seven kinds) with confidence + provenance, deduped
  on (source, target, relationship), sorted by (RELATIONSHIPS index, source, target).

References to slugs outside the corpus never become phantom nodes; they are recorded in
`g.graph["dangling"]` as (declaring skill, relationship, ref). Overlays (e.g. proposed edges
awaiting upstream review) are applied only when passed explicitly to `build_graph`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import RELATIONSHIPS, Edge, Skill

if TYPE_CHECKING:                                       # annotations only; imported lazily below
    import networkx as nx

EDGE_KINDS = ("requires", "related", "conflicts")      # kinds that also become DiGraph edges


@dataclass(frozen=True)
class RelationSpec:
    """How one relationship behaves.

    ordering: "hard" (prerequisite), "soft" (may order, never adds members) or "none".
    symmetric: the claim holds in both directions (stored once, looked up both ways).
    declared_as: where the declaring skill X sits on the edge: "target" -> Edge(R, X),
        "source" -> Edge(X, R).
    """
    ordering: str
    symmetric: bool
    declared_as: str


RELATION_SPECS: dict[str, RelationSpec] = {
    "requires": RelationSpec("hard", False, "target"),
    "recommended_before": RelationSpec("soft", False, "target"),
    "related": RelationSpec("none", False, "source"),
    "conflicts": RelationSpec("none", True, "source"),
    "alternative_to": RelationSpec("none", True, "source"),
    "specializes": RelationSpec("none", False, "source"),
    "supersedes": RelationSpec("none", False, "source"),
}
_REL_INDEX = {r: i for i, r in enumerate(RELATIONSHIPS)}


def _endpoints(declaring: str, ref: str, relationship: str) -> tuple[str, str]:
    """(source, target) of the edge that `declaring` asserts by listing `ref`."""
    if RELATION_SPECS[relationship].declared_as == "target":
        return ref, declaring
    return declaring, ref


def _declaring(edge: Edge) -> tuple[str, str]:
    """(declaring skill, ref) for an edge: the inverse of `_endpoints`."""
    if RELATION_SPECS[edge.relationship].declared_as == "target":
        return edge.target, edge.source
    return edge.source, edge.target


def _declared(s: Skill) -> list[str]:
    """Relationship keys s declares: its frontmatter `declared`, plus any non-empty list.

    Skills built in code (declared=[]) still count as declaring what they list.
    """
    present = set(s.declared) | {r for r in RELATIONSHIPS if getattr(s, r, None)}
    return [r for r in RELATIONSHIPS if r in present]


def _add_kind_edge(g, u: str, v: str, kind: str) -> None:
    """Add a DiGraph edge; `requires` always wins a pair, `conflicts` beats `related`."""
    if kind == "requires":
        g.add_edge(u, v, kind="requires")
    elif not g.has_edge(u, v) or kind == "conflicts" and g.edges[u, v]["kind"] == "related":
        g.add_edge(u, v, kind=kind)


def _check_relationship(relationship: str, where: str) -> None:
    if relationship not in RELATION_SPECS:
        raise ValueError(f"{where}: unknown relationship '{relationship}' "
                         f"(expected one of {', '.join(RELATIONSHIPS)})")


def build_graph(skills: list[Skill], overlays: list[Edge] | None = None) -> nx.DiGraph:
    """Return the typed skill graph.

    Args:
        skills: parsed corpus. Frontmatter edges get confidence "declared" and provenance
            `<skill.provenance>#<relationship>`.
        overlays: extra edges (e.g. from `load_overlay`), added after the frontmatter; they keep
            their own confidence/provenance, lose to a frontmatter edge with the same (source,
            target, relationship), and `requires` overlays become DiGraph requires edges so
            learning paths honor them. Edges with an unknown endpoint are recorded as dangling.

    Returns:
        DiGraph with node attrs (metadata, provenance, declared relationship lists), DiGraph edges
        for requires/related/conflicts, and graph attrs "edges" (list[Edge]), "dangling" (list of
        (declaring skill, relationship, ref)) and "overlays" (sorted overlay provenance strings).

    Raises:
        ValueError: an overlay edge names an unknown relationship.
    """
    import networkx as nx
    g = nx.DiGraph(dangling=[])
    for s in skills:
        g.add_node(s.slug, domain=s.domain, level=s.level, skill_type=s.skill_type,
                   display_name=s.display_name or s.slug, description=s.description,
                   capabilities=list(s.capabilities),
                   related=list(s.related), conflicts=list(s.conflicts),
                   name=s.name, source=s.source, source_version=s.source_version, path=s.path,
                   content_hash=s.content_hash, updated_at=s.updated_at, declared=_declared(s),
                   recommended_before=list(s.recommended_before),
                   alternative_to=list(s.alternative_to), specializes=list(s.specializes),
                   supersedes=list(s.supersedes))
    edges: dict[tuple[str, str, str], Edge] = {}
    dangling: dict[tuple[str, str, str], None] = {}         # ordered set
    for s in skills:
        for rel in RELATIONSHIPS:
            for ref in dict.fromkeys(getattr(s, rel, None) or []):
                if ref not in g:
                    dangling.setdefault((s.slug, rel, ref))
                    continue
                src, dst = _endpoints(s.slug, ref, rel)
                if rel in EDGE_KINDS:
                    _add_kind_edge(g, src, dst, rel)
                edges.setdefault((src, dst, rel), Edge(src, dst, rel, confidence="declared",
                                                       provenance=f"{s.provenance}#{rel}"))
    overlays = list(overlays or [])
    for i, e in enumerate(overlays):
        _check_relationship(e.relationship, f"overlay edge {i} ({e.source} -> {e.target})")
        if e.source not in g or e.target not in g:
            declaring, ref = _declaring(e)
            dangling.setdefault((declaring, e.relationship, ref))
            continue
        key = (e.source, e.target, e.relationship)
        if key in edges:
            continue                                        # frontmatter (or earlier overlay) wins
        if e.relationship in EDGE_KINDS:
            _add_kind_edge(g, e.source, e.target, e.relationship)
        edges[key] = e
    g.graph["dangling"] = list(dangling)
    g.graph["edges"] = sorted(edges.values(),
                              key=lambda e: (_REL_INDEX[e.relationship], e.source, e.target))
    g.graph["overlays"] = sorted({e.provenance for e in overlays if e.provenance})
    return g


def load_overlay(path: str | Path) -> list[Edge]:
    """Read an edge overlay: `{"description"?: str, "edges": [{"source", "target", "relationship",
    "confidence"?, "provenance"?, "evidence"?}, ...]}`.

    Args:
        path: JSON file. Unknown keys are ignored; `evidence` is documentation only.

    Returns:
        Edges in file order; confidence defaults to "proposed", provenance to
        `overlay:<file name>`.

    Raises:
        ValueError: invalid JSON, wrong shape, a missing/empty field, an unknown relationship,
        or an edge from a skill to itself.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{p}: invalid JSON: {e}") from None
    if not isinstance(data, dict) or not isinstance(data.get("edges"), list):
        raise ValueError(f"{p}: expected an object with an 'edges' list")
    if not isinstance(data.get("description", ""), str):
        raise ValueError(f"{p}: 'description' must be a string")
    default_provenance = f"overlay:{p.name}"
    out: list[Edge] = []
    for i, raw in enumerate(data["edges"]):
        where = f"{p}: edges[{i}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: expected an object")
        for name in ("source", "target", "relationship"):
            if not isinstance(raw.get(name), str) or not raw[name].strip():
                raise ValueError(f"{where}: missing or empty '{name}'")
        for name in ("confidence", "provenance"):
            if name in raw and (not isinstance(raw[name], str) or not raw[name].strip()):
                raise ValueError(f"{where}: '{name}' must be a non-empty string")
        _check_relationship(raw["relationship"], where)
        if raw["source"] == raw["target"]:
            raise ValueError(f"{where}: '{raw['source']}' cannot relate to itself")
        out.append(Edge(raw["source"], raw["target"], raw["relationship"],
                        confidence=raw.get("confidence", "proposed"),
                        provenance=raw.get("provenance", default_provenance)))
    return out


def relations(g: nx.DiGraph, slug: str, relationship: str | None = None) -> list[Edge]:
    """Every typed edge touching `slug` (either end), optionally of one relationship.

    Returns:
        Edges in the stored `g.graph["edges"]` order.

    Raises:
        KeyError: unknown slug. ValueError: unknown relationship.
    """
    if slug not in g:
        raise KeyError(slug)
    if relationship is not None:
        _check_relationship(relationship, "relations()")
    return [e for e in g.graph.get("edges", [])
            if slug in (e.source, e.target) and relationship in (None, e.relationship)]


def ordering_graph(g: nx.DiGraph, nodes: Iterable[str],
                   soft: bool = True) -> tuple[nx.DiGraph, list[Edge]]:
    """The ordering constraints among `nodes`: hard requires, plus acyclic soft edges.

    Args:
        g: graph from `build_graph`.
        nodes: slugs to order; slugs not in `g` become isolated nodes.
        soft: also add `recommended_before` edges among `nodes`, one at a time in stored order,
            skipping any that would create a cycle.

    Returns:
        (DiGraph over `nodes` with `kind` edge attrs, soft edges that were dropped). A hard
        requires cycle is left in place for the caller to report.
    """
    import networkx as nx
    keep = set(nodes)
    sub = nx.DiGraph()
    sub.add_nodes_from(sorted(keep))
    sub.add_edges_from(sorted((u, v) for u, v, d in g.edges(data=True)
                              if d.get("kind") == "requires" and u in keep and v in keep),
                       kind="requires")
    dropped: list[Edge] = []
    if soft:
        for e in g.graph.get("edges", []):
            if e.relationship != "recommended_before" or e.source not in keep or e.target not in keep:
                continue
            if sub.has_edge(e.source, e.target):
                continue                                    # already ordered by requires
            if e.source == e.target or nx.has_path(sub, e.target, e.source):
                dropped.append(e)
                continue
            sub.add_edge(e.source, e.target, kind="recommended_before")
    return sub, dropped


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
