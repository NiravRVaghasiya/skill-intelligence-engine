"""GraphRAG: typed skill graph (requires, recommended_before, related, conflicts,
alternative_to, specializes, supersedes) + explained, dependency-ordered learning paths.

`build` turns Skills (and optional edge overlays) into a DiGraph whose `g.graph["edges"]` holds
every typed Edge with its provenance; `paths` orders skills and explains learning paths.
networkx is imported lazily inside the functions.
"""
from .build import (EDGE_KINDS, RELATION_SPECS, RelationSpec, build_graph, find_cycles,
                    load_overlay, ordering_graph, relations, requires_graph)
from .paths import (conflicts_of, learning_path, order_skills, prerequisite_closure,
                    render_learning_path, see_also, superseded_by)

__all__ = ["EDGE_KINDS", "RELATION_SPECS", "RelationSpec", "build_graph", "find_cycles",
           "load_overlay", "ordering_graph", "relations", "requires_graph", "conflicts_of",
           "learning_path", "order_skills", "prerequisite_closure", "render_learning_path",
           "see_also", "superseded_by"]
