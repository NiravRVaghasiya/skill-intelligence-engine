"""Typed relationships, provenance on edges, overlays, soft ordering and explained learning paths."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from sie.graph.build import (RELATION_SPECS, build_graph, find_cycles, load_overlay, ordering_graph,
                             relations)
from sie.graph.paths import (conflicts_of, learning_path, order_skills, render_learning_path,
                             see_also, superseded_by)
from sie.ingest import load_corpus
from sie.models import RELATIONSHIPS, Edge, Skill

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"
PROPOSED = ROOT / "docs" / "proposed_requires.edges.json"
ALL_DECLARED = ["requires", "related", "conflicts"]


def _s(slug, level="intermediate", declared=ALL_DECLARED, **kw):
    """A skill that declares its requires/related/conflicts keys, like every real corpus skill."""
    return Skill(slug=slug, skill_type="workflow", domain="d", level=level, declared=list(declared),
                 source="corp", source_version="v1", path=f"{slug}/SKILL.md", **kw)


def _step(lp, slug):
    return next(s for s in lp["steps"] if s["skill"] == slug)


def _write_overlay(tmp_path, edges, name="extra.edges.json", **top):
    p = tmp_path / name
    p.write_text(json.dumps({"edges": edges, **top}), encoding="utf-8")
    return p


# --- required learning-path cases -----------------------------------------------------------

def test_simple_chain_is_ordered_and_explained():
    g = build_graph([_s("a"), _s("b", requires=["a"]), _s("c", requires=["b"])])
    lp = learning_path(g, "c")
    assert lp["path"] == ["a", "b", "c"]
    assert [(s["skill"], s["position"], s["relation"], s["depth"], s["required_by"]) for s in lp["steps"]] == [
        ("a", 1, "transitive", 2, ["b"]), ("b", 2, "direct", 1, ["c"]), ("c", 3, "target", 0, [])]
    assert _step(lp, "a")["reason"] == "prerequisite of b, which leads to c (transitive)"
    assert _step(lp, "b")["reason"] == "direct prerequisite of c"
    assert _step(lp, "c")["reason"] == "target"
    assert lp["direct"] == ["b"] and lp["transitive"] == ["a"]
    assert lp["complete"] is True and lp["notes"] == []


def test_branching_target_with_two_independent_prerequisites():
    g = build_graph([_s("y"), _s("x", "beginner"), _s("t", requires=["y", "x"]), _s("other")])
    lp = learning_path(g, "t")
    assert lp["path"] == ["x", "y", "t"]                      # beginner first, then slug
    assert lp["direct"] == ["x", "y"] and lp["transitive"] == []
    assert all(_step(lp, s)["depth"] == 1 and _step(lp, s)["required_by"] == ["t"] for s in "xy")


def test_converging_diamond_lists_shared_prerequisite_once_before_both():
    g = build_graph([_s("a"), _s("b", requires=["a"]), _s("c", requires=["a"]), _s("t", requires=["b", "c"])])
    lp = learning_path(g, "t")
    assert lp["path"].count("a") == 1 and lp["path"][0] == "a" and lp["path"][-1] == "t"
    a = _step(lp, "a")
    assert a["relation"] == "transitive" and a["depth"] == 2 and a["required_by"] == ["b", "c"]
    assert a["reason"] == "prerequisite of b, c, which lead to t (transitive)"
    assert a["provenance"] == ["corp@v1:b/SKILL.md#requires", "corp@v1:c/SKILL.md#requires"]


def test_direct_prerequisite_also_required_on_the_way():
    g = build_graph([_s("a"), _s("b", requires=["a"]), _s("t", requires=["a", "b"])])
    lp = learning_path(g, "t")
    a = _step(lp, "a")
    assert lp["path"] == ["a", "b", "t"]
    assert a["relation"] == "direct" and a["depth"] == 1 and a["required_by"] == ["b", "t"]
    assert a["reason"] == "direct prerequisite of t (also required by b)"


def test_cycle_raises_with_the_cycle_spelled_out():
    g = build_graph([_s("a", requires=["b"]), _s("b", requires=["a"]), _s("c", requires=["a"])])
    assert find_cycles(g) == [["a", "b"]]
    with pytest.raises(ValueError, match=r"^requires cycle: a -> b -> a$"):
        learning_path(g, "c")
    with pytest.raises(ValueError, match="requires cycle"):
        order_skills(g, ["a", "b"])
    assert order_skills(g, ["b", "c"]) == ["b", "c"]          # the cycle is outside these nodes


def test_missing_dependency_is_noted_never_fabricated():
    g = build_graph([_s("a"), _s("t", requires=["a", "ghost"])])
    lp = learning_path(g, "t")
    assert lp["path"] == ["a", "t"] and "ghost" not in g
    assert lp["missing_prerequisites"] == [["t", "ghost"]]
    assert lp["notes"] == ["t requires 'ghost', which is not in the corpus; it is omitted"]
    assert lp["complete"] is False
    assert all(s["skill"] != "ghost" for s in lp["steps"]) and "ghost" not in lp["direct"]


def test_unrelated_skill_path_is_just_the_target():
    g = build_graph([_s("t"), _s("u1", requires=["u2"]), _s("u2"), _s("bare", declared=[])])
    lp = learning_path(g, "t")
    assert lp["path"] == ["t"] and lp["direct"] == lp["transitive"] == lp["recommended"] == []
    assert lp["notes"] == ["t declares no prerequisites"] and lp["complete"] is True
    bare = learning_path(g, "bare")
    assert bare["path"] == ["bare"]
    assert bare["notes"] == ["bare does not declare prerequisites; the path may be incomplete"]
    assert bare["complete"] is False


def test_duplicate_dependency_gives_one_edge_and_one_step():
    g = build_graph([_s("a"), _s("t", requires=["a", "a"], related=["a", "a"])])
    assert [(e.source, e.target, e.relationship) for e in g.graph["edges"]] == [
        ("a", "t", "requires"), ("t", "a", "related")]
    lp = learning_path(g, "t")
    assert lp["path"] == ["a", "t"] and _step(lp, "a")["provenance"] == ["corp@v1:t/SKILL.md#requires"]


def test_code_built_skill_listing_requires_counts_as_declared():
    g = build_graph([Skill("a", "workflow", "d", "beginner"),
                     Skill("t", "workflow", "d", "intermediate", requires=["a"])])
    assert g.nodes["t"]["declared"] == ["requires"] and g.nodes["a"]["declared"] == []
    lp = learning_path(g, "t")
    assert lp["notes"] == ["a does not declare prerequisites; the path may be incomplete"]
    assert lp["complete"] is False


# --- typed edges, provenance, symmetric relations ------------------------------------------

def test_every_relationship_becomes_a_typed_edge_in_the_documented_direction():
    kw = {r: ["r"] for r in RELATIONSHIPS}
    g = build_graph([_s("r"), _s("x", declared=RELATIONSHIPS, **kw)])
    got = {e.relationship: (e.source, e.target) for e in g.graph["edges"]}
    assert set(got) == set(RELATIONSHIPS) and set(RELATION_SPECS) == set(RELATIONSHIPS)
    for rel, (src, dst) in got.items():
        expected = ("r", "x") if RELATION_SPECS[rel].declared_as == "target" else ("x", "r")
        assert (src, dst) == expected, rel
    assert [e.relationship for e in g.graph["edges"]] == list(RELATIONSHIPS)   # stored order
    assert all(e.confidence == "declared" and e.provenance == f"corp@v1:x/SKILL.md#{e.relationship}"
               for e in g.graph["edges"])
    assert [r for r, spec in RELATION_SPECS.items() if spec.ordering != "none"] == [
        "requires", "recommended_before"]
    assert RELATION_SPECS["requires"].ordering == "hard"
    assert RELATION_SPECS["recommended_before"].ordering == "soft"
    assert {r for r, spec in RELATION_SPECS.items() if spec.symmetric} == {"conflicts", "alternative_to"}
    # only `requires` is a prerequisite: nothing else pulls r onto x's path
    g2 = build_graph([_s("r"), _s("x", declared=RELATIONSHIPS, **{k: v for k, v in kw.items() if k != "requires"})])
    assert learning_path(g2, "x")["path"] == ["x"]


def test_edges_are_sorted_and_node_attrs_carry_provenance():
    skills = [_s("b", requires=["a"], related=["c"]), _s("a", conflicts=["c"]), _s("c", requires=["a"])]
    g = build_graph(skills)
    assert [(e.relationship, e.source, e.target) for e in g.graph["edges"]] == [
        ("requires", "a", "b"), ("requires", "a", "c"), ("related", "b", "c"), ("conflicts", "a", "c")]
    n = g.nodes["b"]
    assert (n["source"], n["source_version"], n["path"], n["name"]) == ("corp", "v1", "b/SKILL.md", "b")
    assert n["declared"] == ["requires", "related", "conflicts"]
    for key in ("content_hash", "updated_at", "recommended_before", "alternative_to", "specializes",
                "supersedes"):
        assert key in n
    assert g.graph["overlays"] == [] and g.graph["dangling"] == []


def test_dangling_refs_cover_every_relationship_kind():
    g = build_graph([_s("a", supersedes=["old"], alternative_to=["alt"], requires=["x", "x"])])
    assert g.graph["dangling"] == [("a", "requires", "x"), ("a", "alternative_to", "alt"),
                                   ("a", "supersedes", "old")]
    assert g.graph["edges"] == [] and set(g.nodes) == {"a"}


def test_symmetric_relations_are_visible_from_both_ends():
    g = build_graph([_s("a", conflicts=["b"], alternative_to=["c"]), _s("b"), _s("c")])
    assert conflicts_of(g, "a") == {"b"} and conflicts_of(g, "b") == {"a"}
    alt = Edge("a", "c", "alternative_to", "declared", "corp@v1:a/SKILL.md#alternative_to")
    assert relations(g, "c", "alternative_to") == [alt] == relations(g, "a", "alternative_to")
    assert relations(g, "b") == [Edge("a", "b", "conflicts", "declared", "corp@v1:a/SKILL.md#conflicts")]
    assert [e.relationship for e in relations(g, "a")] == ["conflicts", "alternative_to"]


def test_relations_rejects_unknown_slug_and_relationship():
    g = build_graph([_s("a")])
    assert relations(g, "a") == []
    with pytest.raises(KeyError):
        relations(g, "nope")
    with pytest.raises(ValueError, match="unknown relationship 'prereq'"):
        relations(g, "a", "prereq")


def test_superseded_path_member_is_noted():
    g = build_graph([_s("old"), _s("t", requires=["old"]), _s("new", supersedes=["old"]),
                     _s("newer", supersedes=["old"])])
    assert superseded_by(g, "old") == ["new", "newer"] and superseded_by(g, "t") == []
    lp = learning_path(g, "t")
    assert lp["path"] == ["old", "t"]                         # supersedes never changes membership
    assert lp["notes"] == ["old is superseded by new, newer"]


# --- overlays -------------------------------------------------------------------------------

def test_load_overlay_defaults_and_passthrough(tmp_path):
    p = _write_overlay(tmp_path, [
        {"source": "a", "target": "t", "relationship": "requires", "evidence": "quote"},
        {"source": "t", "target": "b", "relationship": "related", "confidence": "reviewed",
         "provenance": "PR #7", "extra": 1}], description="demo")
    assert load_overlay(p) == [Edge("a", "t", "requires", "proposed", "overlay:extra.edges.json"),
                               Edge("t", "b", "related", "reviewed", "PR #7")]


@pytest.mark.parametrize("payload,match", [
    ({"edges": [{"source": "a", "target": "t", "relationship": "prereq"}]}, "unknown relationship 'prereq'"),
    ({"edges": [{"source": "a", "relationship": "requires"}]}, "missing or empty 'target'"),
    ({"edges": [{"source": "", "target": "t", "relationship": "requires"}]}, "missing or empty 'source'"),
    ({"edges": [{"source": "a", "target": "a", "relationship": "requires"}]}, "cannot relate to itself"),
    ({"edges": [{"source": "a", "target": "t", "relationship": "requires", "confidence": 3}]},
     "'confidence' must be a non-empty string"),
    ({"edges": ["a -> t"]}, r"edges\[0\]: expected an object"),
    ({"edges": {"a": "t"}}, "expected an object with an 'edges' list"),
    ([], "expected an object with an 'edges' list"),
    ({"description": 1, "edges": []}, "'description' must be a string"),
])
def test_load_overlay_validates(tmp_path, payload, match):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_overlay(p)


def test_load_overlay_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_overlay(p)


def test_requires_overlay_changes_the_path_and_shows_it_is_proposed(tmp_path):
    skills = [_s("a"), _s("b"), _s("t", requires=["a"])]
    p = _write_overlay(tmp_path, [{"source": "b", "target": "t", "relationship": "requires"},
                                  {"source": "a", "target": "t", "relationship": "requires"}])
    assert learning_path(build_graph(skills), "t")["path"] == ["a", "t"]          # not applied by default
    g = build_graph(skills, overlays=load_overlay(p))
    lp = learning_path(g, "t")
    assert lp["path"] == ["a", "b", "t"] and lp["direct"] == ["a", "b"]
    assert _step(lp, "b")["provenance"] == ["overlay:extra.edges.json [proposed]"]
    assert _step(lp, "b")["reason"] == "direct prerequisite of t (proposed)"
    assert _step(lp, "a")["provenance"] == ["corp@v1:t/SKILL.md#requires"]     # frontmatter wins
    assert g.edges["b", "t"]["kind"] == "requires"
    assert g.graph["overlays"] == ["overlay:extra.edges.json"]
    assert [e.confidence for e in relations(g, "t", "requires")] == ["declared", "proposed"]


def test_overlay_edges_with_unknown_endpoints_are_dangling_not_nodes():
    overlays = [Edge("ghost", "t", "requires", "proposed", "ov"), Edge("t", "nobody", "related", "proposed", "ov")]
    g = build_graph([_s("t")], overlays=overlays)
    assert set(g.nodes) == {"t"} and g.graph["edges"] == []
    assert g.graph["dangling"] == [("t", "requires", "ghost"), ("t", "related", "nobody")]
    lp = learning_path(g, "t")
    assert lp["missing_prerequisites"] == [["t", "ghost"]] and lp["complete"] is False


def test_overlay_with_unknown_relationship_is_rejected():
    with pytest.raises(ValueError, match="unknown relationship 'prereq'"):
        build_graph([_s("a"), _s("b")], overlays=[Edge("a", "b", "prereq")])


def test_overlay_conflicts_and_related_are_honored_in_lookups():
    ov = [Edge("a", "b", "conflicts", "proposed", "ov"), Edge("a", "c", "related", "proposed", "ov")]
    g = build_graph([_s("a", related=["b"]), _s("b"), _s("c")], overlays=ov)
    assert conflicts_of(g, "b") == {"a"} and see_also(g, "a") == ["b", "c"]
    assert learning_path(g, "a")["related"] == ["c"] and learning_path(g, "a")["conflicts"] == ["b"]


# --- soft ordering (recommended_before) -----------------------------------------------------

def test_soft_edges_order_path_members_but_never_add_members():
    # t requires a, z; z is recommended before a (declared on a); outside is recommended before t
    g = build_graph([_s("a", recommended_before=["z"]), _s("z"), _s("outside"),
                     _s("t", requires=["a", "z"], recommended_before=["outside"])])
    lp = learning_path(g, "t")
    assert lp["path"] == ["z", "a", "t"]                      # soft edge overrides the slug tie-break
    assert order_skills(g, ["a", "z", "t"], soft=False) == ["a", "z", "t"]
    assert "outside" not in lp["path"]
    assert lp["recommended"] == [{"skill": "outside", "before": "t",
                                  "provenance": "corp@v1:t/SKILL.md#recommended_before"}]
    assert "  recommended before t: outside" in render_learning_path(g, lp).splitlines()


def test_soft_edge_that_would_create_a_cycle_is_dropped_and_noted():
    g = build_graph([_s("a", recommended_before=["b"]), _s("b", requires=["a"]), _s("t", requires=["b"])])
    sub, dropped = ordering_graph(g, ["a", "b", "t"])
    assert dropped == [Edge("b", "a", "recommended_before", "declared", "corp@v1:a/SKILL.md#recommended_before")]
    assert not sub.has_edge("b", "a")
    lp = learning_path(g, "t")
    assert lp["path"] == ["a", "b", "t"]
    assert lp["notes"] == ["ignored recommended_before b -> a: it would create an ordering cycle"]


def test_soft_cycle_among_soft_edges_drops_the_later_edge():
    g = build_graph([_s("a", recommended_before=["b"]), _s("b", recommended_before=["a"])])
    sub, dropped = ordering_graph(g, ["a", "b"])
    assert [(u, v, d["kind"]) for u, v, d in sub.edges(data=True)] == [("a", "b", "recommended_before")]
    assert [(e.source, e.target) for e in dropped] == [("b", "a")]
    assert order_skills(g, ["b", "a"]) == ["a", "b"]


def test_include_recommended_inserts_soft_prerequisites_without_their_prerequisites():
    g = build_graph([_s("base"), _s("r", requires=["base"]), _s("a"), _s("t", requires=["a"], recommended_before=["r"])])
    default = learning_path(g, "t")
    assert default["path"] == ["a", "t"] and [x["skill"] for x in default["recommended"]] == ["r"]
    lp = learning_path(g, "t", include_recommended=True)
    assert lp["path"] == ["a", "r", "t"] and "base" not in lp["path"]
    r = _step(lp, "r")
    assert (r["relation"], r["depth"], r["required_by"]) == ("recommended", None, [])
    assert r["reason"] == "recommended before t; not a prerequisite"
    assert r["provenance"] == ["corp@v1:t/SKILL.md#recommended_before"]
    assert lp["notes"] == ["r is recommended, but its own prerequisites are not included: base"]
    assert lp["complete"] is False and lp["direct"] == ["a"]
    assert lp["recommended"] == default["recommended"]
    text = render_learning_path(g, lp).splitlines()
    assert text[0] == "[path] learning path to t (3 steps, `requires` + `recommended_before` edges)"
    assert text[2].endswith("  (recommended)")
    json.dumps(lp)                                            # API-safe (depth None -> null)


def test_include_recommended_leaves_off_a_skill_whose_soft_edge_is_dropped_by_a_soft_cycle():
    # x lists zz and zz lists x: (x -> zz) is stored first and kept, so (zz -> x), the edge that
    # made zz a member, is dropped; zz must not land after the target
    g = build_graph([_s("x", recommended_before=["zz"]), _s("zz", recommended_before=["x"])])
    lp = learning_path(g, "x", include_recommended=True)
    assert lp["path"] == ["x"] and lp["path"][-1] == "x"
    assert [s["skill"] for s in lp["steps"]] == ["x"]
    assert [r["skill"] for r in lp["recommended"]] == ["zz"]
    assert "zz is recommended before x, but that conflicts with other ordering; left off the path" \
        in lp["notes"]
    assert learning_path(g, "x")["path"] == ["x"]                     # default unchanged


def test_include_recommended_leaves_off_a_skill_whose_soft_edge_contradicts_requires():
    # r requires m, yet m recommends r before itself: r would sit after m and claim "before m"
    g = build_graph([_s("m", recommended_before=["r"]), _s("r", requires=["m"]), _s("t", requires=["m"])])
    lp = learning_path(g, "t", include_recommended=True)
    assert lp["path"] == ["m", "t"] and lp["path"][-1] == "t"
    assert all("recommended before" not in s["reason"] for s in lp["steps"])
    assert lp["recommended"] == [{"skill": "r", "before": "m",
                                  "provenance": "corp@v1:m/SKILL.md#recommended_before"}]
    assert lp["notes"] == ["r is recommended before m, but that conflicts with other ordering; "
                           "left off the path"]


def test_include_recommended_step_claims_only_the_soft_edges_it_honors():
    # r is recommended before a and before b, but r requires b: only "before a" survives
    g = build_graph([_s("a", recommended_before=["r"]), _s("b", recommended_before=["r"]),
                     _s("r", requires=["b"]), _s("t", requires=["a", "b"])])
    lp = learning_path(g, "t", include_recommended=True)
    assert lp["path"] == ["b", "r", "a", "t"]
    r = _step(lp, "r")
    assert r["reason"] == "recommended before a; not a prerequisite"
    assert r["provenance"] == ["corp@v1:a/SKILL.md#recommended_before"]
    assert lp["notes"] == ["ignored recommended_before r -> b: it would create an ordering cycle"]


def test_transitive_reason_names_only_hard_dependants():
    # a leads to t through b; the included recommended r also requires a, but does not lead to t
    overlay = [Edge("a", "r", "requires", confidence="proposed", provenance="docs/x.md")]
    g = build_graph([_s("a"), _s("b", requires=["a"]), _s("r"),
                     _s("t", requires=["b"], recommended_before=["r"])], overlays=overlay)
    lp = learning_path(g, "t", include_recommended=True)
    assert lp["path"] == ["a", "b", "r", "t"]
    a = _step(lp, "a")
    assert (a["relation"], a["required_by"]) == ("transitive", ["b", "r"])
    assert a["reason"] == ("prerequisite of b, which leads to t (transitive); "
                           "also required by recommended r (proposed)")
    assert a["provenance"] == ["corp@v1:b/SKILL.md#requires", "docs/x.md [proposed]"]
    plain = learning_path(g, "t")
    assert _step(plain, "a")["reason"] == "prerequisite of b, which leads to t (transitive)"


# --- order_skills ---------------------------------------------------------------------------

def test_order_skills_priority_breaks_ties_but_never_beats_requires():
    g = build_graph([_s("a"), _s("b", requires=["a"]), _s("c", "beginner"), _s("d")])
    assert order_skills(g, ["a", "b", "c", "d"]) == ["c", "a", "b", "d"]          # level, then slug
    assert order_skills(g, ["a", "b", "c", "d"], priority={"d": 0, "b": 1, "a": 1, "c": 2}) == [
        "d", "a", "b", "c"]
    assert order_skills(g, ["a", "b"], priority={"b": 0, "a": 5}) == ["a", "b"]   # requires wins
    assert order_skills(g, ["d", "c"], priority={"d": 0}) == ["d", "c"]            # ranked before unranked


def test_order_skills_is_deterministic_and_tolerates_unknown_nodes():
    skills = [_s("a"), _s("b", requires=["a"]), _s("c", recommended_before=["b"]), _s("e", "advanced")]
    nodes = ["a", "b", "c", "e", "zz-not-in-graph"]
    expected = order_skills(build_graph(skills), nodes, priority={"e": 0})
    assert expected == ["e", "a", "b", "c", "zz-not-in-graph"]
    rng = random.Random(0)
    for _ in range(5):
        rng.shuffle(skills)
        rng.shuffle(nodes)
        assert order_skills(build_graph(skills), set(nodes), priority={"e": 0}) == expected


def test_graph_and_paths_do_not_depend_on_hash_seed():
    script = (
        "from sie.graph import build_graph, learning_path, order_skills\n"
        "from sie.models import Skill\n"
        "S = lambda s, **k: Skill(s, 'workflow', 'd', 'intermediate', declared=['requires'], **k)\n"
        "g = build_graph([S('a', requires=['b']), S('b', requires=['a']), S('p'), S('q'),\n"
        "                 S('r', requires=['p', 'q'], recommended_before=['q'], conflicts=['s']), S('s')])\n"
        "print(learning_path(g, 'r'), order_skills(g, {'p', 'q', 'r', 's'}), g.graph['edges'])\n"
        "try:\n    order_skills(g, {'a', 'b', 'p'})\nexcept ValueError as e:\n    print(e)\n")
    outs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True,
                                text=True, check=True).stdout)
    assert len(outs) == 1 and "requires cycle: a -> b -> a" in outs.pop()


def test_build_graph_is_independent_of_skill_order():
    skills = [_s("a", conflicts=["c"]), _s("b", requires=["a"], related=["c", "a"]), _s("c", requires=["a"]),
              _s("t", requires=["b", "c"], recommended_before=["d"]), _s("d")]
    ref = build_graph(skills)
    shuffled = build_graph(list(reversed(skills)))
    assert ref.graph["edges"] == shuffled.graph["edges"]
    assert learning_path(ref, "t") == learning_path(shuffled, "t")


# --- rendering ------------------------------------------------------------------------------

def test_render_learning_path_exact_output():
    g = build_graph([_s("a", conflicts=["x"]), _s("b", requires=["a", "ghost"]), _s("x"),
                     _s("t", requires=["b"], related=["x", "a"]), _s("new", supersedes=["a"])])
    assert render_learning_path(g, learning_path(g, "t")) == "\n".join([
        "[path] learning path to t (3 steps, hard `requires` edges only)",
        "  1. a                          [workflow/d, intermediate]",
        "  2. b                          [workflow/d, intermediate]",
        "  3. t                          [workflow/d, intermediate]  <- target",
        "  see also:  none",
        "  conflicts: x",
        "  WARNING prerequisites not in the corpus (typo?): b requires 'ghost'",
        "  why: a: prerequisite of b, which leads to t (transitive)",
        "  why: b: direct prerequisite of t",
        "  note: b requires 'ghost', which is not in the corpus; it is omitted",
        "  note: a is superseded by new",
    ])


# --- the real corpus ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_skills():
    return load_corpus(CORPUS)


@pytest.fixture(scope="module")
def real(real_skills):
    return build_graph(real_skills)


def test_real_edges_carry_corpus_provenance(real):
    [edge] = relations(real, "rag-evaluation", "requires")
    assert edge == Edge("rag-pipeline", "rag-evaluation", "requires", "declared",
                        "ml-ai-skills@8328c60:rag-evaluation/SKILL.md#requires")
    assert all(e.confidence == "declared" and e.provenance.startswith("ml-ai-skills@8328c60:")
               for e in real.graph["edges"])
    assert real.graph["overlays"] == [] and find_cycles(real) == []
    lp = learning_path(real, "rag-evaluation")
    assert _step(lp, "rag-pipeline")["provenance"] == [edge.provenance]
    assert lp["complete"] is True and lp["notes"] == []


def test_real_corpus_every_target_has_a_consistent_explained_path(real):
    for target in sorted(real.nodes):
        lp = learning_path(real, target)
        assert lp["path"][-1] == target and len(set(lp["path"])) == len(lp["path"])
        assert [s["skill"] for s in lp["steps"]] == lp["path"]
        assert [s["position"] for s in lp["steps"]] == list(range(1, len(lp["path"]) + 1))
        assert set(lp["direct"]) | set(lp["transitive"]) | {target} == set(lp["path"])
        assert lp["complete"] is True                         # every real skill declares `requires`
        assert lp["recommended"] == [] and json.dumps(lp)


def test_proposed_overlay_is_valid_and_acyclic_on_the_real_corpus(real_skills, real):
    edges = load_overlay(PROPOSED)
    assert len(edges) == 9
    assert all((e.relationship, e.confidence, e.provenance) == ("requires", "proposed", "docs/PROPOSED_REQUIRES.md")
               for e in edges)
    assert all(e.source in real and e.target in real for e in edges)
    raw = json.loads(PROPOSED.read_text(encoding="utf-8"))["edges"]
    assert all(r["evidence"].strip() for r in raw)
    g = build_graph(real_skills, overlays=edges)
    assert find_cycles(g) == [] and g.graph["dangling"] == []
    assert g.graph["overlays"] == ["docs/PROPOSED_REQUIRES.md"]
    level = {"beginner": 0, "intermediate": 1, "advanced": 2}
    assert all(level[g.nodes[e.source]["level"]] <= level[g.nodes[e.target]["level"]] for e in edges)


@pytest.mark.parametrize("target,path", [                     # docs/PROPOSED_REQUIRES.md table
    ("cnn-vision", ["neural-net-fundamentals", "cnn-vision"]),
    ("computer-vision", ["pytorch-patterns", "computer-vision"]),
    ("explainability", ["supervised-learning", "explainability"]),
    ("feature-engineering", ["data-preprocessing", "feature-engineering"]),
    ("hyperparameter-tuning", ["model-evaluation", "hyperparameter-tuning"]),
    ("model-optimization", ["pytorch-patterns", "model-optimization"]),
    ("rag-evaluation", ["llm-evaluation", "rag-pipeline", "rag-evaluation"]),
    ("rnn-sequence", ["neural-net-fundamentals", "rnn-sequence"]),
    ("training-deep-models", ["pytorch-patterns", "training-deep-models"]),
])
def test_proposed_overlay_paths_match_the_doc(real_skills, real, target, path):
    g = build_graph(real_skills, overlays=load_overlay(PROPOSED))
    lp = learning_path(g, target)
    assert lp["path"] == path
    assert "docs/PROPOSED_REQUIRES.md [proposed]" in _step(lp, path[0])["provenance"]
    assert learning_path(real, target)["path"] != path        # the default graph is untouched
