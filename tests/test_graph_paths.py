from pathlib import Path

import pytest

from sie.ingest import load_corpus
from sie.models import Skill
from sie.graph.build import build_graph, find_cycles
from sie.graph.paths import learning_path, prerequisite_closure

CORPUS = Path(__file__).resolve().parents[1] / "data" / "skills"


def _corpus():
    return [
        Skill(slug="math", skill_type="reference", domain="foundations", level="beginner"),
        Skill(slug="ml", skill_type="workflow", domain="classical", level="intermediate", requires=["math"]),
        Skill(slug="dl", skill_type="workflow", domain="deep", level="advanced", requires=["ml"]),
    ]


def _s(slug, level="intermediate", **kw):
    return Skill(slug=slug, skill_type="workflow", domain="d", level=level, **kw)


def test_learning_path_is_ordered():
    g = build_graph(_corpus())
    lp = learning_path(g, "dl")
    assert lp["path"].index("math") < lp["path"].index("ml") < lp["path"].index("dl")


def test_no_cycles():
    assert find_cycles(build_graph(_corpus())) == []


def test_diamond_ties_break_by_level_then_slug():
    g = build_graph([_s("a", "beginner"), _s("z", "beginner"), _s("b", "intermediate", requires=["a"]),
                     _s("c", "beginner", requires=["a"]), _s("t", requires=["b", "c", "z"])])
    assert learning_path(g, "t")["path"] == ["a", "c", "z", "b", "t"]


def test_requires_edge_survives_related_and_conflict_on_same_pair():
    g = build_graph([_s("a", related=["b"], conflicts=["b"]), _s("b", requires=["a"])])
    assert g.edges["a", "b"]["kind"] == "requires"
    assert learning_path(g, "b")["path"] == ["a", "b"]
    assert learning_path(g, "b")["path_conflicts"] == [["a", "b"]]


def test_conflicts_flagged_in_either_direction_and_across_path():
    g = build_graph([_s("pre", conflicts=["x"]), _s("t", requires=["pre"]), _s("x"), _s("y", conflicts=["t"])])
    lp = learning_path(g, "t")
    assert lp["conflicts"] == ["x", "y"] and lp["path_conflicts"] == []


def test_related_excludes_path_members_and_keeps_declared_order():
    g = build_graph([_s("p"), _s("q"), _s("r"), _s("t", requires=["p"], related=["r", "p", "q"])])
    assert learning_path(g, "t")["related"] == ["r", "q"]


def test_dangling_refs_do_not_create_phantom_nodes():
    g = build_graph([_s("a", requires=["ghost"], related=["nobody"])])
    assert set(g.nodes) == {"a"}
    assert g.graph["dangling"] == [("a", "requires", "ghost"), ("a", "related", "nobody")]


def test_unknown_target_raises():
    with pytest.raises(KeyError):
        learning_path(build_graph(_corpus()), "nope")
    with pytest.raises(KeyError):
        prerequisite_closure(build_graph(_corpus()), "nope")


def test_cycle_detected_and_blocks_ordering():
    g = build_graph([_s("a", requires=["b"]), _s("b", requires=["a"]), _s("c", requires=["a"])])
    assert find_cycles(g) == [["a", "b"]]
    with pytest.raises(ValueError, match="requires cycle"):
        learning_path(g, "c")


# --- the real ml-ai-skills corpus -------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    return build_graph(load_corpus(CORPUS))


def test_real_corpus_has_no_cycles_or_dangling_refs(real):
    assert find_cycles(real) == [] and real.graph["dangling"] == []
    assert real.number_of_nodes() == 38


@pytest.mark.parametrize("target,path", [
    ("rag-evaluation", ["rag-pipeline", "rag-evaluation"]),
    ("agent-evaluation", ["agents-and-tools", "agent-evaluation"]),
    ("rag-pipeline", ["rag-pipeline"]),                 # no declared prerequisites
    ("fine-tuning-llms", ["fine-tuning-llms"]),
    ("training-deep-models", ["training-deep-models"]),
])
def test_real_learning_paths(real, target, path):
    lp = learning_path(real, target)
    assert lp["path"] == path and lp["path"][-1] == target
    assert lp["conflicts"] == [] and lp["path_conflicts"] == []


def test_real_see_also_comes_from_frontmatter(real):
    assert learning_path(real, "rag-evaluation")["related"] == ["llm-evaluation", "model-evaluation", "ai-ml-security"]
    assert "neural-net-fundamentals" in learning_path(real, "training-deep-models")["related"]


# --- `python -m sie.router --path <slug>` -------------------------------------------------

def test_path_cli_prints_ordered_path_see_also_and_conflicts(monkeypatch, capsys):
    import sys
    from sie.router import main
    monkeypatch.setattr(sys, "argv", ["sie.router", "--skills", str(CORPUS), "--path", "rag-evaluation"])
    main()
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[path] learning path to rag-evaluation (2 steps")
    assert out[1].split()[:2] == ["1.", "rag-pipeline"] and out[2].split()[:2] == ["2.", "rag-evaluation"]
    assert out[2].endswith("<- target")
    assert out[3] == "  see also:  llm-evaluation, model-evaluation, ai-ml-security"
    assert out[4] == "  conflicts: none"


def test_path_cli_unknown_slug_suggests_and_exits(monkeypatch, capsys):
    import sys
    from sie.router import main
    monkeypatch.setattr(sys, "argv", ["sie.router", "--skills", str(CORPUS), "--path", "rag-pipelin"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "unknown skill 'rag-pipelin'; did you mean rag-pipeline" in str(exc.value)


def test_format_learning_path_flags_conflicts_on_path(tmp_path):
    from sie.router import format_learning_path
    for slug, fm in {"a": "conflicts:\n  - b", "b": "requires:\n  - a", "x": "conflicts:\n  - b"}.items():
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "SKILL.md").write_text(
            f"---\ntype: workflow\ndomain: d\nlevel: intermediate\n{fm}\n---\n## Overview\nhi\n", encoding="utf-8")
    text = format_learning_path(str(tmp_path), "b")
    assert "conflicts: x" in text and "WARNING conflicting skills on the path: a <-> b" in text


def test_missing_prerequisite_is_reported_not_silently_dropped(tmp_path):
    from sie.router import format_learning_path
    g = build_graph([_s("a"), _s("t", requires=["a", "rag-pipline"])])
    assert learning_path(g, "t")["missing_prerequisites"] == [["t", "rag-pipline"]]
    for slug, fm in {"a": "", "t": "requires:\n  - a\n  - rag-pipline"}.items():
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "SKILL.md").write_text(
            f"---\ntype: workflow\ndomain: d\nlevel: intermediate\n{fm}\n---\n## Overview\nhi\n", encoding="utf-8")
    assert "WARNING prerequisites not in the corpus (typo?): t requires 'rag-pipline'" in \
        format_learning_path(str(tmp_path), "t")


def test_see_also_is_deduplicated_and_never_lists_a_conflict():
    g = build_graph([_s("a"), _s("x", conflicts=["t"]), _s("y"), _s("t", requires=["a"], related=["x", "y", "y", "a"])])
    lp = learning_path(g, "t")
    assert lp["related"] == ["y"] and lp["conflicts"] == ["x"]


# --- sie.graph.paths.render_learning_path reproduces the CLI output, then explains ---------

@pytest.mark.parametrize("target", ["rag-evaluation", "agent-evaluation", "rag-pipeline", "training-deep-models"])
def test_render_learning_path_starts_with_the_cli_output(real, target):
    from sie.graph.paths import render_learning_path
    from sie.router import format_learning_path
    cli = format_learning_path(str(CORPUS), target).splitlines()
    rendered = render_learning_path(real, learning_path(real, target)).splitlines()
    assert rendered[:len(cli)] == cli
    assert all(line.startswith(("  why: ", "  note: ", "  recommended before ")) for line in rendered[len(cli):])


def test_render_learning_path_matches_cli_on_conflicts_and_missing_prereqs(tmp_path):
    from sie.graph.paths import render_learning_path
    from sie.router import format_learning_path
    for slug, fm in {"a": "conflicts:\n  - b", "b": "requires:\n  - a\n  - ghost", "x": "conflicts:\n  - b"}.items():
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "SKILL.md").write_text(
            f"---\ntype: workflow\ndomain: d\nlevel: intermediate\n{fm}\n---\n## Overview\nhi\n", encoding="utf-8")
    cli = format_learning_path(str(tmp_path), "b").splitlines()
    g = build_graph(load_corpus(tmp_path))
    assert render_learning_path(g, learning_path(g, "b")).splitlines() == cli   # CLI delegates
    assert cli[5:7] == ["  WARNING conflicting skills on the path: a <-> b",
                        "  WARNING prerequisites not in the corpus (typo?): b requires 'ghost'"]
    assert cli[7:] == ["  why: a: direct prerequisite of b",
                       "  note: a does not declare prerequisites; the path may be incomplete",
                       "  note: b requires 'ghost', which is not in the corpus; it is omitted"]


def test_new_learning_path_keys_on_real_paths(real):
    lp = learning_path(real, "agent-evaluation")
    assert lp["direct"] == ["agents-and-tools"] and lp["transitive"] == []
    assert [(s["skill"], s["relation"], s["depth"]) for s in lp["steps"]] == [
        ("agents-and-tools", "direct", 1), ("agent-evaluation", "target", 0)]
    assert lp["complete"] is True and lp["recommended"] == []
    assert learning_path(real, "fine-tuning-llms")["notes"] == ["fine-tuning-llms declares no prerequisites"]
