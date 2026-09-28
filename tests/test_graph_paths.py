from sie.models import Skill
from sie.graph.build import build_graph, find_cycles
from sie.graph.paths import learning_path


def _corpus():
    return [
        Skill(slug="math", skill_type="reference", domain="foundations", level="beginner"),
        Skill(slug="ml", skill_type="workflow", domain="classical", level="intermediate", requires=["math"]),
        Skill(slug="dl", skill_type="workflow", domain="deep", level="advanced", requires=["ml"]),
    ]


def test_learning_path_is_ordered():
    g = build_graph(_corpus())
    lp = learning_path(g, "dl")
    assert lp["path"].index("math") < lp["path"].index("ml") < lp["path"].index("dl")


def test_no_cycles():
    assert find_cycles(build_graph(_corpus())) == []
