"""Collision-case report logic on synthetic runs (the real run needs git history + an index)."""
from eval.collision import _correct_in_top3, _scored_top1, render


def _run(kw, sie):
    return {"commit": "abc1234", "n_skills": 35,
            "keyword": {"intent": "workflow",
                        "route": [{"slug": s, "score": sc, "reasons": "r"} for s, sc in kw]},
            "sie": {"ranking": "rrf", "route": [{"slug": s, "score": 0.03, "section": "Card"} for s in sie]}}


def test_scored_top1_skips_zero_score_prerequisites():
    route = [{"slug": "agents-and-tools", "score": 0.0}, {"slug": "agent-evaluation", "score": 9.0}]
    assert _scored_top1(route) == "agent-evaluation"


def test_correct_in_top3_counts_only_first_three():
    route = [{"slug": s, "score": 5 - i} for i, s in
             enumerate(["computer-vision", "nlp-tasks", "explainability", "model-evaluation"])]
    assert _correct_in_top3(route, by_score=True) == "0/2"
    assert _correct_in_top3(route[::-1], by_score=False).startswith("1/2")


def test_render_marks_wrong_and_correct_top1():
    runs = {"historical": _run([("computer-vision", 7.0), ("model-evaluation", 6.0)],
                               ["model-evaluation", "supervised-learning"]),
            "head": _run([("supervised-learning", 18.0)], ["model-evaluation"])}
    runs["head"]["n_skills"] = 38
    md = render(runs)
    assert "`computer-vision` (**wrong**)" in md
    assert md.count("SIE top-1: `model-evaluation` (correct)") == 2
    assert "ml-ai-skills@abc1234 (35 skills" in md and "data/skills (38 skills" in md
