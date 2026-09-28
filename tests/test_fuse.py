from sie.models import Hit
from sie.index.fuse import best_per_skill, reciprocal_rank_fusion


def _h(slug, section="S", score=0.0):
    return Hit(slug, score, section=section, chunk_id=f"{slug}::{section}")


def test_rrf_prefers_consensus():
    a = [Hit("alpha", 0.9), Hit("beta", 0.8)]
    b = [Hit("alpha", 0.7), Hit("gamma", 0.6)]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].skill_slug == "alpha"   # appears high in both


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_order_and_dedups():
    fused = reciprocal_rank_fusion([[_h("a", "x"), _h("a", "y"), _h("b"), _h("c")]])
    assert [h.skill_slug for h in fused] == ["a", "b", "c"]


def test_many_chunks_of_one_skill_do_not_inflate_it():
    # "many" has 3 chunks in list 1; "one" is #2 in list 1 and #1 in list 2
    l1 = [_h("many", "p"), _h("one"), _h("many", "q"), _h("many", "r")]
    l2 = [_h("one"), _h("many", "p")]
    fused = reciprocal_rank_fusion([l1, l2])
    assert fused[0].skill_slug in {"one", "many"}
    assert abs(fused[0].score - fused[1].score) < 1e-12      # symmetric ranks -> equal scores
    assert [h.skill_slug for h in fused] == ["many", "one"]   # tie broken by slug


def test_ties_break_deterministically_by_slug():
    fused = reciprocal_rank_fusion([[_h("zeta")], [_h("alpha")]])
    assert [h.skill_slug for h in fused] == ["alpha", "zeta"]


def test_fused_hit_keeps_best_ranked_section_and_does_not_mutate():
    l1 = [_h("x", "Gotchas", 0.3), _h("y", "Card")]
    l2 = [_h("y", "Workflow"), _h("x", "Card")]
    before = [(h.skill_slug, h.score) for h in l1 + l2]
    fused = {h.skill_slug: h for h in reciprocal_rank_fusion([l1, l2])}
    assert fused["x"].section == "Gotchas" and fused["y"].section == "Workflow"
    assert [(h.skill_slug, h.score) for h in l1 + l2] == before


def test_rrf_score_formula_and_k():
    fused = reciprocal_rank_fusion([[_h("a")], [_h("b"), _h("a")]], k=10)
    scores = {h.skill_slug: h.score for h in fused}
    assert abs(scores["a"] - (1 / 11 + 1 / 12)) < 1e-12 and abs(scores["b"] - 1 / 11) < 1e-12


def test_best_per_skill():
    assert [h.section for h in best_per_skill([_h("a", "1"), _h("b"), _h("a", "2")])] == ["1", "S"]
