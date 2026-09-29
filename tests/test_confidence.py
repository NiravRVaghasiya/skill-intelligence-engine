"""Every rule of sie/confidence.py, in every retrieval mode, on hand-built rankings."""
import pytest

from sie.confidence import ACTIONS, DEFAULT_POLICY, LEVELS, ConfidencePolicy, assess, summary
from sie.models import MethodEvidence, RankedSkill

HYBRID, DENSE, BM25 = ["dense", "bm25"], ["dense"], ["bm25"]
SIGNAL_KEYS = {"methods", "leaders", "agreement", "support", "similarity", "top_similarity",
               "coverage", "margin", "reranked"}


def rs(slug, rank, score, **evidence):
    """RankedSkill found by the methods named in `evidence` (method=raw score)."""
    ev = [MethodEvidence(method=m, rank=rank, score=s) for m, s in evidence.items()]
    return RankedSkill(slug=slug, rank=rank, score=score, score_type="rrf",
                       methods=[m for m in evidence if m != "rerank"], evidence=ev)


def rrf(rank_a, rank_b=None):
    return 1 / (60 + rank_a) + (1 / (60 + rank_b) if rank_b else 0.0)


def test_policy_defaults_levels_and_actions():
    assert (DEFAULT_POLICY.min_similarity, DEFAULT_POLICY.min_coverage, DEFAULT_POLICY.min_margin) == (0.30, 0.40, 0.03)
    assert LEVELS == ("high", "medium", "low", "none")
    assert ACTIONS == {"high": "route", "medium": "clarify", "low": "clarify", "none": "abstain"}


def test_empty_results_abstain():
    c = assess([], HYBRID, {}, coverage=None, top_similarity=None)
    assert (c.level, c.action, c.ambiguous) == ("none", "abstain", False)
    assert c.reasons == ["no candidate skills retrieved"] and c.competitors == []
    assert c.signals == {"methods": HYBRID, "reranked": False}


# --- high / route ----------------------------------------------------------------------------

def test_hybrid_high_when_both_retrievers_agree_and_evidence_is_strong():
    results = [rs("a", 1, rrf(1, 1), dense=0.62, bm25=9.0), rs("b", 2, rrf(2, 2), dense=0.55, bm25=4.0)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "a"}, coverage=0.8, top_similarity=0.62)
    assert (c.level, c.action, c.ambiguous, c.competitors) == ("high", "route", False, [])
    assert c.reasons == ["dense and bm25 rank a first"]
    assert set(c.signals) == SIGNAL_KEYS
    assert c.signals["agreement"] == 2 and c.signals["support"] == 2
    assert c.signals["similarity"] == 0.62 and c.signals["coverage"] == 0.8
    assert c.signals["leaders"] == {"dense": "a", "bm25": "a"}
    assert c.signals["margin"] == pytest.approx((rrf(1, 1) - rrf(2, 2)) / rrf(1, 1), abs=1e-4)


def test_dense_only_high():
    results = [rs("a", 1, rrf(1), dense=0.70), rs("b", 2, rrf(2), dense=0.50)]
    c = assess(results, DENSE, {"dense": "a"}, coverage=None, top_similarity=0.70)
    assert (c.level, c.action) == ("high", "route") and c.reasons == ["dense ranks a first"]
    assert c.signals["coverage"] is None and c.signals["margin"] == pytest.approx(0.2857, abs=1e-4)


def test_bm25_only_high_uses_coverage_as_strength():
    results = [rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=5.0)]
    c = assess(results, BM25, {"bm25": "a"}, coverage=0.9, top_similarity=None)
    assert (c.level, c.action) == ("high", "route") and c.reasons == ["bm25 ranks a first"]
    assert c.signals["similarity"] is None and c.signals["top_similarity"] is None


def test_bm25_only_without_coverage_can_still_be_high():
    results = [rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=5.0)]
    c = assess(results, BM25, {"bm25": "a"}, coverage=None, top_similarity=None)
    assert c.level == "high"                    # no strength signal at all: nothing is weak


def test_reranked_high_and_absolute_logit_margin():
    results = [rs("a", 1, 4.5, dense=0.6, bm25=8.0, rerank=4.5),
               rs("b", 2, -1.0, dense=0.5, bm25=6.0, rerank=-1.0)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "a"}, coverage=0.7, top_similarity=0.6,
               reranked=True)
    assert (c.level, c.action) == ("high", "route")
    assert c.reasons == ["cross-encoder ranks a first"]
    assert c.signals["margin"] == 5.5 and c.signals["reranked"] is True


def test_single_result_has_no_margin_and_can_be_high():
    c = assess([rs("a", 1, rrf(1), bm25=3.0)], BM25, {"bm25": "a"}, coverage=0.9)
    assert c.level == "high" and c.signals["margin"] is None


# --- none / abstain ---------------------------------------------------------------------------

def test_hybrid_none_only_when_every_signal_is_weak():
    results = [rs("a", 1, rrf(1, 1), dense=0.12, bm25=1.0), rs("b", 2, rrf(2, 2), dense=0.10, bm25=0.5)]
    leaders = {"dense": "a", "bm25": "a"}
    c = assess(results, HYBRID, leaders, coverage=0.1, top_similarity=0.12)
    assert (c.level, c.action, c.ambiguous) == ("none", "abstain", False)
    assert c.reasons == ["no sufficiently good match: best dense similarity 0.12 < 0.30; only 10% "
                         "of the query's term weight matches the best candidate (< 40%)"]
    assert set(c.signals) == SIGNAL_KEYS
    # one strong signal is enough to avoid abstaining
    assert assess(results, HYBRID, leaders, coverage=0.9, top_similarity=0.12).level != "none"
    assert assess(results, HYBRID, leaders, coverage=0.1, top_similarity=0.5).level != "none"


def test_top_similarity_is_any_candidate_not_the_top_result():
    # weak top result, but *something* in the corpus resembles the query: not "none"
    results = [rs("a", 1, rrf(1, 1), dense=0.12, bm25=1.0), rs("b", 2, rrf(2, 2), dense=0.45, bm25=0.5)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "a"}, coverage=0.1, top_similarity=0.45)
    assert c.level == "low" and "dense similarity 0.12 < 0.30" in c.reasons


def test_dense_only_none_without_coverage():
    results = [rs("a", 1, rrf(1), dense=0.2), rs("b", 2, rrf(2), dense=0.1)]
    c = assess(results, DENSE, {"dense": "a"}, coverage=None, top_similarity=0.2)
    assert c.level == "none"
    assert c.reasons == ["no sufficiently good match: best dense similarity 0.20 < 0.30"]


def test_dense_only_with_coverage_needs_both_weak_to_abstain():
    results = [rs("a", 1, rrf(1), dense=0.2), rs("b", 2, rrf(2), dense=0.1)]
    assert assess(results, DENSE, {"dense": "a"}, coverage=0.2, top_similarity=0.2).level == "none"
    strong_lexical = assess(results, DENSE, {"dense": "a"}, coverage=0.9, top_similarity=0.2)
    assert strong_lexical.level == "low"        # still weak dense evidence for the top result


def test_dense_ran_but_returned_nothing_counts_as_weak_dense():
    results = [rs("a", 1, rrf(1), bm25=2.0)]
    c = assess(results, HYBRID, {"bm25": "a"}, coverage=0.1, top_similarity=None)
    assert c.level == "none" and "best dense similarity none < 0.30" in c.reasons[0]


def test_bm25_only_weak_coverage_is_low_never_none():
    # lexical coverage alone can't tell out-of-scope from a long in-scope query
    results = [rs("a", 1, rrf(1), bm25=2.7), rs("b", 2, rrf(2), bm25=1.0)]
    c = assess(results, BM25, {"bm25": "a"}, coverage=0.07, top_similarity=None)
    assert (c.level, c.action, c.ambiguous) == ("low", "clarify", False)
    assert c.reasons == ["bm25 ranks a first",
                         "only 7% of the query's term weight matches a (< 40%)"]


def test_thresholds_come_from_the_policy():
    results = [rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=5.0)]
    strict = ConfidencePolicy(min_coverage=0.95)
    assert assess(results, BM25, {"bm25": "a"}, coverage=0.9).level == "high"
    assert assess(results, BM25, {"bm25": "a"}, coverage=0.9, policy=strict).level == "low"
    loose = ConfidencePolicy(min_similarity=0.05)
    dense = [rs("a", 1, rrf(1), dense=0.2), rs("b", 2, rrf(2), dense=0.1)]
    assert assess(dense, DENSE, {"dense": "a"}, top_similarity=0.2, policy=loose).level == "high"


# --- low / clarify ----------------------------------------------------------------------------

def test_hybrid_low_when_dense_did_not_retrieve_the_top_result():
    results = [rs("a", 1, rrf(1), bm25=9.0), rs("b", 2, rrf(2), dense=0.6)]
    c = assess(results, HYBRID, {"dense": "b", "bm25": "a"}, coverage=0.8, top_similarity=0.6)
    assert (c.level, c.action) == ("low", "clarify")
    assert "dense retrieval did not surface a" in c.reasons
    assert "not retrieved by dense" in c.reasons
    assert "dense ranks b first" in c.reasons
    assert c.competitors == ["b"] and c.ambiguous is True      # low can also be ambiguous
    assert c.signals["support"] == 1 and c.signals["agreement"] == 1


def test_hybrid_low_when_bm25_did_not_retrieve_the_top_result():
    results = [rs("a", 1, rrf(1), dense=0.7), rs("b", 2, rrf(2), dense=0.6, bm25=3.0)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "b"}, coverage=0.5, top_similarity=0.7)
    assert c.level == "low" and "not retrieved by bm25" in c.reasons
    assert "dense retrieval did not surface a" not in c.reasons


def test_hybrid_low_on_weak_top_similarity():
    results = [rs("a", 1, rrf(1, 1), dense=0.25, bm25=9.0), rs("b", 2, rrf(2, 2), dense=0.2, bm25=3.0)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "a"}, coverage=0.9, top_similarity=0.35)
    assert c.level == "low"
    assert c.reasons == ["dense and bm25 rank a first", "dense similarity 0.25 < 0.30"]


def test_reranked_low_when_the_cross_encoder_promotes_a_skill_dense_never_saw():
    results = [rs("a", 1, 6.0, bm25=2.0, rerank=6.0), rs("b", 2, 1.0, dense=0.6, bm25=5.0, rerank=1.0)]
    c = assess(results, HYBRID, {"dense": "b", "bm25": "b"}, coverage=0.6, top_similarity=0.6,
               reranked=True)
    assert c.level == "low"
    assert c.reasons[0] == "cross-encoder ranks a first"
    assert "not retrieved by dense" in c.reasons and c.competitors == ["b"]


# --- medium / clarify -------------------------------------------------------------------------

def test_hybrid_medium_when_retrievers_disagree():
    results = [rs("a", 1, rrf(1, 3), dense=0.6, bm25=8.0), rs("b", 2, rrf(2, 1), dense=0.5, bm25=9.0)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "b"}, coverage=0.7, top_similarity=0.6)
    assert (c.level, c.action, c.ambiguous) == ("medium", "clarify", True)
    assert c.competitors == ["b"] and c.reasons == ["bm25 ranks b first"]
    assert c.signals["agreement"] == 1 and c.signals["leaders"] == {"dense": "a", "bm25": "b"}


def test_exact_rrf_tie_is_ambiguous_even_when_retrievers_agree_on_nothing_else():
    tied = rrf(1, 2)
    results = [rs("a", 1, tied, dense=0.6, bm25=5.0), rs("b", 2, tied, dense=0.5, bm25=6.0),
               rs("c", 3, rrf(3, 3), dense=0.4, bm25=1.0)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "b"}, coverage=0.7, top_similarity=0.6)
    assert c.level == "medium" and c.competitors == ["b"]          # rival and tie: listed once
    assert "score (near-)tie with b" in c.reasons and "bm25 ranks b first" in c.reasons


def test_hybrid_near_but_not_exact_tie_is_not_a_tie():
    # fused scores 1.6% apart: hybrid ties must be exact (RRF ranks 1 and 2 always differ ~1.6%)
    results = [rs("a", 1, rrf(1, 1), dense=0.6, bm25=5.0), rs("b", 2, rrf(2, 2), dense=0.5, bm25=4.9)]
    c = assess(results, HYBRID, {"dense": "a", "bm25": "a"}, coverage=0.7, top_similarity=0.6)
    assert c.level == "high" and c.signals["margin"] < DEFAULT_POLICY.min_margin


def test_single_method_near_tie_uses_raw_scores_not_rrf():
    # RRF 1/61 vs 1/62 differ by 1.6% (< min_margin); only the raw BM25 gap may decide
    clear = [rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=5.0)]
    c = assess(clear, BM25, {"bm25": "a"}, coverage=0.9)
    assert c.level == "high" and c.signals["margin"] == 0.5
    close = [rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=9.9)]
    c = assess(close, BM25, {"bm25": "a"}, coverage=0.9)
    assert (c.level, c.competitors) == ("medium", ["b"]) and c.signals["margin"] == 0.01
    assert c.reasons == ["bm25 ranks a first", "score (near-)tie with b"]


def test_dense_only_near_tie_on_cosine():
    results = [rs("a", 1, rrf(1), dense=0.500), rs("b", 2, rrf(2), dense=0.495)]
    c = assess(results, DENSE, {"dense": "a"}, top_similarity=0.5)
    assert c.level == "medium" and c.competitors == ["b"]


def test_reranked_uses_no_relative_near_tie_but_flags_exact_ties():
    close = [rs("a", 1, 3.00, dense=0.6, bm25=5.0, rerank=3.00), rs("b", 2, 2.99, dense=0.5, bm25=4.0, rerank=2.99)]
    assert assess(close, BM25, {"bm25": "a"}, coverage=0.9, reranked=True).level == "high"
    exact = [rs("a", 1, 3.0, bm25=5.0, rerank=3.0), rs("b", 2, 3.0, bm25=4.0, rerank=3.0)]
    c = assess(exact, BM25, {"bm25": "a"}, coverage=0.9, reranked=True)
    assert c.level == "medium" and c.competitors == ["b"]


def test_margin_undefined_when_evidence_is_missing_or_zero():
    no_ev = [rs("a", 1, rrf(1), bm25=5.0), rs("b", 2, rrf(2))]
    assert assess(no_ev, BM25, {"bm25": "a"}, coverage=0.9).signals["margin"] is None
    zero = [rs("a", 1, rrf(1), bm25=0.0), rs("b", 2, rrf(2), bm25=0.0)]
    assert assess(zero, BM25, {"bm25": "a"}, coverage=0.9).signals["margin"] is None


def test_competitors_are_ordered_by_rank_then_slug():
    results = [rs("a", 1, 0.9, dense=0.6, bm25=5.0, rerank=0.9), rs("m", 2, 0.5, dense=0.5, bm25=4.0, rerank=0.5),
               rs("z", 3, 0.4, dense=0.4, bm25=6.0, rerank=0.4)]
    # dense's leader (z) is listed before bm25's (m) but ranks lower; y/x are not in the results
    c = assess(results, HYBRID, {"dense": "z", "bm25": "m"}, coverage=0.7, top_similarity=0.6,
               reranked=True)
    assert c.competitors == ["m", "z"]
    unranked = assess(results, HYBRID, {"dense": "y", "bm25": "x"}, coverage=0.7, top_similarity=0.6,
                      reranked=True)
    assert unranked.competitors == ["x", "y"]                    # both unranked: by slug
    assert unranked.reasons[1:] == ["dense ranks y first", "bm25 ranks x first"]


def test_leaders_of_methods_that_did_not_run_are_ignored():
    results = [rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=5.0)]
    c = assess(results, BM25, {"bm25": "a", "dense": "b"}, coverage=0.9)
    assert c.level == "high" and c.signals["leaders"] == {"bm25": "a"}


def test_signals_round_to_four_decimals():
    results = [rs("a", 1, rrf(1), dense=0.612345678), rs("b", 2, rrf(2), dense=0.4)]
    c = assess(results, DENSE, {"dense": "a"}, coverage=0.123456, top_similarity=0.612345678)
    assert c.signals["similarity"] == 0.6123 and c.signals["top_similarity"] == 0.6123
    assert c.signals["coverage"] == 0.1235


def test_summary_line():
    c = assess([rs("a", 1, rrf(1), bm25=10.0), rs("b", 2, rrf(2), bm25=5.0)], BM25, {"bm25": "a"}, coverage=0.9)
    assert summary(c) == "high (route): bm25 ranks a first"
    assert summary(assess([], BM25, {})) == "none (abstain): no candidate skills retrieved"
    from sie.models import Confidence
    assert summary(Confidence(level="low", action="clarify")) == "low (clarify): -"
