import json
from pathlib import Path

import pytest

from eval.metrics import (ACTIONS, abstention, aggregate, bootstrap_ci, first_relevant_rank,
                          intent_metrics, mean_metrics, ndcg_at_k, paired_delta_ci, per_query,
                          per_query_ks, percentile, recall_at_k, reciprocal_rank, selective)
from sie.ingest import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_single_gold_metrics():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, "c", 3) == 1.0 and recall_at_k(ranked, "d", 3) == 0.0
    assert reciprocal_rank(ranked, "b") == 0.5 and reciprocal_rank(ranked, "z") == 0.0
    assert ndcg_at_k(ranked, "a", 3) == 1.0 and ndcg_at_k(ranked, "b", 3) == pytest.approx(1 / 1.585, 1e-3)


def test_list_gold_uses_first_acceptable_hit():
    ranked = ["x", "b", "a"]
    assert first_relevant_rank(ranked, ["a", "b"]) == 2
    assert reciprocal_rank(ranked, ["a", "b"]) == 0.5
    assert recall_at_k(ranked, ["a", "b"], 1) == 0.0
    assert first_relevant_rank([], ["a"]) is None


def test_aggregate_means():
    rows = [{"ranked": ["a"], "gold": "a"}, {"ranked": ["b", "a"], "gold": "a"}]
    agg = aggregate(rows, k=3)
    assert agg["recall@3"] == 1.0 and agg["mrr"] == 0.75


def test_bootstrap_is_seeded_and_brackets_mean():
    vals = [1.0, 0.0, 1.0, 1.0, 0.5, 0.0, 1.0]
    lo, hi = bootstrap_ci(vals, n_boot=2000)
    assert (lo, hi) == bootstrap_ci(vals, n_boot=2000)
    assert lo <= sum(vals) / len(vals) <= hi
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_paired_delta():
    d, lo, hi = paired_delta_ci([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], n_boot=500)
    assert d == pytest.approx(1 / 3) and lo <= d <= hi
    with pytest.raises(ValueError):
        paired_delta_ci([1.0], [])


@pytest.mark.parametrize("name", ["queries.jsonl", "queries_source_evals.jsonl"])
def test_query_files_are_well_formed(name):
    slugs = {s.slug: s.domain for s in load_corpus(ROOT / "data" / "skills")}
    rows = [json.loads(l) for l in (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        golds = [r["gold"]] if isinstance(r["gold"], str) else r["gold"]
        assert r["query"].strip() and golds and all(g in slugs for g in golds), r
    if name == "queries.jsonl":
        assert len(rows) >= 40
        assert {slugs[r["gold"] if isinstance(r["gold"], str) else r["gold"][0]] for r in rows} == set(slugs.values())
        assert len({r["query"] for r in rows}) == len(rows)


# -- recall@k / ndcg@k at several cutoffs --------------------------------------------------

def test_per_query_ks_matches_single_cutoff_functions():
    rows = [{"ranked": ["a", "b", "c", "d", "e"], "gold": "d"}, {"ranked": ["x", "a"], "gold": ["a", "q"]},
            {"ranked": [], "gold": "a"}]
    vals = per_query_ks(rows, ks=(1, 3, 5))
    assert list(vals) == ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@1", "ndcg@3", "ndcg@5"]
    assert vals["recall@1"] == [0.0, 0.0, 0.0] and vals["recall@3"] == [0.0, 1.0, 0.0]
    assert vals["recall@5"] == [1.0, 1.0, 0.0] and vals["mrr"] == [0.25, 0.5, 0.0]
    assert vals["ndcg@5"][0] == pytest.approx(1 / 2.3219, 1e-4) and vals["ndcg@3"][0] == 0.0
    old = per_query(rows, k=3)                      # the k=3 keys are unchanged
    assert all(vals[key] == old[key] for key in old)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_per_query_ks_rejects_bad_cutoffs(bad):
    with pytest.raises(ValueError):
        per_query_ks([], ks=(1, bad))


# -- confidence vs correctness ---------------------------------------------------------------

def _conf(correct, level, action):
    return {"correct": correct, "level": level, "action": action}


def test_selective_counts_accuracy_per_level_and_action():
    rows = [_conf(True, "high", "route"), _conf(True, "high", "route"), _conf(False, "high", "route"),
            _conf(True, "medium", "clarify"), _conf(False, "low", "clarify"), _conf(False, "none", "abstain")]
    sel = selective(rows)
    assert sel["n"] == 6 and sel["accuracy"] == pytest.approx(0.5)
    assert list(sel["by_level"]) == ["high", "medium", "low", "none"]
    assert sel["by_level"]["high"] == {"n": 3, "share": 0.5, "accuracy": pytest.approx(2 / 3)}
    assert sel["by_level"]["none"] == {"n": 1, "share": pytest.approx(1 / 6), "accuracy": 0.0}
    assert list(sel["by_action"]) == list(ACTIONS) == ["route", "clarify", "abstain"]
    assert sel["by_action"]["clarify"]["n"] == 2 and sel["by_action"]["clarify"]["accuracy"] == 0.5
    assert sel["coverage"] == 0.5 and sel["routed_accuracy"] == pytest.approx(2 / 3)
    assert sel["abstain_rate"] == pytest.approx(1 / 6)


def test_selective_lists_empty_groups_with_no_accuracy():
    sel = selective([_conf(True, "high", "route")])
    assert sel["by_level"]["medium"] == {"n": 0, "share": 0.0, "accuracy": None}
    assert sel["by_action"]["abstain"] == {"n": 0, "share": 0.0, "accuracy": None}
    empty = selective([])
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["routed_accuracy"] is None
    assert empty["coverage"] == 0.0 and empty["abstain_rate"] == 0.0


def test_selective_reports_unexpected_levels_after_known_ones():
    sel = selective([_conf(True, "high", "route"), _conf(False, "odd", "abstain")])
    assert list(sel["by_level"]) == ["high", "medium", "low", "none", "odd"]


def test_abstention_shares_with_and_without_levels():
    rows = [{"action": "abstain", "level": "none"}, {"action": "abstain", "level": "none"},
            {"action": "route", "level": "high"}, {"action": "clarify", "level": "low"}]
    a = abstention(rows)
    assert a["n"] == 4 and a["by_action"]["abstain"] == {"n": 2, "share": 0.5}
    assert list(a["by_action"]) == ["route", "clarify", "abstain"]
    assert a["by_level"]["none"]["share"] == 0.5 and a["by_level"]["medium"]["n"] == 0
    no_conf = abstention([{"action": "route"}, {"action": "abstain", "level": None}])
    assert no_conf["by_level"] == {} and no_conf["by_action"]["route"]["share"] == 0.5
    assert abstention([]) == {"n": 0, "by_action": {a: {"n": 0, "share": 0.0} for a in ACTIONS},
                              "by_level": {}}


# -- multi-intent ------------------------------------------------------------------------------

def test_intent_metrics_exact_and_partial():
    gold = [["rag-pipeline"], ["rag-evaluation", "llm-evaluation"]]
    assert intent_metrics(["rag-pipeline", "llm-evaluation"], gold) == {
        "recall": 1.0, "precision": 1.0, "exact": 1.0, "count_match": 1.0}
    part = intent_metrics(["rag-pipeline", "model-deployment", "rag-pipeline"], gold)   # dupes ignored
    assert part == {"recall": 0.5, "precision": 0.5, "exact": 0.0, "count_match": 1.0}
    one = intent_metrics(["rag-evaluation"], gold)
    assert one == {"recall": 0.5, "precision": 1.0, "exact": 0.0, "count_match": 0.0}


def test_intent_metrics_one_skill_may_cover_two_intents_but_not_match_the_count():
    m = intent_metrics(["a"], [["a"], ["a", "b"]])
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["exact"] == 1.0
    assert m["count_match"] == 0.0


def test_intent_metrics_nothing_requested():
    assert intent_metrics([], [["a"]]) == {"recall": 0.0, "precision": 0.0, "exact": 0.0,
                                           "count_match": 0.0}
    assert intent_metrics([], []) == {"recall": 1.0, "precision": 1.0, "exact": 1.0, "count_match": 1.0}


def test_mean_metrics():
    rows = [{"recall": 1.0, "exact": 1.0}, {"recall": 0.5, "exact": 0.0}]
    assert mean_metrics(rows) == {"recall": 0.75, "exact": 0.5}
    assert mean_metrics([]) == {}


# -- latency percentiles -----------------------------------------------------------------------

def test_percentile_nearest_rank():
    vals = [15.0, 20.0, 35.0, 40.0, 50.0]
    assert percentile(vals, 0) == 15.0 and percentile(vals, 100) == 50.0
    assert percentile(vals, 30) == 20.0 and percentile(vals, 40) == 20.0     # ceil(0.4 * 5) = 2
    assert percentile(vals, 50) == 35.0 and percentile(vals, 95) == 50.0
    hundred = [float(i) for i in range(100, 0, -1)]                           # unsorted input
    assert percentile(hundred, 7) == 7.0 and percentile(hundred, 95) == 95.0
    assert percentile(hundred, 99.5) == 100.0
    assert percentile([3.0], 50) == 3.0
    assert hundred[0] == 100.0                                                # not mutated


def test_percentile_returns_an_input_value_deterministically():
    vals = [0.3, 0.1, 0.2, 0.4]
    assert percentile(vals, 50) in vals and percentile(vals, 50) == percentile(list(reversed(vals)), 50)


@pytest.mark.parametrize("q", [-1, 100.5])
def test_percentile_rejects_out_of_range_q(q):
    with pytest.raises(ValueError, match="percentile q"):
        percentile([1.0], q)


def test_percentile_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
