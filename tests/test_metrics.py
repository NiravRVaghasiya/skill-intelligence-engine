import json
from pathlib import Path

import pytest

from eval.metrics import (aggregate, bootstrap_ci, first_relevant_rank, ndcg_at_k,
                          paired_delta_ci, recall_at_k, reciprocal_rank)
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
