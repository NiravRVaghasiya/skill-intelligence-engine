"""Retrieval metrics: recall@k, MRR, nDCG, with paired bootstrap confidence intervals.

`gold` is one slug or a list of acceptable slugs (a query where two skills are equally
correct). A query is satisfied by its first acceptable hit, so every metric is a function
of that rank alone and the single-gold case is unchanged.

Beyond ranking quality:
- `selective` / `abstention`: how routing-confidence levels and actions relate to correctness
  (in-scope sets) and how often a system declines (out-of-scope sets);
- `intent_metrics`: a multi-intent request answered with a set of skills;
- `percentile`: nearest-rank percentiles for latency reports.
"""
from __future__ import annotations
import math
import random
from collections import Counter

from sie.confidence import ACTIONS as _LEVEL_ACTIONS, LEVELS

Gold = str | list[str]
ACTIONS = tuple(dict.fromkeys(_LEVEL_ACTIONS[level] for level in LEVELS))   # route, clarify, abstain


def first_relevant_rank(ranked: list[str], gold: Gold) -> int | None:
    """1-based rank of the first acceptable slug, or None if none was retrieved."""
    golds = {gold} if isinstance(gold, str) else set(gold)
    for i, slug in enumerate(ranked, 1):
        if slug in golds:
            return i
    return None


def recall_at_k(ranked: list[str], gold: Gold, k: int) -> float:
    rank = first_relevant_rank(ranked, gold)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(ranked: list[str], gold: Gold) -> float:
    rank = first_relevant_rank(ranked, gold)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked: list[str], gold: Gold, k: int) -> float:
    rank = first_relevant_rank(ranked, gold)
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def per_query(rows: list[dict], k: int = 3) -> dict[str, list[float]]:
    """Metric name -> one value per row (rows carry "ranked" and "gold")."""
    return {
        f"recall@{k}": [recall_at_k(r["ranked"], r["gold"], k) for r in rows],
        "mrr": [reciprocal_rank(r["ranked"], r["gold"]) for r in rows],
        f"ndcg@{k}": [ndcg_at_k(r["ranked"], r["gold"], k) for r in rows],
    }


def aggregate(rows: list[dict], k: int = 3) -> dict:
    n = len(rows) or 1
    return {name: sum(vals) / n for name, vals in per_query(rows, k).items()}


def per_query_ks(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, list[float]]:
    """Like `per_query`, for several cutoffs: recall@k for each k, mrr, then ndcg@k for each k.

    Raises:
        ValueError: a cutoff is not a positive integer.
    """
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"cutoffs must be positive integers, got {k!r}")
    out = {f"recall@{k}": [recall_at_k(r["ranked"], r["gold"], k) for r in rows] for k in ks}
    out["mrr"] = [reciprocal_rank(r["ranked"], r["gold"]) for r in rows]
    out.update({f"ndcg@{k}": [ndcg_at_k(r["ranked"], r["gold"], k) for r in rows] for k in ks})
    return out


def _share(n: int, total: int) -> float:
    return n / total if total else 0.0


def _accuracy(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _ordered(known: tuple[str, ...], seen) -> list[str]:
    """`known` in order, then any other values seen (sorted), so every group is reported."""
    return list(known) + sorted(set(seen) - set(known))


def _groups(rows: list[dict], key: str, known: tuple[str, ...]) -> dict[str, dict]:
    total = len(rows)
    out = {}
    for name in _ordered(known, (r[key] for r in rows)):
        group = [bool(r["correct"]) for r in rows if r[key] == name]
        out[name] = {"n": len(group), "share": _share(len(group), total),
                     "accuracy": _accuracy(group)}
    return out


def selective(rows: list[dict]) -> dict:
    """Confidence vs correctness on in-scope queries (a selective-prediction view).

    Args:
        rows: one per query: "correct" (top-1 is acceptable), "level" (high/medium/low/none)
            and "action" (route/clarify/abstain).

    Returns:
        {"n", "accuracy" (overall top-1), "by_level" / "by_action": {name: {"n", "share",
        "accuracy"}} (every level/action listed, empty ones with n=0 and accuracy None),
        "coverage" (share with action "route"), "routed_accuracy" (top-1 among those; None
        if none), "abstain_rate"}.
    """
    by_action = _groups(rows, "action", ACTIONS)
    routed = by_action["route"]
    return {"n": len(rows), "accuracy": _accuracy([bool(r["correct"]) for r in rows]),
            "by_level": _groups(rows, "level", LEVELS), "by_action": by_action,
            "coverage": routed["share"], "routed_accuracy": routed["accuracy"],
            "abstain_rate": by_action["abstain"]["share"]}


def _counts(values: list[str], known: tuple[str, ...]) -> dict[str, dict]:
    counts = Counter(values)
    return {name: {"n": counts[name], "share": _share(counts[name], len(values))}
            for name in _ordered(known, counts)}


def abstention(rows: list[dict]) -> dict:
    """How a system treats queries no skill fits: the share of each action (and level).

    Args:
        rows: one per query with "action"; "level" may be missing or None for systems
            without a confidence (then "by_level" is empty).

    Returns:
        {"n", "by_action": {action: {"n", "share"}}, "by_level": {level: {"n", "share"}}}.
    """
    levels = [r.get("level") for r in rows]
    return {"n": len(rows), "by_action": _counts([r["action"] for r in rows], ACTIONS),
            "by_level": _counts(levels, LEVELS) if all(v is not None for v in levels) and rows else {}}


def intent_metrics(requested: list[str], gold: list[list[str]]) -> dict[str, float]:
    """Score the skills returned for a multi-intent request against its gold intents.

    Args:
        requested: skills the system returned for the request (duplicates ignored).
        gold: one list of acceptable skills per intent.

    Returns:
        {"recall": share of gold intents covered by >= 1 requested skill, "precision": share
        of requested skills acceptable for some intent (0.0 when nothing was requested for a
        non-empty gold), "exact": 1.0 iff both are 1.0, "count_match": 1.0 iff as many
        distinct skills as gold intents}.
    """
    skills = list(dict.fromkeys(requested))
    acceptable = {s for g in gold for s in g}
    recall = _share(sum(any(s in g for s in skills) for g in gold), len(gold)) if gold else 1.0
    if skills:
        precision = sum(s in acceptable for s in skills) / len(skills)
    else:
        precision = 0.0 if gold else 1.0
    return {"recall": recall, "precision": precision,
            "exact": float(recall == 1.0 and precision == 1.0),
            "count_match": float(len(skills) == len(gold))}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Key-wise mean of metric dicts sharing the same keys (e.g. `intent_metrics` rows).

    An empty list gives {}.
    """
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


def bootstrap_ci(values: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values` (fixed seed -> reproducible)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% of `values` at or below it.

    Deterministic and interpolation-free (always returns one of the inputs).

    Args:
        values: the sample (any order; not modified).
        q: percentile in percent, 0 <= q <= 100 (q=50 -> median, q=95 -> p95, q=0 -> min).

    Returns:
        The value at 1-based rank max(1, ceil(q / 100 * n)) of the sorted sample.

    Raises:
        ValueError: `values` is empty or q is outside [0, 100].
    """
    if not 0 <= q <= 100:
        raise ValueError(f"percentile q must be in [0, 100], got {q!r}")
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) / 100))   # q*n first: exact for integer q
    return ordered[rank - 1]


def paired_delta_ci(a: list[float], b: list[float], **kw) -> tuple[float, float, float]:
    """Mean of (a - b) over paired queries with its bootstrap CI: (delta, lo, hi)."""
    if len(a) != len(b):
        raise ValueError("paired metrics need equal-length inputs")
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, **kw)
    return (sum(diffs) / (len(diffs) or 1), lo, hi)
