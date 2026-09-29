"""Routing confidence: should a caller route, ask a clarifying question, or abstain?

These are **heuristic, uncalibrated** levels, not probabilities. They are rules over evidence
the retrievers already produce, so an agent can tell a clear winner from a contested or
unsupported one. How often each level is right on the benchmark sets is *measured* (not
assumed) by `python -m eval.run_eval`; see benchmarks/RESULTS.md.

Signals (every one is reported in `Confidence.signals`):
    agreement       retrievers whose #1 skill is the top result
    support         retrievers whose candidate pool contains the top result at all
    similarity      best dense cosine among the top result's chunks (dense runs only)
    top_similarity  best dense cosine of *any* candidate: does anything in the corpus look
                    like this query?
    coverage        idf-weighted share of the query's terms that appear in the top result's
                    best-matching chunk (the BM25 index's own idf)
    margin          top-1 vs top-2 score gap (scale depends on the ranking; see `_margin`)
    leaders         each retriever's #1 skill

Rules, first match wins. The "strength" signal is `similarity` when the dense retriever ran,
else `coverage`:
    none / abstain    no candidates; or the dense retriever ran and every available signal
                      is weak (top_similarity < min_similarity and coverage < min_coverage)
    low / clarify     the top result's strength signal is weak, or some retriever did not
                      retrieve it at all
    medium / clarify  ambiguous: another skill is some retriever's #1, or the top two
                      (near-)tie; `competitors` names them
    high / route      every retriever ranks it #1 and its evidence is not weak

Lexical coverage alone never abstains. It falls with query length and paraphrase, so it
cannot tell "nothing in the corpus matches" from "a long in-scope request"; a BM25-only
router therefore says `low` / clarify, not `none`, when coverage is weak. (This one rule was
changed after the first measurement, where BM25-only abstained on 56% of in-scope queries;
benchmarks/RESULTS.md discloses it. The hybrid and dense rules are as first written.)

The thresholds were fixed before any evaluation run and never tuned on the eval sets. The
similarity floor is specific to all-MiniLM-L6-v2 cosine; recalibrate `ConfidencePolicy`
for another embedder or corpus.
"""
from __future__ import annotations
from dataclasses import dataclass

from .models import Confidence, RankedSkill

LEVELS = ("high", "medium", "low", "none")
ACTIONS = {"high": "route", "medium": "clarify", "low": "clarify", "none": "abstain"}


@dataclass(frozen=True)
class ConfidencePolicy:
    """Thresholds for `assess`. Defaults: a priori, for all-MiniLM-L6-v2 embeddings."""
    min_similarity: float = 0.30        # dense cosine below this is weak evidence
    min_coverage: float = 0.40          # idf-weighted query-term coverage below this is weak
    min_margin: float = 0.03            # relative top-1/top-2 gap below this is a near tie
                                        # (single-retriever rankings; RRF ties must be exact)


DEFAULT_POLICY = ConfidencePolicy()


def _evidence_score(skill: RankedSkill, method: str) -> float | None:
    for e in skill.evidence:
        if e.method == method:
            return e.score
    return None


def _margin(results: list[RankedSkill], methods: list[str], reranked: bool) -> float | None:
    """Top-1 vs top-2 gap on a meaningful scale.

    Single retriever: relative gap of its raw scores (cosine / BM25); the fused RRF scores of
    ranks 1 and 2 always differ by ~1.6%. Reranked: absolute cross-encoder logit gap (logits
    can be negative). Hybrid RRF: relative gap of the fused scores (reported, not a rule).
    """
    if len(results) < 2:
        return None
    if reranked:
        return results[0].score - results[1].score
    if len(methods) == 1:
        a, b = (_evidence_score(r, methods[0]) for r in results[:2])
    else:
        a, b = results[0].score, results[1].score
    if a is None or b is None or a == 0:
        return None
    return (a - b) / abs(a)


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def assess(results: list[RankedSkill], methods: list[str], leaders: dict[str, str],
           coverage: float | None = None, top_similarity: float | None = None,
           reranked: bool = False, policy: ConfidencePolicy = DEFAULT_POLICY) -> Confidence:
    """Grade the top result of a routed ranking.

    Args:
        results: the full final ranking, one entry per skill, best first (not cut to k,
            so the runner-up is visible even when k=1).
        methods: retrievers that ran, e.g. ["dense", "bm25"].
        leaders: retriever -> its #1 skill (retrievers that returned nothing are absent).
        coverage: lexical coverage of the top result, None when no BM25 index is available.
        top_similarity: best dense cosine over all candidates, None when dense did not run.
        reranked: the cross-encoder ordered `results`.
        policy: thresholds.

    Returns:
        Confidence with level, advisory action, reasons, competitors and raw signals.
    """
    if not results:
        return Confidence(level="none", action="abstain", reasons=["no candidate skills retrieved"],
                          signals={"methods": list(methods), "reranked": reranked})
    top = results[0]
    dense = "dense" in methods
    similarity = _evidence_score(top, "dense") if dense else None
    agreement = sum(1 for m in methods if leaders.get(m) == top.slug)
    support = sum(1 for m in methods if m in top.methods)
    margin = _margin(results, methods, reranked)
    signals = {"methods": list(methods), "leaders": {m: leaders[m] for m in methods if m in leaders},
               "agreement": agreement, "support": support, "similarity": _round(similarity),
               "top_similarity": _round(top_similarity), "coverage": _round(coverage),
               "margin": _round(margin), "reranked": reranked}

    weak_dense = dense and (top_similarity is None or top_similarity < policy.min_similarity)
    weak_lexical = coverage is not None and coverage < policy.min_coverage
    # Abstaining needs semantic evidence: without a dense signal, weak lexical coverage only
    # lowers confidence (see the module docstring), it never claims "no match".
    if dense and weak_dense and (weak_lexical or coverage is None):
        reasons = [f"best dense similarity "
                   f"{'none' if top_similarity is None else f'{top_similarity:.2f}'} "
                   f"< {policy.min_similarity:.2f}"]
        if coverage is not None:
            reasons.append(f"only {coverage:.0%} of the query's term weight matches the best "
                           f"candidate (< {policy.min_coverage:.0%})")
        return Confidence(level="none", action="abstain",
                          reasons=["no sufficiently good match: " + "; ".join(reasons)],
                          signals=signals)

    rivals = {m: leaders[m] for m in methods if leaders.get(m) not in (None, top.slug)}
    competitors = list(dict.fromkeys(rivals.values()))
    tied: str | None = None
    if len(results) > 1:
        exact_tie = round(results[0].score, 12) == round(results[1].score, 12)
        near_tie = (len(methods) == 1 and not reranked and margin is not None
                    and margin < policy.min_margin)
        if exact_tie or near_tie:
            tied = results[1].slug
            if tied not in competitors:
                competitors.append(tied)
    rank_of = {r.slug: r.rank for r in results}
    competitors.sort(key=lambda s: (rank_of.get(s, len(results) + 1), s))

    reasons: list[str] = []
    if reranked:
        reasons.append(f"cross-encoder ranks {top.slug} first")
    elif agreement == len(methods):
        verb = "ranks" if len(methods) == 1 else "rank"
        reasons.append(f"{' and '.join(methods)} {verb} {top.slug} first")
    weak = False
    if dense:
        if similarity is None:
            weak = True
            reasons.append(f"dense retrieval did not surface {top.slug}")
        elif similarity < policy.min_similarity:
            weak = True
            reasons.append(f"dense similarity {similarity:.2f} < {policy.min_similarity:.2f}")
    elif coverage is not None and coverage < policy.min_coverage:
        weak = True
        reasons.append(f"only {coverage:.0%} of the query's term weight matches {top.slug} "
                       f"(< {policy.min_coverage:.0%})")
    missing = [m for m in methods if m not in top.methods]
    if missing:
        weak = True
        reasons.append(f"not retrieved by {', '.join(missing)}")
    reasons += [f"{m} ranks {lead} first" for m, lead in rivals.items()]
    if tied:
        reasons.append(f"score (near-)tie with {tied}")

    if weak:
        level = "low"
    elif competitors:
        level = "medium"
    else:
        level = "high"
    return Confidence(level=level, action=ACTIONS[level], ambiguous=bool(competitors),
                      reasons=reasons, competitors=competitors, signals=signals)


def summary(conf: Confidence) -> str:
    """One line for CLIs and logs: `high (route): dense and bm25 all rank x first`."""
    why = "; ".join(conf.reasons) if conf.reasons else "-"
    return f"{conf.level} ({conf.action}): {why}"
