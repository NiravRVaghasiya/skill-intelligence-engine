"""Baseline (keyword) vs SIE (hybrid) on the labeled query sets -> benchmarks/.

Writes the `metrics` block of benchmarks/RESULTS.md, benchmarks/results.png (+ a dark
variant) and benchmarks/per_query.jsonl. Every number in those files comes from here.

Usage:
    python -m eval.run_eval               # rebuild indexes, then score every system
    python -m eval.run_eval --no-build    # reuse the persisted data/chroma/ index
"""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sie.chunking import card_chunk
from sie.index.sparse import SparseIndex
from sie.ingest import load_corpus
from sie.models import Hit
from sie.router import HybridRouter, corpus_fingerprint
from sie.rerank import RerankerUnavailable
from eval.baseline import KeywordBaseline
from eval.metrics import bootstrap_ci, first_relevant_rank, paired_delta_ci, per_query
from eval.report import Bar, md_table, render_chart, replace_block

QUERIES = Path("eval/queries.jsonl")
SOURCE_QUERIES = Path("eval/queries_source_evals.jsonl")
OUT = Path("benchmarks")
K = 3
DEPTH = 10          # ranks scored per query (MRR looks this deep)
METRICS = ["top-1", f"recall@{K}", "mrr", f"ndcg@{K}"]
CHART_METRICS = [f"recall@{K}", "mrr", f"ndcg@{K}"]


@dataclass
class System:
    key: str
    label: str
    family: str                      # "keyword" | "sie"
    make: Callable[[], object]       # -> object with .retrieve(query, k) -> list[Hit]


class CardOnlyBM25:
    """Ablation: BM25 over one Card chunk per skill (display name + description + capability
    tags), i.e. roughly the frontmatter fields the keyword router scores, and no body text."""

    def __init__(self, skills_dir: str = "data/skills"):
        self.sparse = SparseIndex()
        self.sparse.build([card_chunk(s) for s in load_corpus(skills_dir)])

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        return self.sparse.search(query, k=k)       # one chunk per skill: already deduped


SYSTEMS = [
    System("keyword", "Keyword router (baseline)", "keyword", lambda: KeywordBaseline()),
    System("keyword-shipped", "Keyword route() as shipped", "keyword",
           lambda: KeywordBaseline(shipped=True)),
    System("bm25-card", "SIE: BM25, Card chunks only", "sie", lambda: CardOnlyBM25()),
    System("bm25", "SIE: BM25 only", "sie", lambda: HybridRouter(mode="sparse", use_reranker=False)),
    System("dense", "SIE: dense only", "sie", lambda: HybridRouter(mode="dense", use_reranker=False)),
    System("hybrid", "SIE: hybrid RRF (--no-rerank)", "sie", lambda: HybridRouter(use_reranker=False)),
    System("hybrid-rerank", "SIE: hybrid + cross-encoder", "sie",
           lambda: HybridRouter(use_reranker=True, strict_rerank=True)),
]
BASELINE = "keyword"


@dataclass
class Scored:
    system: System
    ranked: list[list[str]] | None   # per query; None when the system could not run
    note: str = ""
    scores: list[list[float]] | None = None   # parallel to `ranked`


def load_queries(path: Path = QUERIES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_system(system: System, queries: list[dict]) -> Scored:
    """Top-DEPTH ranked slugs for every query; a system missing its model is 'pending'."""
    try:
        r = system.make()
        hits = [r.retrieve(q["query"], k=DEPTH) for q in queries]
    except RerankerUnavailable as e:
        return Scored(system, None, note=str(e))
    return Scored(system, [[h.skill_slug for h in hs] for hs in hits],
                  scores=[[h.score for h in hs] for hs in hits])


def tie_aware_top1(scored: Scored, queries: list[dict]) -> tuple[int, float]:
    """(#queries whose top-1 is an exact score tie, expected top-1 under random tie order).

    Every system breaks exact ties deterministically (by slug); this shows how much of
    its strict top-1 that choice decides. Ties are compared at 9 decimals.
    """
    tied, expected = 0, 0.0
    for ranked, scores, q in zip(scored.ranked, scores_of(scored), queries):
        if not ranked:
            continue
        top = round(scores[0], 9)
        group = [slug for slug, sc in zip(ranked, scores) if round(sc, 9) == top]
        golds = {q["gold"]} if isinstance(q["gold"], str) else set(q["gold"])
        tied += len(group) > 1
        expected += sum(slug in golds for slug in group) / len(group)
    return tied, expected / len(queries)


def scores_of(scored: Scored) -> list[list[float]]:
    return scored.scores or [[0.0] * len(r) for r in scored.ranked]


def metric_values(scored: Scored, queries: list[dict]) -> dict[str, list[float]]:
    rows = [{"ranked": r, "gold": q["gold"]} for r, q in zip(scored.ranked, queries)]
    vals = per_query(rows, k=K)
    vals["top-1"] = per_query(rows, k=1)["recall@1"]
    return vals


def _fmt_ci(vals: list[float]) -> str:
    lo, hi = bootstrap_ci(vals)
    return f"{sum(vals) / len(vals):.3f} <sub>[{lo:.2f}, {hi:.2f}]</sub>"


def headline_key(results: list[Scored]) -> str:
    """The full pipeline if its reranker ran, otherwise the best runnable SIE config."""
    by_key = {s.system.key: s for s in results}
    return "hybrid-rerank" if by_key["hybrid-rerank"].ranked is not None else "hybrid"


def metrics_table(results: list[Scored], queries: list[dict], headline: str) -> str:
    rows = []
    for s in results:
        name = f"**{s.system.label}**" if s.system.key == headline else s.system.label
        if s.ranked is None:
            rows.append([name] + ["_pending_"] * len(METRICS))
            continue
        vals = metric_values(s, queries)
        rows.append([name] + [_fmt_ci(vals[m]) for m in METRICS])
    return md_table(["System"] + METRICS, rows)


def delta_table(results: list[Scored], queries: list[dict], headline: str) -> str:
    by_key = {s.system.key: s for s in results}
    a, b = metric_values(by_key[headline], queries), metric_values(by_key[BASELINE], queries)
    rows = []
    for m in METRICS:
        d, lo, hi = paired_delta_ci(a[m], b[m])
        rows.append([m, f"{sum(b[m]) / len(b[m]):.3f}", f"{sum(a[m]) / len(a[m]):.3f}",
                     f"{d:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]"])
    return md_table(["metric", "keyword", by_key[headline].system.label, "delta", "95% CI (paired)"], rows)


def win_loss(results: list[Scored], queries: list[dict], headline: str) -> str:
    by_key = {s.system.key: s for s in results}
    a = metric_values(by_key[headline], queries)["mrr"]
    b = metric_values(by_key[BASELINE], queries)["mrr"]
    wins, losses = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    return f"per-query reciprocal rank: SIE better on **{wins}**, keyword better on **{losses}**, tied on {len(a) - wins - losses}"


def breakdown(results: list[Scored], queries: list[dict], headline: str, field: str) -> str:
    by_key = {s.system.key: s for s in results}
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(queries):
        groups.setdefault(q.get(field, "-"), []).append(i)
    a, b = metric_values(by_key[headline], queries), metric_values(by_key[BASELINE], queries)
    rows = []
    for g, idx in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        mean = lambda v: sum(v[i] for i in idx) / len(idx)
        rows.append([g, str(len(idx)), f"{mean(b['top-1']):.2f}", f"{mean(a['top-1']):.2f}",
                     f"{mean(b['mrr']):.2f}", f"{mean(a['mrr']):.2f}"])
    return md_table([field, "n", "keyword top-1", "SIE top-1", "keyword MRR", "SIE MRR"], rows)


def misses(results: list[Scored], queries: list[dict], headline: str) -> str:
    """Every query where the headline system misses the top-3 — nothing hidden."""
    by_key = {s.system.key: s for s in results}
    rows = []
    for q, sie, kw in zip(queries, by_key[headline].ranked, by_key[BASELINE].ranked):
        if first_relevant_rank(sie[:K], q["gold"]) is None:
            gold = q["gold"] if isinstance(q["gold"], str) else " / ".join(q["gold"])
            rank = first_relevant_rank(sie, q["gold"])
            rows.append([q["query"].replace("|", "/"), f"`{gold}`", ", ".join(sie[:3]),
                         str(rank or f">{DEPTH}"), ", ".join(kw[:3])])
    if not rows:
        return "_none_"
    return md_table(["query", "gold", "SIE top-3", "SIE rank", "keyword top-3"], rows, align="lllcl")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def provenance(results: list[Scored]) -> str:
    import importlib.metadata as md
    r = HybridRouter(use_reranker=False)
    chunks = r._load_chunks()
    pending = [s for s in results if s.ranked is None]
    lines = [
        f"- corpus: `data/skills` (38 skills, {len(chunks)} chunks, fingerprint `{corpus_fingerprint(chunks)}`)",
        f"- queries: `{QUERIES.as_posix()}` sha256 `{_sha(QUERIES)}`, "
        f"`{SOURCE_QUERIES.as_posix()}` sha256 `{_sha(SOURCE_QUERIES)}`",
        f"- dense: all-MiniLM-L6-v2 via `{r.dense.embedder}` embedder, ChromaDB "
        f"{md.version('chromadb')} (cosine HNSW, persisted to `data/chroma/`); sparse: rank-bm25 "
        f"{md.version('rank-bm25')}; fusion: RRF k=60 over the top-20 chunks of each retriever",
        f"- keyword baseline: vendored verbatim, see `eval/baseline/SOURCE.md`",
    ]
    lines += [f"- **{s.system.label}: pending** — {s.note}" for s in pending]
    return "\n".join(lines)


def tie_table(results: list[Scored], queries: list[dict]) -> str:
    rows = []
    for s in results:
        if s.ranked is None:
            continue
        tied, expected = tie_aware_top1(s, queries)
        rows.append([s.system.label, f"{_mean(s, queries, 'top-1'):.3f}", f"{expected:.3f}", str(tied)])
    return md_table(["System", "top-1 (ties by slug)", "top-1 (expected, random tie order)",
                     "queries with a tied top-1"], rows)


def section(title: str, results: list[Scored], queries: list[dict], headline: str) -> str:
    return "\n\n".join([
        f"### {title} (n={len(queries)})",
        metrics_table(results, queries, headline),
        f"Headline vs baseline, paired over the same queries — {win_loss(results, queries, headline)}:",
        delta_table(results, queries, headline),
        "Tie sensitivity: every system breaks exact score ties alphabetically by slug.",
        tie_table(results, queries),
    ])


def write_per_query(results: list[Scored], sets: dict[str, list[dict]]) -> None:
    with (OUT / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for set_name, queries in sets.items():
            for i, q in enumerate(queries):
                row = {"set": set_name, "query": q["query"], "gold": q["gold"],
                       "domain": q.get("domain"), "style": q.get("style") or q.get("category")}
                for s in results[set_name]:
                    if s.ranked is not None:
                        ranked = s.ranked[i]
                        row[s.system.key] = {"rank": first_relevant_rank(ranked, q["gold"]),
                                             "top3": ranked[:3]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bars(results: list[Scored], queries: list[dict], headline: str) -> list[Bar]:
    bars = []
    for s in results:
        if s.ranked is None:
            bars.append(Bar(s.system.label, s.system.family, None,
                            note="pending: cross-encoder weights unavailable"))
            continue
        vals = metric_values(s, queries)
        bars.append(Bar(s.system.label, s.system.family,
                        {m: (sum(vals[m]) / len(vals[m]), *bootstrap_ci(vals[m])) for m in CHART_METRICS},
                        labelled=s.system.key in (BASELINE, headline)))
    return bars


def chart(results: dict[str, list[Scored]], sets: dict[str, list[dict]], headline: str) -> None:
    rows = [(f"Main set (n={len(sets['main'])}, eval/queries.jsonl)",
             _bars(results["main"], sets["main"], headline)),
            (f"Source-authored set (n={len(sets['source'])}, the ml-ai-skills repo's own evals/)",
             _bars(results["source"], sets["source"], headline))]
    for theme, name in (("light", "results.png"), ("dark", "results-dark.png")):
        render_chart(rows, CHART_METRICS, "Skill routing: keyword router vs SIE",
                     "bar = mean over queries, whisker = 95% bootstrap CI; higher is better",
                     OUT / name, theme=theme)


def _mean(s: Scored, queries: list[dict], metric: str) -> float:
    vals = metric_values(s, queries)[metric]
    return sum(vals) / len(vals)


def observations(results: dict[str, list[Scored]], sets: dict[str, list[dict]], headline: str) -> str:
    """Takeaways computed from the same numbers as the tables, so the prose can't drift."""
    lines = []
    for name, title in (("main", "Main"), ("source", "Source-authored")):
        qs = sets[name]
        by = {s.system.key: s for s in results[name] if s.ranked is not None}
        t1 = {key: _mean(by[key], qs, "top-1") for key in (BASELINE, "bm25-card", "bm25")}
        lines.append(
            f"- **{title} set, where the lexical gain comes from:** BM25 over the Card chunks "
            f"alone (about the fields the keyword router reads) takes top-1 from "
            f"{t1[BASELINE]:.3f} to {t1['bm25-card']:.3f} (better term weighting on the same "
            f"text). Indexing the body sections too "
            f"{'raises' if t1['bm25'] > t1['bm25-card'] else 'lowers'} it to {t1['bm25']:.3f}.")
    for name, title in (("main", "Main"), ("source", "Source-authored")):
        qs = sets[name]
        by = {s.system.key: s for s in results[name] if s.ranked is not None}
        best, other = sorted(("bm25", "dense"), key=lambda key: -_mean(by[key], qs, "mrr"))
        mrr = {key: _mean(by[key], qs, "mrr") for key in (best, other, headline, BASELINE)}
        lines.append(
            f"- **{title} set:** the stronger single retriever is "
            f"**{by[best].system.label.split(': ')[1]}** (MRR {mrr[best]:.3f} vs {mrr[other]:.3f}). "
            f"{by[headline].system.label} is {mrr[headline] - mrr[best]:+.3f} MRR against it and "
            f"{mrr[headline] - mrr[BASELINE]:+.3f} against the keyword router.")
    kw = metric_values({s.system.key: s for s in results["main"]}[BASELINE], sets["main"])["top-1"]
    by_style: dict[str, list[float]] = {}
    for q, v in zip(sets["main"], kw):
        by_style.setdefault(q.get("style", "-"), []).append(v)
    means = {style: sum(v) / len(v) for style, v in by_style.items()}
    hi, lo = max(means, key=means.get), min(means, key=means.get)
    lines.append(f"- The keyword router is strongest on `{hi}` queries (top-1 {means[hi]:.2f}; `-` marks "
                 f"the scaffold's original 12) and weakest on `{lo}` queries (top-1 {means[lo]:.2f}).")
    return "\n".join(lines)


def report(results: dict[str, list[Scored]], sets: dict[str, list[dict]]) -> str:
    main_q, src_q = sets["main"], sets["source"]
    headline = headline_key(results["main"])
    label = {s.system.key: s.system.label for s in results["main"]}[headline]
    return "\n\n".join([
        f"_Generated by `python -m eval.run_eval`; do not edit by hand. Headline system: "
        f"**{label}**. Cells are mean <sub>[95% bootstrap CI]</sub>; top-1 = recall@1._",
        "### Observations (computed)\n\n" + observations(results, sets, headline),
        "![results](results.png)",
        section("Main set — `eval/queries.jsonl`", results["main"], main_q, headline),
        section("Source-authored set — the ml-ai-skills repo's own `evals/` cases",
                results["source"], src_q, headline),
        "### Breakdown (main set)",
        breakdown(results["main"], main_q, headline, "domain"),
        breakdown(results["main"], main_q, headline, "style"),
        breakdown(results["main"], main_q, headline, "set"),
        f"<details><summary>Every main-set query where {label} misses the top-{K}</summary>\n\n"
        f"{misses(results['main'], main_q, headline)}\n\n</details>",
        f"<details><summary>Every source-set query where {label} misses the top-{K}</summary>\n\n"
        f"{misses(results['source'], src_q, headline)}\n\n</details>",
        "### Provenance\n\n" + provenance(results["main"]),
    ])


def readme_headline(results: dict[str, list[Scored]], sets: dict[str, list[dict]]) -> str:
    """The README's opening claim, computed (so adding reranker weights updates it too)."""
    headline = headline_key(results["main"])
    m = {s.system.key: s for s in results["main"]}
    src = {s.system.key: s for s in results["source"]}
    pct = lambda s, qs: f"{100 * _mean(s, qs, 'top-1'):.1f}%"
    return (f"**Top-1 routing accuracy goes from {pct(m[BASELINE], sets['main'])} to "
            f"{pct(m[headline], sets['main'])}** on {len(sets['main'])} labeled queries (keyword "
            f"router -> {m[headline].system.label}), and from {pct(src[BASELINE], sets['source'])} to "
            f"{pct(src[headline], sets['source'])} on the {len(sets['source'])} queries taken from the "
            f"source repo's own eval cases. _Generated by `python -m eval.run_eval`._")


def readme_block(results: dict[str, list[Scored]], sets: dict[str, list[dict]]) -> str:
    """Compact headline table + computed takeaways for README.md (same run as RESULTS.md)."""
    headline = headline_key(results["main"])
    rows = []
    for s in results["main"]:
        if s.system.key == "keyword-shipped":
            continue
        name = f"**{s.system.label}**" if s.system.key == headline else s.system.label
        if s.ranked is None:
            rows.append([name] + ["pending"] * len(METRICS))
        else:
            rows.append([name] + [f"{_mean(s, sets['main'], m):.3f}" for m in METRICS])
    src = {s.system.key: s for s in results["source"]}
    return "\n\n".join([
        f"Main set: {len(sets['main'])} labeled queries across all 8 domains (every skill covered).",
        md_table(["System"] + METRICS, rows),
        f"On the {len(sets['source'])} queries taken from the source repo's own `evals/` cases: "
        f"keyword top-1 **{_mean(src[BASELINE], sets['source'], 'top-1'):.3f}** / MRR "
        f"{_mean(src[BASELINE], sets['source'], 'mrr'):.3f} vs SIE top-1 "
        f"**{_mean(src[headline], sets['source'], 'top-1'):.3f}** / MRR "
        f"{_mean(src[headline], sets['source'], 'mrr'):.3f}. "
        f"CIs, paired deltas, tie sensitivity, ablations and every miss: "
        f"[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). _Generated by `python -m eval.run_eval`._",
        "**What the ablations say (computed):**\n\n" + observations(results, sets, headline),
    ])


def print_summary(results: dict[str, list[Scored]], sets: dict[str, list[dict]]) -> None:
    for set_name, queries in sets.items():
        print(f"\n[eval] {set_name} set (n={len(queries)})")
        print(f"  {'system':32s} " + " ".join(f"{m:>9s}" for m in METRICS))
        for s in results[set_name]:
            if s.ranked is None:
                print(f"  {s.system.label:32s} pending ({s.note.split(';')[0]})")
                continue
            vals = metric_values(s, queries)
            print(f"  {s.system.label:32s} " + " ".join(f"{sum(vals[m]) / len(vals[m]):9.3f}" for m in METRICS))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the keyword baseline and SIE variants.")
    ap.add_argument("--no-build", action="store_true", help="reuse the persisted dense index")
    args = ap.parse_args()
    if not args.no_build:
        print("[eval] building indexes ...")
        HybridRouter(use_reranker=False).build()
    sets = {"main": load_queries(QUERIES), "source": load_queries(SOURCE_QUERIES)}
    results = {name: [score_system(s, qs) for s in SYSTEMS] for name, qs in sets.items()}
    print_summary(results, sets)
    OUT.mkdir(exist_ok=True)
    write_per_query(results, sets)
    chart(results, sets, headline_key(results["main"]))
    replace_block(OUT / "RESULTS.md", "metrics", report(results, sets))
    replace_block(Path("README.md"), "headline", readme_headline(results, sets))
    replace_block(Path("README.md"), "benchmarks", readme_block(results, sets))
    print(f"\n[eval] wrote {OUT / 'RESULTS.md'}, {OUT / 'results.png'}, {OUT / 'per_query.jsonl'}")


if __name__ == "__main__":
    main()
