"""Baseline (keyword) vs SIE (hybrid) on the labeled query set -> benchmarks/RESULTS.md.

Usage:
    python -m eval.run_eval
"""
from __future__ import annotations
import json
from pathlib import Path

from sie.router import HybridRouter
from eval.metrics import aggregate

QUERIES = Path("eval/queries.jsonl")


def load_queries() -> list[dict]:
    return [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_sie(k: int = 3) -> dict:
    r = HybridRouter()
    rows = []
    for q in load_queries():
        ranked = [h.skill_slug for h in r.retrieve(q["query"], k=10)]
        rows.append({"ranked": ranked, "gold": q["gold"]})
    return aggregate(rows, k=k)


def main() -> None:
    print("[eval] building indexes ...")
    HybridRouter().build()
    print("[eval] scoring SIE ...")
    sie = eval_sie()
    print("SIE:", sie)
    # TODO: plug the source repo's scripts/router.py as the keyword baseline and compare.


if __name__ == "__main__":
    main()
