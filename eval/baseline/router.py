#!/usr/bin/env python3
"""Deterministic, metadata-driven skill router.

No embeddings, no LLM call, no external service — a scored keyword match
against each skill's `capabilities` + `display_name` + `description`, plus a
soft bonus/penalty for whether the skill's `type` (workflow/reference)
matches the task's classified intent. This is intentionally the simplest
thing that could work (Phase 23: don't overengineer); see docs/ROUTING.md
for the full algorithm description, its known failure modes, and when you'd
actually need something smarter (e.g. semantic/embedding-based retrieval)
instead of extending this heuristic further.

Usage:
    python scripts/router.py "Build a fraud detection model and evaluate it
        under severe class imbalance."
    python scripts/router.py --json "explain attention"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skills_lib as sl  # noqa: E402

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "to", "of", "in", "on", "for", "and", "or", "with",
    "my", "me", "i", "it", "its", "as", "at", "by", "from", "how", "do",
    "does", "can", "should", "would", "will", "using", "use", "into", "under",
}

REFERENCE_CUES = (
    "what is", "what's", "what are", "explain", "how does", "how do",
    "why does", "why do", "why is", "difference between", "understand",
    "intuition", "concept of", "meaning of",
)
WORKFLOW_CUES = (
    "build", "train", "evaluate", "deploy", "clean", "tune", "fine-tune",
    "finetune", "implement", "write", "create", "fix", "debug", "optimize",
    "preprocess", "compare", "calibrate", "ship", "monitor", "retrain",
    "set up", "design a pipeline", "reduce", "handle",
)


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def classify_intent(task_text: str) -> str:
    """Returns 'reference' or 'workflow'. Ties (or no cue at all) default to
    'workflow' because an agent given a plain topic name ("model calibration")
    with no imperative verb is more often about to *do* something with it."""
    t = task_text.lower()
    ref_hits = sum(1 for cue in REFERENCE_CUES if cue in t)
    wf_hits = sum(1 for cue in WORKFLOW_CUES if cue in t)
    if ref_hits > wf_hits:
        return "reference"
    return "workflow"


@dataclass
class RouteHit:
    slug: str
    score: float
    reasons: list[str]
    type: str
    domain: str


def score_skill(skill: sl.Skill, task_tokens: set[str], intent: str) -> RouteHit:
    fm = skill.frontmatter
    reasons: list[str] = []
    score = 0.0

    capabilities = [c.lower() for c in (fm.get("capabilities") or [])]
    cap_hits = [c for c in capabilities if any(word in task_tokens for word in tokenize(c)) or c in " ".join(task_tokens)]
    if cap_hits:
        score += 3 * len(cap_hits)
        reasons.append(f"capability match: {', '.join(cap_hits)}")

    name_tokens = tokenize(fm.get("display_name", "") + " " + skill.slug.replace("-", " "))
    name_overlap = name_tokens & task_tokens
    if name_overlap:
        score += 2 * len(name_overlap)
        reasons.append(f"name overlap: {', '.join(sorted(name_overlap))}")

    desc_tokens = tokenize(fm.get("description", ""))
    desc_overlap = desc_tokens & task_tokens
    if desc_overlap:
        score += 1 * len(desc_overlap)
        reasons.append(f"description overlap: {', '.join(sorted(desc_overlap))}")

    if fm.get("type") == intent:
        score += 2
        reasons.append(f"type matches classified intent ({intent})")
    else:
        score -= 1

    return RouteHit(slug=skill.slug, score=score, reasons=reasons, type=fm.get("type", ""), domain=fm.get("domain", ""))


def expand_with_requires(skills_by_slug: dict[str, sl.Skill], ordered_slugs: list[str]) -> list[str]:
    """Prepend hard prerequisites (`requires`) that weren't already selected,
    preserving relative order and avoiding duplicates."""
    result: list[str] = []
    seen: set[str] = set()

    def visit(slug: str) -> None:
        if slug in seen or slug not in skills_by_slug:
            return
        for req in skills_by_slug[slug].frontmatter.get("requires") or []:
            visit(req)
        if slug not in seen:
            seen.add(slug)
            result.append(slug)

    for slug in ordered_slugs:
        visit(slug)
    return result


def route(task_text: str, top_k: int = 5, min_score: float = 1.0) -> tuple[str, list[RouteHit]]:
    skills = sl.load_all_skills()
    skills_by_slug = sl.by_slug(skills)
    task_tokens = tokenize(task_text)
    intent = classify_intent(task_text)

    hits = [score_skill(s, task_tokens, intent) for s in skills]
    hits = [h for h in hits if h.score >= min_score]
    hits.sort(key=lambda h: (-h.score, h.slug))
    top = hits[:top_k]

    expanded_order = expand_with_requires(skills_by_slug, [h.slug for h in top])
    hits_by_slug = {h.slug: h for h in top}
    final = [hits_by_slug.get(slug) or RouteHit(slug, 0.0, ["pulled in as a hard prerequisite"],
                                                 skills_by_slug[slug].frontmatter.get("type", ""),
                                                 skills_by_slug[slug].frontmatter.get("domain", ""))
             for slug in expanded_order]
    return intent, final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="free-text description of what the user wants to do")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    intent, hits = route(args.task, top_k=args.top_k)

    if args.json:
        print(json.dumps({
            "task": args.task, "intent": intent,
            "route": [{"slug": h.slug, "score": h.score, "type": h.type, "domain": h.domain, "reasons": h.reasons}
                      for h in hits],
        }, indent=2))
        return 0

    print(f"intent: {intent}")
    if not hits:
        print("no skill scored above the minimum threshold — this task may need a skill that doesn't exist yet")
        return 0
    for h in hits:
        print(f"  [{h.type or '?':9s}] {h.slug:28s} score={h.score:<5.1f} {'; '.join(h.reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
