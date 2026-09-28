"""Keyword-router baseline: the source repo's scripts/router.py, vendored verbatim.

`router.py` and `skills_lib.py` in this package are byte-identical copies from
ml-ai-skills (see SOURCE.md for commit + sha256). This adapter only points them at a
corpus directory; it never re-implements their scoring.

Two views of the baseline:
    ranking   every skill scored by the router's own score_skill()/classify_intent(),
              sorted exactly as route() sorts ((-score, slug)), with no min-score cut
              and no requires-expansion -> a full ranking (the fairest input to MRR/nDCG)
    shipped   route(task, top_k=5) exactly as a consumer gets it (min_score=1.0,
              prerequisites prepended)
"""
from __future__ import annotations
import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from sie.models import Hit

VENDORED = Path(__file__).resolve().parent


def _exec_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod           # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def load_router(scripts_dir: str | Path = VENDORED) -> ModuleType:
    """Import router.py from `scripts_dir` bound to that dir's own skills_lib.py.

    Several copies (vendored, a live checkout, a historical `git archive`) can be loaded
    side by side: each gets a private skills_lib and sys.path/sys.modules are restored.
    """
    scripts_dir = Path(scripts_dir).resolve()
    tag = f"{abs(hash(str(scripts_dir))):x}"
    saved_path, saved_lib = list(sys.path), sys.modules.pop("skills_lib", None)
    try:
        sys.modules["skills_lib"] = _exec_module(f"_kw_skills_lib_{tag}", scripts_dir / "skills_lib.py")
        return _exec_module(f"_kw_router_{tag}", scripts_dir / "router.py")
    finally:
        sys.path[:] = saved_path
        sys.modules.pop("skills_lib", None)
        if saved_lib is not None:
            sys.modules["skills_lib"] = saved_lib


class KeywordBaseline:
    def __init__(self, skills_dir: str | Path = "data/skills", scripts_dir: str | Path = VENDORED,
                 shipped: bool = False):
        """
        Args:
            skills_dir: corpus root (one <slug>/SKILL.md per skill).
            scripts_dir: directory holding router.py + skills_lib.py.
            shipped: use route() as shipped instead of the full scored ranking.
        """
        self.router = load_router(scripts_dir)
        root = Path(skills_dir).resolve()
        # route() calls sl.load_all_skills() with its default root; bind ours instead
        self.router.sl.load_all_skills = functools.partial(self.router.sl.load_all_skills, root)
        self.skills = self.router.sl.load_all_skills()
        self.shipped = shipped

    def intent(self, query: str) -> str:
        return self.router.classify_intent(query)

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        """Ranked skills for `query`; `snippet` carries the router's own match reasons."""
        if self.shipped:
            _, hits = self.router.route(query, top_k=5)
        else:
            tokens, intent = self.router.tokenize(query), self.router.classify_intent(query)
            hits = [self.router.score_skill(s, tokens, intent) for s in self.skills]
            hits.sort(key=lambda h: (-h.score, h.slug))
        return [Hit(skill_slug=h.slug, score=float(h.score), snippet="; ".join(h.reasons))
                for h in hits[:k]]
