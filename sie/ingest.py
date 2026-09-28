"""Parse the ml-ai-skills corpus: YAML frontmatter + body sections -> Skill objects.

Usage:
    python -m sie.ingest --skills data/skills
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

try:
    import frontmatter  # python-frontmatter
except ImportError:  # pragma: no cover
    frontmatter = None

from .models import Skill

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def parse_skill_file(path: Path) -> Skill:
    """Parse a single SKILL.md into a Skill dataclass."""
    if frontmatter is None:
        raise RuntimeError("python-frontmatter not installed; pip install -r requirements.txt")
    post = frontmatter.load(path)
    meta = post.metadata
    slug = path.parent.name if path.name.upper() == "SKILL.MD" else path.stem
    return Skill(
        slug=meta.get("slug", slug),
        skill_type=meta.get("type", "reference"),
        domain=meta.get("domain", "unknown"),
        level=meta.get("level", "intermediate"),
        capabilities=_as_list(meta.get("capabilities")),
        requires=_as_list(meta.get("requires")),
        related=_as_list(meta.get("related")),
        conflicts=_as_list(meta.get("conflicts")),
        risk_level=meta.get("risk_level", "low"),
        evidence_level=meta.get("evidence_level", "established-practice"),
        body=post.content,
    )


def load_corpus(skills_dir: str | Path) -> list[Skill]:
    """Load every SKILL.md under skills_dir."""
    root = Path(skills_dir)
    paths = sorted(root.glob("**/SKILL.md")) or sorted(root.glob("**/*.md"))
    skills = []
    for p in paths:
        if p.name.startswith("_"):
            continue
        try:
            skills.append(parse_skill_file(p))
        except Exception as e:  # pragma: no cover
            print(f"[ingest] skipped {p}: {e}")
    return skills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="data/skills", help="path to ml-ai-skills corpus")
    args = ap.parse_args()
    corpus = load_corpus(args.skills)
    print(f"[ingest] loaded {len(corpus)} skills from {args.skills}")
    for s in corpus[:5]:
        print(f"  - {s.slug} [{s.skill_type}/{s.domain}] requires={s.requires}")


if __name__ == "__main__":
    main()
