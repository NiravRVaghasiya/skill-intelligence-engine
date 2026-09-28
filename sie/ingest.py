"""Parse the ml-ai-skills corpus: YAML frontmatter + body sections -> Skill objects.

Usage:
    python -m sie.ingest --skills data/skills
    python -m sie.ingest --skills data/skills --expect 38   # exit 1 on any skip / count mismatch
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    import frontmatter  # python-frontmatter
except ImportError:  # pragma: no cover
    frontmatter = None

from .models import Skill

VALID_TYPES = {"workflow", "reference"}
VALID_LEVELS = {"beginner", "intermediate", "advanced"}


@dataclass
class LoadReport:
    """Everything load_corpus saw: loaded skills, skipped/ignored files (with reasons), warnings."""
    root: str
    skills: list[Skill] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (path, reason) — load failures
    ignored: list[tuple[str, str]] = field(default_factory=list)   # (path, reason) — template/duplicate
    warnings: list[str] = field(default_factory=list)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def _text(value) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def parse_skill_file(path: Path) -> Skill:
    """Parse a single SKILL.md into a Skill dataclass.

    Raises:
        ValueError: the file has no YAML frontmatter block.
    """
    if frontmatter is None:
        raise RuntimeError("python-frontmatter not installed; pip install -r requirements.txt")
    post = frontmatter.load(path)
    meta = post.metadata
    if not meta:
        raise ValueError("no YAML frontmatter")
    folder = path.parent.name if path.name.upper() == "SKILL.MD" else path.stem
    return Skill(
        slug=str(meta.get("slug") or folder),
        skill_type=str(meta.get("type", "reference")),
        domain=str(meta.get("domain", "unknown")),
        level=str(meta.get("level", "intermediate")),
        capabilities=_as_list(meta.get("capabilities")),
        requires=_as_list(meta.get("requires")),
        related=_as_list(meta.get("related")),
        conflicts=_as_list(meta.get("conflicts")),
        risk_level=str(meta.get("risk_level", "low")),
        evidence_level=str(meta.get("evidence_level", "established-practice")),
        body=post.content,
        display_name=_text(meta.get("display_name")),
        description=_text(meta.get("description")),
    )


def _is_ignored(path: Path, root: Path) -> bool:
    """Template (`_TEMPLATE/`) and hidden (`.git/`, `.claude/`) dirs are never skills."""
    return any(part.startswith(("_", ".")) for part in path.relative_to(root).parent.parts)


def _candidate_paths(root: Path) -> list[Path]:
    """SKILL.md files under root, shallowest first so canonical copies beat plugin re-bundles."""
    paths = list(root.glob("**/SKILL.md")) or [
        p for p in root.glob("**/*.md") if p.name.upper() != "README.MD"]
    return sorted(paths, key=lambda p: (len(p.parts), str(p)))


def _validate(skills: list[Skill]) -> list[str]:
    """Non-fatal metadata problems: unknown enum values and dangling slug references."""
    known = {s.slug for s in skills}
    out: list[str] = []
    for s in skills:
        if s.skill_type not in VALID_TYPES:
            out.append(f"{s.slug}: unknown type '{s.skill_type}'")
        if s.level not in VALID_LEVELS:
            out.append(f"{s.slug}: unknown level '{s.level}'")
        for kind in ("requires", "related", "conflicts"):
            for ref in getattr(s, kind):
                if ref not in known:
                    out.append(f"{s.slug}: {kind} -> '{ref}' is not a skill in this corpus")
                elif ref == s.slug:
                    out.append(f"{s.slug}: {kind} references itself")
    return out


def load_corpus_report(skills_dir: str | Path) -> LoadReport:
    """Load every SKILL.md under skills_dir, recording every skip and metadata warning.

    Args:
        skills_dir: corpus root (vendored `data/skills` or a full ml-ai-skills checkout).

    Returns:
        LoadReport with skills sorted by slug. Unparseable files land in `skipped`;
        templates and duplicate slugs (the shallowest copy wins) land in `ignored`.
        Nothing is dropped without a recorded reason.
    """
    root = Path(skills_dir)
    report = LoadReport(root=str(skills_dir))
    if not root.is_dir():
        report.skipped.append((str(root), "corpus directory not found"))
        return report
    seen: dict[str, Path] = {}
    for p in _candidate_paths(root):
        if _is_ignored(p, root):
            report.ignored.append((str(p), "template/hidden directory"))
            continue
        try:
            skill = parse_skill_file(p)
        except Exception as e:
            report.skipped.append((str(p), f"parse error: {e}"))
            continue
        if skill.slug in seen:
            report.ignored.append((str(p), f"duplicate slug '{skill.slug}' (kept {seen[skill.slug]})"))
            continue
        seen[skill.slug] = p
        report.skills.append(skill)
    report.skills.sort(key=lambda s: s.slug)
    report.warnings = _validate(report.skills)
    return report


def load_corpus(skills_dir: str | Path) -> list[Skill]:
    """Load every SKILL.md under skills_dir; load failures are always printed (to stderr)."""
    report = load_corpus_report(skills_dir)
    for path, reason in report.skipped:
        print(f"[ingest] skipped {path}: {reason}", file=sys.stderr)
    return report.skills


def _fmt(items: list[str]) -> str:
    return ", ".join(items) if items else "-"


def format_report(report: LoadReport) -> str:
    """Human-readable load report: one row per skill, then skips, warnings, and totals."""
    lines = [f"[ingest] loaded {len(report.skills)} skills from {report.root} "
             f"({len(report.skipped)} skipped, {len(report.ignored)} ignored, "
             f"{len(report.warnings)} warnings)",
             f"  {'slug':26s} {'type':9s} {'domain':15s} {'level':12s} "
             f"{'requires':18s} {'conflicts':10s} related"]
    for s in report.skills:
        lines.append(f"  {s.slug:26s} {s.skill_type:9s} {s.domain:15s} {s.level:12s} "
                     f"{_fmt(s.requires):18s} {_fmt(s.conflicts):10s} {_fmt(s.related)}")
    lines += [f"  SKIPPED {path}: {reason}" for path, reason in report.skipped]
    lines += [f"  ignored {path}: {reason}" for path, reason in report.ignored]
    lines += [f"  WARNING {w}" for w in report.warnings]
    by_domain = Counter(s.domain for s in report.skills)
    by_type = Counter(s.skill_type for s in report.skills)
    lines.append("  domains: " + ", ".join(f"{d}={n}" for d, n in by_domain.most_common()))
    lines.append("  types:   " + ", ".join(f"{t}={n}" for t, n in by_type.most_common()))
    lines.append("  edges:   " + ", ".join(
        f"{kind}={sum(len(getattr(s, kind)) for s in report.skills)}"
        for kind in ("requires", "related", "conflicts")))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the skill corpus and print a load report.")
    ap.add_argument("--skills", default="data/skills", help="path to ml-ai-skills corpus")
    ap.add_argument("--expect", type=int, default=None,
                    help="exit 1 unless exactly this many skills load with zero load failures")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # never crash on a cp1252 console
    report = load_corpus_report(args.skills)
    print(format_report(report))
    if args.expect is not None and (len(report.skills) != args.expect or report.skipped):
        print(f"[ingest] FAIL: expected {args.expect} skills and 0 skipped", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
