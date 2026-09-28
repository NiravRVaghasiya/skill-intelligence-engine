"""One-command project setup: vendor the ml-ai-skills corpus, then build indexes.

Copies every <slug>/SKILL.md folder from a source ml-ai-skills checkout into
data/skills/, then (optionally) builds the dense + sparse indexes.

Usage:
    python scripts/setup_corpus.py --source "C:/Users/nrvhari/Desktop/AmazonQuick/ml-ai-skills"
    python scripts/setup_corpus.py --source ../ml-ai-skills --no-build
    python scripts/setup_corpus.py --source ../ml-ai-skills --link   # symlink instead of copy
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "skills"


def find_skill_dirs(source: Path) -> list[Path]:
    """Return the canonical skill dirs: top-level <slug>/SKILL.md only.

    The ml-ai-skills repo re-bundles the same SKILL.md files under plugins/
    (per-domain packaging), so a recursive glob finds ~114 files for 38 skills.
    Take only the canonical copies that sit directly under `source`.
    """
    canonical = sorted(p.parent for p in source.glob("*/SKILL.md")
                       if not p.parent.name.startswith("_"))
    if canonical:
        return canonical
    # fallback for a flattened checkout: recurse but skip plugin duplicates
    return sorted({p.parent for p in source.glob("**/SKILL.md")
                   if not p.parent.name.startswith("_")
                   and "plugins" not in p.parts})


def vendor(source: Path, link: bool = False) -> int:
    if not source.exists():
        sys.exit(f"[setup] source not found: {source}")
    skill_dirs = find_skill_dirs(source)
    if not skill_dirs:
        sys.exit(f"[setup] no SKILL.md files found under {source}")
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in skill_dirs:
        target = DEST / d.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
        if link:
            try:
                target.symlink_to(d, target_is_directory=True)
            except OSError:
                shutil.copytree(d, target)  # Windows without symlink privilege
        else:
            shutil.copytree(d, target)
        n += 1
    print(f"[setup] vendored {n} skills into {DEST}")
    return n


def build_indexes() -> None:
    try:
        from sie.router import HybridRouter
    except Exception as e:  # pragma: no cover
        print(f"[setup] skipping build (deps not installed?): {e}")
        print("        run: pip install -r requirements.txt")
        return
    print("[setup] building dense + sparse indexes ...")
    count = HybridRouter(skills_dir=str(DEST)).build()
    print(f"[setup] indexed {count} chunks -> data/chroma/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Vendor the ml-ai-skills corpus and build indexes.")
    ap.add_argument("--source", required=True, help="path to an ml-ai-skills checkout")
    ap.add_argument("--link", action="store_true", help="symlink skill folders instead of copying")
    ap.add_argument("--no-build", action="store_true", help="vendor only; skip index build")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))  # so `import sie` works when run from anywhere
    n = vendor(Path(args.source).resolve(), link=args.link)
    if n != 38:
        print(f"[setup] WARNING: expected 38 skills, vendored {n}. Check the source path.")
    if not args.no_build:
        build_indexes()
    print("[setup] done. Try:  python -m sie.router \"impute missing values\"")


if __name__ == "__main__":
    main()
