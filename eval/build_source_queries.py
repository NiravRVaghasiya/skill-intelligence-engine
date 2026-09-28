"""Extract the source repo's own eval cases as a second, independently-authored query set.

Each ml-ai-skills `evals/<skill>/<case>.yaml` holds a user message (`input`) written to
exercise one skill (`skill`). Those labels are used verbatim — none are edited here.

Usage:
    python -m eval.build_source_queries --source ../ml-ai-skills
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

OUT = Path("eval/queries_source_evals.jsonl")


def _commit(source: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def extract(source: Path) -> list[dict]:
    """One {"query", "gold", "category", "source"} row per eval case, sorted by path."""
    import yaml   # PyYAML ships with python-frontmatter
    commit = _commit(source)
    rows = []
    for path in sorted((source / "evals").glob("*/*.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows.append({
            "query": " ".join(case["input"].split()),
            "gold": case["skill"],
            "category": case["category"],
            "source": f"ml-ai-skills@{commit}:{path.relative_to(source).as_posix()}",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="../ml-ai-skills", help="ml-ai-skills checkout")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rows = extract(Path(args.source))
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"[source-queries] wrote {len(rows)} cases -> {args.out}")


if __name__ == "__main__":
    main()
