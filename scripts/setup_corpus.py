"""One-command project setup: vendor a skill corpus, write its manifest, then build indexes.

Copies every canonical <slug>/SKILL.md folder from a source checkout into --dest
(default data/skills/), writes <dest>/corpus.toml (name, version, source, license) so every
loaded skill carries provenance, then (optionally) builds the dense + sparse indexes into
--persist-dir (default <repo>/data/chroma, where the engine and API look by default).

Usage:
    python scripts/setup_corpus.py --source "C:/Users/nrvhari/Desktop/AmazonQuick/ml-ai-skills"
    python scripts/setup_corpus.py --source ../ml-ai-skills --no-build --expect 38
    python scripts/setup_corpus.py --source ../ml-ai-skills --link   # symlink instead of copy
    python scripts/setup_corpus.py --source ../my-skills --dest data/my-skills \\
        --name my-skills --version v1.2 --source-url https://example.com/my-skills --no-build

Manifest values, highest precedence first: command-line flags; the source's own corpus.toml
(name, version, source, license, [fields]; its `root` is where skill folders are read from);
the corpus.toml already in --dest when it describes the same corpus (same name: source,
license, [fields]); then the source directory name and `git rev-parse --short HEAD`.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import date, time as dtime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "skills"
PERSIST = REPO_ROOT / "data" / "chroma"      # == sie.engine.DEFAULT_PERSIST_DIR
MANIFEST_NAME = "corpus.toml"       # == sie.corpus.MANIFEST_NAME (not imported: runs pre-install)
DEFAULT_PATTERN = "**/SKILL.md"     # == sie.corpus.DEFAULT_PATTERN
IGNORE_PREFIXES = ("_", ".")        # == sie.corpus.DEFAULT_IGNORE_PREFIXES
_TOML_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\t": "\\t", "\r": "\\r",
                 "\b": "\\b", "\f": "\\f"}


def _excluded(rel_parts: tuple[str, ...]) -> bool:
    """A folder (parts relative to the source) that never holds a live skill to vendor:
    plugin re-bundles (`plugins/`) and template/draft/hidden folders (`_x/`, `.x/`)."""
    return "plugins" in rel_parts or any(p.startswith(IGNORE_PREFIXES) for p in rel_parts)


def find_skill_dirs(source: Path) -> list[Path]:
    """Return the canonical skill dirs: top-level <slug>/SKILL.md only.

    Some corpora re-bundle the same SKILL.md files under plugins/ (per-domain packaging),
    so a recursive glob can find several copies per skill. Take only the canonical copies
    that sit directly under `source`; for a nested checkout with none, every SKILL.md folder
    outside plugins/ and `_`/`.`-prefixed folders (checked relative to `source`).
    """
    canonical = sorted(p.parent for p in source.glob("*/SKILL.md")
                       if not p.parent.name.startswith(IGNORE_PREFIXES))
    if canonical:
        return canonical
    return sorted({p.parent for p in source.glob("**/SKILL.md")
                   if not _excluded(p.relative_to(source).parent.parts)})


def _left_out(source: Path, skill_dirs: list[Path]) -> list[str]:
    """SKILL.md folders under `source` that are neither vendored nor inside a vendored folder
    nor excluded on purpose (plugins/, `_`/`.` prefixes), as source-relative posix paths."""
    chosen = set(skill_dirs)
    return sorted(p.parent.relative_to(source).as_posix() for p in source.glob("**/SKILL.md")
                  if not _excluded(p.relative_to(source).parent.parts)
                  and not chosen.intersection(p.parents))


def _collisions(source: Path, skill_dirs: list[Path]) -> dict[str, list[str]]:
    """Folder names shared by several skill dirs (casefolded: they clash on Windows/macOS too)."""
    by_name: dict[str, list[str]] = {}
    for d in skill_dirs:
        by_name.setdefault(d.name.casefold(), []).append(d.relative_to(source).as_posix())
    return {name: dirs for name, dirs in sorted(by_name.items()) if len(dirs) > 1}


def vendor(source: Path, link: bool = False, dest: Path = DEST) -> int:
    """Copy (or symlink) every canonical skill folder of `source` into `dest`.

    Returns:
        Number of skill folders vendored. Exits on a missing/empty source, source == dest, or
        two skill folders that would land on the same <dest>/<name> (nothing is copied then).
    """
    if not source.exists():
        sys.exit(f"[setup] source not found: {source}")
    if source.resolve() == dest.resolve():
        sys.exit(f"[setup] --source and --dest are the same directory: {dest}")
    skill_dirs = find_skill_dirs(source)
    if not skill_dirs:
        sys.exit(f"[setup] no SKILL.md files found under {source}")
    clashes = _collisions(source, skill_dirs)
    if clashes:
        sys.exit("[setup] skill folders with the same name would overwrite each other in "
                 f"{dest}: " + "; ".join(" vs ".join(dirs) for dirs in clashes.values())
                 + ". Rename one, or vendor them into separate --dest directories.")
    left = _left_out(source, skill_dirs)
    if left:
        print(f"[setup] WARNING: {len(left)} nested SKILL.md folder(s) are not vendored (only "
              f"top-level <slug>/SKILL.md folders are): {', '.join(left)}")
    dest.mkdir(parents=True, exist_ok=True)
    for d in skill_dirs:
        target = dest / d.name
        if target.exists() or target.is_symlink():
            shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
        if link:
            try:
                target.symlink_to(d, target_is_directory=True)
            except OSError:
                shutil.copytree(d, target)  # Windows without symlink privilege
        else:
            shutil.copytree(d, target)
    n = len(skill_dirs)                     # distinct targets: collisions were rejected above
    print(f"[setup] vendored {n} skills into {dest}")
    _warn_stale(dest, {d.name for d in skill_dirs})
    return n


def _warn_stale(dest: Path, vendored: set[str]) -> None:
    """Skill folders already in dest that the source no longer has are reported, not deleted."""
    stale = sorted(p.parent.name for p in dest.glob("*/SKILL.md")
                   if p.parent.name not in vendored and not p.parent.name.startswith(IGNORE_PREFIXES))
    if stale:
        print(f"[setup] WARNING: {len(stale)} skill folder(s) in {dest} are not in the source "
              f"(left in place): {', '.join(stale)}")


def _git(source: Path, *args: str) -> str | None:
    """stdout of `git -C source <args>`, or None when git is missing or the command fails."""
    try:
        out = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True,
                             errors="replace", timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_version(source: Path) -> str:
    """Commit the source's files come from: `git rev-parse --short HEAD`, plus "-dirty" when
    the source tree has uncommitted changes (`git status --porcelain -- .` is non-empty).

    Returns "" when git or the repo is absent, or when none of the source's files are tracked
    by the repo git finds (e.g. an unpacked copy inside some other working tree, whose HEAD
    says nothing about these files).
    """
    head = (_git(source, "rev-parse", "--short", "HEAD") or "").strip()
    if not head or not (_git(source, "ls-files", "--", ".") or "").strip():
        return ""
    status = _git(source, "status", "--porcelain", "--", ".")
    return f"{head}-dirty" if status and status.strip() else head


def previous_manifest(path: Path, name: str) -> dict:
    """The existing manifest's values if it describes the same corpus (`name`), else {}."""
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    return data if data.get("name") == name else {}


def source_manifest(source: Path) -> dict:
    """The source checkout's own corpus.toml, {} when absent or unreadable (with a warning)."""
    path = source / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        print(f"[setup] WARNING: ignoring the source's unreadable {path}: {e}")
        return {}


def skills_root(source: Path, manifest: dict) -> Path:
    """Where the source's skill folders live: its manifest's `root` (relative to the source),
    else the source itself. Warns when the manifest's [layout] can't be reproduced by vendoring
    <slug>/SKILL.md folders under the default layout."""
    layout = manifest.get("layout")
    if isinstance(layout, dict):
        pattern = layout.get("pattern", DEFAULT_PATTERN)
        prefixes = layout.get("ignore_prefixes", list(IGNORE_PREFIXES))
        if pattern != DEFAULT_PATTERN or prefixes != list(IGNORE_PREFIXES):
            print(f"[setup] WARNING: the source's {MANIFEST_NAME} sets [layout] pattern = "
                  f"{pattern!r}, ignore_prefixes = {prefixes!r}; setup vendors <slug>/SKILL.md "
                  f"folders under the default layout, so the vendored corpus may differ. "
                  f"Compare with: python -m sie.ingest --skills \"{source}\"")
    root = manifest.get("root")
    if not isinstance(root, str) or root.strip() in ("", "."):
        return source
    path = (source / root.strip().replace("\\", "/")).resolve()
    if path.is_dir():
        print(f"[setup] reading skill folders from {path} (root = {root!r} in the source's "
              f"{MANIFEST_NAME})")
        return path
    print(f"[setup] WARNING: the source's {MANIFEST_NAME} sets root = {root!r}, which is not a "
          f"directory; reading skill folders from {source}")
    return source


def _src_str(manifest: dict, key: str) -> str:
    """A string value of the source manifest ("" if absent); other types are warned about."""
    value = manifest.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    print(f"[setup] WARNING: ignoring {key} = {value!r} in the source's {MANIFEST_NAME} "
          f"(expected a string)")
    return ""


def _src_version(manifest: dict) -> str:
    """The source manifest's version as sie reads it (int -> "2", date -> ISO); floats and
    other types are ignored with a warning (a float has already lost digits: 1.10 -> 1.1)."""
    value = manifest.get("version")
    if value is None or isinstance(value, str):
        return (value or "").strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (date, dtime)):
        return value.isoformat()
    print(f"[setup] WARNING: ignoring version = {value!r} in the source's {MANIFEST_NAME} "
          f"(quote it, e.g. version = \"1.10\")")
    return ""


def _toml_str(value: str) -> str:
    """A TOML basic string (quotes, backslashes and control characters escaped)."""
    out = []
    for ch in value:
        if ch in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def manifest_text(name: str, version: str, source_url: str = "", license: str = "",
                  fields: dict[str, str] | None = None, version_from_git: bool = False) -> str:
    """corpus.toml content for a vendored corpus (layout: <slug>/SKILL.md under the root)."""
    comment = "          # upstream commit the files were vendored from" if version_from_git else ""
    lines = ["# Corpus manifest written by scripts/setup_corpus.py. SIE reads any corpus with",
             "# this layout; nothing in sie/ depends on these values.",
             f"name = {_toml_str(name)}",
             f"version = {_toml_str(version)}{comment}",
             f"source = {_toml_str(source_url)}",
             f"license = {_toml_str(license)}",
             "",
             "[layout]",
             f"pattern = {_toml_str(DEFAULT_PATTERN)}"]
    if fields:
        lines += ["", "[fields]"] + [f"{k} = {_toml_str(v)}" for k, v in sorted(fields.items())]
    return "\n".join(lines) + "\n"


def write_manifest(dest: Path, text: str) -> Path:
    """Atomically replace <dest>/corpus.toml (the old file stays intact until the new one is whole)."""
    path = dest / MANIFEST_NAME
    tmp = path.with_name(MANIFEST_NAME + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def _manifest_for(args: argparse.Namespace, source: Path, dest: Path,
                  src: dict[str, Any] | None = None) -> str:
    """Resolve name/version/source/license/[fields]: flags > the source's corpus.toml (`src`)
    > the old dest manifest of the same corpus > the directory name / git."""
    src = src or {}
    from_src: list[str] = []

    def pick(flag: str | None, key: str) -> str:
        if flag is not None:
            return flag
        value = _src_version(src) if key == "version" else _src_str(src, key)
        if value:
            from_src.append(key)
        return value

    name = args.name or pick(None, "name") or source.name
    prev = previous_manifest(dest / MANIFEST_NAME, name)
    version = pick(args.version, "version")
    from_git = args.version is None and not version
    if from_git:
        version = git_version(source)
    kept: list[str] = []

    def keep(flag: str | None, key: str) -> str:
        value = pick(flag, key)
        if flag is None and not value and prev.get(key):
            kept.append(key)
            value = str(prev[key])
        return value

    source_url = keep(args.source_url, "source")
    license = keep(args.license, "license")
    fields = src.get("fields") if isinstance(src.get("fields"), dict) else {}
    if fields:
        from_src.append("[fields]")
    elif isinstance(prev.get("fields"), dict) and prev["fields"]:
        fields = prev["fields"]
        kept.append("[fields]")
    if from_src:
        print(f"[setup] took {', '.join(from_src)} from the source's {MANIFEST_NAME}")
    if kept:
        print(f"[setup] kept {', '.join(kept)} from the existing {MANIFEST_NAME} of '{name}'")
    return manifest_text(name, version, source_url, license,
                         {str(k): str(v) for k, v in fields.items()},
                         version_from_git=from_git and bool(version))


def check_load(dest: Path, vendored: int, folders: Iterable[str] = ()) -> list[str]:
    """Load the vendored corpus the way SIE will, and say what loaded and what didn't.

    Every load failure (report.skipped) is printed, and so is every vendored folder in
    `folders` that yields no loaded skill (a count check alone misses a broken skill when
    stale folders in dest make up the number).

    Returns:
        The vendored folder names with no loaded skill ([] when sie's deps are missing).
        Exits 1 when the manifest in dest is invalid.
    """
    try:
        from sie.ingest import load_corpus_report
    except Exception as e:  # pragma: no cover
        print(f"[setup] skipping load check (deps not installed?): {e}")
        return []
    try:
        report = load_corpus_report(dest)
    except (OSError, ValueError) as e:
        sys.exit(f"[setup] FAIL: {dest} does not load: {e}")
    info = report.corpus
    version = f"@{info.version}" if info.version else " (no version)"
    print(f"[setup] {len(report.skills)} skills load from {dest} "
          f"(corpus {info.name}{version}, fingerprint {info.fingerprint})")
    for path, reason in report.skipped:
        print(f"[setup] WARNING: skipped {path}: {reason}")
    loaded = {PurePosixPath(s.path).parts[0] for s in report.skills if s.path}
    missing = sorted(set(folders) - loaded)
    see = f"see: python -m sie.ingest --skills \"{dest}\""
    if missing:
        print(f"[setup] WARNING: {len(missing)} vendored folder(s) load no skill: "
              f"{', '.join(missing)}; {see}")
    if len(report.skills) < vendored:
        print(f"[setup] WARNING: only {len(report.skills)} of {vendored} vendored skills load; {see}")
    return missing


def build_indexes(dest: Path = DEST, persist_dir: Path = PERSIST) -> None:
    """Build the dense + sparse indexes of `dest` into `persist_dir` (repo-anchored default)."""
    try:
        from sie.router import HybridRouter
    except Exception as e:  # pragma: no cover
        print(f"[setup] skipping build (deps not installed?): {e}")
        print("        run: pip install -r requirements.txt")
        return
    print("[setup] building dense + sparse indexes ...")
    count = HybridRouter(skills_dir=str(dest), persist_dir=str(persist_dir)).build()
    print(f"[setup] indexed {count} chunks -> {persist_dir}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Vendor a skill corpus, write its manifest, build indexes.")
    ap.add_argument("--source", required=True, help="path to a skill corpus checkout")
    ap.add_argument("--dest", default=str(DEST), help="vendor into this directory (default data/skills)")
    ap.add_argument("--persist-dir", default=str(PERSIST),
                    help="build the index into this directory (default <repo>/data/chroma)")
    ap.add_argument("--name", default=None,
                    help="corpus name (default: the source's corpus.toml, else its directory name)")
    ap.add_argument("--version", default=None,
                    help="corpus version (default: the source's corpus.toml, else "
                         "`git rev-parse --short HEAD` of the source (+ '-dirty'), else '')")
    ap.add_argument("--source-url", default=None, help="where the corpus is published (default '')")
    ap.add_argument("--license", default=None, help="corpus license (default '')")
    ap.add_argument("--no-manifest", action="store_true", help=f"do not write <dest>/{MANIFEST_NAME}")
    ap.add_argument("--expect", type=int, default=None,
                    help="exit 1 unless exactly this many skill folders were vendored and all load")
    ap.add_argument("--link", action="store_true", help="symlink skill folders instead of copying")
    ap.add_argument("--no-build", action="store_true", help="vendor only; skip index build")
    args = ap.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))  # so `import sie` works when run from anywhere
    source, dest = Path(args.source).resolve(), Path(args.dest).resolve()
    src_manifest = source_manifest(source)
    files = skills_root(source, src_manifest)
    n = vendor(files, link=args.link, dest=dest)
    if not args.no_manifest:
        print(f"[setup] wrote {write_manifest(dest, _manifest_for(args, source, dest, src_manifest))}")
    elif (dest / MANIFEST_NAME).exists():
        print(f"[setup] --no-manifest: left the existing {dest / MANIFEST_NAME} untouched")
    missing = check_load(dest, n, [d.name for d in find_skill_dirs(files)])
    if args.expect is not None and n != args.expect:
        print(f"[setup] FAIL: expected {args.expect} skills, vendored {n}. Check the source path.",
              file=sys.stderr)
        sys.exit(1)
    if args.expect is not None and missing:
        print(f"[setup] FAIL: {len(missing)} vendored folder(s) load no skill: {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(1)
    if not args.no_build:
        build_indexes(dest, Path(args.persist_dir).resolve())
    print("[setup] done. Try:  python -m sie.router \"impute missing values\"")


if __name__ == "__main__":
    main()
