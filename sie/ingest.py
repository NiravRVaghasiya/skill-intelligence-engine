"""Load a skill corpus: any directory of SKILL.md files with YAML frontmatter -> Skill objects.

The corpus may describe itself with a `corpus.toml` manifest (name, version, source, layout,
frontmatter key overrides; see sie/corpus.py); without one it is named after its directory.
Every Skill records its provenance: corpus name and version, path relative to the corpus
root, and a content hash. Nothing is dropped silently: every skipped or ignored file and
every metadata problem lands in the LoadReport.

Discovery is the same on every OS: the corpus root is walked (entering symlinked folders,
never looping), each file's root-relative path is matched against the layout pattern (`**` =
zero or more folders, `*`, `?`, `[...]`; case-sensitive, except that a fixed file name such as
SKILL.md matches in any casing), and `Skill.path` keeps the on-disk spelling.

Usage:
    python -m sie.ingest --skills data/skills
    python -m sie.ingest --skills data/skills --expect 38   # exit 1 on any skip / count mismatch
    python -m sie.ingest --skills path/to/corpus --manifest path/to/corpus.toml
"""
from __future__ import annotations
import argparse
import fnmatch
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, time as dtime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

try:
    import frontmatter  # python-frontmatter
except ImportError:  # pragma: no cover
    frontmatter = None

from .corpus import DEFAULT_PATTERN, CorpusConfig, corpus_info, hash_bytes, load_config
from .models import RELATIONSHIPS, CorpusInfo, Skill

VALID_TYPES = {"workflow", "reference"}
VALID_LEVELS = {"beginner", "intermediate", "advanced"}

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_GLOB_CHARS = set("*?[")
_NO_CONFIG = CorpusConfig(name="", root="")      # parse_skill_file without a corpus


@dataclass
class LoadReport:
    """Everything load_corpus saw: loaded skills, skipped/ignored files (with reasons), warnings."""
    root: str
    skills: list[Skill] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (path, reason) — load failures
    ignored: list[tuple[str, str]] = field(default_factory=list)   # (path, reason) — template/duplicate
    warnings: list[str] = field(default_factory=list)
    corpus: CorpusInfo | None = None        # identity + content fingerprint of what loaded
    config: CorpusConfig | None = None      # how the corpus was read (manifest or defaults)


def _as_list(value: Any) -> list[str]:
    """A frontmatter list: YAML list, comma-separated string, or a single scalar.

    A mapping contributes its keys (callers warn about it); empty items are dropped.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (set, frozenset)):     # YAML !!set: no stable order of its own
        value = sorted(value, key=str)
    elif not isinstance(value, (list, tuple, dict)):
        value = [value]
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def _text(value) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _iso(value: Any) -> str:
    """Dates/times -> ISO-8601; anything else -> its stripped string form."""
    if isinstance(value, (date, dtime)):        # datetime is a date subclass
        return value.isoformat()
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    """Frontmatter value -> something `json.dumps` (and a strict JSON encoder) always accepts."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (date, dtime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {_json_key(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(v) for v in value), key=repr)
    return str(value)


def _json_key(key: Any) -> str:
    return key.isoformat() if isinstance(key, (date, dtime)) else str(key)


def _pick(meta: dict, keys: tuple[str, ...], used: set[str],
          accept: Callable[[str, Any], bool] | None = None) -> Any:
    """First non-empty value among `keys` (optionally passing `accept`), marking what it consumed.

    Consumed: the chosen key and any empty alias. Non-empty aliases that lost stay unconsumed,
    so their values survive in `Skill.metadata`.
    """
    chosen = None
    for key in keys:
        if key not in meta:
            continue
        value = meta[key]
        if not _present(value):
            used.add(key)
        elif chosen is None and (accept is None or accept(key, value)):
            used.add(key)
            chosen = value
    return chosen


def _pattern_suffix(pattern: str) -> tuple[str, str]:
    """(fixed file name, "*suffix" suffix) of a glob's last component; "" where not applicable."""
    last = pattern.replace("\\", "/").rsplit("/", 1)[-1]
    if not _GLOB_CHARS & set(last):
        return last, ""
    if last.startswith("*") and not _GLOB_CHARS & set(last[1:]):
        return "", last[1:]
    return "", ""


def _file_slug(path: Path, pattern: str) -> str:
    """Fallback slug: the folder for fixed-name files (SKILL.md), else the file name minus
    the pattern's suffix (`x.skill.md` under `*.skill.md` -> `x`), else the stem."""
    fixed, suffix = _pattern_suffix(pattern)
    if path.name.upper() == "SKILL.MD" or (fixed and path.name.lower() == fixed.lower()):
        return Path(os.path.abspath(path)).parent.name    # normalized, so a root of '.' works
    if suffix and path.name.endswith(suffix) and len(path.name) > len(suffix):
        return path.name[:-len(suffix)]
    return path.stem


def _slug(meta: dict, path: Path, config: CorpusConfig, used: set[str]) -> str:
    """id > slug > name (only when slug-like) > folder/file; a [fields] slug key is read first."""
    override = config.fields.get("slug")

    def accept(key: str, value: Any) -> bool:
        return key != "name" or key == override or bool(_SLUG_RE.match(str(value).strip()))

    value = _pick(meta, config.keys_for("slug"), used, accept)
    return str(value).strip() if value is not None else _file_slug(path, config.pattern)


def _rel_path(path: Path, root: Path | None) -> str:
    """Path relative to the corpus root (posix); `<folder>/<file>` when there is no root."""
    if root is not None:
        for p, r in ((path, root), (path.resolve(), root.resolve())):
            try:
                return p.relative_to(r).as_posix()
            except ValueError:
                continue
    return PurePosixPath(path.parent.name, path.name).as_posix()


def _both(keys: list[str]) -> str:
    quoted = [f"'{k}'" for k in keys]
    if len(quoted) == 2:
        return f"both {quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]} all"


def _relation(meta: dict, slug: str, kind: str, config: CorpusConfig, used: set[str],
              warnings: list[str]) -> tuple[list[str], bool]:
    """Merged, de-duplicated refs of one relationship across its aliases; (refs, declared)."""
    keys = [k for k in config.keys_for(kind) if k in meta]
    if len(keys) > 1:
        warnings.append(f"{slug}: {_both(keys)} given; merged")
    refs: list[str] = []
    for key in keys:
        used.add(key)
        if isinstance(meta[key], dict):
            warnings.append(f"{slug}: '{key}' is a mapping, expected a list; using its keys")
        refs += _as_list(meta[key])
    counts = Counter(refs)
    unique = list(dict.fromkeys(refs))
    for ref in unique:
        if counts[ref] > 1:
            times = "twice" if counts[ref] == 2 else f"{counts[ref]} times"
            warnings.append(f"{slug}: {kind} lists '{ref}' {times}")
    return unique, bool(keys)


def _read_skill(path: Path, root: Path | None, config: CorpusConfig) -> tuple[Skill, list[str]]:
    """Parse one skill file into (Skill, per-file warnings). See parse_skill_file."""
    if frontmatter is None:
        raise RuntimeError("python-frontmatter not installed; pip install -r requirements.txt")
    raw = path.read_bytes()
    # same text frontmatter.load() sees (universal newlines), minus a UTF-8 BOM if present.
    # parse(), not loads(): Post(**metadata) rejects keys YAML reads as non-strings (on:, 1:).
    meta, content = frontmatter.parse(
        raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n"))
    if not meta:
        raise ValueError("no YAML frontmatter")
    used: set[str] = set()
    warnings: list[str] = []

    def scalar(canonical: str, default: Any = "") -> Any:
        value = _pick(meta, config.keys_for(canonical), used)
        return default if value is None else value

    slug = _slug(meta, path, config, used)
    relations: dict[str, list[str]] = {}
    declared: list[str] = []
    for kind in RELATIONSHIPS:
        relations[kind], is_declared = _relation(meta, slug, kind, config, used, warnings)
        if is_declared:
            declared.append(kind)
    caps_keys = [k for k in config.keys_for("capabilities") if k in meta]
    used.update(caps_keys)
    skill = Skill(
        slug=slug,
        skill_type=str(scalar("skill_type", "reference")),
        domain=str(scalar("domain", "unknown")),
        level=str(scalar("level", "intermediate")),
        capabilities=[c for k in caps_keys for c in _as_list(meta[k])],
        risk_level=str(scalar("risk_level", "low")),
        evidence_level=str(scalar("evidence_level", "established-practice")),
        body=content,
        display_name=_text(scalar("display_name", None)),
        description=_text(scalar("description", None)),
        source=config.name,
        source_version=config.version,
        path=_rel_path(path, root),
        content_hash=hash_bytes(raw),
        version=_iso(scalar("version")),
        updated_at=_iso(scalar("updated_at")),
        declared=declared,
        **relations,
    )
    skill.metadata = {_json_key(k): _json_safe(v) for k, v in meta.items() if k not in used}
    warnings += [f"{slug}: frontmatter key {k!r} is not a string (YAML read it as "
                 f"{type(k).__name__}); kept in metadata as '{_json_key(k)}'"
                 for k in meta if not isinstance(k, str)]
    return skill, warnings


def parse_skill_file(path: Path, root: str | Path | None = None,
                     config: CorpusConfig | None = None) -> Skill:
    """Parse a single SKILL.md into a Skill dataclass.

    Args:
        path: the skill file.
        root: corpus root; `Skill.path` is relative to it (else `<folder>/<file>`).
        config: corpus config supplying `source`/`source_version` and [fields] overrides;
            None -> no corpus identity (`source == ""`) and default frontmatter keys.

    Returns:
        Skill with provenance filled in. Per-file warnings (duplicate refs, merged alias
        keys) are only reported through load_corpus_report.

    Raises:
        ValueError: the file has no YAML frontmatter block.
    """
    path = Path(path)
    return _read_skill(path, Path(root) if root is not None else None, config or _NO_CONFIG)[0]


def _ignore_reason(path: Path, root: Path, prefixes: tuple[str, ...]) -> str | None:
    """Template (`_TEMPLATE/`) and hidden (`.git/`, `.claude/`) dirs/files are never skills."""
    if not prefixes:
        return None
    rel = path.relative_to(root)
    if any(part.startswith(prefixes) for part in rel.parent.parts):
        return "template/hidden directory"
    if rel.name.startswith(prefixes):
        return "template/hidden file"
    return None


def _walk(root: Path, report: LoadReport | None = None) -> list[tuple[str, ...]]:
    """Root-relative parts of every file under `root`, entering symlinked folders too.

    A folder that resolves to one of its own enclosing folders (a symlink loop) is not
    entered and is recorded in `report.ignored`; an unreadable folder in `report.skipped`.
    """
    top = os.fspath(root)
    state: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {top: ((), frozenset())}
    files: list[tuple[str, ...]] = []

    def unreadable(e: OSError) -> None:
        if report is not None:
            report.skipped.append((str(e.filename), f"cannot read directory: {e.strerror or e}"))

    for dirpath, dirnames, filenames in os.walk(top, followlinks=True, onerror=unreadable):
        rel, enclosing = state.pop(dirpath)
        enclosing = enclosing | {os.path.realpath(dirpath)}
        keep = []
        for name in sorted(dirnames):
            sub = os.path.join(dirpath, name)
            if os.path.realpath(sub) in enclosing:
                if report is not None:
                    report.ignored.append((str(root.joinpath(*rel, name)),
                                           "symlink loop (points at an enclosing folder; not followed)"))
                continue
            state[sub] = (rel + (name,), enclosing)
            keep.append(name)
        dirnames[:] = keep
        files += [rel + (name,) for name in filenames]
    return files


def _pattern_parts(pattern: str) -> tuple[str, ...]:
    return tuple(c for c in pattern.replace("\\", "/").split("/") if c not in ("", "."))


def _glob_match(parts: tuple[str, ...], comps: tuple[str, ...]) -> bool:
    """Does a root-relative path (`parts`) match a glob split into components (`comps`)?

    `**` matches zero or more folders (as the last component: any file below); `*`, `?` and
    `[...]` match within one name, case-sensitively. A last component without wildcards (a
    fixed file name such as SKILL.md) matches in any casing, so discovery is OS-independent.
    """
    if not comps:
        return not parts
    head, rest = comps[0], comps[1:]
    if head == "**":
        if not rest:
            return bool(parts)
        return any(_glob_match(parts[i:], rest) for i in range(len(parts)))
    if not parts:
        return False
    if not rest:
        if len(parts) != 1:
            return False
        if not _GLOB_CHARS & set(head):
            return parts[0].casefold() == head.casefold()
    return fnmatch.fnmatchcase(parts[0], head) and _glob_match(parts[1:], rest)


def _candidate_paths(root: Path, pattern: str = DEFAULT_PATTERN,
                     report: LoadReport | None = None) -> list[Path]:
    """Files matching `pattern`, shallowest first so canonical copies beat plugin re-bundles.

    With the default pattern and no SKILL.md anywhere, falls back to every non-README `*.md`
    (any casing). Symlink loops and unreadable folders are recorded in `report` when given.
    """
    comps = _pattern_parts(pattern)
    files = _walk(root, report)
    matched = [f for f in files if _glob_match(f, comps)]
    if not matched and pattern == DEFAULT_PATTERN:
        matched = [f for f in files
                   if f[-1].casefold().endswith(".md") and f[-1].upper() != "README.MD"]
    paths = [root.joinpath(*f) for f in matched]
    paths = [p for p in paths if p.is_file() or p.is_symlink()]   # dangling link -> skipped
    return sorted(paths, key=lambda p: (len(p.parts), p.as_posix()))


def _validate(skills: list[Skill]) -> list[str]:
    """Non-fatal metadata problems: unknown enum values and dangling/self slug references."""
    known = {s.slug for s in skills}
    out: list[str] = []
    for s in skills:
        if s.skill_type not in VALID_TYPES:
            out.append(f"{s.slug}: unknown type '{s.skill_type}'")
        if s.level not in VALID_LEVELS:
            out.append(f"{s.slug}: unknown level '{s.level}'")
        for kind in RELATIONSHIPS:
            for ref in getattr(s, kind):
                if ref not in known:
                    out.append(f"{s.slug}: {kind} -> '{ref}' is not a skill in this corpus")
                elif ref == s.slug:
                    out.append(f"{s.slug}: {kind} references itself")
    return out


def load_corpus_report(skills_dir: str | Path, manifest: str | Path | None = None) -> LoadReport:
    """Load every skill file of a corpus, recording every skip and metadata warning.

    Args:
        skills_dir: corpus root (a directory of SKILL.md files, optionally with corpus.toml).
        manifest: explicit corpus.toml; its directory (or its `root`) is then the corpus root.

    Returns:
        LoadReport with skills sorted by slug and `corpus` set (name, version, fingerprint).
        Unparseable files land in `skipped`; templates and duplicate slugs (the shallowest
        copy wins) land in `ignored`. Nothing is dropped without a recorded reason.

    Raises:
        FileNotFoundError / ValueError: an explicit manifest is missing, or a manifest is invalid.
    """
    config = load_config(skills_dir, manifest)
    root = Path(config.root)
    report = LoadReport(root=config.root, config=config)
    if not root.is_dir():
        report.skipped.append((str(root), "corpus directory not found"))
        report.corpus = corpus_info(config, report.skills)
        return report
    seen: dict[str, Path] = {}
    file_warnings: dict[str, list[str]] = {}
    for p in _candidate_paths(root, config.pattern, report):
        reason = _ignore_reason(p, root, config.ignore_prefixes)
        if reason:
            report.ignored.append((str(p), reason))
            continue
        try:
            skill, warnings = _read_skill(p, root, config)
        except Exception as e:
            report.skipped.append((str(p), f"parse error: {e}"))
            continue
        if skill.slug in ("", ".", ".."):
            report.skipped.append((str(p), f"could not derive a slug (got '{skill.slug}'); "
                                           "set id, slug or a slug-like name"))
            continue
        if skill.slug in seen:
            report.ignored.append((str(p), f"duplicate slug '{skill.slug}' (kept {seen[skill.slug]})"))
            continue
        seen[skill.slug] = p
        file_warnings[skill.slug] = warnings
        report.skills.append(skill)
    report.skills.sort(key=lambda s: s.slug)
    report.warnings = [w for s in report.skills for w in file_warnings[s.slug]]
    report.warnings += _validate(report.skills)
    report.corpus = corpus_info(config, report.skills)
    return report


def load_corpus(skills_dir: str | Path, manifest: str | Path | None = None) -> list[Skill]:
    """Load every SKILL.md under skills_dir; load failures are always printed (to stderr)."""
    report = load_corpus_report(skills_dir, manifest)
    for path, reason in report.skipped:
        print(f"[ingest] skipped {path}: {reason}", file=sys.stderr)
    return report.skills


def _fmt(items: list[str]) -> str:
    return ", ".join(items) if items else "-"


def _corpus_label(info: CorpusInfo | None) -> str:
    if info is None:
        return ""
    version = f"@{info.version}" if info.version else ""
    return f"(corpus {info.name}{version}, fingerprint {info.fingerprint}) "


def format_report(report: LoadReport) -> str:
    """Human-readable load report: one row per skill, then skips, warnings, and totals."""
    lines = [f"[ingest] loaded {len(report.skills)} skills from {report.root} "
             f"{_corpus_label(report.corpus)}"
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
        f"{kind}={sum(len(getattr(s, kind)) for s in report.skills)}" for kind in RELATIONSHIPS))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a skill corpus and print a load report.")
    ap.add_argument("--skills", default="data/skills",
                    help="corpus root: a directory of SKILL.md files (optionally with corpus.toml)")
    ap.add_argument("--manifest", default=None,
                    help="explicit corpus.toml (default: <skills>/corpus.toml if present)")
    ap.add_argument("--expect", type=int, default=None,
                    help="exit 1 unless exactly this many skills load with zero load failures")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # never crash on a cp1252 console
    try:
        report = load_corpus_report(args.skills, args.manifest)
    except (OSError, ValueError) as e:
        print(f"[ingest] FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print(format_report(report))
    if args.expect is not None and (len(report.skills) != args.expect or report.skipped):
        print(f"[ingest] FAIL: expected {args.expect} skills and 0 skipped", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
