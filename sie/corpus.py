"""Corpus configuration and identity: where a skill corpus lives, what it is, what it contains.

SIE reads any directory of skill files with YAML frontmatter. A corpus may describe itself
with an optional `corpus.toml` manifest at its root (or passed explicitly); every key is
optional:

    name = "my-skills"               # default: the corpus directory name
    version = "1.4.0"                # release or commit; integers are accepted (2 -> "2");
                                     #   quote decimals (unquoted 1.10 is a float, 1.1)
    source = "https://example.com/my-skills"
    license = "MIT"
    root = "skills"                  # relative to the manifest's directory ('/' or '\')

    [layout]
    pattern = "**/SKILL.md"          # glob, relative to root, selecting skill files
                                     #   (see sie.ingest: `**`, `*`, `?`, `[...]`)
    ignore_prefixes = ["_", "."]     # path parts starting with these are never skills

    [fields]                         # canonical field -> frontmatter key, for corpora that
    requires = "depends"             #   spell a key differently (see FIELD_ALIASES)

Unknown top-level keys are ignored so newer manifests still load; malformed values raise
ValueError. `content_fingerprint` identifies *what* was loaded (file contents), independent of
file order, paths on disk, and load time.
"""
from __future__ import annotations
import hashlib
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .models import CorpusInfo, Skill

MANIFEST_NAME = "corpus.toml"
DEFAULT_PATTERN = "**/SKILL.md"
DEFAULT_IGNORE_PREFIXES: tuple[str, ...] = ("_", ".")

# Canonical Skill field -> frontmatter keys read for it, in order. Scalar fields take the first
# non-empty key; relationship lists merge every key present. A [fields] entry replaces the list
# with its one key (for `slug` it is read first, then these), and a key claimed by a [fields]
# entry is dropped from every other field's list, so one frontmatter key feeds one field.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "slug": ("id", "slug", "name"),
    "skill_type": ("type", "skill_type"),
    "domain": ("domain",),
    "level": ("level",),
    "capabilities": ("capabilities",),
    "display_name": ("display_name",),
    "description": ("description",),
    "risk_level": ("risk_level",),
    "evidence_level": ("evidence_level",),
    "version": ("version",),
    "updated_at": ("updated_at", "updated", "last_updated", "last_verified"),
    "requires": ("requires", "prerequisites", "depends_on"),
    "recommended_before": ("recommended_before", "soft_requires", "recommended"),
    "related": ("related", "see_also"),
    "conflicts": ("conflicts", "conflicts_with"),
    "alternative_to": ("alternative_to", "alternatives"),
    "specializes": ("specializes", "specialization_of"),
    "supersedes": ("supersedes", "replaces"),
}


@dataclass(frozen=True)
class CorpusConfig:
    """How to read one corpus: identity (name/version/source/license) and layout.

    `fields` is excluded from hashing (dicts are unhashable); treat it as read-only.
    """
    name: str
    root: str
    version: str = ""
    source_url: str = ""
    license: str = ""
    pattern: str = DEFAULT_PATTERN
    ignore_prefixes: tuple[str, ...] = DEFAULT_IGNORE_PREFIXES
    fields: dict[str, str] = field(default_factory=dict, hash=False)
    manifest: str = ""              # path of the corpus.toml that was read, "" if none

    def keys_for(self, canonical: str) -> tuple[str, ...]:
        """Frontmatter keys to read for a canonical field, in order, [fields] override applied.

        A key that [fields] assigns to a *different* field is never read here (e.g. with
        `recommended_before = "requires"`, `requires` no longer reads the `requires` key).

        Raises:
            KeyError: `canonical` is not a field in FIELD_ALIASES.
        """
        override = self.fields.get(canonical)
        claimed = {k for c, k in self.fields.items() if c != canonical}
        chain = tuple(k for k in FIELD_ALIASES[canonical] if k not in claimed and k != override)
        if not override:
            return chain
        if canonical == "slug":         # the override is read first, then the usual chain
            return (override,) + chain
        return (override,)


def load_config(skills_dir: str | Path, manifest: str | Path | None = None) -> CorpusConfig:
    """Resolve a corpus's configuration.

    Precedence: an explicit `manifest` (a file, or a directory holding corpus.toml) wins and
    its corpus root defaults to the manifest's directory (so `skills_dir` is not used);
    else `<skills_dir>/corpus.toml` if present (root defaults to `skills_dir`); else
    defaults, named after the directory.

    Args:
        skills_dir: corpus root used when no explicit manifest is given.
        manifest: optional path to a corpus.toml.

    Returns:
        CorpusConfig; `root` is the directory to scan, `manifest` the file read ("" if none).

    Raises:
        FileNotFoundError: an explicit manifest path does not exist.
        ValueError: the manifest is not valid TOML or a value has the wrong type.
    """
    if manifest is not None and str(manifest).strip():
        path = Path(manifest)
        if path.is_dir():
            path = path / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"corpus manifest not found: {path}")
        base, default_root = path.parent, path.parent.as_posix()
    else:
        path = Path(skills_dir) / MANIFEST_NAME
        if not path.is_file():
            return CorpusConfig(name=_dir_name(skills_dir), root=str(skills_dir))
        base, default_root = Path(skills_dir), str(skills_dir)
    return _parse_manifest(_read_toml(path), path, base, default_root)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"invalid corpus manifest {path}: {e}") from e


def _parse_manifest(data: dict[str, Any], path: Path, base: Path, default_root: str) -> CorpusConfig:
    """Validate a decoded corpus.toml into a CorpusConfig (unknown top-level keys ignored)."""
    def bad(msg: str) -> ValueError:
        return ValueError(f"invalid corpus manifest {path}: {msg}")

    root_value = _opt_str(data, "root", bad).replace("\\", "/")    # same manifest on every OS
    root = (base / root_value).as_posix() if root_value else default_root
    name = _opt_str(data, "name", bad)
    if "name" in data and not name:
        raise bad("'name' must be a non-empty string")
    layout = data.get("layout", {})
    if not isinstance(layout, dict):
        raise bad(f"[layout] must be a table, got {type(layout).__name__}")
    return CorpusConfig(
        name=name or _dir_name(root),
        root=root,
        version=_version(data.get("version"), bad),
        source_url=_opt_str(data, "source", bad),
        license=_opt_str(data, "license", bad),
        pattern=_pattern(layout.get("pattern", DEFAULT_PATTERN), bad),
        ignore_prefixes=_prefixes(layout.get("ignore_prefixes", DEFAULT_IGNORE_PREFIXES), bad),
        fields=_fields(data.get("fields", {}), bad),
        manifest=path.as_posix(),
    )


def _opt_str(data: dict[str, Any], key: str, bad) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise bad(f"'{key}' must be a string, got {type(value).__name__}")
    return value.strip()


def _version(value: Any, bad) -> str:
    """Manifest `version` as a string: integers and TOML dates are accepted and stringified.

    A TOML float is rejected: it has already lost digits (`version = 1.10` reads as 1.1).
    """
    if value is None:
        return ""
    if isinstance(value, bool):         # bool is an int subclass; `version = true` is a mistake
        raise bad("'version' must be a string or number, got bool")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        raise bad(f"'version' is a TOML float ({value!r}), which loses digits (1.10 reads as "
                  f"1.1); quote it, e.g. version = \"1.10\"")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (date, dtime)):  # datetime is a date subclass
        return value.isoformat()
    raise bad(f"'version' must be a string or number, got {type(value).__name__}")


def _pattern(value: Any, bad) -> str:
    if not isinstance(value, str) or not value.strip():
        raise bad("[layout] pattern must be a non-empty string")
    value = value.strip()
    parts = value.replace("\\", "/").split("/")
    if (value.startswith(("/", "\\")) or PureWindowsPath(value).drive
            or PurePosixPath(value).is_absolute()):
        raise bad(f"[layout] pattern must be relative to the corpus root, got '{value}'")
    if ".." in parts:
        raise bad(f"[layout] pattern must stay inside the corpus root, got '{value}'")
    return "/".join(parts)              # '\' separators work on every OS, not only Windows


def _prefixes(value: Any, bad) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or \
            not all(isinstance(p, str) and p for p in value):
        raise bad("[layout] ignore_prefixes must be a list of non-empty strings")
    return tuple(value)


def _fields(value: Any, bad) -> dict[str, str]:
    """[fields] table: canonical field -> frontmatter key; each key may be mapped once."""
    if not isinstance(value, dict):
        raise bad(f"[fields] must be a table, got {type(value).__name__}")
    out: dict[str, str] = {}
    owner: dict[str, str] = {}          # frontmatter key -> the canonical field it feeds
    for canonical, key in value.items():
        if canonical not in FIELD_ALIASES:
            raise bad(f"unknown field '{canonical}' in [fields]; expected one of: "
                      + ", ".join(sorted(FIELD_ALIASES)))
        if not isinstance(key, str) or not key.strip():
            raise bad(f"[fields] {canonical} must be a non-empty string (a frontmatter key)")
        key = key.strip()
        if key in owner:
            raise bad(f"[fields] maps both '{owner[key]}' and '{canonical}' to '{key}'")
        out[canonical], owner[key] = key, canonical
    return out


def _dir_name(path: str | Path) -> str:
    """Corpus name fallback: the directory's own name (resolved, so '.' works)."""
    return Path(path).resolve().name or "corpus"


def hash_bytes(data: bytes) -> str:
    """sha256 hex of file bytes with CRLF normalized to LF (same hash on every OS checkout)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def content_hash(path: str | Path) -> str:
    """sha256 hex of a file's bytes, CRLF normalized to LF."""
    return hash_bytes(Path(path).read_bytes())


def content_fingerprint(skills: Iterable[Skill]) -> str:
    """Order-independent identity of a corpus's contents: 16 hex of sha256 over (slug, hash).

    Changes when any skill's file content changes or a skill is added/removed; never depends
    on load order, absolute paths, or load time.
    """
    lines = sorted(f"{s.slug}\0{s.content_hash}\n" for s in skills)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def corpus_info(config: CorpusConfig, skills: list[Skill]) -> CorpusInfo:
    """Identity of a loaded corpus. `loaded_at` is informational only (never fingerprinted)."""
    return CorpusInfo(name=config.name, root=config.root, version=config.version,
                      source_url=config.source_url, license=config.license,
                      manifest=config.manifest, fingerprint=content_fingerprint(skills),
                      n_skills=len(skills), loaded_at=_utc_now())
