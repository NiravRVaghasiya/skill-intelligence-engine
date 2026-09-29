"""Evaluation query sets: a TOML manifest of labeled JSONL files, validated against the corpus.

`eval/datasets.toml` lists every set `python -m eval.run_eval` scores, so an external or larger
benchmark needs a manifest entry, not code (`--datasets other.toml`). Each `[[dataset]]` table:

    name          unique id (metrics.json, per_query.jsonl)
    path          JSONL file; a relative path resolves against the working directory (run the
                  harness from the repo root, like every other eval command)
    kind          "routing" | "out_of_scope" | "multi_intent" (row shapes below)
    origin        who wrote the queries, one of ORIGINS; or instead
    origin_field  a row field whose value `origin_map` (a table) maps to an origin
    description   one line for the report
    corpus        optional: the corpus whose slugs the labels use; when another corpus is
                  loaded the set is skipped, with the reason printed
    label, title, chart_title
                  optional display names (report prose, section heading, chart row; `{n}` in
                  chart_title becomes the number of queries)

Row shapes (one JSON object per line; "query" is always a non-blank string):
    routing       "gold": slug | [slug, ...]       any listed skill is correct
    out_of_scope  "gold": null                      no skill in the corpus fits
    multi_intent  "gold": [[slug, ...], ...]        acceptable skills per intent, one list per
                                                    intent; an optional "n_intents" must match
A row's own "origin" overrides the manifest. Every gold slug must exist in the corpus.

Usage:
    python -m eval.datasets                        # validate eval/datasets.toml, print a summary
    python -m eval.datasets --datasets other.toml
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("eval/datasets.toml")
KINDS = ("routing", "out_of_scope", "multi_intent")
ORIGINS = ("scaffold", "synthetic-llm", "source-authored", "human-collected")
ORIGIN_DESCRIPTIONS = {
    "scaffold": "pre-existing hand-written queries from the project scaffold (process unrecorded)",
    "synthetic-llm": "written by LLM agents from the skill catalog alone, without running any "
                     "retriever, and frozen before scoring",
    "source-authored": "written by the corpus author as behavioral eval cases (independent of "
                       "SIE, not of the corpus)",
    "human-collected": "independently collected real user queries",
}
_KEYS = {"name", "path", "kind", "origin", "origin_field", "origin_map", "description", "corpus",
         "label", "title", "chart_title"}


@dataclass(frozen=True)
class DatasetSpec:
    """One manifest entry: where a query set lives, what shape it has, who wrote it."""
    name: str
    path: str
    kind: str
    description: str = ""
    origin: str = ""                                   # "" -> origin_field / row-level origin
    origin_field: str = ""
    origin_map: tuple[tuple[str, str], ...] = ()       # (field value, origin) pairs
    corpus: str = ""                                   # "" -> labels fit any corpus
    label: str = ""
    title: str = ""
    chart_title: str = ""


@dataclass
class Dataset:
    """A loaded, validated query set."""
    spec: DatasetSpec
    rows: list[dict]
    sha256: str                                        # of the file bytes, full hex
    row_origins: list[str]                             # resolved origin per row

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def path(self) -> str:
        return self.spec.path

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def label(self) -> str:
        return self.spec.label or self.spec.name

    @property
    def title(self) -> str:
        return self.spec.title or f"{self.label} — `{self.spec.path}`"

    @property
    def chart_title(self) -> str:
        template = self.spec.chart_title or f"{self.label} (n={{n}}, {self.spec.path})"
        return template.replace("{n}", str(self.n))

    def origins(self) -> dict[str, int]:
        """Origin -> number of rows, in ORIGINS order."""
        counts = Counter(self.row_origins)
        return {o: counts[o] for o in ORIGINS if counts[o]}


def golds(gold: str | list[str]) -> list[str]:
    """The acceptable slugs of one routing label (a slug or a list of slugs)."""
    return [gold] if isinstance(gold, str) else list(gold)


def _bad(where: str, msg: str) -> ValueError:
    return ValueError(f"{where}: {msg}")


def _str(entry: dict, key: str, where: str, required: bool = False) -> str:
    value = entry.get(key, "")
    if not isinstance(value, str):
        raise _bad(where, f"'{key}' must be a string, got {type(value).__name__}")
    if required and not value.strip():
        raise _bad(where, f"'{key}' is required")
    return value


def _origin(value: Any, where: str) -> str:
    if value not in ORIGINS:
        raise _bad(where, f"unknown origin {value!r} (expected one of {', '.join(ORIGINS)})")
    return value


def _spec(entry: Any, index: int) -> DatasetSpec:
    where = f"dataset {index + 1}"
    if not isinstance(entry, dict):
        raise _bad(where, "each [[dataset]] must be a table")
    name = _str(entry, "name", where, required=True)
    where = f"dataset '{name}'"
    unknown = sorted(set(entry) - _KEYS)
    if unknown:
        raise _bad(where, f"unknown keys {', '.join(unknown)} (allowed: {', '.join(sorted(_KEYS))})")
    kind = _str(entry, "kind", where, required=True)
    if kind not in KINDS:
        raise _bad(where, f"unknown kind {kind!r} (expected one of {', '.join(KINDS)})")
    origin = _str(entry, "origin", where)
    if origin:
        _origin(origin, where)
    origin_field = _str(entry, "origin_field", where)
    raw_map = entry.get("origin_map", {})
    if not isinstance(raw_map, dict):
        raise _bad(where, "'origin_map' must be a table")
    if raw_map and not origin_field:
        raise _bad(where, "'origin_map' needs 'origin_field'")
    origin_map = tuple(sorted((str(k), _origin(v, f"{where} origin_map[{k!r}]"))
                              for k, v in raw_map.items()))
    return DatasetSpec(name=name, path=_str(entry, "path", where, required=True), kind=kind,
                       description=_str(entry, "description", where), origin=origin,
                       origin_field=origin_field, origin_map=origin_map,
                       corpus=_str(entry, "corpus", where), label=_str(entry, "label", where),
                       title=_str(entry, "title", where),
                       chart_title=_str(entry, "chart_title", where))


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[DatasetSpec]:
    """Parse and validate a datasets manifest.

    Returns:
        The specs in manifest order.

    Raises:
        FileNotFoundError: no such manifest. ValueError: malformed TOML, no [[dataset]],
        duplicate names, unknown keys/kinds/origins, or wrong value types.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: invalid TOML: {e}") from e
    entries = data.get("dataset")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: no [[dataset]] tables")
    specs = [_spec(entry, i) for i, entry in enumerate(entries)]
    dupes = sorted(n for n, c in Counter(s.name for s in specs).items() if c > 1)
    if dupes:
        raise ValueError(f"{path}: duplicate dataset names: {', '.join(dupes)}")
    return specs


def read_rows(path: str | Path) -> tuple[list[dict], str]:
    """(rows, sha256 of the file bytes) of a JSONL file; blank lines are skipped.

    Raises:
        FileNotFoundError: missing file. ValueError: a line is not a JSON object.
    """
    data = Path(path).read_bytes()
    rows = []
    for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
        rows.append(row)
    return rows, hashlib.sha256(data).hexdigest()


def _slug_list(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(s, str) and s.strip() for s in value))


def _gold_slugs(kind: str, gold: Any, where: str) -> list[str]:
    """Validate one row's gold label for `kind`; returns the slugs it names."""
    if kind == "routing":
        if isinstance(gold, str) and gold.strip():
            return [gold]
        if _slug_list(gold):
            return list(gold)
        raise ValueError(f"{where}: routing gold must be a slug or a non-empty list of slugs")
    if kind == "out_of_scope":
        if gold is not None:
            raise ValueError(f"{where}: out_of_scope gold must be null")
        return []
    if not (isinstance(gold, list) and gold and all(_slug_list(g) for g in gold)):
        raise ValueError(f"{where}: multi_intent gold must be a non-empty list of non-empty "
                         f"slug lists")
    return [s for g in gold for s in g]


def _row_origin(spec: DatasetSpec, row: dict, where: str) -> str:
    if "origin" in row:
        return _origin(row["origin"], where)
    if spec.origin_field:
        value = row.get(spec.origin_field)
        mapped = dict(spec.origin_map).get(str(value)) if value is not None else None
        if mapped is None and spec.origin:
            return spec.origin
        if mapped is None:
            raise ValueError(f"{where}: {spec.origin_field}={value!r} has no origin_map entry")
        return mapped
    if spec.origin:
        return spec.origin
    raise ValueError(f"{where}: no origin (set 'origin' or 'origin_field' in the manifest, "
                     f"or an 'origin' field on the row)")


def load_dataset(spec: DatasetSpec, slugs: set[str] | None = None) -> Dataset:
    """Read and validate one query set.

    Args:
        spec: the manifest entry.
        slugs: the loaded corpus's skill slugs; None skips the gold-slug check.

    Raises:
        FileNotFoundError: missing file. ValueError: bad row shape or origin, an empty set, or
        gold slugs that are not skills of the corpus (all of them are listed).
    """
    rows, sha = read_rows(spec.path)
    if not rows:
        raise ValueError(f"dataset '{spec.name}': {spec.path} has no rows")
    origins, unknown = [], set()
    for i, row in enumerate(rows, 1):
        where = f"dataset '{spec.name}' row {i}"
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{where}: 'query' must be a non-blank string")
        if "gold" not in row:
            raise ValueError(f"{where}: missing 'gold'")
        named = _gold_slugs(spec.kind, row["gold"], where)
        if spec.kind == "multi_intent" and "n_intents" in row and row["n_intents"] != len(row["gold"]):
            raise ValueError(f"{where}: n_intents={row['n_intents']!r} but gold lists "
                             f"{len(row['gold'])} intents")
        if slugs is not None:
            unknown.update(s for s in named if s not in slugs)
        origins.append(_row_origin(spec, row, where))
    if unknown:
        raise ValueError(f"dataset '{spec.name}': gold slugs not in the corpus: "
                         f"{', '.join(sorted(unknown))}")
    return Dataset(spec=spec, rows=rows, sha256=sha, row_origins=origins)


def load_datasets(manifest: str | Path = DEFAULT_MANIFEST, slugs: set[str] | None = None,
                  corpus_name: str | None = None) -> tuple[list[Dataset], list[str]]:
    """Load every set of a manifest that applies to the loaded corpus.

    Args:
        manifest: datasets.toml path.
        slugs: skill slugs of the loaded corpus (gold labels are checked against them).
        corpus_name: the loaded corpus's name; a set declaring another `corpus` is skipped.

    Returns:
        (datasets in manifest order, one printable reason per skipped set).
    """
    loaded, skipped = [], []
    for spec in load_manifest(manifest):
        if spec.corpus and corpus_name is not None and spec.corpus != corpus_name:
            skipped.append(f"skipped dataset '{spec.name}': its labels are for corpus "
                           f"'{spec.corpus}', but '{corpus_name}' is loaded")
            continue
        loaded.append(load_dataset(spec, slugs))
    return loaded, skipped


def main() -> None:
    from sie.ingest import load_corpus_report
    ap = argparse.ArgumentParser(description="Validate an evaluation datasets manifest.")
    ap.add_argument("--datasets", default=str(DEFAULT_MANIFEST), help="datasets manifest (TOML)")
    ap.add_argument("--skills", default="data/skills", help="corpus the labels refer to")
    args = ap.parse_args()
    report = load_corpus_report(args.skills)
    name = report.corpus.name if report.corpus else None
    try:
        datasets, skipped = load_datasets(args.datasets, {s.slug for s in report.skills}, name)
    except (OSError, ValueError) as e:
        raise SystemExit(f"[datasets] FAIL: {e}")
    for ds in datasets:
        origins = ", ".join(f"{o}={n}" for o, n in ds.origins().items())
        print(f"[datasets] {ds.name:14s} {ds.kind:13s} n={ds.n:<4d} {origins:32s} "
              f"sha256 {ds.sha256[:12]}  {ds.path}")
    for reason in skipped:
        print(f"[datasets] {reason}")


if __name__ == "__main__":
    main()
