import json
import os
import re
import sys
from pathlib import Path

import pytest

from sie import ingest
from sie.ingest import format_report, load_corpus, load_corpus_report, main, parse_skill_file
from sie.models import RELATIONSHIPS

CORPUS = Path(__file__).resolve().parents[1] / "data" / "skills"


def _write_skill(root: Path, slug: str, frontmatter: str, body: str = "## Workflow\nsteps") -> Path:
    path = root / slug / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return path


_FULL = """name: rag-pipeline
display_name: RAG Pipeline
description: >
  Use when the user wants to build
  a retrieval pipeline.
type: workflow
domain: llm
level: intermediate
capabilities:
  - chunking-strategy
  - vector-indexing
requires:
conflicts:
related:
  - prompt-engineering"""


def test_parses_all_frontmatter_fields(tmp_path):
    s = parse_skill_file(_write_skill(tmp_path, "rag-pipeline", _FULL))
    assert (s.slug, s.skill_type, s.domain, s.level) == ("rag-pipeline", "workflow", "llm", "intermediate")
    assert s.display_name == "RAG Pipeline"
    assert s.description == "Use when the user wants to build a retrieval pipeline."
    assert s.capabilities == ["chunking-strategy", "vector-indexing"]
    assert s.requires == [] and s.conflicts == []          # empty YAML keys -> [] not None
    assert s.related == ["prompt-engineering"]
    assert s.body.startswith("## Workflow")


def test_slug_falls_back_to_folder_name(tmp_path):
    s = parse_skill_file(_write_skill(tmp_path, "my-skill", "type: reference\ndomain: foundations"))
    assert s.slug == "my-skill"


def test_malformed_file_is_reported_not_dropped(tmp_path):
    _write_skill(tmp_path, "good", "type: workflow\ndomain: llm")
    bad = tmp_path / "bad" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text("no frontmatter here\n", encoding="utf-8")
    report = load_corpus_report(tmp_path)
    assert [s.slug for s in report.skills] == ["good"]
    assert len(report.skipped) == 1 and report.skipped[0][0].endswith("SKILL.md")
    assert "no YAML frontmatter" in report.skipped[0][1]


def test_load_corpus_prints_skips(tmp_path, capsys):
    bad = tmp_path / "bad" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text("---\n: [unclosed\n---\nbody", encoding="utf-8")
    assert load_corpus(tmp_path) == []
    assert "skipped" in capsys.readouterr().err


def test_template_and_duplicates_are_ignored_with_reason(tmp_path):
    _write_skill(tmp_path, "rag-pipeline", "type: workflow\ndomain: llm")
    _write_skill(tmp_path / "plugins" / "llm" / "skills", "rag-pipeline", "type: workflow\ndomain: llm")
    _write_skill(tmp_path, "_TEMPLATE", "type: workflow\ndomain: x")
    report = load_corpus_report(tmp_path)
    assert [s.slug for s in report.skills] == ["rag-pipeline"]
    reasons = sorted(r for _, r in report.ignored)
    assert len(reasons) == 2
    assert reasons[0].startswith("duplicate slug 'rag-pipeline'") and reasons[1] == "template/hidden directory"
    assert report.skipped == []


def test_dangling_reference_and_bad_enum_warn(tmp_path):
    _write_skill(tmp_path, "a", "type: guide\ndomain: llm\nlevel: expert\nrequires:\n  - ghost")
    warnings = load_corpus_report(tmp_path).warnings
    assert any("requires -> 'ghost'" in w for w in warnings)
    assert any("unknown type 'guide'" in w for w in warnings)
    assert any("unknown level 'expert'" in w for w in warnings)


def test_missing_corpus_dir_is_a_skip(tmp_path):
    report = load_corpus_report(tmp_path / "nope")
    assert report.skills == [] and report.skipped[0][1] == "corpus directory not found"


def test_expect_flag_fails_on_count_mismatch(tmp_path, monkeypatch, capsys):
    _write_skill(tmp_path, "a", "type: workflow\ndomain: llm")
    monkeypatch.setattr(sys, "argv", ["sie.ingest", "--skills", str(tmp_path), "--expect", "38"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_real_corpus_loads_38_clean():
    report = load_corpus_report(CORPUS)
    assert len(report.skills) == 38
    assert report.skipped == [] and report.ignored == [] and report.warnings == []
    by_slug = {s.slug: s for s in report.skills}
    assert len({s.domain for s in report.skills}) == 8
    assert by_slug["rag-evaluation"].requires == ["rag-pipeline"]
    assert by_slug["agent-evaluation"].requires == ["agents-and-tools"]
    assert sum(len(s.requires) for s in report.skills) == 2
    assert all(s.skill_type in {"workflow", "reference"} for s in report.skills)
    assert all(s.description.startswith("Use when") for s in report.skills)


# ---- provenance, manifests, aliases (corpus-agnostic ingestion) ------------------------------

def _frontmatter_value(path: Path, key: str) -> str:
    match = re.search(rf"^{key}:\s*(\S+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{key} not in {path}"
    return match.group(1)


def test_real_corpus_provenance():
    report = load_corpus_report(CORPUS)
    s = {x.slug: x for x in report.skills}["rag-evaluation"]
    assert (s.source, s.source_version, s.path) == ("ml-ai-skills", "8328c60", "rag-evaluation/SKILL.md")
    assert s.provenance == "ml-ai-skills@8328c60:rag-evaluation/SKILL.md"
    assert re.fullmatch(r"[0-9a-f]{64}", s.content_hash)
    assert s.updated_at == _frontmatter_value(CORPUS / "rag-evaluation" / "SKILL.md", "last_verified")
    assert {"requires", "related", "conflicts"} <= set(s.declared)
    assert s.declared == [k for k in RELATIONSHIPS if k in s.declared]      # RELATIONSHIPS order
    assert "lifecycle" in s.metadata and json.loads(json.dumps(s.metadata)) == s.metadata
    consumed = {"name", "type", "domain", "level", "description", "display_name", "requires",
                "related", "conflicts", "capabilities", "last_verified", "risk_level",
                "evidence_level"}
    for skill in report.skills:
        assert skill.path == f"{skill.slug}/SKILL.md"           # slug == folder in this corpus
        assert not consumed & set(skill.metadata)               # consumed keys aren't duplicated
        json.dumps(skill.metadata, allow_nan=False)
    assert report.corpus.name == "ml-ai-skills" and report.corpus.version == "8328c60"
    assert report.corpus.n_skills == 38 and re.fullmatch(r"[0-9a-f]{16}", report.corpus.fingerprint)
    assert report.config.name == "ml-ai-skills"


def test_format_report_names_the_corpus_and_every_relationship():
    report = load_corpus_report(CORPUS)
    lines = format_report(report).splitlines()
    assert lines[0] == (f"[ingest] loaded 38 skills from {CORPUS} (corpus ml-ai-skills@8328c60, "
                        f"fingerprint {report.corpus.fingerprint}) (0 skipped, 0 ignored, 0 warnings)")
    edges = next(line for line in lines if line.startswith("  edges:"))
    kinds = [part.split("=")[0] for part in edges.split(":", 1)[1].strip().split(", ")]
    assert kinds == list(RELATIONSHIPS) and "requires=2," in edges


def test_no_manifest_uses_directory_name(tmp_path):
    corpus = tmp_path / "team-skills"
    _write_skill(corpus, "alpha", "type: workflow\ndomain: llm")
    report = load_corpus_report(corpus)
    s = report.skills[0]
    assert (report.corpus.name, report.corpus.version, report.corpus.manifest) == ("team-skills", "", "")
    assert (s.source, s.source_version, s.path) == ("team-skills", "", "alpha/SKILL.md")
    assert s.provenance == "team-skills:alpha/SKILL.md"
    assert format_report(report).splitlines()[0].startswith(
        f"[ingest] loaded 1 skills from {corpus} (corpus team-skills, fingerprint ")


def test_parse_without_root_or_config_has_no_corpus_identity(tmp_path):
    s = parse_skill_file(_write_skill(tmp_path, "solo", "type: workflow\ndomain: llm"))
    assert (s.source, s.source_version, s.path) == ("", "", "solo/SKILL.md")
    assert s.provenance == "unknown:solo/SKILL.md"
    assert re.fullmatch(r"[0-9a-f]{64}", s.content_hash) and s.declared == []


def test_explicit_manifest_custom_pattern_and_field_overrides(tmp_path):
    files = tmp_path / "corpus"
    (files / "guides").mkdir(parents=True)
    (files / "guides" / "alpha.skill.md").write_text(
        "---\nkey: alpha\nkind: workflow\nneeds: [beta]\nrequires: [ignored]\n---\nbody\n",
        encoding="utf-8")
    (files / "beta.skill.md").write_text("---\ntype: workflow\n---\nbody\n", encoding="utf-8")
    (files / "README.md").write_text("---\ntype: workflow\n---\nnot a skill\n", encoding="utf-8")
    (files / "gamma").mkdir()
    (files / "gamma" / "SKILL.md").write_text("---\ntype: workflow\n---\nx\n", encoding="utf-8")
    manifest = tmp_path / "conf" / "acme.toml"
    manifest.parent.mkdir()
    manifest.write_text('name = "acme"\nversion = 3\nroot = "../corpus"\n'
                        '[layout]\npattern = "**/*.skill.md"\n'
                        '[fields]\nslug = "key"\nrequires = "needs"\nskill_type = "kind"\n',
                        encoding="utf-8")
    report = load_corpus_report("does-not-matter", manifest=manifest)
    by_slug = {s.slug: s for s in report.skills}
    assert sorted(by_slug) == ["alpha", "beta"]              # README / SKILL.md don't match
    alpha, beta = by_slug["alpha"], by_slug["beta"]
    assert alpha.requires == ["beta"] and alpha.skill_type == "workflow"
    assert alpha.metadata == {"requires": ["ignored"]}       # the replaced alias is plain metadata
    assert alpha.declared == ["requires"]
    assert alpha.provenance == "acme@3:guides/alpha.skill.md"
    assert beta.skill_type == "reference"                    # `type` is not read once remapped
    assert beta.metadata == {"type": "workflow"}
    assert beta.path == "beta.skill.md"                      # slug = file name minus pattern suffix
    assert report.corpus.name == "acme" and report.corpus.manifest.endswith("acme.toml")
    assert report.warnings == [] and report.skipped == [] and report.ignored == []


def test_invalid_manifest_raises_and_cli_fails_cleanly(tmp_path, monkeypatch, capsys):
    _write_skill(tmp_path, "a", "type: workflow\ndomain: llm")
    (tmp_path / "corpus.toml").write_text('[fields]\nrequirez = "x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field 'requirez'"):
        load_corpus_report(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sie.ingest", "--skills", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1 and "unknown field 'requirez'" in capsys.readouterr().err


def test_cli_manifest_flag(tmp_path, monkeypatch, capsys):
    _write_skill(tmp_path / "files", "a", "type: workflow\ndomain: llm")
    manifest = tmp_path / "m.toml"
    manifest.write_text('name = "flagged"\nversion = "9"\nroot = "files"\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["sie.ingest", "--manifest", str(manifest), "--expect", "1"])
    main()
    assert "(corpus flagged@9, fingerprint " in capsys.readouterr().out


def test_relationship_aliases_merge_with_warning(tmp_path):
    fm = ("skill_type: reference\ndomain: llm\nrequires: [b]\nprerequisites: c\n"
          "soft_requires: [b]\nsee_also: [c]\nconflicts_with: [b]\nalternatives: [c]\n"
          "specialization_of: [b]\nreplaces: [c]")
    _write_skill(tmp_path, "a", fm)
    for slug in ("b", "c"):
        _write_skill(tmp_path, slug, "type: workflow\ndomain: llm")
    report = load_corpus_report(tmp_path)
    a = report.skills[0]
    assert a.skill_type == "reference"
    assert a.requires == ["b", "c"]                          # read in alias order, merged
    assert (a.recommended_before, a.related, a.conflicts) == (["b"], ["c"], ["b"])
    assert (a.alternative_to, a.specializes, a.supersedes) == (["c"], ["b"], ["c"])
    assert a.declared == list(RELATIONSHIPS)
    assert a.metadata == {}
    assert report.warnings == ["a: both 'requires' and 'prerequisites' given; merged"]


def test_three_aliases_warning(tmp_path):
    _write_skill(tmp_path, "a", "requires: [b]\nprerequisites: [b]\ndepends_on: []")
    _write_skill(tmp_path, "b", "type: workflow")
    warnings = load_corpus_report(tmp_path).warnings
    assert warnings[:2] == ["a: 'requires', 'prerequisites' and 'depends_on' all given; merged",
                            "a: requires lists 'b' twice"]


def test_duplicate_refs_are_deduped_with_warning_before_validation(tmp_path):
    _write_skill(tmp_path, "a", "type: workflow\ndomain: llm\nrequires: [b, b, ghost]\n"
                                "related: [b, b, b]\nsupersedes: [a]")
    _write_skill(tmp_path, "b", "type: workflow\ndomain: llm")
    report = load_corpus_report(tmp_path)
    a = report.skills[0]
    assert a.requires == ["b", "ghost"] and a.related == ["b"]
    assert report.warnings == ["a: requires lists 'b' twice",
                               "a: related lists 'b' 3 times",
                               "a: requires -> 'ghost' is not a skill in this corpus",
                               "a: supersedes references itself"]


def test_every_relationship_kind_is_validated(tmp_path):
    fm = "\n".join(f"{kind}: [ghost-{i}]" for i, kind in enumerate(RELATIONSHIPS))
    _write_skill(tmp_path, "a", "type: workflow\n" + fm)
    warnings = load_corpus_report(tmp_path).warnings
    assert warnings == [f"a: {kind} -> 'ghost-{i}' is not a skill in this corpus"
                        for i, kind in enumerate(RELATIONSHIPS)]


def test_empty_and_malformed_relationship_values(tmp_path):
    s = parse_skill_file(_write_skill(tmp_path, "a", "requires:\n  -\n  - b\nrelated: 'c, , d'"))
    assert s.requires == ["b"] and s.related == ["c", "d"]
    assert s.declared == ["requires", "related"]
    _write_skill(tmp_path, "b", "requires:\n  a: '>=1'")      # a mapping: its keys, with a warning
    report = load_corpus_report(tmp_path)
    assert [x.requires for x in report.skills] == [["b"], ["a"]] and report.skipped == []
    assert "b: 'requires' is a mapping, expected a list; using its keys" in report.warnings


def test_crlf_file_parses_and_hashes_like_lf(tmp_path):
    text = b"---\ntype: workflow\ndomain: llm\nrequires: [b]\n---\n## Workflow\nsteps\n"
    lf, crlf = tmp_path / "lf" / "a" / "SKILL.md", tmp_path / "crlf" / "a" / "SKILL.md"
    for path, data in ((lf, text), (crlf, text.replace(b"\n", b"\r\n"))):
        path.parent.mkdir(parents=True)
        path.write_bytes(data)                  # bytes: text mode would translate newlines
    a, b = parse_skill_file(lf), parse_skill_file(crlf)
    assert b"\r\n" in crlf.read_bytes() and b"\r" not in lf.read_bytes()
    assert a.content_hash == b.content_hash
    assert (a.body, a.requires) == (b.body, b.requires)


def test_utf8_bom_file_loads(tmp_path):
    path = tmp_path / "a" / "SKILL.md"
    path.parent.mkdir()
    path.write_bytes(b"\xef\xbb\xbf---\ntype: workflow\n---\nbody\n")
    assert parse_skill_file(path).skill_type == "workflow"


@pytest.mark.parametrize("fm, slug", [
    ("id: from-id\nslug: from-slug\nname: from-name", "from-id"),
    ("slug: from-slug\nname: from-name", "from-slug"),
    ("name: from-name", "from-name"),
    ("name: RAG Pipeline", "folder"),                 # not slug-like -> falls through
    ("name: rag_pipeline.v2", "rag_pipeline.v2"),
    ("id: ''\nname: from-name", "from-name"),         # empty id is skipped
    ("type: workflow", "folder"),
])
def test_slug_resolution_order(tmp_path, fm, slug):
    assert parse_skill_file(_write_skill(tmp_path, "folder", fm)).slug == slug


def test_unused_identity_keys_stay_in_metadata(tmp_path):
    s = parse_skill_file(_write_skill(tmp_path, "folder", "id: x\nslug: y\nname: RAG Pipeline"))
    assert s.slug == "x" and s.metadata == {"slug": "y", "name": "RAG Pipeline"}
    s = parse_skill_file(_write_skill(tmp_path, "other", "name: RAG Pipeline"))
    assert s.slug == "other" and s.metadata == {"name": "RAG Pipeline"}


def test_version_updated_at_and_json_safe_metadata(tmp_path):
    fm = ("version: 2\nupdated_at: 2026-01-02T03:04:05\nlast_verified: 2025-01-01\n"
          "owner: {team: ml, since: 2024-05-06}\nscore: .nan\ntags: !!set {b: null, a: null}\n"
          "lifecycle: stable")
    s = parse_skill_file(_write_skill(tmp_path, "a", fm))
    assert s.version == "2" and s.updated_at == "2026-01-02T03:04:05"
    assert s.metadata == {"last_verified": "2025-01-01",   # lost to updated_at, value kept
                          "owner": {"team": "ml", "since": "2024-05-06"},
                          "score": "nan", "tags": ["a", "b"], "lifecycle": "stable"}
    json.dumps(s.metadata, allow_nan=False)


@pytest.mark.parametrize("key", ["updated_at", "updated", "last_updated", "last_verified"])
def test_updated_at_aliases(tmp_path, key):
    s = parse_skill_file(_write_skill(tmp_path, "a", f"{key}: 2026-09-21"))
    assert s.updated_at == "2026-09-21" and key not in s.metadata


def test_empty_scalar_keys_are_consumed_and_default(tmp_path):
    s = parse_skill_file(_write_skill(tmp_path, "a", "type:\ndomain:\nversion:\nupdated_at:"))
    assert (s.skill_type, s.domain, s.version, s.updated_at) == ("reference", "unknown", "", "")
    assert s.metadata == {}


def test_custom_ignore_prefixes(tmp_path):
    _write_skill(tmp_path, "_underscore-ok", "type: workflow")
    _write_skill(tmp_path / "drafts", "wip", "type: workflow")
    (tmp_path / "corpus.toml").write_text('[layout]\nignore_prefixes = ["drafts"]\n', encoding="utf-8")
    report = load_corpus_report(tmp_path)
    assert [s.slug for s in report.skills] == ["_underscore-ok"]
    assert [r for _, r in report.ignored] == ["template/hidden directory"]


def test_hidden_file_names_are_ignored_for_custom_patterns(tmp_path):
    (tmp_path / "a.skill.md").write_text("---\ntype: workflow\n---\n", encoding="utf-8")
    (tmp_path / "_draft.skill.md").write_text("---\ntype: workflow\n---\n", encoding="utf-8")
    (tmp_path / "corpus.toml").write_text('[layout]\npattern = "*.skill.md"\n', encoding="utf-8")
    report = load_corpus_report(tmp_path)
    assert [s.slug for s in report.skills] == ["a"]
    assert report.ignored[0][1] == "template/hidden file"


def test_missing_corpus_dir_still_reports_corpus(tmp_path):
    report = load_corpus_report(tmp_path / "nope")
    assert report.corpus.name == "nope" and report.corpus.n_skills == 0


# ---- OS-independent discovery -----------------------------------------------------------------

def _link_dir(link: Path, target: Path) -> None:
    """A directory symlink, or a junction on Windows without symlink privilege; else skip."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError):
        pytest.skip("cannot create directory links here")


def test_symlinked_skill_folders_are_loaded(tmp_path):
    corpus, elsewhere = tmp_path / "corpus", tmp_path / "elsewhere"
    _write_skill(corpus, "alpha", "type: workflow")
    _write_skill(elsewhere, "beta", "type: workflow")
    _link_dir(corpus / "beta", elsewhere / "beta")          # what `setup_corpus.py --link` makes
    report = load_corpus_report(corpus)
    assert [(s.slug, s.path) for s in report.skills] == [("alpha", "alpha/SKILL.md"),
                                                         ("beta", "beta/SKILL.md")]
    assert report.skipped == [] and report.ignored == []


def test_symlink_loops_are_recorded_not_followed(tmp_path):
    _write_skill(tmp_path, "alpha", "type: workflow")
    loop = tmp_path / "alpha" / "loop"
    _link_dir(loop, tmp_path)                               # alpha/loop -> the corpus root
    try:
        report = load_corpus_report(tmp_path)
        assert [s.slug for s in report.skills] == ["alpha"]
        assert report.ignored == [(str(loop),
                                   "symlink loop (points at an enclosing folder; not followed)")]
    finally:                                                # don't leave a loop for tmp cleanup
        try:
            os.unlink(loop)
        except OSError:
            os.rmdir(loop)                                  # a Windows junction


def test_a_linked_duplicate_folder_is_a_recorded_duplicate(tmp_path):
    _write_skill(tmp_path, "alpha", "type: workflow")
    (tmp_path / "plugins").mkdir()
    _link_dir(tmp_path / "plugins" / "alpha", tmp_path / "alpha")
    report = load_corpus_report(tmp_path)
    assert [s.path for s in report.skills] == ["alpha/SKILL.md"]    # shallowest copy wins
    assert [r for _, r in report.ignored] == [
        f"duplicate slug 'alpha' (kept {tmp_path / 'alpha' / 'SKILL.md'})"]


def test_unreadable_folders_are_recorded(tmp_path, monkeypatch):
    _write_skill(tmp_path, "alpha", "type: workflow")
    real_walk = os.walk

    def walk(top, **kwargs):
        kwargs["onerror"](PermissionError(13, "Permission denied", str(tmp_path / "locked")))
        yield from real_walk(top, **kwargs)

    monkeypatch.setattr(ingest.os, "walk", walk)
    report = load_corpus_report(tmp_path)
    assert [s.slug for s in report.skills] == ["alpha"]
    assert report.skipped == [(str(tmp_path / "locked"), "cannot read directory: Permission denied")]


def test_fixed_file_name_matches_any_casing_and_keeps_the_on_disk_name(tmp_path):
    for slug, name in (("alpha", "skill.md"), ("beta", "Skill.MD"), ("gamma", "SKILL.md")):
        (tmp_path / slug).mkdir()
        (tmp_path / slug / name).write_text("---\ntype: workflow\n---\nbody\n", encoding="utf-8")
    report = load_corpus_report(tmp_path)
    assert [(s.slug, s.path) for s in report.skills] == [
        ("alpha", "alpha/skill.md"), ("beta", "beta/Skill.MD"), ("gamma", "gamma/SKILL.md")]


def test_markdown_fallback_is_case_insensitive(tmp_path):
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "Intro.MD").write_text("---\ntype: workflow\n---\nx\n", encoding="utf-8")
    (tmp_path / "Readme.md").write_text("---\ntype: workflow\n---\nx\n", encoding="utf-8")
    report = load_corpus_report(tmp_path)
    assert [(s.slug, s.path) for s in report.skills] == [("Intro", "guides/Intro.MD")]


@pytest.mark.parametrize("pattern, path, expected", [
    ("**/SKILL.md", "SKILL.md", True),                  # `**` matches zero folders
    ("**/SKILL.md", "a/b/c/SKILL.md", True),
    ("**/SKILL.md", "a/skill.md", True),                # a fixed file name matches any casing
    ("**/SKILL.md", "a/SKILL.md.bak", False),
    ("*/SKILL.md", "a/SKILL.md", True),
    ("*/SKILL.md", "a/b/SKILL.md", False),              # `*` never crosses '/'
    ("*/SKILL.md", "SKILL.md", False),
    ("skills/**/SKILL.md", "skills/SKILL.md", True),
    ("skills/**/SKILL.md", "skills/x/y/SKILL.md", True),
    ("skills/**/SKILL.md", "Skills/x/SKILL.md", False),  # folder names are case-sensitive
    ("skills/**/SKILL.md", "other/skills/x/SKILL.md", False),
    ("**/*.skill.md", "a/b.skill.md", True),
    ("**/*.skill.md", "a/b.Skill.md", False),           # wildcard names are case-sensitive
    ("?/SKILL.md", "a/SKILL.md", True),
    ("?/SKILL.md", "ab/SKILL.md", False),
    ("[ab]*/SKILL.md", "beta/SKILL.md", True),
    ("[!ab]*/SKILL.md", "beta/SKILL.md", False),
    ("[!ab]*/SKILL.md", "gamma/SKILL.md", True),
    ("docs/**", "docs/a/b.md", True),                   # a trailing `**`: any file below
    ("docs/**", "docs", False),
    ("./x/*.md", "x/a.md", True),
])
def test_glob_match(pattern, path, expected):
    assert ingest._glob_match(tuple(path.split("/")), ingest._pattern_parts(pattern)) is expected


def test_nested_pattern_through_a_manifest(tmp_path):
    for rel in ("skills/a/SKILL.md", "skills/x/b/SKILL.md", "other/c/SKILL.md", "skills/d/NOTES.md"):
        path = tmp_path.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\ntype: workflow\n---\nbody\n", encoding="utf-8")
    (tmp_path / "corpus.toml").write_text('[layout]\npattern = "skills/**/SKILL.md"\n',
                                          encoding="utf-8")
    report = load_corpus_report(tmp_path)
    assert [s.path for s in report.skills] == ["skills/a/SKILL.md", "skills/x/b/SKILL.md"]
    assert report.skipped == [] and report.ignored == []


# ---- frontmatter keys and slugs -------------------------------------------------------------

def test_non_string_frontmatter_keys_are_kept_not_fatal(tmp_path):
    # `on` is a YAML bool, the date a date, 7 an int; `content`/`handler` clashed with Post()
    fm = "type: workflow\non: push\n2024-01-01: note\n7: x\ncontent: c\nhandler: h"
    _write_skill(tmp_path, "a", fm)
    report = load_corpus_report(tmp_path)
    assert report.skipped == []
    a = report.skills[0]
    assert a.body.startswith("## Workflow") and a.skill_type == "workflow"
    assert a.metadata == {"True": "push", "2024-01-01": "note", "7": "x",
                          "content": "c", "handler": "h"}
    json.dumps(a.metadata, allow_nan=False)
    assert report.warnings == [
        "a: frontmatter key True is not a string (YAML read it as bool); kept in metadata as 'True'",
        "a: frontmatter key datetime.date(2024, 1, 1) is not a string (YAML read it as date); "
        "kept in metadata as '2024-01-01'",
        "a: frontmatter key 7 is not a string (YAML read it as int); kept in metadata as '7'"]


def test_skill_at_a_root_given_as_dot_is_named_after_its_folder(tmp_path, monkeypatch):
    repo = tmp_path / "solo-skill"
    repo.mkdir()
    (repo / "SKILL.md").write_text("---\ntype: workflow\nname: Solo Skill\n---\nbody\n",
                                   encoding="utf-8")
    monkeypatch.chdir(repo)
    report = load_corpus_report(".")
    assert [(s.slug, s.path) for s in report.skills] == [("solo-skill", "SKILL.md")]
    assert report.skipped == [] and report.warnings == []
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")
    assert parse_skill_file(Path("..") / "SKILL.md").slug == "solo-skill"      # not '..'


@pytest.mark.parametrize("value", ["'.'", "'..'"])
def test_an_underivable_slug_is_skipped_with_a_reason(tmp_path, value):
    _write_skill(tmp_path, "good", "type: workflow")
    _write_skill(tmp_path, "bad", f"type: workflow\nid: {value}")
    report = load_corpus_report(tmp_path)
    assert [s.slug for s in report.skills] == ["good"]
    assert len(report.skipped) == 1 and report.skipped[0][0].endswith("SKILL.md")
    assert report.skipped[0][1].startswith("could not derive a slug")
