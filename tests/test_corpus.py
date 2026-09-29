import importlib.util
import re
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from sie.corpus import (FIELD_ALIASES, MANIFEST_NAME, CorpusConfig, content_fingerprint,
                        content_hash, corpus_info, load_config)
from sie.ingest import load_corpus_report
from sie.models import RELATIONSHIPS, Skill

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"


def _manifest(directory: Path, text: str, name: str = MANIFEST_NAME) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _skill(slug: str, content_hash: str = "") -> Skill:
    return Skill(slug=slug, skill_type="workflow", domain="d", level="beginner",
                 content_hash=content_hash)


# ---- load_config ----------------------------------------------------------------------------

def test_no_manifest_defaults_to_directory_name(tmp_path):
    corpus = tmp_path / "team-skills"
    corpus.mkdir()
    cfg = load_config(corpus)
    assert (cfg.name, cfg.version, cfg.manifest, cfg.root) == ("team-skills", "", "", str(corpus))
    assert (cfg.pattern, cfg.ignore_prefixes, cfg.fields) == ("**/SKILL.md", ("_", "."), {})


def test_missing_directory_still_gets_a_config(tmp_path):
    cfg = load_config(tmp_path / "nope")
    assert cfg.name == "nope" and cfg.root == str(tmp_path / "nope")


def test_auto_discovered_manifest(tmp_path):
    _manifest(tmp_path, 'name = "acme"\nversion = 2\nsource = "https://x.test/acme"\n'
                        'license = "Apache-2.0"\nfuture_key = [1, 2]\n\n[future_table]\na = 1\n')
    cfg = load_config(tmp_path)
    assert (cfg.name, cfg.version, cfg.source_url, cfg.license) == \
        ("acme", "2", "https://x.test/acme", "Apache-2.0")     # numeric version -> "2"
    assert cfg.root == str(tmp_path)                            # auto-discovered: root = skills_dir
    assert cfg.manifest.endswith(MANIFEST_NAME)


@pytest.mark.parametrize("value, expected", [('"1.4.0"', "1.4.0"), ('"1.10"', "1.10"), ("2", "2"),
                                             ("2026-09-21", "2026-09-21"), ('"  v3 "', "v3")])
def test_version_values_are_strings(tmp_path, value, expected):
    _manifest(tmp_path, f"version = {value}\n")
    assert load_config(tmp_path).version == expected


@pytest.mark.parametrize("value", ["1.5", "1.10", "2.0"])
def test_float_version_is_rejected_not_rewritten(tmp_path, value):
    _manifest(tmp_path, f"version = {value}\n")        # 1.10 would silently become "1.1"
    with pytest.raises(ValueError, match=r"'version' is a TOML float .*quote it"):
        load_config(tmp_path)


def test_manifest_without_name_uses_root_dir_name(tmp_path):
    _manifest(tmp_path / "my-corpus", 'version = "x"\n')
    assert load_config(tmp_path / "my-corpus").name == "my-corpus"


def test_explicit_manifest_wins_and_roots_at_its_directory(tmp_path):
    _manifest(tmp_path / "auto", 'name = "auto"\n')
    explicit = _manifest(tmp_path / "elsewhere", 'name = "explicit"\n', name="custom.toml")
    cfg = load_config(tmp_path / "auto", manifest=explicit)
    assert cfg.name == "explicit"
    assert Path(cfg.root) == tmp_path / "elsewhere"
    assert Path(cfg.manifest) == explicit


def test_explicit_manifest_root_is_relative_to_the_manifest(tmp_path):
    explicit = _manifest(tmp_path / "conf", 'name = "c"\nroot = "../skills"\n')
    cfg = load_config("ignored", manifest=explicit)
    assert Path(cfg.root).resolve() == (tmp_path / "skills").resolve()


def test_auto_discovered_manifest_root_key(tmp_path):
    _manifest(tmp_path, 'root = "inner"\n')
    assert Path(load_config(tmp_path).root) == tmp_path / "inner"


def test_manifest_may_be_given_as_its_directory(tmp_path):
    _manifest(tmp_path / "c", 'name = "by-dir"\n')
    assert load_config("unused", manifest=tmp_path / "c").name == "by-dir"


def test_explicit_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus manifest not found"):
        load_config(tmp_path, manifest=tmp_path / "missing.toml")


def test_layout_and_fields(tmp_path):
    _manifest(tmp_path, '[layout]\npattern = "**/*.skill.md"\nignore_prefixes = ["drafts"]\n'
                        '[fields]\nrequires = "needs"\nslug = "key"\n')
    cfg = load_config(tmp_path)
    assert cfg.pattern == "**/*.skill.md" and cfg.ignore_prefixes == ("drafts",)
    assert cfg.fields == {"requires": "needs", "slug": "key"}
    assert cfg.keys_for("requires") == ("needs",)                   # replaces the alias list
    assert cfg.keys_for("slug") == ("key", "id", "slug", "name")    # read first, then the chain
    assert cfg.keys_for("related") == FIELD_ALIASES["related"]


@pytest.mark.parametrize("text, message", [
    ("name = \n", "invalid corpus manifest"),                          # not TOML
    ("name = 5\n", "'name' must be a string"),
    ('name = ""\n', "'name' must be a non-empty string"),
    ("version = true\n", "'version' must be a string or number, got bool"),
    ("version = [1]\n", "'version' must be a string or number"),
    ("source = 1\n", "'source' must be a string"),
    ("license = {a = 1}\n", "'license' must be a string"),
    ("root = 3\n", "'root' must be a string"),
    ('layout = "flat"\n', "[layout] must be a table"),
    ("[layout]\npattern = 3\n", "pattern must be a non-empty string"),
    ('[layout]\npattern = ""\n', "pattern must be a non-empty string"),
    ('[layout]\npattern = "/abs/*.md"\n', "must be relative"),
    ('[layout]\npattern = "C:/x/*.md"\n', "must be relative"),
    ('[layout]\npattern = "../*/SKILL.md"\n', "must stay inside"),
    ('[layout]\nignore_prefixes = "_"\n', "ignore_prefixes must be a list"),
    ('[layout]\nignore_prefixes = [""]\n', "ignore_prefixes must be a list"),
    ('fields = ["requires"]\n', "[fields] must be a table"),
    ('[fields]\nrequirez = "needs"\n', "unknown field 'requirez'"),
    ("[fields]\nrequires = 3\n", "requires must be a non-empty string"),
    ('[fields]\nrequires = "deps"\nrelated = "deps"\n', "maps both 'requires' and 'related'"),
])
def test_invalid_manifest_raises_value_error(tmp_path, text, message):
    _manifest(tmp_path, text)
    with pytest.raises(ValueError, match=re.escape(message)):
        load_config(tmp_path)


def test_a_key_claimed_by_fields_leaves_its_default_field():
    cfg = CorpusConfig(name="x", root="r", fields={"recommended_before": "requires"})
    assert cfg.keys_for("recommended_before") == ("requires",)
    assert cfg.keys_for("requires") == ("prerequisites", "depends_on")     # 'requires' is claimed
    cfg = CorpusConfig(name="x", root="r", fields={"domain": "type"})
    assert (cfg.keys_for("domain"), cfg.keys_for("skill_type")) == (("type",), ("skill_type",))
    cfg = CorpusConfig(name="x", root="r", fields={"display_name": "name", "slug": "key"})
    assert cfg.keys_for("slug") == ("key", "id", "slug")                   # override still first


def _two_skills(root: Path, alpha: str, beta: str) -> None:
    for slug, fm in (("alpha", alpha), ("beta", beta)):
        (root / slug).mkdir(parents=True)
        (root / slug / "SKILL.md").write_text(f"---\n{fm}\n---\nbody\n", encoding="utf-8")


def test_remapped_key_feeds_one_relationship_only(tmp_path):
    _two_skills(tmp_path, "type: workflow", "type: workflow\nrequires: [alpha]")
    _manifest(tmp_path, '[fields]\nrecommended_before = "requires"\n')
    report = load_corpus_report(tmp_path)
    beta = {s.slug: s for s in report.skills}["beta"]
    assert (beta.requires, beta.recommended_before) == ([], ["alpha"])    # not a hard edge too
    assert beta.declared == ["recommended_before"] and report.warnings == []


def test_remapped_scalar_key_feeds_one_field_only(tmp_path):
    _two_skills(tmp_path, "type: nlp", "type: cv\nskill_type: workflow")
    _manifest(tmp_path, '[fields]\ndomain = "type"\n')
    by_slug = {s.slug: s for s in load_corpus_report(tmp_path).skills}
    assert (by_slug["alpha"].domain, by_slug["alpha"].skill_type) == ("nlp", "reference")
    assert (by_slug["beta"].domain, by_slug["beta"].skill_type) == ("cv", "workflow")


def test_backslash_pattern_and_root_are_normalized(tmp_path):
    skill = tmp_path / "sub" / "skills" / "sk" / "a" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\ntype: workflow\n---\nbody\n", encoding="utf-8")
    _manifest(tmp_path, "root = 'sub\\skills'\n[layout]\npattern = 'sk\\*\\SKILL.md'\n")
    cfg = load_config(tmp_path)
    assert cfg.pattern == "sk/*/SKILL.md"
    assert cfg.root == (tmp_path / "sub" / "skills").as_posix()
    report = load_corpus_report(tmp_path)                   # the same result on every OS
    assert [(s.slug, s.path) for s in report.skills] == [("a", "sk/a/SKILL.md")]


def test_every_relationship_is_overridable():
    assert set(RELATIONSHIPS) <= set(FIELD_ALIASES)
    keys = [k for aliases in FIELD_ALIASES.values() for k in aliases]
    assert len(keys) == len(set(keys))          # default alias lists never overlap


def test_config_is_hashable_and_comparable():
    a = CorpusConfig(name="x", root="r", fields={"requires": "needs"})
    b = CorpusConfig(name="x", root="r", fields={"requires": "needs"})
    assert a == b and hash(a) == hash(b)


def test_vendored_manifest():
    cfg = load_config(CORPUS)
    assert (cfg.name, cfg.version, cfg.license) == ("ml-ai-skills", "8328c60", "MIT")
    assert cfg.source_url == "https://github.com/NiravRVaghasiya/ml-ai-skills"
    assert cfg.pattern == "**/SKILL.md" and cfg.fields == {}


# ---- hashing / identity ---------------------------------------------------------------------

def test_content_hash_normalizes_crlf(tmp_path):
    lf, crlf, other = tmp_path / "lf.md", tmp_path / "crlf.md", tmp_path / "other.md"
    lf.write_bytes(b"---\nname: a\n---\nbody\n")
    crlf.write_bytes(b"---\r\nname: a\r\n---\r\nbody\r\n")
    other.write_bytes(b"---\nname: a\n---\nbody!\n")
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash(lf))
    assert content_hash(lf) == content_hash(crlf) == content_hash(str(crlf))
    assert content_hash(other) != content_hash(lf)


def test_fingerprint_is_order_independent_and_content_sensitive():
    skills = [_skill("a", "1" * 64), _skill("b", "2" * 64), _skill("c", "3" * 64)]
    fp = content_fingerprint(skills)
    assert re.fullmatch(r"[0-9a-f]{16}", fp)
    assert content_fingerprint(reversed(skills)) == fp
    assert content_fingerprint([_skill("a", "1" * 64), _skill("b", "9" * 64),
                                _skill("c", "3" * 64)]) != fp       # one file changed
    assert content_fingerprint(skills[:2]) != fp                     # one skill removed
    assert content_fingerprint([]) == content_fingerprint([])


def test_corpus_info_carries_identity_and_utc_timestamp(tmp_path):
    cfg = CorpusConfig(name="acme", root=str(tmp_path), version="7", source_url="u",
                       license="MIT", manifest="m.toml")
    skills = [_skill("a", "1" * 64)]
    info = corpus_info(cfg, skills)
    assert (info.name, info.root, info.version, info.source_url, info.license, info.manifest,
            info.n_skills) == ("acme", str(tmp_path), "7", "u", "MIT", "m.toml", 1)
    assert info.fingerprint == content_fingerprint(skills)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", info.loaded_at)
    assert corpus_info(cfg, skills).fingerprint == info.fingerprint  # time never fingerprinted


def test_real_corpus_fingerprint_is_stable_and_file_order_free():
    a, b = load_corpus_report(CORPUS), load_corpus_report(CORPUS)
    assert a.corpus.fingerprint == b.corpus.fingerprint
    assert a.corpus.fingerprint == content_fingerprint(list(reversed(a.skills)))
    assert a.corpus.n_skills == 38


# ---- scripts/setup_corpus.py ----------------------------------------------------------------

@pytest.fixture
def setup_corpus(monkeypatch):
    spec = importlib.util.spec_from_file_location("setup_corpus_under_test",
                                                  ROOT / "scripts" / "setup_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "git_version", lambda source: "abc1234")
    return mod


def _source(tmp_path: Path, name: str = "team-skills") -> Path:
    src = tmp_path / name
    for slug in ("alpha", "beta"):
        (src / slug).mkdir(parents=True)
        (src / slug / "SKILL.md").write_text(
            f"---\nname: {slug}\ntype: workflow\ndomain: d\nlevel: beginner\n---\n## Workflow\nx\n",
            encoding="utf-8")
    (src / "_TEMPLATE").mkdir()
    (src / "_TEMPLATE" / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    return src


def test_setup_writes_manifest_without_building(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--expect", "2"])
    assert sorted(p.name for p in dest.iterdir()) == ["alpha", "beta", "corpus.toml"]
    cfg = load_config(dest)
    assert (cfg.name, cfg.version, cfg.source_url, cfg.license) == ("team-skills", "abc1234", "", "")
    report = load_corpus_report(dest)
    assert [s.provenance for s in report.skills] == ["team-skills@abc1234:alpha/SKILL.md",
                                                     "team-skills@abc1234:beta/SKILL.md"]
    assert report.warnings == [] and report.skipped == []
    out = capsys.readouterr().out
    assert "vendored 2 skills" in out and "building" not in out
    assert f"2 skills load from {dest} (corpus team-skills@abc1234, fingerprint " \
           f"{report.corpus.fingerprint})" in out


def test_setup_warns_when_vendored_skills_do_not_load(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    (src / "beta" / "SKILL.md").write_text("no frontmatter\n", encoding="utf-8")
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    assert "WARNING: only 1 of 2 vendored skills load" in capsys.readouterr().out


def test_setup_fails_when_the_dest_manifest_is_invalid(setup_corpus, tmp_path):
    src, dest = _source(tmp_path), tmp_path / "dest"
    _manifest(dest, '[fields]\nrequirez = "x"\n')
    with pytest.raises(SystemExit, match="unknown field 'requirez'"):
        setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--no-manifest"])


def test_setup_flags_override_defaults(setup_corpus, tmp_path):
    src, dest = _source(tmp_path), tmp_path / "dest"
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--name", "acme",
                       "--version", "v1.2", "--source-url", "https://x.test/acme",
                       "--license", "MIT"])
    cfg = load_config(dest)
    assert (cfg.name, cfg.version, cfg.source_url, cfg.license) == \
        ("acme", "v1.2", "https://x.test/acme", "MIT")


def test_setup_no_manifest(setup_corpus, tmp_path):
    src, dest = _source(tmp_path), tmp_path / "dest"
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--no-manifest"])
    assert not (dest / MANIFEST_NAME).exists()
    assert load_config(dest).name == "dest"


def test_setup_expect_mismatch_exits_1(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    with pytest.raises(SystemExit) as exc:
        setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--expect", "38"])
    assert exc.value.code == 1
    assert "expected 38 skills, vendored 2" in capsys.readouterr().err


def test_setup_revendor_keeps_source_and_license_of_same_corpus(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    _manifest(dest, 'name = "team-skills"\nversion = "old"\nsource = "https://x.test/t"\n'
                    'license = "MIT"\n[fields]\nrequires = "needs"\n')
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    cfg = load_config(dest)
    assert (cfg.version, cfg.source_url, cfg.license) == ("abc1234", "https://x.test/t", "MIT")
    assert cfg.fields == {"requires": "needs"}
    assert "kept source, license, [fields]" in capsys.readouterr().out


def test_setup_does_not_carry_values_across_corpora(setup_corpus, tmp_path):
    src, dest = _source(tmp_path), tmp_path / "dest"
    _manifest(dest, 'name = "other-corpus"\nsource = "https://x.test/other"\nlicense = "MIT"\n')
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    cfg = load_config(dest)
    assert (cfg.name, cfg.source_url, cfg.license) == ("team-skills", "", "")


def test_setup_refuses_source_equal_to_dest(setup_corpus, tmp_path):
    src = _source(tmp_path)
    with pytest.raises(SystemExit, match="same directory"):
        setup_corpus.main(["--source", str(src), "--dest", str(src), "--no-build"])
    assert (src / "alpha" / "SKILL.md").exists()


def test_setup_reports_stale_folders_without_deleting(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    (dest / "gone").mkdir(parents=True)
    (dest / "gone" / "SKILL.md").write_text("---\nname: gone\n---\n", encoding="utf-8")
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    assert "not in the source (left in place): gone" in capsys.readouterr().out
    assert (dest / "gone" / "SKILL.md").exists()


def test_manifest_text_round_trips_through_toml(setup_corpus):
    tricky = 'we"ird \\ name\nwith\ttabs\x01'
    text = setup_corpus.manifest_text(tricky, "1", "https://x.test/a?b=1", "MIT",
                                      {"requires": "needs"}, version_from_git=True)
    data = tomllib.loads(text)
    assert data["name"] == tricky and data["version"] == "1"
    assert data["layout"] == {"pattern": "**/SKILL.md"} and data["fields"] == {"requires": "needs"}


def test_git_version_parses_or_degrades(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("setup_corpus_git",
                                                  ROOT / "scripts" / "setup_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # unpatched git_version
    calls = []

    def git(tracked="alpha/SKILL.md\n", status=""):
        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            out = {"rev-parse": "8328c60\n", "ls-files": tracked, "status": status}[cmd[3]]
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return fake_run

    monkeypatch.setattr(mod.subprocess, "run", git())
    assert mod.git_version(tmp_path) == "8328c60"
    assert calls[0][:2] == ["git", "-C"] and calls[0][-3:] == ["rev-parse", "--short", "HEAD"]
    assert [c[3:] for c in calls[1:]] == [["ls-files", "--", "."], ["status", "--porcelain", "--", "."]]
    # an untracked copy inside some other repo: that repo's HEAD is not this corpus's version
    monkeypatch.setattr(mod.subprocess, "run", git(tracked=""))
    assert mod.git_version(tmp_path) == ""
    # uncommitted edits: the vendored bytes are not HEAD's
    monkeypatch.setattr(mod.subprocess, "run", git(status=" M alpha/SKILL.md\n"))
    assert mod.git_version(tmp_path) == "8328c60-dirty"
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 128, "", "not a repo"))
    assert mod.git_version(tmp_path) == ""

    def no_git(cmd, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(mod.subprocess, "run", no_git)
    assert mod.git_version(tmp_path) == ""


def _git_cmd(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_version_on_real_repos(tmp_path):
    spec = importlib.util.spec_from_file_location("setup_corpus_realgit",
                                                  ROOT / "scripts" / "setup_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    outer = tmp_path / "outer"
    outer.mkdir()
    _git_cmd(outer, "init", "-q")
    (outer / "README").write_text("x\n", encoding="utf-8")
    _git_cmd(outer, "add", "README")
    _git_cmd(outer, "commit", "-q", "-m", "init")
    src = _source(outer)                                   # untracked, inside another repo
    assert mod.git_version(src) == ""
    _git_cmd(outer, "add", "-A")
    _git_cmd(outer, "commit", "-q", "-m", "skills")
    head = subprocess.run(["git", "-C", str(outer), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert mod.git_version(src) == head
    (src / "alpha" / "SKILL.md").write_text("---\nname: alpha\n---\nedited\n", encoding="utf-8")
    assert mod.git_version(src) == f"{head}-dirty"


def test_setup_uses_the_source_manifest(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    (src / "beta" / "SKILL.md").write_text(
        "---\nname: beta\ntype: workflow\ndomain: d\nlevel: beginner\ndepends: [alpha]\n---\nx\n",
        encoding="utf-8")
    _manifest(src, 'name = "acme"\nversion = "2.1"\nlicense = "Apache-2.0"\n'
                   '[fields]\nrequires = "depends"\n')
    direct = {s.slug: s for s in load_corpus_report(src).skills}
    assert direct["beta"].requires == ["alpha"]
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--expect", "2"])
    cfg = load_config(dest)
    assert (cfg.name, cfg.version, cfg.license, cfg.fields) == \
        ("acme", "2.1", "Apache-2.0", {"requires": "depends"})
    vendored = {s.slug: s for s in load_corpus_report(dest).skills}
    assert vendored["beta"].requires == ["alpha"]           # the edge survives vendoring
    assert "took name, version, license, [fields] from the source's corpus.toml" in \
        capsys.readouterr().out


def test_setup_flags_beat_the_source_manifest_which_beats_the_old_one(setup_corpus, tmp_path):
    src, dest = _source(tmp_path), tmp_path / "dest"
    _manifest(src, 'name = "acme"\nlicense = "Apache-2.0"\n')
    _manifest(dest, 'name = "acme"\nsource = "https://x.test/old"\nlicense = "MIT"\n')
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build",
                       "--version", "v9"])
    cfg = load_config(dest)
    assert (cfg.name, cfg.version, cfg.source_url, cfg.license) == \
        ("acme", "v9", "https://x.test/old", "Apache-2.0")


def test_setup_reads_skill_folders_from_the_source_manifest_root(setup_corpus, tmp_path):
    repo = tmp_path / "repo"
    _source(repo, "skills")
    _manifest(repo, 'name = "acme"\nroot = "skills"\n')
    dest = tmp_path / "dest"
    setup_corpus.main(["--source", str(repo), "--dest", str(dest), "--no-build", "--expect", "2"])
    assert [s.slug for s in load_corpus_report(dest).skills] == ["alpha", "beta"]
    assert load_config(dest).name == "acme"


def test_setup_warns_about_a_source_layout_it_cannot_reproduce(setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    _manifest(src, '[layout]\npattern = "*/SKILL.md"\n')
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    assert "sets [layout] pattern = '*/SKILL.md'" in capsys.readouterr().out


def _nested(src: Path, rel: str, name: str) -> None:
    (src / rel).mkdir(parents=True)
    (src / rel / "SKILL.md").write_text(
        f"---\nname: {name}\ntype: workflow\ndomain: d\nlevel: beginner\n---\nx\n", encoding="utf-8")


def test_fallback_discovery_is_relative_to_the_source(setup_corpus, tmp_path):
    src = tmp_path / "plugins" / "marketplace" / "team-skills"     # e.g. ~/.claude/plugins/...
    _nested(src, "skills/alpha", "alpha")
    _nested(src, "skills/beta", "beta")
    _nested(src, "plugins/llm/skills/alpha", "alpha")              # a re-bundle inside: skipped
    _nested(src, "skills/_drafts/wip", "wip")                      # a draft: skipped like ingest
    _nested(src, "skills/.hidden/secret", "secret")
    assert [d.name for d in setup_corpus.find_skill_dirs(src)] == ["alpha", "beta"]


def test_top_level_discovery_skips_hidden_folders_and_warns_about_nested_ones(
        setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    _nested(src, ".cache", "cache")
    _nested(src, "extra/gamma", "gamma")                           # not top-level: left out
    _nested(src, "alpha/examples", "alpha-example")                # inside alpha: copied along
    assert [d.name for d in setup_corpus.find_skill_dirs(src)] == ["alpha", "beta"]
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    assert "1 nested SKILL.md folder(s) are not vendored (only top-level <slug>/SKILL.md " \
           "folders are): extra/gamma" in capsys.readouterr().out
    assert not (dest / ".cache").exists() and (dest / "alpha" / "examples" / "SKILL.md").exists()


def test_setup_refuses_same_named_skill_folders(setup_corpus, tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _nested(src, "skills/nlp/evaluation", "nlp-evaluation")
    _nested(src, "skills/cv/evaluation", "cv-evaluation")
    with pytest.raises(SystemExit, match="skills/cv/evaluation vs skills/nlp/evaluation"):
        setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build"])
    assert not dest.exists()                                       # nothing was copied


def test_setup_reports_every_skip_even_when_stale_folders_fill_the_count(
        setup_corpus, tmp_path, capsys):
    src, dest = _source(tmp_path), tmp_path / "dest"
    (dest / "gone").mkdir(parents=True)
    (dest / "gone" / "SKILL.md").write_text("---\nname: gone\ntype: workflow\n---\n",
                                            encoding="utf-8")
    (src / "beta" / "SKILL.md").write_text("no frontmatter\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        setup_corpus.main(["--source", str(src), "--dest", str(dest), "--no-build", "--expect", "2"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "2 skills load from" in captured.out                    # alpha + the stale 'gone'
    assert "WARNING: skipped " in captured.out and "no YAML frontmatter" in captured.out
    assert "1 vendored folder(s) load no skill: beta" in captured.out
    assert "1 vendored folder(s) load no skill: beta" in captured.err


def test_setup_builds_the_index_under_the_repo_from_any_cwd(setup_corpus, tmp_path, monkeypatch,
                                                            capsys):
    built = []

    class FakeRouter:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def build(self):
            return 7

    monkeypatch.setitem(sys.modules, "sie.router", types.SimpleNamespace(HybridRouter=FakeRouter))
    monkeypatch.chdir(tmp_path)
    src, dest = _source(tmp_path), tmp_path / "dest"
    setup_corpus.main(["--source", str(src), "--dest", str(dest)])
    assert built[-1] == {"skills_dir": str(dest), "persist_dir": str(ROOT / "data" / "chroma")}
    assert f"indexed 7 chunks -> {ROOT / 'data' / 'chroma'}" in capsys.readouterr().out
    setup_corpus.main(["--source", str(src), "--dest", str(dest), "--persist-dir", "idx"])
    assert built[-1]["persist_dir"] == str(tmp_path / "idx")
