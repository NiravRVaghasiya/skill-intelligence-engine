import sys
from pathlib import Path

import pytest

from sie.ingest import load_corpus, load_corpus_report, main, parse_skill_file

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
