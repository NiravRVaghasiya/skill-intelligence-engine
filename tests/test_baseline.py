"""The vendored keyword baseline is byte-identical to the source and behaves like it."""
import hashlib
import sys
from pathlib import Path

from eval.baseline import VENDORED, KeywordBaseline, load_router
from sie.ingest import load_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"
PINNED = {  # must match eval/baseline/SOURCE.md
    "router.py": "15529d76a655a5f0644c7d66de6ec6708144fb340e8a32bbc5624e575c5e11c6",
    "skills_lib.py": "44b0a45a9e8a1a0c002d994d7de1fd513fcead2de25e3546e4d3d3ba5b098c5b",
}
COLLISION_QUERY = "Build a fraud detection model and evaluate it under severe class imbalance"


def test_vendored_files_are_unmodified():
    for name, sha in PINNED.items():
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == sha, name
        assert sha in (VENDORED / "SOURCE.md").read_text(encoding="utf-8")


def test_documented_query_matches_source_output_at_head():
    # docs/ROUTING.md (post Phase-15 audit): supervised-learning #1 with score 18
    hits = KeywordBaseline(CORPUS).retrieve(COLLISION_QUERY, k=4)
    assert [(h.skill_slug, h.score) for h in hits] == [
        ("supervised-learning", 18.0), ("model-deployment", 14.0),
        ("ml-monitoring", 12.0), ("computer-vision", 10.0)]


def test_shipped_route_prepends_prerequisites():
    hits = KeywordBaseline(CORPUS, shipped=True).retrieve(COLLISION_QUERY)
    slugs = [h.skill_slug for h in hits]
    assert slugs.index("agents-and-tools") == slugs.index("agent-evaluation") - 1


def test_full_ranking_covers_every_skill_deterministically():
    b = KeywordBaseline(CORPUS)
    a1 = [h.skill_slug for h in b.retrieve("explain attention", k=100)]
    assert len(a1) == 38 and a1 == [h.skill_slug for h in b.retrieve("explain attention", k=100)]
    assert b.intent("explain attention") == "reference"


def test_loader_isolates_skills_lib():
    r1, r2 = load_router(), load_router(VENDORED)
    assert r1.sl is not r2.sl and "skills_lib" not in sys.modules


def test_ingest_agrees_with_source_parser_on_all_skills():
    ours = {s.slug: s for s in load_corpus(CORPUS)}
    theirs = {s.slug: s.frontmatter for s in load_router().sl.load_all_skills(CORPUS)}
    assert ours.keys() == theirs.keys() and len(ours) == 38
    for slug, fm in theirs.items():
        s = ours[slug]
        assert (s.skill_type, s.domain, s.level) == (fm["type"], fm["domain"], fm["level"]), slug
        for field in ("requires", "related", "conflicts", "capabilities"):
            assert getattr(s, field) == (fm.get(field) or []), (slug, field)
        assert s.description == fm["description"] and s.display_name == fm["display_name"]
