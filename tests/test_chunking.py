from sie.models import Skill
from sie.chunking import split_sections, chunk_skill


def test_split_sections_basic():
    body = "## Overview\nhello\n## Gotchas\nbe careful"
    secs = dict(split_sections(body))
    assert "Overview" in secs and "Gotchas" in secs
    assert secs["Overview"] == "hello"


def test_card_chunk_drops_negative_scope_clause():
    s = Skill(slug="rag", skill_type="workflow", domain="llm", level="i", display_name="RAG Pipeline",
              description="Use when building retrieval. NOT for fine-tuning (see fine-tuning-llms).",
              capabilities=["vector-indexing"], body="## Overview\nhi")
    card = chunk_skill(s)[0]
    assert card.section == "Card" and card.chunk_id == "rag::Card::0"
    assert "Use when building retrieval." in card.text and "vector indexing" in card.text
    assert "fine-tuning" not in card.text


def test_body_chunks_carry_skill_header():
    s = Skill(slug="rag", skill_type="workflow", domain="llm", level="i", display_name="RAG Pipeline",
              body="## Gotchas\nwatch out")
    body = chunk_skill(s, include_card=False)
    assert [c.text for c in body] == ["RAG Pipeline - Gotchas\nwatch out"]


def test_real_corpus_chunk_ids_unique():
    from pathlib import Path
    from sie.ingest import load_corpus
    chunks = [c for s in load_corpus(Path(__file__).resolve().parents[1] / "data" / "skills")
              for c in chunk_skill(s)]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert sum(c.section == "Card" for c in chunks) == 38


def test_code_fence_not_split():
    body = "## Workflow\n```python\nx = 1\n\ny = 2\n```\n"
    chunks = chunk_skill(Skill(slug="s", skill_type="workflow", domain="d", level="l", body=body))
    joined = "\n".join(c.text for c in chunks)
    assert "x = 1" in joined and "y = 2" in joined


def test_repeated_and_reserved_section_titles_get_unique_ids():
    s = Skill(slug="d", skill_type="workflow", domain="x", level="i", description="Use when x.",
              body="## Example\none\n## Example\ntwo\n## Card\nthree")
    ids = [c.chunk_id for c in chunk_skill(s)]
    assert ids == ["d::Card::0", "d::Example::0", "d::Example#2::0", "d::Card#2::0"]
    assert [c.section for c in chunk_skill(s)][1:] == ["Example", "Example", "Card"]
