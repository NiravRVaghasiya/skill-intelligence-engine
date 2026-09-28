from sie.models import Skill
from sie.chunking import split_sections, chunk_skill


def test_split_sections_basic():
    body = "## Overview\nhello\n## Gotchas\nbe careful"
    secs = dict(split_sections(body))
    assert "Overview" in secs and "Gotchas" in secs
    assert secs["Overview"] == "hello"


def test_code_fence_not_split():
    body = "## Workflow\n```python\nx = 1\n\ny = 2\n```\n"
    chunks = chunk_skill(Skill(slug="s", skill_type="workflow", domain="d", level="l", body=body))
    joined = "\n".join(c.text for c in chunks)
    assert "x = 1" in joined and "y = 2" in joined
