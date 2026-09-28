"""Optional LLM-as-judge: score whether a retrieved skill is faithful to the query."""
from __future__ import annotations
import json

from sie.llm import LLMClient

_SYSTEM = "You are a strict evaluator. Answer ONLY with JSON."
_TEMPLATE = 'Query: {query}\nRetrieved skill: {slug}\nSkill summary: {summary}\n\nIs this skill the correct, faithful match for the query?\nReturn JSON: {{"faithful": true|false, "reason": "..."}}'


def judge(query: str, slug: str, summary: str, client: LLMClient | None = None) -> dict:
    client = client or LLMClient()
    raw = client.complete(_TEMPLATE.format(query=query, slug=slug, summary=summary), system=_SYSTEM)
    try:
        return json.loads(raw)
    except Exception:
        return {"faithful": None, "reason": raw}
