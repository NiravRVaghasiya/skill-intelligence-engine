"""Skill Intelligence Engine: a corpus-agnostic intelligence layer over SKILL.md skill libraries.

SIE indexes, retrieves, ranks, explains, evaluates and composes skills from an *external*
skill corpus (any directory of SKILL.md files with YAML frontmatter). It is not a skill
library and ships no skill content.

    from sie import Engine                  # imported lazily: `import sie` stays cheap
    engine = Engine(skills_dir="path/to/skills", mode="sparse", use_reranker=False)
    engine.route("evaluate my RAG answers").top.slug
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"
__all__ = ["Engine", "__version__"]

if TYPE_CHECKING:
    from .engine import Engine


def __getattr__(name: str) -> Any:
    """Lazy `sie.Engine` (PEP 562), so importing a submodule never loads the whole engine."""
    if name == "Engine":
        from .engine import Engine
        return Engine
    raise AttributeError(f"module 'sie' has no attribute {name!r}")
