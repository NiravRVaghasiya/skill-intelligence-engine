"""Multi-intent composition: query -> candidate intents -> skills -> dependencies -> ordered plan.

An opt-in layer over single-intent routing, deliberately conservative (NOT an autonomous
planner):
- `split_intents` splits only at explicit boundaries (sentence ends, newlines, list bullets)
  and at a comma / connector ("and", "then", ...) that is followed by a task verb from
  `ACTION_VERBS`, so noun conjunctions ("precision and recall", "train and test sets") stay
  together. A pronoun in a later part ("evaluate it") is resolved to the first part's (short)
  object. A query with one part is routed verbatim. Stdlib `re`, deterministic.
- `compose` routes each intent on its own and keeps only its top skill. A resolved part whose
  skill an earlier intent already took is also routed as written, and that result is used
  when it selects a new skill (the copied object must not pull every part back to intent 1's
  skill). An intent whose routing confidence says "abstain" selects nothing; ambiguous or weak
  selections are kept but flagged in `notes`. Prerequisites come only from the graph's
  `requires` edges (the corpus's declarations, plus overlay edges only when the caller built
  the graph with them); nothing is inferred from retrieval. Order: hard `requires` first, then
  soft `recommended_before`, then the order the intents were stated.

Usage:
    python -m sie.compose "Build a RAG pipeline, evaluate it, deploy it, and monitor it"
    python -m sie.compose "..." --mode sparse --no-rerank      # BM25 only: no models needed
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import islice
from typing import TYPE_CHECKING, Protocol

from .graph.build import ordering_graph
from .graph.paths import conflicts_of, order_skills, prerequisite_closure, superseded_by
from .models import Composition, Edge, Intent, PlanStep, RouteResult

if TYPE_CHECKING:                                       # annotations only; imported lazily
    import networkx as nx

# Generic English task verbs (not domain vocabulary). A soft boundary splits a sentence only
# when the text after it starts with one of these.
ACTION_VERBS = frozenset({
    "add", "analyze", "annotate", "audit", "benchmark", "build", "calibrate", "choose", "clean",
    "compare", "compress", "configure", "containerize", "create", "debug", "deploy", "design",
    "detect", "diagnose", "distill", "document", "evaluate", "explain", "export", "fine-tune",
    "finetune", "fix", "forecast", "generate", "harden", "implement", "improve", "index",
    "integrate", "investigate", "label", "launch", "log", "measure", "migrate", "monitor",
    "optimize", "pick", "plan", "prepare", "preprocess", "profile", "prototype", "prune",
    "quantize", "rank", "reduce", "refactor", "release", "retrain", "review", "run", "scale",
    "secure", "select", "serve", "set", "ship", "summarize", "test", "track", "train", "tune",
    "validate", "verify", "version", "visualize", "write",
})
# ACTION_VERBS words that in ML text usually head a noun phrase ("test set", "log loss", "train
# split", "index size"). After a soft boundary they start a new task only when the next word is
# one that follows a verb, not a noun ("test whether ...", "log each run", "set up ...").
_NOUN_LIKE_VERBS = frozenset({
    "test", "log", "train", "index", "label", "version", "run", "set", "plan", "track", "profile",
    "scale", "release", "review", "design", "document", "rank", "export",
})
_VERB_FOLLOWERS = frozenset({
    "the", "a", "an", "each", "every", "all", "our", "my", "your", "its", "their", "this", "that",
    "these", "those", "it", "them", "whether", "if", "how", "up", "to", "on", "with",
    "it's", "it’s",                                  # the common misspelling of "its"
})
# Words stripped from the start of a split-off part ("..., and then deploy it").
CONNECTORS = ("and", "then", "also", "finally", "next", "afterwards", "after that", "lastly")

# Every pattern below runs in linear time on arbitrary input (no backtracking over long runs).
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")                 # "fine-tune", "model's"
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")                # "- ", "* ", "1. ", "2) "
_SENTENCE_END = re.compile(r"(?<![.!?;])[.!?;]+\s+")             # starts only at a run's start
_SOFT = re.compile(r",|\s+(?:and\s+then|after\s+that|as\s+well\s+as|and|then|also)(?=\s)", re.I)
_LEADING = re.compile(r"(?:[\s,;:]+|(?:%s)(?=[\s,;:]|$))+"
                      % "|".join(c.replace(" ", r"\s+") for c in CONNECTORS), re.I)
_TRAILING = " .,;:!?…"                                           # str.rstrip set
_COMMA_RUN = re.compile(r",(?:\s*,)+")
_PRONOUN = re.compile(r"(?<![\w'’-])(it|them|this|that)(?![\w'’-])", re.I)
_NEXT_WORD = re.compile(r"\s*(" + _WORD.pattern + ")")
# Abbreviations whose "." never ends a sentence ("e.g. BM25", "et al. on" are not two intents).
# "etc." may end one; see `_continues`.
_ABBREVIATIONS = frozenset({"e.g", "i.e", "vs", "cf", "viz", "approx", "incl", "esp", "al"})
# A resolved pronoun copies the first part's object into later parts; a longer object would
# swamp the later part's own words, so it is cut to its head phrase (see `_cap_object`).
_MAX_OBJECT_WORDS = 6
_DETERMINERS = frozenset({
    "a", "an", "the", "our", "my", "your", "its", "their", "his", "her", "this", "that", "these",
    "those", "some", "any", "each", "every", "all",
})
# Prepositions, conjunctions and relative words: where a noun phrase's head part ends.
_PHRASE_BREAKS = frozenset({
    "about", "above", "across", "after", "against", "along", "among", "around", "as", "at",
    "before", "behind", "below", "beneath", "beside", "between", "beyond", "by", "despite",
    "during", "except", "for", "from", "in", "inside", "into", "like", "near", "of", "off", "on",
    "onto", "out", "over", "per", "since", "through", "throughout", "to", "toward", "towards",
    "under", "until", "upon", "using", "via", "with", "within", "without",
    "and", "or", "but", "nor", "so", "yet", "because", "although", "though", "while", "whereas",
    "if", "unless", "once", "when", "whenever", "where", "wherever",
    "that", "which", "who", "whom", "whose", "what",
})
# "this"/"that" followed by one of these (or by nothing / punctuation) is a pronoun; followed
# by any other word it is a determiner ("deploy this model") and is left alone.
_AFTER_PRONOUN = frozenset({
    "about", "across", "after", "again", "against", "along", "and", "as", "at", "before", "by",
    "during", "for", "from", "if", "in", "into", "later", "now", "on", "once", "onto", "or",
    "over", "so", "then", "through", "to", "too", "under", "until", "using", "via", "when",
    "where", "while", "with", "within", "without",
})


class Router(Protocol):
    """What `compose` needs from a router: `HybridRouter.route`, or any stand-in."""

    def route(self, query: str, k: int | None = None) -> RouteResult: ...


# -- intent splitting ---------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Collapse whitespace and strip trailing punctuation."""
    return " ".join(text.split()).rstrip(_TRAILING)


def _skip_connectors(text: str, pos: int = 0, end: int | None = None) -> int:
    """Index just past any separators / connector words at the start of text[pos:end]."""
    m = _LEADING.match(text, pos, len(text) if end is None else end)
    return m.end() if m else pos


def _clean(piece: str) -> str:
    """A split-off part as written: whitespace collapsed, leading connectors and trailing
    punctuation stripped."""
    text = _normalize(piece)                        # first, so "Then." is all connector
    return text[_skip_connectors(text):]


def _lines(query: str) -> list[str]:
    """Non-blank lines with a leading list bullet removed (newlines are hard boundaries)."""
    lines = (_BULLET.sub("", line, count=1) for line in query.splitlines())
    return [line for line in lines if line.strip()]


def _token_before(line: str, end: int) -> str:
    """The short token (letters, digits, inner dots) that ends at line[end], lowercased
    ("e.g", "vs", "etc"); "" when it is longer than 9 characters."""
    i = end
    while i > 0 and end - i <= 8 and (line[i - 1].isalnum() or line[i - 1] == "."):
        i -= 1
    return "" if i > 0 and line[i - 1].isalnum() else line[i:end].lower()


def _continues(line: str, m: re.Match) -> bool:
    """Does the sentence go on after the "." that `m` (a `_SENTENCE_END` match) starts at?

    Always after an abbreviation ("e.g.", "vs.", "et al."); after "etc." only when a lowercase
    word follows ("outliers, etc. before training"), since "etc. Train a model" ends a sentence.
    """
    if m.group(0).rstrip() != ".":
        return False
    token = _token_before(line, m.start())
    return token in _ABBREVIATIONS or (token == "etc" and m.end() < len(line)
                                       and line[m.end()].islower())


def _sentences(line: str) -> list[str]:
    """Split at `[.!?;]` + whitespace, except inside a sentence ("e.g.", "et al.", "etc. and")."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(line):
        if _continues(line, m):
            continue
        out.append(line[start:m.start()])
        start = m.end()
    out.append(line[start:])
    return out


def _has_words(text: str, pos: int, end: int, n: int) -> bool:
    return sum(1 for _ in islice(_WORD.finditer(text, pos, end), n)) >= n


def _starts_task(sentence: str, verb: re.Match | None) -> bool:
    """Does the word `verb` (after a soft boundary) start a new task?

    It must be an `ACTION_VERBS` word with another word after it somewhere (the right side
    keeps >= 2 words); a noun homograph (`_NOUN_LIKE_VERBS`: "test", "log", ...) only when the
    very next word is one that follows verbs ("test the ...", "log each ...", "set up").
    """
    if verb is None or (word := verb.group(0).lower()) not in ACTION_VERBS:
        return False
    if _WORD.search(sentence, verb.end()) is None:
        return False
    if word not in _NOUN_LIKE_VERBS:
        return True
    nxt = _NEXT_WORD.match(sentence, verb.end())
    return nxt is not None and nxt.group(1).lower() in _VERB_FOLLOWERS


def _soft_split(sentence: str) -> list[str]:
    """Split one sentence at commas / connectors that introduce a new task.

    A boundary is accepted only if the text after it (connector words skipped) starts a task
    (`_starts_task`: an `ACTION_VERBS` word, noun homographs only before a verb-follower word),
    and both the part before it (since the last accepted boundary) and the rest of the sentence
    keep >= 2 word tokens. Greedy, left to right.
    """
    sentence = _COMMA_RUN.sub(",", " ".join(sentence.split()))
    pieces, start = [], 0
    left_from = _skip_connectors(sentence, start)    # the current part's first non-connector
    head, head_ok = -1, False
    for m in _SOFT.finditer(sentence):
        if m.end() > head:           # candidates inside one run of connectors share a head
            head = _skip_connectors(sentence, m.end())
            head_ok = _starts_task(sentence, _WORD.match(sentence, head))
        if not head_ok or not _has_words(sentence, min(left_from, m.start()), m.start(), 2):
            continue                                                           # left: >= 2 words
        pieces.append(sentence[start:m.start()])
        start = m.end()
        left_from = _skip_connectors(sentence, start)
    pieces.append(sentence[start:])
    return pieces


def _cap_object(obj: str) -> str:
    """`obj` when it has at most `_MAX_OBJECT_WORDS` words; else its head phrase.

    The head phrase is the text before the first preposition / conjunction / relative word
    that follows the head (the first non-determiner word): "a LoRA on Llama 3 8B using our
    past support replies so it picks up our tone" -> "a LoRA". "" (leave the pronoun
    unresolved) when there is no such word within the first `_MAX_OBJECT_WORDS` + 1 words.
    """
    head_seen, cut, prev_end = False, None, 0
    for n, w in enumerate(_WORD.finditer(obj)):
        word = w.group(0).lower()
        if cut is None and head_seen and word in _PHRASE_BREAKS:
            cut = prev_end
        if n == _MAX_OBJECT_WORDS:                      # a word too many to copy the object whole
            return obj[:cut] if cut is not None else ""
        head_seen = head_seen or word not in _DETERMINERS
        prev_end = w.end()
    return obj


def _object(first: str) -> str:
    """The first part minus its leading action verb ("set up" drops "up"), capped to a short
    head phrase (`_cap_object`); "" if no verb."""
    verb = _WORD.match(first)
    if verb is None or verb.group(0).lower() not in ACTION_VERBS:
        return ""
    rest = first[verb.end():]
    if verb.group(0).lower() == "set":
        rest = re.sub(r"^\s+up\b", "", rest, flags=re.I)
    return _cap_object(rest.strip())


def _is_pronoun(segment: str, m: re.Match) -> bool:
    """it/them always; this/that only when not used as a determiner ("this model")."""
    if m.group(1).lower() in ("it", "them"):
        return True
    nxt = _NEXT_WORD.match(segment, m.end())
    return nxt is None or nxt.group(1).lower() in _AFTER_PRONOUN


def _resolve(segment: str, obj: str) -> str:
    """Replace the first standalone pronoun in `segment` with `obj`."""
    for m in _PRONOUN.finditer(segment):
        if _is_pronoun(segment, m):
            return segment[:m.start()] + obj + segment[m.end():]
    return segment


def _split(query: str, max_intents: int | None) -> tuple[list[tuple[str, str]], int]:
    """`split_intents`, plus the number of parts before truncation (0 for no words, 1 for a
    single intent). Truncates before resolving pronouns, so the work stays linear."""
    if not _WORD.search(query or ""):
        return [], 0
    pieces = [p for line in _lines(query) for s in _sentences(line) for p in _soft_split(s)]
    parts = [c for c in map(_clean, pieces) if _WORD.search(c)]
    if len(parts) < 2:
        text = query.strip()                    # verbatim: exactly what plain routing gets
        return [(text, text)], 1
    total, obj = len(parts), _object(parts[0])
    if max_intents is not None:
        parts = parts[:max_intents]
    return [(p, _resolve(p, obj) if i and obj else p) for i, p in enumerate(parts)], total


def split_intents(query: str, max_intents: int | None = None) -> list[tuple[str, str]]:
    """Split a query into candidate intents, conservatively.

    1. Hard boundaries: newlines, list bullets (`- `, `* `, `1.`, `1)` at line start), and
       `.`/`!`/`?`/`;` followed by whitespace (not after "e.g.", "i.e.", "vs.", "et al.", ...,
       nor after "etc." when a lowercase word follows).
    2. Soft boundaries inside a sentence: `,` and " and ", " then ", " and then ",
       " after that ", " also ", " as well as ", accepted only when the text after them
       (leading `CONNECTORS` skipped) starts with an `ACTION_VERBS` word and both sides keep
       >= 2 word tokens; otherwise the conjunction joins nouns ("precision and recall"). A
       verb that is usually a noun in ML text ("test", "log", "train", "index", ...) counts
       only before a word that follows verbs ("test the ...", "log each ...", "set up").
    3. Each part: whitespace collapsed, leading connectors and trailing punctuation stripped.
    4. When the first part starts with an action verb, the first standalone it/them/this/that
       of every later part is replaced by the first part's object (the part minus its verb,
       and "up" after "set"); "this"/"that" used as a determiner ("this model") is kept. An
       object longer than 6 words is cut to its head phrase ("a LoRA on Llama 3 8B using ..."
       -> "a LoRA"), or the pronoun is left unresolved when there is no clear cut.

    Args:
        query: free text.
        max_intents: keep only the first N parts; None (default) returns every part.

    Returns:
        [(part as written, query to route)], in query order. With fewer than two parts:
        [(query stripped, query stripped)], so a single intent is routed exactly as plain
        routing would route it. [] when the query has no words.

    Raises:
        ValueError: max_intents < 1.
    """
    if max_intents is not None:
        _check_count("max_intents", max_intents)
    return _split(query, max_intents)[0]


def routed_texts(query: str, max_intents: int = 5) -> list[str]:
    """Every text `compose(..., query, max_intents=max_intents)` may pass to `router.route`.

    That is the verbatim (stripped) query when it has a single intent; otherwise, for each
    routed part, its query to route and, when a pronoun was resolved, the part as written
    (compose's fallback when the resolved text selects an already-selected skill).

    Args:
        query: free text, as given to `compose`.
        max_intents: as given to `compose`.

    Returns:
        Deduplicated texts in first-use order ([] when the query has no words).

    Raises:
        ValueError: max_intents < 1.
    """
    _check_count("max_intents", max_intents)
    parts = _split(query, max_intents)[0]
    return list(dict.fromkeys(t for text, routed in parts for t in (routed, text)))


# -- composition --------------------------------------------------------------------------

def _check_count(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _intent_list(indexes: list[int]) -> str:
    """Label for intent indexes: `intent 2`, `intents 2, 3`."""
    return f"intent{'s' if len(indexes) > 1 else ''} {', '.join(map(str, indexes))}"


def _selection(result: RouteResult) -> str | None:
    """The top skill, unless the routing confidence says abstain."""
    top = result.top
    return top.slug if top is not None and result.confidence.action != "abstain" else None


def _route_intents(router: Router, parts: list[tuple[str, str]], k: int) -> list[Intent]:
    """Route each part (identical text is routed once); select its top skill.

    A part whose pronoun was resolved is routed resolved first. When that selects a skill an
    earlier intent already selected (the copied object pulled it back, e.g. "deploy a RAG
    pipeline" -> the RAG skill), the part is also routed as written; that result (and the
    as-written text as `Intent.query`) is used when it selects a skill no earlier intent did.
    """
    cache: dict[str, RouteResult] = {}

    def route(text: str) -> RouteResult:
        if text not in cache:
            cache[text] = router.route(text, k=k)
        return cache[text]

    chosen: set[str] = set()                              # skills earlier intents selected
    intents = []
    for i, (text, routed) in enumerate(parts, 1):
        query, result = routed, route(routed)
        selected = _selection(result)
        if routed != text and selected in chosen:
            as_written = route(text)
            alternative = _selection(as_written)
            if alternative is not None and alternative not in chosen:
                query, result, selected = text, as_written, alternative
        if selected is not None:
            chosen.add(selected)
        intents.append(Intent(index=i, text=text, query=query, result=result, selected=selected))
    return intents


def _shown(text: str) -> str:
    """Text for one-line notes and summaries: whitespace (incl. newlines) collapsed."""
    return " ".join(text.split())


def _intent_notes(it: Intent) -> list[str]:
    """Caveats about one intent's selection."""
    label = f"intent {it.index} ('{_shown(it.text)}')"
    conf = it.result.confidence
    if it.selected is None:
        return [f"{label} has no sufficiently good match"]
    if conf.ambiguous:
        rivals = f" vs {', '.join(conf.competitors)}" if conf.competitors else ""
        return [f"{label} is ambiguous: {it.selected}{rivals}; confirm before acting"]
    if conf.level == "low":
        return [f"{label} matched {it.selected} only weakly (low confidence); confirm before acting"]
    return []


def _required_by(graph: nx.DiGraph, slug: str, order: list[str]) -> list[str]:
    """Plan skills (plan order) with a `requires` edge from `slug`."""
    return [m for m in order if m != slug and graph.has_edge(slug, m)
            and graph.edges[slug, m].get("kind") == "requires"]


def _unconfirmed(edges: dict[tuple[str, str, str], Edge], slug: str, required_by: list[str]) -> str:
    """` (proposed)` when a requires edge behind `required_by` is not a corpus declaration."""
    labels = sorted({e.confidence for m in required_by
                     if (e := edges.get((slug, m, "requires"))) and e.confidence != "declared"})
    return f" ({', '.join(labels)})" if labels else ""


def _requested_reason(indexes: list[int], by_index: dict[int, Intent], in_graph: bool,
                      required_by: list[str], marker: str) -> str:
    parts = [f"requested by {_intent_list(indexes)}"]
    ambiguous = [i for i in indexes if by_index[i].result.confidence.ambiguous]
    if ambiguous:
        rivals = list(dict.fromkeys(c for i in ambiguous
                                    for c in by_index[i].result.confidence.competitors))
        parts.append(f"ambiguous for {_intent_list(ambiguous)}"
                     + (f" (vs {', '.join(rivals)})" if rivals else ""))
    weak = [i for i in indexes if i not in ambiguous and by_index[i].result.confidence.level == "low"]
    if weak:
        parts.append(f"low confidence for {_intent_list(weak)}")
    if not in_graph:
        parts.append("not in the skill graph, so its prerequisites are unknown")
    if required_by:
        parts.append(f"also required by {', '.join(required_by)}{marker}")
    return "; ".join(parts)


def _plan_notes(graph: nx.DiGraph, requested: dict[str, list[int]], order: list[str],
                conflicts: list[list[str]], dropped: list[Edge],
                include_prerequisites: bool) -> list[str]:
    """Plan-level caveats: merges, unknown skills, declaration gaps, conflicts, supersession."""
    notes = [f"{_intent_list(idx)} all map to {s}; merged into one step"
             for s, idx in requested.items() if len(idx) > 1]
    notes += [f"{s} is not in the skill graph (the router's corpus differs from the graph's); "
              f"its prerequisites and relationships are unknown" for s in requested if s not in graph]
    known = [s for s in order if s in graph]
    if include_prerequisites:
        notes += [f"{s} does not declare prerequisites; the plan may be incomplete"
                  for s in known if "requires" not in (graph.nodes[s].get("declared") or [])]
        missing: dict[str, list[str]] = {}
        for s, rel, ref in graph.graph.get("dangling", []):
            if rel == "requires" and s in graph and ref not in missing.get(s, []):
                missing.setdefault(s, []).append(ref)
        notes += [f"{s} requires '{ref}', which is not in the corpus; it is omitted"
                  for s in known for ref in missing.get(s, [])]
    notes += [f"{a} conflicts with {b}; review before combining them" for a, b in conflicts]
    notes += [f"{s} is superseded by {', '.join(sup)}" for s in known if (sup := superseded_by(graph, s))]
    notes += [f"ignored recommended_before {e.source} -> {e.target}: it would create an "
              f"ordering cycle" for e in dropped]
    return notes


def compose(router: Router, graph: nx.DiGraph, query: str, k: int = 3, max_intents: int = 5,
            include_prerequisites: bool = True) -> Composition:
    """Turn a (possibly multi-part) query into a dependency-ordered plan of skills.

    Args:
        router: anything with `route(query, k) -> RouteResult` (e.g. `HybridRouter`).
        graph: skill graph from `sie.graph.build_graph`, ideally over the router's corpus.
        query: free text; split with `split_intents`.
        k: skills each intent's routing returns (only the top one is selected).
        max_intents: route at most this many intents; the rest are counted in a note.
        include_prerequisites: add the `requires` closure of every selected skill.

    Returns:
        Composition: every routed intent with its RouteResult and selection; one PlanStep per
        distinct skill (role "requested" or "prerequisite", 1-based positions) ordered by hard
        `requires`, then acyclic `recommended_before`, then intent order; the graph's edges
        among plan skills (stored order); conflicting pairs; unmatched intent indexes; notes
        (per intent in intent order, then plan-level).

    Raises:
        ValueError: k or max_intents < 1, or a `requires` cycle among the plan's skills.
    """
    _check_count("k", k)
    _check_count("max_intents", max_intents)
    parts, total = _split(query, max_intents)
    if not parts:
        return Composition(query=query, intents=[], steps=[],
                           notes=["empty query: nothing to compose"])
    intents = _route_intents(router, parts, k)
    notes = [n for it in intents for n in _intent_notes(it)]
    if (extra := total - max_intents) > 0:
        notes.append(f"{extra} more intent{'s were' if extra > 1 else ' was'} not routed "
                     f"(max_intents={max_intents})")

    requested: dict[str, list[int]] = {}              # first-mention order
    for it in intents:
        if it.selected is not None:
            requested.setdefault(it.selected, []).append(it.index)
    needed_by: dict[str, list[str]] = {}              # prerequisite -> requested skills needing it
    if include_prerequisites:
        for r in requested:
            if r in graph:
                for p in sorted(prerequisite_closure(graph, r)):
                    needed_by.setdefault(p, []).append(r)
    nodes = list(requested) + sorted(p for p in needed_by if p not in requested)
    # hard requires wins, soft edges next, then the earliest intent that asks for / needs it
    priority = {s: min(requested.get(s, []) + [i for r in needed_by.get(s, []) for i in requested[r]])
                for s in nodes}
    order = order_skills(graph, nodes, priority=priority, soft=True)
    dropped = ordering_graph(graph, nodes, soft=True)[1]

    plan = set(order)
    edges = [e for e in graph.graph.get("edges", []) if e.source in plan and e.target in plan]
    by_key = {(e.source, e.target, e.relationship): e for e in edges}
    by_index = {it.index: it for it in intents}
    steps = []
    for position, slug in enumerate(order, 1):
        required_by = _required_by(graph, slug, order) if slug in graph else []
        marker = _unconfirmed(by_key, slug, required_by)
        if slug in requested:
            steps.append(PlanStep(slug=slug, position=position, role="requested",
                                  intents=list(requested[slug]), required_by=required_by,
                                  reason=_requested_reason(requested[slug], by_index, slug in graph,
                                                           required_by, marker)))
        else:
            steps.append(PlanStep(slug=slug, position=position, role="prerequisite",
                                  intents=sorted({i for r in needed_by[slug] for i in requested[r]}),
                                  required_by=required_by,
                                  reason=f"required by {', '.join(required_by or needed_by[slug])}{marker}"))
    conflicts = [list(p) for p in sorted({tuple(sorted((s, c))) for s in order if s in graph
                                          for c in conflicts_of(graph, s) if c in plan})]
    if not order:
        notes.append("no intent matched a skill; the plan is empty")
    notes += _plan_notes(graph, requested, order, conflicts, dropped, include_prerequisites)
    return Composition(query=query, intents=intents, steps=steps, edges=edges, conflicts=conflicts,
                       unmatched=[it.index for it in intents if it.selected is None], notes=notes)


def composition_summary(c: Composition) -> list[str]:
    """Short printable lines: a header, one line per intent, one per plan step, the notes."""
    n_intents, n_steps = len(c.intents), len(c.steps)
    lines = [f"[compose] {n_intents} intent{'s' if n_intents != 1 else ''} -> "
             f"{n_steps} skill{'s' if n_steps != 1 else ''} in the plan"]
    for it in c.intents:
        text, query = _shown(it.text), _shown(it.query)       # one line even for verbatim input
        routed = f" (routed as '{query}')" if query != text else ""
        conf = it.result.confidence
        lines.append(f"  intent {it.index}: '{text}'{routed} -> {it.selected or 'no match'} "
                     f"[{conf.level}/{conf.action}]")
    lines += [f"  {s.position}. {s.slug:26s} {s.role:12s} {s.reason}" for s in c.steps]
    lines += [f"  note: {note}" for note in c.notes]
    return lines


def main(argv: list[str] | None = None) -> None:
    from .graph.build import build_graph
    from .ingest import load_corpus_report
    from .router import MODES, HybridRouter
    ap = argparse.ArgumentParser(description="Compose a multi-part request into an ordered skill plan.")
    ap.add_argument("query", help="task description, possibly with several parts")
    ap.add_argument("--skills", default="data/skills")
    ap.add_argument("--manifest", metavar="PATH", help="corpus.toml (default <skills>/corpus.toml)")
    ap.add_argument("-k", type=int, default=3, help="skills each intent's routing returns")
    ap.add_argument("--max-intents", type=int, default=5, help="route at most N intents")
    ap.add_argument("--no-prereqs", action="store_true", help="do not add `requires` prerequisites")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    ap.add_argument("--mode", choices=MODES, default="hybrid", help="retrievers to use")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")       # never crash on a cp1252 console
    router = HybridRouter(skills_dir=args.skills, use_reranker=not args.no_rerank, mode=args.mode,
                          manifest=args.manifest)
    try:
        graph = build_graph(load_corpus_report(args.skills, args.manifest).skills)
        c = compose(router, graph, args.query, k=args.k, max_intents=args.max_intents,
                    include_prerequisites=not args.no_prereqs)
    except ImportError as e:                          # e.g. hybrid mode on a core-only install
        from .engine import _missing_dependency
        sys.exit(f"[compose] {_missing_dependency(e)}")
    except (OSError, RuntimeError, ValueError) as e:
        sys.exit(f"[compose] {e}")
    print("\n".join(composition_summary(c)))


if __name__ == "__main__":
    main()
