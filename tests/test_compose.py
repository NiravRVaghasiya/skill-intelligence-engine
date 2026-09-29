"""Multi-intent composition: intent splitting, per-intent routing, prerequisites, ordering, notes.

Routing is scripted with a FakeRouter (routed text -> ranking + confidence) over synthetic
graphs; one integration test uses the real corpus with the BM25-only router (no models).
"""
from pathlib import Path

import pytest

from sie.compose import ACTION_VERBS, compose, composition_summary, main, routed_texts, split_intents
from sie.confidence import ACTIONS
from sie.graph.build import build_graph
from sie.ingest import load_corpus
from sie.models import Confidence, Edge, RankedSkill, RouteResult, Skill

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "skills"
ALL_DECLARED = ["requires", "related", "conflicts"]
RAG = "Build a RAG system, evaluate it, deploy it, and monitor it."


def _s(slug, level="intermediate", declared=ALL_DECLARED, **kw):
    """A skill that declares its requires/related/conflicts keys, like every real corpus skill."""
    return Skill(slug=slug, skill_type="workflow", domain="d", level=level, declared=list(declared),
                 source="corp", source_version="v1", path=f"{slug}/SKILL.md", **kw)


class FakeRouter:
    """Routes from a script: routed text -> (ranked slugs, confidence level, competitors)."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def route(self, query, k=None):
        self.calls.append((query, k))
        if query not in self.script:
            raise AssertionError(f"unexpected routed query {query!r}")
        entry = self.script[query]                   # [slugs] or ([slugs], level, competitors)
        slugs, level, competitors = entry if isinstance(entry, tuple) else (entry, "high", [])
        results = [RankedSkill(slug=s, rank=i, score=1.0 / i, score_type="bm25", methods=["bm25"])
                   for i, s in enumerate(slugs, 1)]
        conf = Confidence(level=level if results else "none",
                          action=ACTIONS[level if results else "none"],
                          ambiguous=bool(competitors), competitors=list(competitors),
                          reasons=[f"scripted {level}"])
        return RouteResult(query=query, results=results[:k], confidence=conf, mode="sparse")


def _rag_graph(**extra):
    """rag-evaluation requires rag-pipeline; everything else independent unless `extra` says so."""
    kw = {"rag-evaluation": {"requires": ["rag-pipeline"]}}
    for slug, fields in extra.items():
        kw.setdefault(slug, {}).update(fields)
    slugs = ["rag-pipeline", "rag-evaluation", "model-deployment", "ml-monitoring", "python-for-ml",
             "other"]
    return build_graph([_s(s, **kw.get(s, {})) for s in slugs])


RAG_SCRIPT = {
    "Build a RAG system": ["rag-pipeline", "rag-evaluation"],
    "evaluate a RAG system": ["rag-evaluation", "rag-pipeline"],
    "deploy a RAG system": ["model-deployment", "rag-pipeline"],
    "monitor a RAG system": ["ml-monitoring", "model-deployment"],
}


def _slugs(c):
    return [s.slug for s in c.steps]


def _step(c, slug):
    return next(s for s in c.steps if s.slug == slug)


# --- split_intents ---------------------------------------------------------------------------

def test_split_rag_example_resolves_pronouns_to_the_first_object():
    assert split_intents(RAG) == [
        ("Build a RAG system", "Build a RAG system"), ("evaluate it", "evaluate a RAG system"),
        ("deploy it", "deploy a RAG system"), ("monitor it", "monitor a RAG system")]


@pytest.mark.parametrize("query", [
    "compare precision and recall for my classifier",       # noun conjunction
    "clean and impute data",                                # left side has one token
    "train and evaluate a model",                           # left side has one token
    "Build the index and deploy",                           # right side has one token
    "Compare retrievers, e.g. BM25 and dense retrieval",    # "e.g." is not a sentence end
    "Summarize results, precision, recall and F1",          # commas before nouns
])
def test_split_keeps_a_single_intent_whole(query):
    assert split_intents(query) == [(query, query)]


def test_split_single_intent_is_routed_verbatim():
    # only stripped: no whitespace collapsing, trailing "?" kept (plain routing sees the same)
    assert split_intents("  How do I   evaluate a RAG\tsystem?  ") == [
        ("How do I   evaluate a RAG\tsystem?", "How do I   evaluate a RAG\tsystem?")]
    assert split_intents("Evaluate my RAG answers!") == [("Evaluate my RAG answers!",) * 2]


def test_split_soft_boundary_needs_an_action_verb_after_it():
    assert split_intents("Evaluate the model and deploy it") == [
        ("Evaluate the model", "Evaluate the model"), ("deploy it", "deploy the model")]
    assert split_intents("Monitor latency and log the errors") == [
        ("Monitor latency", "Monitor latency"), ("log the errors", "log the errors")]
    # verbs and connectors are case-insensitive; "fine-tune" is one word
    assert split_intents("Build the index AND Fine-tune it") == [
        ("Build the index", "Build the index"), ("Fine-tune it", "Fine-tune the index")]


def test_split_hard_boundaries_sentences_newlines_and_bullets():
    assert split_intents("Build a RAG system. Then evaluate it! Deploy it; monitor it?") == [
        ("Build a RAG system", "Build a RAG system"), ("evaluate it", "evaluate a RAG system"),
        ("Deploy it", "Deploy a RAG system"), ("monitor it", "monitor a RAG system")]
    listed = "- build a RAG system\n* evaluate it\n\n1. deploy it\n2) monitor it\n"
    assert split_intents(listed) == [
        ("build a RAG system", "build a RAG system"), ("evaluate it", "evaluate a RAG system"),
        ("deploy it", "deploy a RAG system"), ("monitor it", "monitor a RAG system")]
    assert split_intents("Build a RAG system\r\nevaluate it") == [
        ("Build a RAG system", "Build a RAG system"), ("evaluate it", "evaluate a RAG system")]
    # a hard boundary splits even without an action verb
    assert split_intents("What is RAG? How do I evaluate it?") == [
        ("What is RAG", "What is RAG"), ("How do I evaluate it", "How do I evaluate it")]
    # a part that is only a connector is dropped; one real part left -> the whole query
    assert split_intents("Bake bread.\nThen.") == [("Bake bread.\nThen.", "Bake bread.\nThen.")]


def test_split_strips_leading_connectors_and_drops_empty_parts():
    assert split_intents("Build a RAG system, and then, finally, deploy it") == [
        ("Build a RAG system", "Build a RAG system"), ("deploy it", "deploy a RAG system")]
    assert split_intents("Build a model. After that, evaluate it. Lastly.") == [
        ("Build a model", "Build a model"), ("evaluate it", "evaluate a model")]


def test_split_pronoun_resolution_rules():
    # only when the first part starts with an action verb
    assert split_intents("My RAG system is slow. Evaluate it.") == [
        ("My RAG system is slow", "My RAG system is slow"), ("Evaluate it", "Evaluate it")]
    # "set up" drops "up" from the object; only the first pronoun is replaced
    assert split_intents("Set up experiment tracking, then log it and version it")[1:] == [
        ("log it", "log experiment tracking"), ("version it", "version experiment tracking")]
    assert split_intents("Build a RAG system. Deploy it next to it")[1] == (
        "Deploy it next to it", "Deploy a RAG system next to it")
    # "this"/"that" as a determiner or conjunction is left alone; "them" is a pronoun
    assert split_intents("Train a classifier, then deploy this model")[1] == (
        "deploy this model", "deploy this model")
    assert split_intents("Train a classifier, then verify that it works")[1] == (
        "verify that it works", "verify that a classifier works")
    assert split_intents("Fine-tune a model, then evaluate that on held-out data")[1] == (
        "evaluate that on held-out data", "evaluate a model on held-out data")
    assert split_intents("Build a RAG system, then deploy them")[1] == (
        "deploy them", "deploy a RAG system")
    assert split_intents("Build a RAG system, then test it's latency")[1] == (
        "test it's latency", "test it's latency")                   # "it's" is not "it"


def test_split_empty_and_wordless_queries():
    assert split_intents("") == [] and split_intents("   ...  ") == []


def test_split_returns_every_part_unless_capped():
    q = "Build a model. Evaluate it. Deploy it. Monitor it. Explain it. Audit it. Secure it."
    assert len(split_intents(q)) == 7
    assert split_intents(q, max_intents=2) == split_intents(q)[:2]
    with pytest.raises(ValueError):
        split_intents(q, max_intents=0)


def test_split_is_deterministic():
    assert [split_intents(RAG) for _ in range(3)] == [split_intents(RAG)] * 3


ADVERSARIAL = {                                           # short ids: long ids break PYTEST_CURRENT_TEST
    "connector-run": "Build the index " + "and " * 20000 + "deploy it",   # was quadratic
    "then-run": "Build the index, " + "then " * 20000 + "deploy",
    "dot-run": "." * 200000 + "x",
    "long-token": "x" * 200000 + " y. z",
    "abbreviations": "Build the index " + "e.g. " * 20000,
    "dash-run": "Build the index, deploy it " + "-" * 200000,
}


@pytest.mark.parametrize("query", list(ADVERSARIAL.values()), ids=list(ADVERSARIAL))
def test_split_runs_in_linear_time_on_adversarial_input(query):
    import time
    start = time.perf_counter()
    split_intents(query)
    assert time.perf_counter() - start < 5.0               # ~0.05 s here; quadratic took minutes


def test_action_verbs_are_lowercase_single_words():
    assert all(v == v.lower() and " " not in v for v in ACTION_VERBS)


@pytest.mark.parametrize("query", [
    "Split the data into train, validation and test sets",     # "test" + noun
    "Compare validation accuracy and test accuracy",
    "Plot accuracy and log loss per epoch",                   # "log loss"
    "Make sure the train and test splits do not overlap",
    "Measure query latency and index size",                   # "index size"
])
def test_split_noun_homographs_do_not_start_a_task(query):
    assert split_intents(query) == [(query, query)]


@pytest.mark.parametrize("query, second", [
    ("Build a tool-calling agent, then test whether it calls the right tools",
     "test whether it calls the right tools"),
    ("Set up experiment tracking and log each run's params", "log each run's params"),
    ("Clean the data, then train a first model", "train a first model"),
    ("Fine-tune a model, then export to ONNX", "export to ONNX"),
    ("Pick a vector store, then set up the index", "set up the index"),
    ("Build a RAG system, then test it's latency", "test it's latency"),
])
def test_split_noun_homographs_used_as_verbs_still_split(query, second):
    parts = split_intents(query)
    assert len(parts) == 2 and parts[1][0] == second


def test_split_et_al_and_etc_do_not_end_a_sentence_mid_clause():
    for q in ("Handle missing values, outliers, etc. before training",
              "Replicate Vaswani et al. on my data"):
        assert split_intents(q) == [(q, q)]
    # "etc." followed by a capitalized word still ends the sentence
    assert split_intents("Clean the data, handle outliers, etc. Train a model") == [
        ("Clean the data, handle outliers, etc", "Clean the data, handle outliers, etc"),
        ("Train a model", "Train a model")]


def test_split_long_object_is_cut_to_its_head_phrase():
    q = ("Fine-tune a LoRA on Llama 3 8B using our past support replies so it picks up our tone, "
         "then evaluate it")
    assert split_intents(q)[1] == ("evaluate it", "evaluate a LoRA")
    assert split_intents("Set up experiment tracking with MLflow for all our training runs, "
                         "then log it")[1] == ("log it", "log experiment tracking")
    # up to 6 words: copied whole, prepositions included
    assert split_intents("Build a RAG system for support tickets, then deploy it")[1] == (
        "deploy it", "deploy a RAG system for support tickets")
    # longer, with no preposition / conjunction / relative word to cut at: left unresolved
    assert split_intents("Build a very large multilingual retrieval augmented generation system, "
                         "then deploy it")[1] == ("deploy it", "deploy it")


def test_split_resolves_pronouns_only_in_the_parts_it_returns(monkeypatch):
    import sie.compose as compose_mod
    calls = []
    real = compose_mod._resolve
    monkeypatch.setattr(compose_mod, "_resolve", lambda seg, obj: calls.append(seg) or real(seg, obj))
    q = "Build a model. " + "Test it. " * 1000
    assert len(split_intents(q, max_intents=3)) == 3 and len(calls) == 2
    calls.clear()
    router = FakeRouter({"Build a model": ["rag-pipeline"], "Test a model": ["rag-evaluation"]})
    c = compose(router, _rag_graph(), q, max_intents=2)
    assert len(c.intents) == 2 and len(calls) == 1
    assert c.notes == ["999 more intents were not routed (max_intents=2)"]


def test_routed_texts_lists_every_text_compose_may_route():
    assert routed_texts(RAG) == [
        "Build a RAG system", "evaluate a RAG system", "evaluate it", "deploy a RAG system",
        "deploy it", "monitor a RAG system", "monitor it"]
    assert routed_texts(RAG, max_intents=2) == [
        "Build a RAG system", "evaluate a RAG system", "evaluate it"]
    assert routed_texts("  Evaluate my RAG answers?  ") == ["Evaluate my RAG answers?"]
    assert routed_texts("Deploy the model. Deploy the model.") == ["Deploy the model"]
    assert routed_texts("   ") == []
    with pytest.raises(ValueError):
        routed_texts(RAG, max_intents=0)


# --- compose ---------------------------------------------------------------------------------

def test_single_intent_is_one_step_identical_to_plain_routing_top1():
    q = "Build a RAG system"
    router = FakeRouter({q: ["rag-pipeline", "rag-evaluation"]})
    c = compose(router, _rag_graph(), q)
    assert not c.multi_intent and len(c.intents) == 1
    assert c.intents[0].query == q and c.intents[0].selected == router.route(q, k=3).top.slug
    assert [(s.slug, s.position, s.role, s.intents, s.required_by) for s in c.steps] == [
        ("rag-pipeline", 1, "requested", [1], [])]
    assert c.steps[0].reason == "requested by intent 1"
    assert c.notes == [] and c.unmatched == [] and c.conflicts == []


def test_single_intent_routes_the_query_verbatim_so_a_near_tie_cannot_flip():
    # plain routing of the stripped query and routing without the "?" disagree on top-1
    q = "Which skill evaluates RAG answers?"
    router = FakeRouter({q: ["rag-evaluation", "rag-pipeline"],
                         "Which skill evaluates RAG answers": ["rag-pipeline", "rag-evaluation"]})
    c = compose(router, _rag_graph(), f"  {q}\n")
    assert router.calls == [(q, 3)]
    assert c.intents[0].query == q and c.intents[0].selected == router.route(q, k=3).top.slug
    assert _step(c, "rag-evaluation").role == "requested"


# "deploy/monitor a RAG pipeline" collapse onto the RAG skills (the copied object dominates, as
# in hybrid routing); routed as written, "deploy it" / "monitor it" find their own skills.
RAG_PIPELINE = "Build a RAG pipeline, evaluate it, deploy it, and monitor it"
COLLAPSING_SCRIPT = {
    "Build a RAG pipeline": ["rag-pipeline", "rag-evaluation"],
    "evaluate a RAG pipeline": ["rag-evaluation", "rag-pipeline"],
    "deploy a RAG pipeline": ["rag-pipeline", "model-deployment"],
    "deploy it": ["model-deployment", "other"],
    "monitor a RAG pipeline": ["rag-evaluation", "ml-monitoring"],
    "monitor it": ["ml-monitoring", "other"],
}


def test_pronoun_collapse_falls_back_to_the_part_as_written():
    router = FakeRouter(COLLAPSING_SCRIPT)
    c = compose(router, _rag_graph(), RAG_PIPELINE)
    assert [(it.text, it.query, it.selected) for it in c.intents] == [
        ("Build a RAG pipeline", "Build a RAG pipeline", "rag-pipeline"),
        ("evaluate it", "evaluate a RAG pipeline", "rag-evaluation"),     # resolved text kept
        ("deploy it", "deploy it", "model-deployment"),
        ("monitor it", "monitor it", "ml-monitoring")]
    assert c.intents[2].result.query == "deploy it"
    assert _slugs(c) == ["rag-pipeline", "rag-evaluation", "model-deployment", "ml-monitoring"]
    assert not any("merged" in n for n in c.notes)
    # "evaluate it" is never routed: its resolved text already found a new skill
    assert [q for q, _ in router.calls] == [
        "Build a RAG pipeline", "evaluate a RAG pipeline", "deploy a RAG pipeline", "deploy it",
        "monitor a RAG pipeline", "monitor it"]
    assert {q for q, _ in router.calls} <= set(routed_texts(RAG_PIPELINE))
    assert composition_summary(c)[3] == "  intent 3: 'deploy it' -> model-deployment [high/route]"


@pytest.mark.parametrize("as_written, why", [
    ([], "abstains"),
    (["rag-pipeline"], "an earlier intent's skill"),
])
def test_pronoun_collapse_keeps_the_resolved_result_when_the_fallback_is_no_better(as_written, why):
    router = FakeRouter({"Build a RAG pipeline": ["rag-pipeline"],
                         "deploy a RAG pipeline": ["rag-pipeline"], "deploy it": as_written})
    c = compose(router, _rag_graph(), "Build a RAG pipeline, then deploy it")
    assert (c.intents[1].query, c.intents[1].selected) == ("deploy a RAG pipeline", "rag-pipeline")
    assert c.notes == ["intents 1, 2 all map to rag-pipeline; merged into one step"]


def test_routed_texts_covers_every_route_call():
    for script, q in [(COLLAPSING_SCRIPT, RAG_PIPELINE), (RAG_SCRIPT, RAG),
                      ({"Bake bread.\nThen.": []}, "Bake bread.\nThen.")]:
        router = FakeRouter(script)
        compose(router, _rag_graph(), q)
        assert {text for text, _ in router.calls} <= set(routed_texts(q))


def test_two_intents_follow_mention_order_without_edges():
    router = FakeRouter({"Deploy the model": ["model-deployment"],
                         "monitor the model": ["ml-monitoring"]})
    c = compose(router, _rag_graph(), "Deploy the model and monitor it")
    assert c.multi_intent and [it.query for it in c.intents] == ["Deploy the model", "monitor the model"]
    assert _slugs(c) == ["model-deployment", "ml-monitoring"]
    assert [s.intents for s in c.steps] == [[1], [2]]
    assert router.calls == [("Deploy the model", 3), ("monitor the model", 3)]   # k passed through


def test_rag_example_prerequisites_pulled_in_and_ordered():
    g = _rag_graph(**{"model-deployment": {"requires": ["python-for-ml"]}})
    c = compose(FakeRouter(RAG_SCRIPT), g, RAG)
    assert [it.selected for it in c.intents] == ["rag-pipeline", "rag-evaluation", "model-deployment",
                                                "ml-monitoring"]
    assert _slugs(c) == ["rag-pipeline", "rag-evaluation", "python-for-ml", "model-deployment",
                         "ml-monitoring"]
    assert [s.position for s in c.steps] == [1, 2, 3, 4, 5]
    pre = _step(c, "python-for-ml")
    assert (pre.role, pre.intents, pre.required_by, pre.reason) == (
        "prerequisite", [3], ["model-deployment"], "required by model-deployment")
    rp = _step(c, "rag-pipeline")
    assert rp.role == "requested" and rp.required_by == ["rag-evaluation"]
    assert rp.reason == "requested by intent 1; also required by rag-evaluation"
    assert [(e.source, e.target, e.relationship) for e in c.edges] == [
        ("python-for-ml", "model-deployment", "requires"), ("rag-pipeline", "rag-evaluation", "requires")]


def test_prerequisite_order_wins_over_mention_order():
    q = "Evaluate a RAG system, build it, monitor it, and deploy it"
    g = _rag_graph(**{"ml-monitoring": {"requires": ["model-deployment"]}})
    router = FakeRouter({"Evaluate a RAG system": ["rag-evaluation"], "build a RAG system": ["rag-pipeline"],
                         "monitor a RAG system": ["ml-monitoring"], "deploy a RAG system": ["model-deployment"]})
    c = compose(router, g, q)
    assert [it.selected for it in c.intents] == ["rag-evaluation", "rag-pipeline", "ml-monitoring",
                                                "model-deployment"]
    # hard requires beat the stated order; then intent order (a prerequisite inherits the
    # earliest intent that needs it, so rag-pipeline/rag-evaluation come before deployment)
    assert _slugs(c) == ["rag-pipeline", "rag-evaluation", "model-deployment", "ml-monitoring"]
    assert all(s.role == "requested" for s in c.steps)
    assert _step(c, "model-deployment").required_by == ["ml-monitoring"]


def test_ambiguous_intent_is_kept_and_flagged():
    router = FakeRouter({"Build a RAG system": ["rag-pipeline"],
                         "evaluate a RAG system": (["rag-evaluation", "other"], "medium", ["other"])})
    c = compose(router, _rag_graph(), "Build a RAG system, then evaluate it")
    assert c.intents[1].selected == "rag-evaluation" and c.unmatched == []
    assert c.notes == ["intent 2 ('evaluate it') is ambiguous: rag-evaluation vs other; "
                       "confirm before acting"]
    assert _step(c, "rag-evaluation").reason == "requested by intent 2; ambiguous for intent 2 (vs other)"


def test_low_confidence_selection_is_kept_and_flagged():
    router = FakeRouter({"Build a RAG system": (["rag-pipeline"], "low", [])})
    c = compose(router, _rag_graph(), "Build a RAG system")
    assert _slugs(c) == ["rag-pipeline"]
    assert c.notes == ["intent 1 ('Build a RAG system') matched rag-pipeline only weakly "
                       "(low confidence); confirm before acting"]
    assert c.steps[0].reason == "requested by intent 1; low confidence for intent 1"


def test_unrelated_intent_abstains_and_others_are_kept():
    router = FakeRouter({"Build a RAG system": ["rag-pipeline"], "Bake bread": []})
    c = compose(router, _rag_graph(), "Build a RAG system. Bake bread.")
    assert [it.selected for it in c.intents] == ["rag-pipeline", None]
    assert c.unmatched == [2] and _slugs(c) == ["rag-pipeline"]
    assert c.notes == ["intent 2 ('Bake bread') has no sufficiently good match"]


def test_notes_and_summary_show_a_multiline_intent_on_one_line():
    c = compose(FakeRouter({"Bake bread.\nThen.": []}), _rag_graph(), "Bake bread.\nThen.")
    assert c.intents[0].text == "Bake bread.\nThen."
    assert c.notes[0] == "intent 1 ('Bake bread. Then.') has no sufficiently good match"
    assert all("\n" not in line for line in composition_summary(c))


def test_abstain_with_candidates_selects_nothing():
    router = FakeRouter({"Bake bread": (["other"], "none", [])})
    c = compose(router, _rag_graph(), "Bake bread")
    assert c.intents[0].selected is None and c.steps == [] and c.unmatched == [1]
    assert c.notes == ["intent 1 ('Bake bread') has no sufficiently good match",
                       "no intent matched a skill; the plan is empty"]


def test_duplicate_intents_merge_into_one_step():
    router = FakeRouter({"Evaluate the retriever": ["rag-evaluation"],
                         "evaluate the generator": ["rag-evaluation"],
                         "deploy the retriever": ["model-deployment"]})
    c = compose(router, _rag_graph(), "Evaluate the retriever, then evaluate the generator, "
                                     "then deploy it", include_prerequisites=False)
    assert _slugs(c) == ["rag-evaluation", "model-deployment"]
    assert _step(c, "rag-evaluation").intents == [1, 2]
    assert _step(c, "rag-evaluation").reason == "requested by intents 1, 2"
    assert c.notes == ["intents 1, 2 all map to rag-evaluation; merged into one step"]


def test_identical_routed_text_is_routed_once():
    router = FakeRouter({"Deploy the model": ["model-deployment"]})
    c = compose(router, _rag_graph(), "Deploy the model. Deploy the model.")
    assert len(c.intents) == 2 and router.calls == [("Deploy the model", 3)]
    assert _slugs(c) == ["model-deployment"] and _step(c, "model-deployment").intents == [1, 2]


def test_prerequisites_come_only_from_declared_requires():
    g = build_graph([
        _s("a", "beginner"), _s("b", requires=["a"]),
        _s("t", requires=["b"], related=["rel"], conflicts=["con"], recommended_before=["soft"],
           alternative_to=["alt"], specializes=["gen"], supersedes=["old"]),
        _s("rel"), _s("con"), _s("soft"), _s("alt"), _s("gen"), _s("old")])
    c = compose(FakeRouter({"Train the target": ["t"]}), g, "Train the target")
    assert _slugs(c) == ["a", "b", "t"]                          # transitive requires only
    a, b = _step(c, "a"), _step(c, "b")
    assert (a.role, a.intents, a.required_by, a.reason) == ("prerequisite", [1], ["b"], "required by b")
    assert (b.role, b.required_by, b.reason) == ("prerequisite", ["t"], "required by t")
    assert {e.relationship for e in c.edges} == {"requires"}     # other relations point outside

    plain = compose(FakeRouter({"Train the target": ["t"]}), g, "Train the target",
                    include_prerequisites=False)
    assert _slugs(plain) == ["t"] and plain.edges == []


def test_shared_prerequisite_appears_once_with_union_of_intents():
    g = build_graph([_s("base"), _s("x", requires=["base"]), _s("y", requires=["base"]), _s("z")])
    router = FakeRouter({"Build the z thing": ["z"], "build the x thing": ["x"], "build the y thing": ["y"]})
    c = compose(router, g, "Build the z thing, build the x thing, build the y thing")
    assert _slugs(c) == ["z", "base", "x", "y"]
    base = _step(c, "base")
    assert base.intents == [2, 3] and base.required_by == ["x", "y"] and base.reason == "required by x, y"


def test_requested_prerequisite_keeps_its_own_intents_but_moves_up():
    g = build_graph([_s("p"), _s("t", requires=["p"]), _s("u")])
    router = FakeRouter({"Train the t model": ["t"], "tune the u model": ["u"], "prepare the p data": ["p"]})
    c = compose(router, g, "Train the t model, tune the u model, prepare the p data")
    assert _slugs(c) == ["p", "t", "u"]
    p = _step(c, "p")
    assert (p.role, p.intents, p.required_by) == ("requested", [3], ["t"])
    assert p.reason == "requested by intent 3; also required by t"


def test_conflicts_are_flagged_as_sorted_pairs():
    g = build_graph([_s("zeta", conflicts=["alpha"]), _s("alpha"), _s("mid")])
    router = FakeRouter({"Train with zeta": ["zeta"], "train with alpha": ["alpha"], "train with mid": ["mid"]})
    c = compose(router, g, "Train with zeta, train with alpha, train with mid")
    assert c.conflicts == [["alpha", "zeta"]]
    assert c.notes == ["alpha conflicts with zeta; review before combining them"]
    assert [(e.source, e.target, e.relationship) for e in c.edges] == [("zeta", "alpha", "conflicts")]


def test_superseded_skill_is_noted():
    g = build_graph([_s("old"), _s("new", supersedes=["old"])])
    c = compose(FakeRouter({"Train the old way": ["old"]}), g, "Train the old way")
    assert _slugs(c) == ["old"] and c.notes == ["old is superseded by new"]


def test_skill_missing_from_graph_is_listed_without_dependencies():
    router = FakeRouter({"Build a RAG system": ["rag-pipeline"], "deploy a RAG system": ["ghost"]})
    c = compose(router, _rag_graph(), "Build a RAG system, then deploy it")
    assert _slugs(c) == ["rag-pipeline", "ghost"]
    ghost = _step(c, "ghost")
    assert ghost.role == "requested" and ghost.required_by == []
    assert ghost.reason == "requested by intent 2; not in the skill graph, so its prerequisites are unknown"
    assert c.notes == ["ghost is not in the skill graph (the router's corpus differs from the graph's); "
                       "its prerequisites and relationships are unknown"]


def test_missing_and_undeclared_prerequisites_are_noted_never_fabricated():
    g = build_graph([_s("t", requires=["typo"]), _s("bare", declared=[])])
    router = FakeRouter({"Train the t model": ["t"], "tune the bare model": ["bare"]})
    c = compose(router, g, "Train the t model, tune the bare model")
    assert _slugs(c) == ["t", "bare"]                            # intent order
    assert c.notes == ["bare does not declare prerequisites; the plan may be incomplete",
                       "t requires 'typo', which is not in the corpus; it is omitted"]
    assert compose(router, g, "Train the t model, tune the bare model",
                   include_prerequisites=False).notes == []


def test_soft_edges_order_but_never_add_members():
    # "late" is recommended before "early": it moves ahead of the stated order
    g = build_graph([_s("early", recommended_before=["late"]), _s("late"), _s("unasked"),
                     _s("x", recommended_before=["unasked"])])
    router = FakeRouter({"Train the early model": ["early"], "tune the late model": ["late"],
                         "test the x model": ["x"]})
    c = compose(router, g, "Train the early model, tune the late model, test the x model")
    assert _slugs(c) == ["late", "early", "x"]                   # "unasked" is not added
    assert [(e.source, e.target, e.relationship) for e in c.edges] == [
        ("late", "early", "recommended_before")]


def test_soft_edge_cycle_is_dropped_and_noted():
    # a soft edge b -> a (declared on a) contradicts a -> b (requires)
    g = build_graph([_s("a", recommended_before=["b"]), _s("b", requires=["a"])])
    c = compose(FakeRouter({"Train the b model": ["b"]}), g, "Train the b model")
    assert _slugs(c) == ["a", "b"]
    assert c.notes == ["ignored recommended_before b -> a: it would create an ordering cycle"]


def test_requires_cycle_among_plan_skills_raises():
    g = build_graph([_s("a", requires=["b"]), _s("b", requires=["a"])])
    with pytest.raises(ValueError, match="requires cycle"):
        compose(FakeRouter({"Train the a model": ["a"]}), g, "Train the a model")


def test_overlay_prerequisite_is_marked_as_not_declared():
    overlay = [Edge("rag-pipeline", "model-deployment", "requires", confidence="proposed",
                    provenance="docs/x.md")]
    g = build_graph([_s("rag-pipeline"), _s("model-deployment")], overlays=overlay)
    c = compose(FakeRouter({"Deploy a RAG system": ["model-deployment"]}), g, "Deploy a RAG system")
    assert _slugs(c) == ["rag-pipeline", "model-deployment"]
    assert _step(c, "rag-pipeline").reason == "required by model-deployment (proposed)"
    assert c.edges == overlay


def test_truncation_routes_only_max_intents_and_notes_the_rest():
    q = "Build a model. Evaluate it. Deploy it. Monitor it. Explain it. Audit it. Secure it."
    texts = ["Build a model"] + [f"{v} a model" for v in ("Evaluate", "Deploy", "Monitor", "Explain")]
    slugs = ["rag-pipeline", "rag-evaluation", "model-deployment", "ml-monitoring", "other"]
    script = {text: [slug] for text, slug in zip(texts, slugs)}      # distinct: no fallback routing
    router = FakeRouter(script)
    c = compose(router, _rag_graph(), q)
    assert len(c.intents) == 5 and len(router.calls) == 5
    assert c.notes[0] == "2 more intents were not routed (max_intents=5)"
    one = compose(FakeRouter(script), _rag_graph(), "Build a model. Evaluate it.", max_intents=1)
    assert len(one.intents) == 1 and one.notes == ["1 more intent was not routed (max_intents=1)"]


def test_noun_conjunction_is_routed_as_one_intent():
    q = "compare precision and recall for my classifier"
    router = FakeRouter({q: ["other"]})
    c = compose(router, _rag_graph(), q)
    assert len(c.intents) == 1 and router.calls == [(q, 3)] and _slugs(c) == ["other"]


def test_pronouns_are_resolved_before_routing():
    router = FakeRouter(RAG_SCRIPT)
    c = compose(router, _rag_graph(), RAG)
    assert [q for q, _ in router.calls] == list(RAG_SCRIPT)
    assert [(it.index, it.text, it.query) for it in c.intents] == [
        (1, "Build a RAG system", "Build a RAG system"), (2, "evaluate it", "evaluate a RAG system"),
        (3, "deploy it", "deploy a RAG system"), (4, "monitor it", "monitor a RAG system")]


def test_notes_order_intents_first_then_plan_level():
    g = build_graph([_s("a", conflicts=["b"]), _s("b")])
    router = FakeRouter({"Train the a model": ["a"], "bake the bread": [],
                         "Train the b model": (["b", "a"], "medium", ["a"]),
                         "Tune the a model": ["a"]})
    c = compose(router, g, "Train the a model. bake the bread. Train the b model. Tune the a model.")
    assert c.notes == [
        "intent 2 ('bake the bread') has no sufficiently good match",
        "intent 3 ('Train the b model') is ambiguous: b vs a; confirm before acting",
        "intents 1, 4 all map to a; merged into one step",
        "a conflicts with b; review before combining them"]


def test_compose_is_deterministic():
    g = _rag_graph(**{"model-deployment": {"requires": ["python-for-ml"]}})
    runs = [compose(FakeRouter(RAG_SCRIPT), g, RAG) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert composition_summary(runs[0]) == composition_summary(runs[1])


def test_empty_query_and_invalid_arguments():
    router = FakeRouter({})
    c = compose(router, _rag_graph(), "   ")
    assert (c.intents, c.steps, c.notes) == ([], [], ["empty query: nothing to compose"])
    assert router.calls == []
    for kw in ({"k": 0}, {"max_intents": 0}, {"k": True}):
        with pytest.raises(ValueError):
            compose(router, _rag_graph(), "Build a RAG system", **kw)


def test_composition_summary_lines():
    router = FakeRouter({"Build a RAG system": ["rag-pipeline"], "evaluate a RAG system": ["rag-evaluation"],
                         "Bake bread": []})
    c = compose(router, _rag_graph(), "Build a RAG system, then evaluate it. Bake bread.")
    assert composition_summary(c) == [
        "[compose] 3 intents -> 2 skills in the plan",
        "  intent 1: 'Build a RAG system' -> rag-pipeline [high/route]",
        "  intent 2: 'evaluate it' (routed as 'evaluate a RAG system') -> rag-evaluation [high/route]",
        "  intent 3: 'Bake bread' -> no match [none/abstain]",
        f"  1. {'rag-pipeline':26s} {'requested':12s} requested by intent 1; also required by rag-evaluation",
        f"  2. {'rag-evaluation':26s} {'requested':12s} requested by intent 2",
        "  note: intent 3 ('Bake bread') has no sufficiently good match"]


# --- real corpus, BM25-only router (no models) -------------------------------------------------

@pytest.fixture(scope="module")
def sparse_router():
    from sie.router import HybridRouter
    return HybridRouter(skills_dir=str(CORPUS), mode="sparse", use_reranker=False)


def _without_timings(c):
    """Everything a Composition decided; RouteResult.timings_ms is wall-clock, so dropped."""
    intents = [(it.index, it.text, it.query, it.selected, it.result.results, it.result.confidence)
               for it in c.intents]
    return intents, c.steps, c.edges, c.conflicts, c.unmatched, c.notes


def test_real_corpus_rag_request_orders_pipeline_before_evaluation(sparse_router):
    q = "Build a RAG pipeline, evaluate it, deploy it, and monitor it"
    g = build_graph(load_corpus(CORPUS))
    c = compose(sparse_router, g, q)
    assert len(c.intents) >= 3
    assert c.intents[1].query == "evaluate a RAG pipeline"
    slugs = _slugs(c)
    assert slugs and len(slugs) == len(set(slugs)) and all(s in g for s in slugs)
    if "rag-pipeline" in slugs and "rag-evaluation" in slugs:
        assert slugs.index("rag-pipeline") < slugs.index("rag-evaluation")
    assert _without_timings(compose(sparse_router, g, q)) == _without_timings(c)


def test_real_corpus_rag_request_keeps_deploy_and_monitor(sparse_router):
    # the resolved "deploy/monitor a RAG pipeline" land on the RAG skills again; the
    # as-written fallback keeps the deployment and monitoring tasks in the plan
    g = build_graph(load_corpus(CORPUS))
    c = compose(sparse_router, g, RAG_PIPELINE)
    selected = [it.selected for it in c.intents]
    assert "model-deployment" in selected and "ml-monitoring" in selected
    assert len(set(selected)) == len(selected) == 4
    assert {it.query for it in c.intents} <= set(routed_texts(RAG_PIPELINE))


def test_single_intent_on_the_real_corpus_matches_plain_routing(sparse_router):
    g = build_graph(load_corpus(CORPUS))
    for q in ("How do I evaluate my RAG answers?", "  deploy a model behind an API.\n"):
        c = compose(sparse_router, g, q)
        assert len(c.intents) == 1 and c.intents[0].query == q.strip()
        assert c.intents[0].selected == sparse_router.route(q.strip(), k=3).top.slug


def test_cli_reports_a_missing_dependency_with_the_install_hint(monkeypatch):
    from sie.router import HybridRouter

    def no_chromadb(self, *a, **kw):
        raise ModuleNotFoundError("No module named 'chromadb'", name="chromadb")
    monkeypatch.setattr(HybridRouter, "route", no_chromadb)
    with pytest.raises(SystemExit) as exc:
        main(["Build a RAG pipeline", "--skills", str(CORPUS), "--mode", "sparse", "--no-rerank"])
    msg = str(exc.value.code)
    assert msg.startswith("[compose] missing dependency: No module named 'chromadb'")
    assert "install the 'retrieval' extra" in msg


def test_cli_prints_the_summary(capsys):
    main(["Build a RAG pipeline, then evaluate it", "--skills", str(CORPUS), "--mode", "sparse",
          "--no-rerank"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[compose] 2 intents -> ") and out[0].endswith(" in the plan")
    assert out[1].startswith("  intent 1: 'Build a RAG pipeline' -> ")
    assert out[2].startswith("  intent 2: 'evaluate it' (routed as 'evaluate a RAG pipeline') -> ")
