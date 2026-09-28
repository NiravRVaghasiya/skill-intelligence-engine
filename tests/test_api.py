"""Every endpoint via TestClient, with the router mocked (no index, no models)."""
import pytest
from fastapi.testclient import TestClient

from sie import api as api_mod
from sie.models import Hit


class FakeRouter:
    def __init__(self, error=None):
        self.error, self.reranker, self.calls, self.last_reranked = error, None, [], False

    def retrieve(self, query, k=5, pool=20):
        self.calls.append((query, k) if pool == 20 else (query, k, pool))
        if self.error:
            raise self.error
        return [Hit("data-preprocessing", 0.91234, section="Card"),
                Hit("feature-engineering", 0.5, section="Workflow")][:k]


@pytest.fixture
def client(monkeypatch):
    fake = FakeRouter()
    monkeypatch.setattr(api_mod, "_router", lambda: fake)
    c = TestClient(api_mod.api)
    c.fake = fake
    return c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "version": "0.1.0", "skills": 38}


def test_search_returns_ranked_results(client):
    r = client.get("/search", params={"q": "impute missing values", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "impute missing values" and body["reranked"] is False
    assert body["results"][0] == {"slug": "data-preprocessing", "score": 0.9123, "section": "Card"}
    assert client.fake.calls == [("impute missing values", 2)]


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 999}])
def test_search_validates_params(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_search_unbuilt_index_is_503(monkeypatch):
    monkeypatch.setattr(api_mod, "_router", lambda: FakeRouter(RuntimeError("dense index not built")))
    r = TestClient(api_mod.api).get("/search", params={"q": "x"})
    assert r.status_code == 503 and "not built" in r.json()["detail"]


def test_learning_path(client):
    r = client.get("/learning-path", params={"target": "rag-evaluation"})
    assert r.status_code == 200
    assert r.json()["path"] == ["rag-pipeline", "rag-evaluation"]
    assert "llm-evaluation" in r.json()["related"]


def test_learning_path_unknown_target_is_404(client):
    r = client.get("/learning-path", params={"target": "nope"})
    assert r.status_code == 404 and "unknown skill" in r.json()["detail"]


def test_skill_detail(client):
    body = client.get("/skill/rag-evaluation").json()
    assert body["slug"] == "rag-evaluation" and body["domain"] == "llm"
    assert body["requires"] == ["rag-pipeline"] and body["required_by"] == []
    assert body["description"].startswith("Use when")
    assert client.get("/skill/rag-pipeline").json()["required_by"] == ["rag-evaluation"]


def test_skill_unknown_is_404(client):
    assert client.get("/skill/nope").status_code == 404


def _write(root, slug, fm):
    (root / slug).mkdir(parents=True)
    (root / slug / "SKILL.md").write_text(
        f"---\ntype: workflow\ndomain: d\nlevel: intermediate\n{fm}\n---\n## Overview\nhi\n", encoding="utf-8")


@pytest.fixture
def corpus_dir(monkeypatch, tmp_path):
    """Point the API at a scratch corpus; restore the real cached graph afterwards."""
    monkeypatch.setattr(api_mod, "SKILLS_DIR", str(tmp_path))
    api_mod._graph_cached.cache_clear()
    yield tmp_path
    api_mod._graph_cached.cache_clear()


def test_missing_corpus_is_503_and_not_cached(client, corpus_dir):
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"
    assert client.get("/learning-path", params={"target": "a"}).status_code == 503
    assert client.get("/skill/a").status_code == 503
    _write(corpus_dir, "a", "")                        # corpus appears: no restart needed
    assert client.get("/health").json() == {"status": "ok", "version": "0.1.0", "skills": 1}


def test_skill_relationships_match_learning_path(client, corpus_dir):
    _write(corpus_dir, "t", "related:\n  - x\n  - y\n  - y\n  - nobody\nrequires:\n  - ghost")
    _write(corpus_dir, "x", "conflicts:\n  - t")
    _write(corpus_dir, "y", "")
    body = client.get("/skill/t").json()
    assert body["conflicts"] == ["x"] and body["related"] == ["y"]
    assert body["dangling"] == [["related", "nobody"], ["requires", "ghost"]]
    lp = client.get("/learning-path", params={"target": "t"}).json()
    assert lp["conflicts"] == body["conflicts"] and lp["related"] == body["related"]
    assert lp["missing_prerequisites"] == [["t", "ghost"]]


def test_search_passes_pool(client):
    body = client.get("/search", params={"q": "x", "k": 2, "pool": 40}).json()
    assert body["pool"] == 40
    assert client.get("/search", params={"q": "x", "pool": 0}).status_code == 422
