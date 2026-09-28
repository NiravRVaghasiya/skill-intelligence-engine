"""Model-availability probes, with the network and hub library faked."""
import pytest

from sie import hub


@pytest.mark.parametrize("name,expected", [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", False), ("all-MiniLM-L6-v2", False),
    ("data/models/ms-marco-MiniLM-L-6-v2", True), ("./ce", True), ("/models/ce", True),
    ("~/ce", True), (r"C:\models\ce", True), ("C:/models/ce", True),
])
def test_looks_like_path(name, expected):
    assert hub.looks_like_path(name) is expected


def test_local_dir_counts_as_cached(tmp_path):
    assert hub.cached_locally(str(tmp_path)) and hub.unavailable_reason(str(tmp_path)) is None


def test_missing_path_never_probes_the_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "hub_reachable", lambda timeout=3.0: pytest.fail("probed the hub"))
    assert "local model path not found" in hub.unavailable_reason(str(tmp_path / "nope"))


def _capture_probe(monkeypatch):
    seen = []

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(hub.socket, "create_connection", lambda addr, timeout: seen.append(addr) or Conn())
    return seen


@pytest.mark.parametrize("endpoint,addr", [
    (None, ("huggingface.co", 443)), ("http://127.0.0.1:8765", ("127.0.0.1", 8765)),
    ("http://mirror.local", ("mirror.local", 80)), ("mirror.example", ("mirror.example", 443)),
])
def test_probe_honors_endpoint_scheme_and_port(monkeypatch, endpoint, addr):
    for v in hub._PROXY_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    if endpoint:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
    else:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    seen = _capture_probe(monkeypatch)
    assert hub.hub_reachable() and seen == [addr]


def test_offline_and_proxy_short_circuit(monkeypatch):
    seen = _capture_probe(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hub.hub_reachable() is False
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    assert hub.hub_reachable() is True and seen == []


def test_unreachable_hub(monkeypatch):
    def refuse(addr, timeout):
        raise OSError("dns")
    for v in hub._PROXY_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hub.socket, "create_connection", refuse)
    assert "unreachable" in hub.unavailable_reason("org/model-that-is-not-cached")
