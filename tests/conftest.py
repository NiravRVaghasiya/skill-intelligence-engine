"""Test-suite guardrails (CLAUDE.md G3/G4): no heavy model libraries, no network.

Any test that reaches for chromadb / sentence-transformers / torch / onnxruntime /
transformers / huggingface_hub, or opens a non-loopback socket, fails loudly instead of
silently downloading a model.
"""
import importlib.abc
import os
import socket
import sys

import pytest

BLOCKED = {"chromadb", "sentence_transformers", "torch", "onnxruntime", "transformers",
           "huggingface_hub"}
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class _HeavyImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"'{name}' is blocked in tests: mock the dense index / reranker (G3/G4)")
        return None


sys.meta_path.insert(0, _HeavyImportBlocker())


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise NetworkBlocked(f"network access to {address!r} attempted in a test")
        return real_connect(self, address)

    real_getaddrinfo = socket.getaddrinfo

    def guarded_dns(host, *args, **kwargs):
        if host not in _LOOPBACK and host is not None:
            raise NetworkBlocked(f"DNS lookup of {host!r} attempted in a test")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_dns)
