"""Test-suite guardrails (CLAUDE.md G3/G4): no heavy model libraries, no network.

Any test that reaches for chromadb / sentence-transformers / torch / onnxruntime /
transformers / huggingface_hub, or opens a non-loopback socket, fails loudly instead of
silently downloading a model. Both guards are installed when this file is imported, so they
also cover collection, test-module import code and fixtures of every scope.

Opt-in exception: with SIE_INTEGRATION=1 the heavy-import blocker is not installed, so the
model-backed tests in tests/integration/ can run against locally cached models and the
persisted index. The network stays blocked either way (models must already be cached). In
that mode only tests marked `integration` run: the unit tests assume the blocker.
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


INTEGRATION = os.environ.get("SIE_INTEGRATION") == "1"
if not INTEGRATION:
    sys.meta_path.insert(0, _HeavyImportBlocker())


def pytest_collection_modifyitems(config, items):
    if not INTEGRATION:
        return
    skip = pytest.mark.skip(reason="SIE_INTEGRATION=1 runs only integration tests "
                                   "(unit tests assume the heavy-import blocker)")
    for item in items:
        if item.get_closest_marker("integration") is None:
            item.add_marker(skip)


class NetworkBlocked(RuntimeError):
    pass


def _install_network_guard() -> None:
    """Block non-loopback connects and DNS for the whole session.

    Installed when this conftest is imported, before collection, so test-module import code
    and fixtures of every scope (module-scoped ones run before function-scoped autouse
    fixtures) are covered, not only test bodies. Idempotent.
    """
    if getattr(socket.socket.connect, "_sie_guard", False):
        return
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise NetworkBlocked(f"network access to {address!r} attempted in a test")
        return real_connect(self, address)

    def guarded_dns(host, *args, **kwargs):
        if host not in _LOOPBACK and host is not None:
            raise NetworkBlocked(f"DNS lookup of {host!r} attempted in a test")
        return real_getaddrinfo(host, *args, **kwargs)

    guarded._sie_guard = guarded_dns._sie_guard = True
    socket.socket.connect = guarded
    socket.getaddrinfo = guarded_dns


_install_network_guard()
