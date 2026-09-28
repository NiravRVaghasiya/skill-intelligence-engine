"""Canaries: the conftest guardrails really do block model libraries and the network."""
import importlib
import socket

import pytest

from tests.conftest import BLOCKED, NetworkBlocked


@pytest.mark.parametrize("name", sorted(BLOCKED))
def test_heavy_imports_are_blocked(name):
    with pytest.raises(ImportError, match="blocked in tests"):
        importlib.import_module(name)


def test_network_is_blocked():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("pypi.org", 443)
