"""Model-availability checks shared by the dense embedder and the cross-encoder.

They run *before* importing torch (~30 s cold start on Windows) so a missing model fails
in seconds: a local directory, the Hugging Face cache, or a reachable hub.
"""
from __future__ import annotations
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


def looks_like_path(name: str) -> bool:
    """A local-path spelling (./x, /x, ~/x, C:\\x, a/b/c) rather than an HF repo id (org/name)."""
    return (name.startswith((".", "/", "~", "\\")) or "\\" in name
            or re.match(r"^[A-Za-z]:", name) is not None or name.count("/") > 1)


def cached_locally(name: str) -> bool:
    """True if `name` is a local model dir or a repo id already in the Hugging Face cache."""
    if Path(name).expanduser().is_dir():
        return True
    if looks_like_path(name):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str)
    except Exception:               # hub library missing, or not a valid repo id
        return False


def hub_reachable(timeout: float = 3.0) -> bool:
    """Fast TCP probe of HF_ENDPOINT (scheme/port honored); assume reachable behind a proxy."""
    if os.getenv("HF_HUB_OFFLINE") == "1":
        return False
    if any(os.getenv(v) for v in _PROXY_VARS):
        return True                 # can't cheaply probe through a proxy; let the hub client try
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    url = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
    port = url.port or (80 if url.scheme == "http" else 443)
    try:
        socket.create_connection((url.hostname or "huggingface.co", port), timeout=timeout).close()
        return True
    except OSError:
        return False


def unavailable_reason(name: str) -> str | None:
    """None if model `name` can be loaded right now, else a short reason why not."""
    if cached_locally(name):
        return None
    if looks_like_path(name):
        return f"local model path not found: {name}"
    if not hub_reachable():
        return "no local weights and the Hugging Face hub is unreachable"
    return None
