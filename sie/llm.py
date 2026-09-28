"""Swappable LLM client. Defaults to a no-op so the engine runs with no API key."""
from __future__ import annotations
import os


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "noop")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def complete(self, prompt: str, system: str = "") -> str:
        if self.provider == "noop":
            return "[llm-disabled] set LLM_PROVIDER=openai|ollama to enable generation."
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            r = client.chat.completions.create(model=self.model, messages=msgs)
            return r.choices[0].message.content
        if self.provider == "ollama":
            import urllib.request, json
            url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
            data = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())["response"]
        raise ValueError(f"unknown provider: {self.provider}")
