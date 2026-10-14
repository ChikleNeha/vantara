"""Ollama client: turns model outputs into plain language and answers questions about the portfolio.

The LLM never computes predictions. It only rewrites facts we hand it, and every call has a
deterministic fallback so the platform works with Ollama switched off.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

NARRATIVE_SYSTEM = (
    "You are an analyst writing for a retail marketing manager. Explain a customer's churn risk in 3 short sentences "
    "of plain English using ONLY the facts in the JSON. Do not invent numbers, products or reasons. "
    "No jargon such as SHAP or log-odds. End with one concrete suggested action."
)
ASSISTANT_SYSTEM = (
    "You answer questions about a retail customer portfolio using ONLY the context JSON provided. "
    "If the context does not contain the answer, say so plainly. Be concise and quote numbers exactly as given."
)


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaClient:
    def __init__(self, cfg: Config):
        o = cfg.ollama
        self.base_url, self.model, self.timeout = o["base_url"].rstrip("/"), o["model"], float(o["timeout_seconds"])

    def status(self) -> dict[str, Any]:
        """Reachability and whether the configured model is pulled."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            has = any(n == self.model or n.startswith(f"{self.model}:") for n in names)
            return {"available": True, "model": self.model, "model_pulled": has, "installed": names}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "model": self.model, "model_pulled": False, "error": str(exc)[:120]}

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailable(str(exc)) from exc

    def narrate(self, facts: dict[str, Any]) -> str:
        """Rewrite structured churn facts as a short narrative."""
        return self._chat(NARRATIVE_SYSTEM, json.dumps(facts, default=str))

    def ask(self, question: str, context: dict[str, Any]) -> str:
        """Answer a free-text question grounded in a context dictionary built from the database."""
        return self._chat(ASSISTANT_SYSTEM, f"Context JSON:\n{json.dumps(context, default=str)}\n\nQuestion: {question}")
