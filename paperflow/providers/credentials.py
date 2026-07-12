"""Credential resolution for OpenAI-compatible PaperFlow providers.

PaperFlow uses the OpenAI Python SDK for several compatible APIs.  The
credentials for daily LLM generation and embeddings intentionally resolve
independently so a text-only gateway (for example, DeepSeek) cannot
accidentally be used for embeddings, and neither setting is used by the
PaperDaily Codex CLI adapter.
"""

from __future__ import annotations

import os
from typing import Literal

OpenAICompatibleKind = Literal["llm", "embed"]


def _first_configured(*names: str) -> str:
    """Return the first non-empty environment value in precedence order."""

    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def resolve_openai_compatible_credentials(kind: OpenAICompatibleKind) -> tuple[str, str | None]:
    """Resolve a provider-specific API key and base URL.

    Precedence is deliberately separate for the two PaperFlow API paths:

    * LLM: ``PAPERFLOW_LLM_*`` -> ``PAPERFLOW_OPENAI_*`` -> ``OPENAI_*``
    * embedding: ``PAPERFLOW_EMBED_*`` -> ``PAPERFLOW_OPENAI_*`` -> ``OPENAI_*``

    ``OPENAI_*`` remains a backward-compatible fallback.  This helper is not
    used by PaperDaily's Codex CLI provider, whose authentication contract is
    intentionally independent.
    """

    if kind == "llm":
        api_key = _first_configured(
            "PAPERFLOW_LLM_API_KEY",
            "PAPERFLOW_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
        base_url = _first_configured(
            "PAPERFLOW_LLM_BASE_URL",
            "PAPERFLOW_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        )
    elif kind == "embed":
        api_key = _first_configured(
            "PAPERFLOW_EMBED_API_KEY",
            "PAPERFLOW_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
        base_url = _first_configured(
            "PAPERFLOW_EMBED_BASE_URL",
            "PAPERFLOW_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        )
    else:  # pragma: no cover - Literal callers are statically constrained.
        raise ValueError(f"Unsupported OpenAI-compatible credential kind: {kind!r}")

    return api_key, base_url or None


__all__ = ["resolve_openai_compatible_credentials"]
