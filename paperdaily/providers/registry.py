"""Registry and auto-selection for PaperDaily agent providers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .base import AgentProvider, AgentProviderError
from .claude import ClaudeProvider
from .codex import CodexProvider

DEFAULT_PROVIDER_ORDER = ("codex", "claude")


def _normalize_provider_name(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def create_agent_provider(
    name: str,
    *,
    command: str | None = None,
    environment: Mapping[str, str] | None = None,
    max_budget_usd: float | None = None,
    isolated_home: bool = False,
) -> AgentProvider:
    normalized = _normalize_provider_name(name)
    if normalized == "codex":
        return CodexProvider(
            command=command or "codex",
            environment=environment,
            isolated_home=isolated_home,
        )
    if normalized in {"claude", "claude_code"}:
        return ClaudeProvider(
            command=command or "claude",
            environment=environment,
            max_budget_usd=max_budget_usd,
            isolated_home=isolated_home,
        )
    raise AgentProviderError(
        "registry",
        "unsupported_provider",
        f"Unsupported agent provider: {name}",
    )


def build_agent_provider(
    name: str | None = None,
    *,
    preferred_order: Sequence[str] = DEFAULT_PROVIDER_ORDER,
    commands: Mapping[str, str] | None = None,
    environments: Mapping[str, Mapping[str, str]] | None = None,
    max_budget_usd: float | None = None,
    isolated_home: bool = False,
) -> AgentProvider:
    requested = _normalize_provider_name(
        name or os.environ.get("PAPERDAILY_AGENT_PROVIDER") or "auto"
    )
    command_map = dict(commands or {})
    environment_map = dict(environments or {})

    if requested not in {"", "auto"}:
        return create_agent_provider(
            requested,
            command=command_map.get(requested),
            environment=environment_map.get(requested),
            max_budget_usd=max_budget_usd,
            isolated_home=isolated_home,
        )

    diagnostics: list[dict[str, Any]] = []
    for candidate_name in preferred_order:
        normalized = _normalize_provider_name(candidate_name)
        provider = create_agent_provider(
            normalized,
            command=command_map.get(normalized),
            environment=environment_map.get(normalized),
            max_budget_usd=max_budget_usd,
            isolated_home=isolated_home,
        )
        diagnosis = provider.diagnose()
        diagnostics.append(diagnosis)
        if diagnosis.get("ready", diagnosis.get("available", False)):
            return provider

    raise AgentProviderError(
        "registry",
        "no_available_provider",
        "No ready Codex or Claude CLI provider was found",
        stderr=diagnostics,
    )


def diagnose_agent_providers(
    *,
    commands: Mapping[str, str] | None = None,
    environments: Mapping[str, Mapping[str, str]] | None = None,
    isolated_home: bool = False,
) -> list[dict[str, Any]]:
    command_map = dict(commands or {})
    environment_map = dict(environments or {})
    return [
        create_agent_provider(
            name,
            command=command_map.get(name),
            environment=environment_map.get(name),
            isolated_home=isolated_home,
        ).diagnose()
        for name in DEFAULT_PROVIDER_ORDER
    ]


def list_agent_providers() -> tuple[str, ...]:
    return DEFAULT_PROVIDER_ORDER
