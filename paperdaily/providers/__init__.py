"""Secure CLI Agent providers used by PaperDaily deep reading."""

from .base import AgentProvider, AgentProviderError
from .claude import ClaudeProvider
from .codex import CodexProvider
from .registry import (
    build_agent_provider,
    create_agent_provider,
    diagnose_agent_providers,
    list_agent_providers,
)

__all__ = [
    "AgentProvider",
    "AgentProviderError",
    "ClaudeProvider",
    "CodexProvider",
    "build_agent_provider",
    "create_agent_provider",
    "diagnose_agent_providers",
    "list_agent_providers",
]
