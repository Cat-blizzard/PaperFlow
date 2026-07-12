"""Codex CLI adapter for schema-constrained, read-only deep reading."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .base import (
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    AgentProvider,
    AgentProviderError,
    build_subprocess_environment,
    environment_value,
    isolated_agent_home,
    load_json_schema,
    load_prompt,
    parse_json_object,
    prepare_run_paths,
    resolve_executable,
    run_subprocess,
    validate_required_fields,
)


class CodexProvider(AgentProvider):
    name = "codex"

    def __init__(
        self,
        command: str = "codex",
        *,
        environment: Mapping[str, str] | None = None,
        isolated_home: bool = False,
    ) -> None:
        self.command = command
        self._environment = dict(environment or {})
        self.isolated_home = bool(isolated_home)

    def _resolved_command(self) -> str | None:
        return resolve_executable(self.command)

    def is_available(self) -> bool:
        return self._resolved_command() is not None

    def _api_key(self) -> str:
        return environment_value("OPENAI_API_KEY", overrides=self._environment)

    def diagnose(self) -> dict[str, Any]:
        resolved = self._resolved_command()
        has_api_key = bool(self._api_key())
        if self.isolated_home:
            auth_mode = "openai_api_key_isolated" if has_api_key else "missing_for_isolated_mode"
            ready = resolved is not None and has_api_key
            home_isolation = "api_key_only"
        else:
            auth_mode = "environment" if has_api_key else "codex_home"
            ready = resolved is not None
            home_isolation = "disabled_shared_home"
        return {
            "provider": self.name,
            "available": resolved is not None,
            "ready": ready,
            "command": resolved,
            "auth_mode": auth_mode,
            "home_isolation": home_isolation,
            "sandbox": "read-only",
            "ephemeral": True,
        }

    def run_structured(
        self,
        *,
        workspace: Path,
        schema_file: Path,
        output_file: Path,
        prompt: str | None = None,
        prompt_file: Path | None = None,
        timeout_seconds: int = DEFAULT_AGENT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        resolved_workspace, resolved_schema, resolved_output = prepare_run_paths(
            self.name,
            workspace,
            schema_file,
            output_file,
        )
        schema = load_json_schema(self.name, resolved_schema)
        prompt_text = load_prompt(self.name, prompt=prompt, prompt_file=prompt_file)
        executable = self._resolved_command()
        if executable is None:
            raise AgentProviderError(
                self.name,
                "command_not_found",
                f"Codex CLI command was not found: {self.command}",
            )
        if self.isolated_home and not self._api_key():
            raise AgentProviderError(
                self.name,
                "authentication_missing",
                "Codex isolated-home mode requires OPENAI_API_KEY; it cannot use ChatGPT/CODEX_HOME login",
            )

        command = [
            executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(resolved_schema),
            "--output-last-message",
            str(resolved_output),
            "-C",
            str(resolved_workspace),
            "-",
        ]
        isolation_context = (
            isolated_agent_home(resolved_workspace, provider=self.name)
            if self.isolated_home
            else nullcontext({})
        )
        with isolation_context as isolation_overrides:
            overrides = {**self._environment, **isolation_overrides}
            completed = run_subprocess(
                provider=self.name,
                command=command,
                display_command=command,
                prompt=prompt_text,
                workspace=resolved_workspace,
                timeout_seconds=timeout_seconds,
                environment=build_subprocess_environment(
                    provider_keys=("OPENAI_API_KEY",),
                    extra_allowed_keys=tuple(isolation_overrides),
                    overrides=overrides,
                ),
            )

        if not resolved_output.is_file():
            raise AgentProviderError(
                self.name,
                "missing_output",
                "Codex completed without writing --output-last-message",
                command=command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            output_text = resolved_output.read_text(encoding="utf-8")
        except OSError as exc:
            raise AgentProviderError(
                self.name,
                "output_unavailable",
                f"Unable to read Codex output: {resolved_output}",
                command=command,
                stderr=exc,
            ) from exc
        result = parse_json_object(self.name, output_text, source="Codex output")
        validate_required_fields(self.name, result, schema)
        return result
