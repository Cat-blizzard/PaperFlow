"""Claude Code CLI adapter for schema-constrained, read-only deep reading."""

from __future__ import annotations

import json
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
    write_json_output,
)


class ClaudeProvider(AgentProvider):
    name = "claude"

    def __init__(
        self,
        command: str = "claude",
        *,
        environment: Mapping[str, str] | None = None,
        max_budget_usd: float | None = None,
        isolated_home: bool = False,
    ) -> None:
        self.command = command
        self._environment = dict(environment or {})
        self.max_budget_usd = max_budget_usd
        self.isolated_home = bool(isolated_home)

    def _resolved_command(self) -> str | None:
        return resolve_executable(self.command)

    def _api_key(self) -> str:
        return environment_value("ANTHROPIC_API_KEY", overrides=self._environment)

    def is_available(self) -> bool:
        return self._resolved_command() is not None

    def diagnose(self) -> dict[str, Any]:
        resolved = self._resolved_command()
        has_api_key = bool(self._api_key())
        return {
            "provider": self.name,
            "available": resolved is not None,
            "ready": resolved is not None and has_api_key,
            "command": resolved,
            "auth_mode": (
                "anthropic_api_key_isolated"
                if has_api_key and self.isolated_home
                else "anthropic_api_key"
                if has_api_key
                else "missing_for_bare_mode"
            ),
            "home_isolation": "api_key_only" if self.isolated_home else "disabled_shared_home",
            "bare": True,
            "allowed_tools": ["Read"],
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
                f"Claude CLI command was not found: {self.command}",
            )
        if not self._api_key():
            raise AgentProviderError(
                self.name,
                "authentication_missing",
                "Claude --bare mode requires ANTHROPIC_API_KEY",
            )

        compact_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        command = [
            executable,
            "--bare",
            "--disable-slash-commands",
            "-p",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Read",
            "--allowedTools",
            "Read",
            "--output-format",
            "json",
            "--json-schema",
            compact_schema,
            "--no-session-persistence",
        ]
        if self.max_budget_usd is not None:
            if self.max_budget_usd <= 0:
                raise AgentProviderError(
                    self.name,
                    "invalid_budget",
                    "max_budget_usd must be positive",
                )
            command.extend(["--max-budget-usd", str(self.max_budget_usd)])

        display_command = ["<json-schema>" if item == compact_schema else item for item in command]
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
                display_command=display_command,
                prompt=prompt_text,
                workspace=resolved_workspace,
                timeout_seconds=timeout_seconds,
                environment=build_subprocess_environment(
                    provider_keys=("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
                    extra_allowed_keys=tuple(isolation_overrides),
                    overrides=overrides,
                ),
            )

        response = parse_json_object(self.name, completed.stdout, source="Claude process output")
        if response.get("is_error") is True:
            raise AgentProviderError(
                self.name,
                "model_error",
                str(response.get("result") or response.get("error") or "Claude returned an error result"),
                command=display_command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        structured: Any = response.get("structured_output")
        if structured is None:
            result_value = response.get("result")
            if isinstance(result_value, str) and result_value.strip():
                structured = parse_json_object(self.name, result_value, source="Claude result")
            elif not any(key in response for key in ("type", "subtype", "result", "is_error")):
                structured = response
        if not isinstance(structured, dict):
            raise AgentProviderError(
                self.name,
                "missing_output",
                "Claude response did not contain structured_output",
                command=display_command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        validate_required_fields(self.name, structured, schema)
        write_json_output(self.name, resolved_output, structured)
        return dict(structured)
