"""Shared contracts and safe subprocess helpers for agent-backed deep reading."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_AGENT_TIMEOUT_SECONDS = 30 * 60
_MAX_ERROR_OUTPUT_CHARS = 4_000
_PROCESS_TREE_KILL_TIMEOUT_SECONDS = 10
_PIPE_DRAIN_TIMEOUT_SECONDS = 3

_COMMON_ENVIRONMENT_KEYS = {
    "ALLUSERSPROFILE",
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PROGRAMDATA",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERNAME",
    "USERPROFILE",
    "WINDIR",
}


def _limited_text(value: Any) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    if len(text) <= _MAX_ERROR_OUTPUT_CHARS:
        return text
    return text[:_MAX_ERROR_OUTPUT_CHARS] + "\n...[truncated]"


def environment_value(
    key: str,
    *,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Read an environment value case-insensitively.

    Windows treats variable names case-insensitively, while ``os.environ`` and
    test-provided mappings do not always preserve one spelling.  Provider
    authentication checks must agree with the child environment builder.
    """

    wanted = str(key).upper()
    for source in (overrides or {}, os.environ):
        for candidate, value in source.items():
            if str(candidate).upper() == wanted:
                return str(value or "")
    return ""


def _redact_subprocess_output(
    value: Any,
    *,
    prompt: str,
    environment: Mapping[str, str],
) -> str:
    """Remove stdin and credential values before subprocess output reaches logs."""

    text = _limited_text(value)
    sensitive_values = [str(prompt or "")]
    secret_markers = ("API_KEY", "AUTH_TOKEN", "PASSWORD", "SECRET")
    sensitive_values.extend(
        str(item)
        for key, item in environment.items()
        if any(marker in key.upper() for marker in secret_markers)
    )
    for sensitive in sorted({item for item in sensitive_values if item}, key=len, reverse=True):
        text = text.replace(sensitive, "[REDACTED]")
    return text


class AgentProviderError(RuntimeError):
    """A machine-readable provider failure that avoids embedding the prompt."""

    def __init__(
        self,
        provider: str,
        code: str,
        message: str,
        *,
        command: Sequence[str] | None = None,
        returncode: int | None = None,
        stdout: Any = None,
        stderr: Any = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.message = message
        self.command = tuple(str(part) for part in (command or ()))
        self.returncode = returncode
        self.stdout = _limited_text(stdout)
        self.stderr = _limited_text(stderr)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "code": self.code,
            "message": self.message,
        }
        if self.command:
            payload["command"] = list(self.command)
        if self.returncode is not None:
            payload["returncode"] = self.returncode
        if self.stdout:
            payload["stdout"] = self.stdout
        if self.stderr:
            payload["stderr"] = self.stderr
        return payload


class AgentProvider(ABC):
    """Interface for a CLI agent that returns schema-constrained JSON."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider executable can be resolved."""

    @abstractmethod
    def diagnose(self) -> dict[str, Any]:
        """Return non-mutating installation/authentication diagnostics."""

    @abstractmethod
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
        """Run the provider and persist/return one JSON object."""


def resolve_executable(command: str) -> str | None:
    """Resolve commands portably, including npm-generated ``.CMD`` shims."""

    raw = str(command or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(raw)


def build_subprocess_environment(
    *,
    provider_keys: Sequence[str] = (),
    extra_allowed_keys: Sequence[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an allowlisted child environment so unrelated secrets stay hidden."""

    # Windows environment names are case-insensitive even though ``dict`` is
    # not. Merge overrides by canonical name as well as comparing the
    # allowlist case-insensitively, otherwise ``systemroot`` and ``SystemRoot``
    # can reach CreateProcess as ambiguous duplicate entries.
    merged: dict[str, tuple[str, str]] = {}
    for key, value in os.environ.items():
        merged[str(key).upper()] = (str(key), str(value))
    for key, value in (overrides or {}).items():
        merged[str(key).upper()] = (str(key), str(value))

    allowed = {
        str(key).upper()
        for key in (
            _COMMON_ENVIRONMENT_KEYS
            | {str(key) for key in provider_keys}
            | {str(key) for key in extra_allowed_keys}
        )
    }
    return {
        original_key: value
        for canonical_key, (original_key, value) in merged.items()
        if canonical_key in allowed and value != ""
    }


@contextlib.contextmanager
def isolated_agent_home(
    workspace: Path,
    *,
    provider: str,
) -> Iterator[dict[str, str]]:
    """Provide an empty, temporary home/config tree for an API-key-only run.

    This deliberately *does not* copy the caller's home, Codex/Claude config,
    or user temp directories.  The tree is made below the paper workspace so
    the child only receives paths owned by this individual deep-read job; it
    is removed when the subprocess returns or raises.  Authentication is
    intentionally not written here: callers must pass a direct API key in the
    child environment.

    It is credential/config isolation, not an OS container.  The providers
    still keep their read-only tool/sandbox restrictions separately.
    """

    resolved_workspace = Path(workspace).expanduser().resolve(strict=False)
    if not resolved_workspace.is_dir():
        raise AgentProviderError(
            provider,
            "isolation_setup_failed",
            f"Cannot create an isolated home outside a missing workspace: {resolved_workspace}",
        )

    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix=".paperdaily-agent-home-",
            dir=str(resolved_workspace),
        )
    except OSError as exc:
        raise AgentProviderError(
            provider,
            "isolation_setup_failed",
            "Unable to create the temporary isolated provider home",
            stderr=exc,
        ) from exc

    cleanup_error: OSError | None = None
    try:
        root = Path(temporary_directory.name).resolve(strict=False)
        try:
            root.relative_to(resolved_workspace)
        except ValueError as exc:  # Defensive: ``dir=workspace`` should ensure this.
            raise AgentProviderError(
                provider,
                "isolation_setup_failed",
                "Temporary isolated provider home escaped the paper workspace",
                stderr=exc,
            ) from exc

        locations = {
            "home": root / "home",
            "appdata": root / "appdata",
            "localappdata": root / "localappdata",
            "codex": root / "codex",
            "claude": root / "claude",
            "tmp": root / "tmp",
            "xdg_config": root / "xdg-config",
            "xdg_cache": root / "xdg-cache",
            "xdg_data": root / "xdg-data",
            "xdg_runtime": root / "xdg-runtime",
        }
        try:
            for path in locations.values():
                path.mkdir(mode=0o700, parents=True, exist_ok=False)
                with contextlib.suppress(OSError):
                    path.chmod(0o700)
        except OSError as exc:
            raise AgentProviderError(
                provider,
                "isolation_setup_failed",
                "Unable to prepare empty isolated provider directories",
                stderr=exc,
            ) from exc

        home = locations["home"]
        drive, home_path = os.path.splitdrive(str(home))
        overrides = {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(locations["appdata"]),
            "LOCALAPPDATA": str(locations["localappdata"]),
            "CODEX_HOME": str(locations["codex"]),
            "CLAUDE_CONFIG_DIR": str(locations["claude"]),
            "TEMP": str(locations["tmp"]),
            "TMP": str(locations["tmp"]),
            "TMPDIR": str(locations["tmp"]),
            "XDG_CONFIG_HOME": str(locations["xdg_config"]),
            "XDG_CACHE_HOME": str(locations["xdg_cache"]),
            "XDG_DATA_HOME": str(locations["xdg_data"]),
            "XDG_RUNTIME_DIR": str(locations["xdg_runtime"]),
            "NPM_CONFIG_USERCONFIG": str(root / "npmrc"),
        }
        # A few Windows CLIs reconstruct a home using HOMEDRIVE + HOMEPATH.
        # Give them the temporary home rather than inheriting the real one.
        if drive:
            overrides["HOMEDRIVE"] = drive
            overrides["HOMEPATH"] = home_path or "\\"
        else:
            overrides["HOMEDRIVE"] = ""
            overrides["HOMEPATH"] = ""
        yield overrides
    finally:
        try:
            temporary_directory.cleanup()
        except OSError as exc:  # pragma: no cover - platform/file-lock dependent.
            cleanup_error = exc
        # Do not mask the primary provider failure.  On a successful run,
        # surface cleanup trouble so callers do not assume isolation artifacts
        # were definitely removed.
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise AgentProviderError(
                provider,
                "isolation_cleanup_failed",
                "The temporary isolated provider home could not be removed",
                stderr=cleanup_error,
            ) from cleanup_error


def prepare_run_paths(
    provider: str,
    workspace: Path,
    schema_file: Path,
    output_file: Path,
) -> tuple[Path, Path, Path]:
    """Validate immutable inputs and prepare a fresh output location."""

    resolved_workspace = Path(workspace).expanduser().resolve()
    resolved_schema = Path(schema_file).expanduser().resolve()
    resolved_output = Path(output_file).expanduser().resolve()

    if not resolved_workspace.is_dir():
        raise AgentProviderError(
            provider,
            "invalid_workspace",
            f"Workspace directory does not exist: {resolved_workspace}",
        )
    if not resolved_schema.is_file():
        raise AgentProviderError(
            provider,
            "invalid_schema",
            f"JSON schema file does not exist: {resolved_schema}",
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_output.unlink(missing_ok=True)
    except OSError as exc:
        raise AgentProviderError(
            provider,
            "output_unavailable",
            f"Unable to prepare output file: {resolved_output}",
            stderr=exc,
        ) from exc
    return resolved_workspace, resolved_schema, resolved_output


def load_json_schema(provider: str, schema_file: Path) -> dict[str, Any]:
    try:
        value = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentProviderError(
            provider,
            "invalid_schema",
            f"Unable to read JSON schema: {schema_file}",
            stderr=exc,
        ) from exc
    if not isinstance(value, dict):
        raise AgentProviderError(provider, "invalid_schema", "The JSON schema root must be an object")
    return value


def load_prompt(
    provider: str,
    *,
    prompt: str | None,
    prompt_file: Path | None,
) -> str:
    if prompt is not None and prompt_file is not None:
        raise AgentProviderError(
            provider,
            "invalid_prompt",
            "Provide prompt or prompt_file, not both",
        )
    if prompt_file is not None:
        path = Path(prompt_file).expanduser().resolve()
        if not path.is_file():
            raise AgentProviderError(
                provider,
                "invalid_prompt",
                f"Prompt file does not exist: {path}",
            )
        try:
            value = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AgentProviderError(
                provider,
                "invalid_prompt",
                f"Unable to read prompt file: {path}",
                stderr=exc,
            ) from exc
    else:
        value = str(prompt or "")
    if not value.strip():
        raise AgentProviderError(provider, "invalid_prompt", "Prompt must not be empty")
    return value


def parse_json_object(provider: str, text: str, *, source: str) -> dict[str, Any]:
    normalized = str(text or "").strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise AgentProviderError(
            provider,
            "invalid_json",
            f"{source} did not contain valid JSON",
            stdout=normalized,
            stderr=exc,
        ) from exc
    if not isinstance(value, dict):
        raise AgentProviderError(
            provider,
            "invalid_json",
            f"{source} must contain a JSON object",
            stdout=normalized,
        )
    return value


def validate_required_fields(
    provider: str,
    result: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    required = schema.get("required")
    if not isinstance(required, list):
        return
    missing = [str(key) for key in required if str(key) not in result]
    if missing:
        raise AgentProviderError(
            provider,
            "schema_validation_failed",
            f"Structured output is missing required fields: {', '.join(missing)}",
        )


def write_json_output(provider: str, output_file: Path, result: Mapping[str, Any]) -> None:
    try:
        output_file.write_text(
            json.dumps(dict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AgentProviderError(
            provider,
            "output_unavailable",
            f"Unable to write structured output: {output_file}",
            stderr=exc,
        ) from exc


def _platform_is_windows() -> bool:
    return os.name == "nt"


def _process_is_running(process: subprocess.Popen[str]) -> bool:
    try:
        return process.poll() is None
    except OSError:
        return True


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    environment: Mapping[str, str],
) -> None:
    """Force-stop a provider and every descendant without exposing its stdin."""

    if not _process_is_running(process):
        return

    if _platform_is_windows():
        # npm ``.CMD`` shims start node.exe as a child. Killing only cmd.exe
        # leaves node holding stdout/stderr pipe handles, which can make a
        # subsequent communicate() wait forever. taskkill /T closes the full
        # tree; all output is discarded and only the numeric PID is passed.
        kill_environment = {
            key: value
            for key, value in environment.items()
            if key.upper() in {item.upper() for item in _COMMON_ENVIRONMENT_KEYS}
        }
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=kill_environment,
                timeout=_PROCESS_TREE_KILL_TIMEOUT_SECONDS,
                check=False,
                shell=False,
            )
    else:
        # Popen starts a new session below, so the provider PID is also the
        # process-group ID. SIGKILL prevents grandchildren from retaining the
        # captured pipes after a timeout.
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)

    if _process_is_running(process):
        with contextlib.suppress(OSError):
            process.kill()


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            stream.close()


def _drain_terminated_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Collect already-buffered output without allowing inherited pipes to hang."""

    try:
        stdout, stderr = process.communicate(timeout=_PIPE_DRAIN_TIMEOUT_SECONDS)
        return str(stdout or ""), str(stderr or "")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        _close_process_pipes(process)
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=1)
        return "", ""


def run_subprocess(
    *,
    provider: str,
    command: Sequence[str],
    display_command: Sequence[str] | None,
    prompt: str,
    workspace: Path,
    timeout_seconds: int,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    if int(timeout_seconds) <= 0:
        raise AgentProviderError(provider, "invalid_timeout", "timeout_seconds must be positive")
    safe_command = tuple(display_command or command)
    command_args = [str(part) for part in command]
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": str(workspace),
        "env": dict(environment),
        "encoding": "utf-8",
        "errors": "replace",
        "shell": False,
    }
    if _platform_is_windows():
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(command_args, **popen_kwargs)
    except OSError as exc:
        raise AgentProviderError(
            provider,
            "launch_failed",
            f"Unable to launch provider command: {command[0]}",
            command=safe_command,
            stderr=exc,
        ) from exc

    try:
        stdout, stderr = process.communicate(input=str(prompt), timeout=int(timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process, environment=environment)
        drained_stdout, drained_stderr = _drain_terminated_process(process)
        timeout_stdout = exc.stdout if exc.stdout is not None else drained_stdout
        timeout_stderr = exc.stderr if exc.stderr is not None else drained_stderr
        raise AgentProviderError(
            provider,
            "timeout",
            f"Provider exceeded the {timeout_seconds}s timeout",
            command=safe_command,
            stdout=_redact_subprocess_output(
                timeout_stdout,
                prompt=prompt,
                environment=environment,
            ),
            stderr=_redact_subprocess_output(
                timeout_stderr,
                prompt=prompt,
                environment=environment,
            ),
        ) from exc

    returncode = process.returncode
    if returncode is None:
        returncode = process.wait()
    returncode = int(returncode)
    completed = subprocess.CompletedProcess(command_args, returncode, stdout=stdout, stderr=stderr)
    if returncode != 0:
        raise AgentProviderError(
            provider,
            "process_failed",
            f"Provider exited with status {returncode}",
            command=safe_command,
            returncode=returncode,
            stdout=_redact_subprocess_output(stdout, prompt=prompt, environment=environment),
            stderr=_redact_subprocess_output(stderr, prompt=prompt, environment=environment),
        )
    return completed
