"""Typed YAML configuration for the PaperDaily extension."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .topics import Topic, TopicConfigError, parse_topics


class ConfigError(ValueError):
    """Raised when a PaperDaily configuration file is invalid."""


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _as_string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ConfigError(f"{name} must be a string or list of strings")
    return [str(item).strip() for item in values if str(item or "").strip()]


def _resolve_path(value: Any, *, base_dir: Path | None, default: str) -> Path:
    path = Path(str(value or default)).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False) if base_dir is not None else path


@dataclass
class DailyConfig:
    """Settings for normal daily recommendation runs."""

    # ``0`` means no cap for a one-day digest.  Catch-up windows retain their
    # separate ``catchup.max_papers_per_run`` default.
    default_limit: int = 0
    rerank_limit: int = 30
    llm_rerank_enabled: bool = True
    llm_rerank_weight: float = 0.25
    llm_rerank_max_tokens: int = 3000
    llm_rerank_input_cost_per_million_tokens: float = 0.0
    llm_rerank_output_cost_per_million_tokens: float = 0.0
    summary_language: str = "zh-CN"
    generate_chinese_summary: bool = True
    channels: list[str] = field(default_factory=lambda: ["terminal", "markdown"])
    arxiv_page_size: int = 200
    arxiv_max_results: int = 5000
    arxiv_request_delay_seconds: float = 3.0
    arxiv_cache_ttl_hours: int = 24
    arxiv_rss_enabled: bool = True
    arxiv_rss_include_cross_list: bool = True
    arxiv_rss_cache_ttl_minutes: int = 15
    arxiv_api_id_batch_size: int = 20
    mmr_lambda: float = 0.75

    def __post_init__(self) -> None:
        self.default_limit = int(self.default_limit)
        self.rerank_limit = int(self.rerank_limit)
        self.llm_rerank_enabled = bool(self.llm_rerank_enabled)
        self.llm_rerank_weight = float(self.llm_rerank_weight)
        self.llm_rerank_max_tokens = int(self.llm_rerank_max_tokens)
        self.llm_rerank_input_cost_per_million_tokens = float(
            self.llm_rerank_input_cost_per_million_tokens
        )
        self.llm_rerank_output_cost_per_million_tokens = float(
            self.llm_rerank_output_cost_per_million_tokens
        )
        self.summary_language = str(self.summary_language or "zh-CN").strip()
        self.channels = _as_string_list(self.channels, "daily.channels")
        self.arxiv_page_size = int(self.arxiv_page_size)
        self.arxiv_max_results = int(self.arxiv_max_results)
        self.arxiv_request_delay_seconds = float(self.arxiv_request_delay_seconds)
        self.arxiv_cache_ttl_hours = int(self.arxiv_cache_ttl_hours)
        self.arxiv_rss_enabled = bool(self.arxiv_rss_enabled)
        self.arxiv_rss_include_cross_list = bool(self.arxiv_rss_include_cross_list)
        self.arxiv_rss_cache_ttl_minutes = int(self.arxiv_rss_cache_ttl_minutes)
        self.arxiv_api_id_batch_size = int(self.arxiv_api_id_batch_size)
        self.mmr_lambda = float(self.mmr_lambda)
        if self.default_limit < 0:
            raise ConfigError("daily.default_limit must be zero or positive")
        if self.default_limit > 0 and self.rerank_limit < self.default_limit:
            raise ConfigError("daily.rerank_limit must be greater than or equal to default_limit")
        if not 0.0 <= self.llm_rerank_weight <= 1.0:
            raise ConfigError("daily.llm_rerank_weight must be between 0 and 1")
        if self.llm_rerank_max_tokens <= 0:
            raise ConfigError("daily.llm_rerank_max_tokens must be positive")
        if min(
            self.llm_rerank_input_cost_per_million_tokens,
            self.llm_rerank_output_cost_per_million_tokens,
        ) < 0:
            raise ConfigError("daily.llm rerank token costs cannot be negative")
        if self.arxiv_page_size <= 0 or self.arxiv_max_results <= 0:
            raise ConfigError("daily arXiv page/result limits must be positive")
        if self.arxiv_request_delay_seconds < 0 or self.arxiv_cache_ttl_hours < 0:
            raise ConfigError("daily arXiv delay/cache TTL cannot be negative")
        if self.arxiv_rss_cache_ttl_minutes < 0:
            raise ConfigError("daily arXiv RSS cache TTL cannot be negative")
        if not 1 <= self.arxiv_api_id_batch_size <= 100:
            raise ConfigError("daily arXiv API ID batch size must be between 1 and 100")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ConfigError("daily.mmr_lambda must be between 0 and 1")

    @property
    def limit(self) -> int:
        """Compatibility alias used by small CLI call sites."""

        return self.default_limit

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DailyConfig:
        raw = _as_mapping(data, "daily")
        return cls(
            default_limit=raw.get("default_limit", raw.get("limit", 0)),
            rerank_limit=raw.get("rerank_limit", 30),
            llm_rerank_enabled=raw.get(
                "llm_rerank_enabled",
                raw.get("enable_llm_rerank", True),
            ),
            llm_rerank_weight=raw.get("llm_rerank_weight", raw.get("rerank_weight", 0.25)),
            llm_rerank_max_tokens=raw.get("llm_rerank_max_tokens", 3000),
            llm_rerank_input_cost_per_million_tokens=raw.get(
                "llm_rerank_input_cost_per_million_tokens", 0.0
            ),
            llm_rerank_output_cost_per_million_tokens=raw.get(
                "llm_rerank_output_cost_per_million_tokens", 0.0
            ),
            summary_language=raw.get("summary_language", raw.get("language", "zh-CN")),
            generate_chinese_summary=bool(raw.get("generate_chinese_summary", True)),
            channels=raw.get("channels", ["terminal", "markdown"]),
            arxiv_page_size=raw.get("arxiv_page_size", 200),
            arxiv_max_results=raw.get("arxiv_max_results", 5000),
            arxiv_request_delay_seconds=raw.get("arxiv_request_delay_seconds", 3.0),
            arxiv_cache_ttl_hours=raw.get("arxiv_cache_ttl_hours", 24),
            arxiv_rss_enabled=bool(raw.get("arxiv_rss_enabled", True)),
            arxiv_rss_include_cross_list=bool(raw.get("arxiv_rss_include_cross_list", True)),
            arxiv_rss_cache_ttl_minutes=raw.get("arxiv_rss_cache_ttl_minutes", 15),
            arxiv_api_id_batch_size=raw.get("arxiv_api_id_batch_size", 20),
            mmr_lambda=raw.get("mmr_lambda", 0.75),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_limit": self.default_limit,
            "rerank_limit": self.rerank_limit,
            "llm_rerank_enabled": self.llm_rerank_enabled,
            "llm_rerank_weight": self.llm_rerank_weight,
            "llm_rerank_max_tokens": self.llm_rerank_max_tokens,
            "llm_rerank_input_cost_per_million_tokens": self.llm_rerank_input_cost_per_million_tokens,
            "llm_rerank_output_cost_per_million_tokens": self.llm_rerank_output_cost_per_million_tokens,
            "summary_language": self.summary_language,
            "generate_chinese_summary": self.generate_chinese_summary,
            "channels": list(self.channels),
            "arxiv_page_size": self.arxiv_page_size,
            "arxiv_max_results": self.arxiv_max_results,
            "arxiv_request_delay_seconds": self.arxiv_request_delay_seconds,
            "arxiv_cache_ttl_hours": self.arxiv_cache_ttl_hours,
            "arxiv_rss_enabled": self.arxiv_rss_enabled,
            "arxiv_rss_include_cross_list": self.arxiv_rss_include_cross_list,
            "arxiv_rss_cache_ttl_minutes": self.arxiv_rss_cache_ttl_minutes,
            "arxiv_api_id_batch_size": self.arxiv_api_id_batch_size,
            "mmr_lambda": self.mmr_lambda,
        }


@dataclass
class CatchupConfig:
    """Policy knobs for missed-run planning."""

    interactive: bool = True
    auto_catchup_days: int = 2
    first_run_days: int = 7
    default_window_days: int = 7
    max_window_days: int = 30
    max_papers_per_run: int = 30
    overflow_mode: str = "curated"

    def __post_init__(self) -> None:
        self.auto_catchup_days = int(self.auto_catchup_days)
        self.first_run_days = int(self.first_run_days)
        self.default_window_days = int(self.default_window_days)
        self.max_window_days = int(self.max_window_days)
        self.max_papers_per_run = int(self.max_papers_per_run)
        self.overflow_mode = str(self.overflow_mode or "curated").strip().lower()
        if not 0 <= self.auto_catchup_days <= 2:
            raise ConfigError("catchup.auto_catchup_days must be between 0 and 2")
        if min(self.first_run_days, self.default_window_days, self.max_window_days) <= 0:
            raise ConfigError("catchup window sizes must be positive")
        if self.default_window_days > self.max_window_days:
            raise ConfigError("catchup.default_window_days cannot exceed max_window_days")
        if self.max_papers_per_run <= 0:
            raise ConfigError("catchup.max_papers_per_run must be positive")
        if self.overflow_mode not in {"curated", "weekly", "weekly_digest", "top", "all"}:
            raise ConfigError("catchup.overflow_mode is not supported")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> CatchupConfig:
        raw = _as_mapping(data, "catchup")
        overflow = raw.get("overflow", {})
        overflow_mapping = overflow if isinstance(overflow, Mapping) else {}
        return cls(
            interactive=bool(raw.get("interactive", True)),
            auto_catchup_days=raw.get("auto_catchup_days", 2),
            first_run_days=raw.get("first_run_days", 7),
            default_window_days=raw.get("default_window_days", 7),
            max_window_days=raw.get("max_window_days", raw.get("maximum_recommended_days", 30)),
            max_papers_per_run=raw.get(
                "max_papers_per_run",
                overflow_mapping.get("max_papers", 30),
            ),
            overflow_mode=raw.get("overflow_mode", overflow_mapping.get("mode", "curated")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interactive": self.interactive,
            "auto_catchup_days": self.auto_catchup_days,
            "first_run_days": self.first_run_days,
            "default_window_days": self.default_window_days,
            "max_window_days": self.max_window_days,
            "max_papers_per_run": self.max_papers_per_run,
            "overflow_mode": self.overflow_mode,
        }


@dataclass
class ProvidersConfig:
    """Agent-provider selection plus provider-specific options."""

    default_provider: str = "auto"
    fallback_order: list[str] = field(default_factory=lambda: ["codex", "claude"])
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_provider = str(self.default_provider or "auto").strip().lower()
        self.fallback_order = _as_string_list(self.fallback_order, "providers.fallback_order")
        normalized: dict[str, dict[str, Any]] = {}
        for name, options in dict(self.settings or {}).items():
            if options is None:
                normalized[str(name)] = {}
            elif isinstance(options, Mapping):
                normalized[str(name)] = dict(options)
            else:
                raise ConfigError(f"providers.{name} must be a mapping")
        self.settings = normalized

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ProvidersConfig:
        raw = dict(_as_mapping(data, "providers"))
        default_provider = raw.pop("default_provider", raw.pop("default", "auto"))
        fallback_order = raw.pop("fallback_order", ["codex", "claude"])
        nested_settings = raw.pop("settings", {})
        if nested_settings and not isinstance(nested_settings, Mapping):
            raise ConfigError("providers.settings must be a mapping")
        settings = dict(nested_settings or {})
        settings.update(raw)
        return cls(
            default_provider=default_provider,
            fallback_order=fallback_order,
            settings=settings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_provider": self.default_provider,
            "fallback_order": list(self.fallback_order),
            **{name: dict(options) for name, options in self.settings.items()},
        }


@dataclass
class DeepReadConfig:
    """Security options for full-paper agent runs.

    ``isolated_home`` is deliberately opt-in.  The default preserves existing
    Codex ChatGPT/CODEX_HOME login behaviour; callers that enable it must use
    a direct provider API key because no user home or provider config is made
    available to the child process.
    """

    isolated_home: bool = False
    max_pdf_pages: int = 100
    max_extracted_text_chars: int = 500_000

    def __post_init__(self) -> None:
        if not isinstance(self.isolated_home, bool):
            raise ConfigError("deep_read.isolated_home must be a boolean")
        for attribute in ("max_pdf_pages", "max_extracted_text_chars"):
            value = getattr(self, attribute)
            if isinstance(value, bool):
                raise ConfigError(f"deep_read.{attribute} must be a positive integer")
            try:
                normalized = int(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"deep_read.{attribute} must be a positive integer") from exc
            if normalized <= 0:
                raise ConfigError(f"deep_read.{attribute} must be a positive integer")
            setattr(self, attribute, normalized)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DeepReadConfig:
        raw = _as_mapping(data, "deep_read")
        value = raw.get("isolated_home", False)
        if not isinstance(value, bool):
            raise ConfigError("deep_read.isolated_home must be a boolean")
        return cls(
            isolated_home=value,
            max_pdf_pages=raw.get("max_pdf_pages", 100),
            max_extracted_text_chars=raw.get("max_extracted_text_chars", 500_000),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "isolated_home": self.isolated_home,
            "max_pdf_pages": self.max_pdf_pages,
            "max_extracted_text_chars": self.max_extracted_text_chars,
        }


@dataclass
class PaperDailyConfig:
    """Complete application configuration."""

    user_id: str = "default"
    timezone: str = "Asia/Shanghai"
    output_dir: Path = Path("data/output")
    database: Path = Path("data/paperflow.db")
    daily: DailyConfig = field(default_factory=DailyConfig)
    catchup: CatchupConfig = field(default_factory=CatchupConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    deep_read: DeepReadConfig = field(default_factory=DeepReadConfig)
    topics: list[Topic] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.user_id = str(self.user_id or "").strip()
        self.timezone = str(self.timezone or "").strip()
        self.output_dir = Path(self.output_dir)
        self.database = Path(self.database)
        if not self.user_id:
            raise ConfigError("user_id is required")
        if not self.timezone:
            raise ConfigError("timezone is required")

        topic_ids: set[str] = set()
        for topic in self.topics:
            if topic.id in topic_ids:
                raise ConfigError(f"duplicate topic id: {topic.id}")
            topic_ids.add(topic.id)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        base_dir: Path | None = None,
    ) -> PaperDailyConfig:
        raw = _as_mapping(data, "config")
        user = _as_mapping(raw.get("user"), "user")
        runtime = _as_mapping(raw.get("runtime"), "runtime")
        try:
            topics = parse_topics(raw.get("topics", []))
        except TopicConfigError as exc:
            raise ConfigError(str(exc)) from exc

        return cls(
            user_id=raw.get("user_id", user.get("id", user.get("user_id", "default"))),
            timezone=raw.get("timezone", user.get("timezone", "Asia/Shanghai")),
            output_dir=_resolve_path(
                raw.get("output_dir", runtime.get("output_dir")),
                base_dir=base_dir,
                default="data/output",
            ),
            database=_resolve_path(
                raw.get("database", runtime.get("database")),
                base_dir=base_dir,
                default="data/paperflow.db",
            ),
            daily=DailyConfig.from_dict(raw.get("daily")),
            catchup=CatchupConfig.from_dict(raw.get("catchup")),
            providers=ProvidersConfig.from_dict(raw.get("providers")),
            deep_read=DeepReadConfig.from_dict(raw.get("deep_read")),
            topics=topics,
        )

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        def display(path: Path) -> str:
            if relative_to is not None:
                try:
                    return str(path.relative_to(relative_to))
                except ValueError:
                    pass
            return str(path)

        return {
            "user_id": self.user_id,
            "timezone": self.timezone,
            "output_dir": display(self.output_dir),
            "database": display(self.database),
            "daily": self.daily.to_dict(),
            "catchup": self.catchup.to_dict(),
            "topics": [topic.to_dict() for topic in self.topics],
        }


# Short alias for callers that prefer ``Config``.
Config = PaperDailyConfig


def load_config(path: str | Path) -> PaperDailyConfig:
    """Load and validate a UTF-8 YAML configuration file."""

    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return PaperDailyConfig.from_dict(payload, base_dir=config_path.parent)


def save_config(config: PaperDailyConfig, path: str | Path) -> Path:
    """Serialize a validated configuration as UTF-8 YAML."""

    config_path = Path(path).expanduser().resolve(strict=False)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict(relative_to=config_path.parent)
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


__all__ = [
    "CatchupConfig",
    "Config",
    "ConfigError",
    "DailyConfig",
    "DeepReadConfig",
    "PaperDailyConfig",
    "ProvidersConfig",
    "load_config",
    "save_config",
]
