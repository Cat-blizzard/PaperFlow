"""Local GUI adapters for the PaperDaily workflow.

The desktop server deliberately keeps PaperDaily separate from the legacy
PaperFlow daily-push pipeline.  It reads the configured PaperDaily user and
SQLite state directly, while long-running arXiv and Codex work runs in daemon
threads so a browser request never owns a model or network operation.
"""

from __future__ import annotations

from copy import deepcopy
import re
import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from paperdaily.cli import DEFAULT_CONFIG_PATH
from paperdaily.channels import render_digest_markdown
from paperdaily.config import PaperDailyConfig, load_config, save_config
from paperdaily.deep_read import DeepReadService
from paperdaily.identifiers import canonicalize_arxiv_id
from paperdaily.models import Digest, Recommendation
from paperdaily.providers import build_agent_provider, diagnose_agent_providers
from paperdaily.service import PaperDailyService, RunOutcome
from paperdaily.storage import PaperDailyStore
from paperdaily.summaries import ChineseSummaryService
from paperdaily.topics import Topic
from paperflow.providers import build_embedding_provider, build_llm_provider

MAX_NOTE_CHARS = 180_000
MAX_TASK_ERROR_CHARS = 1_600
USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEFAULT_TOPIC_CATEGORIES = ["cs.RO", "cs.AI", "cs.CV", "cs.LG", "cs.CL"]
DEFAULT_ACRONYM_CONTEXT = ["robot", "robotic", "manipulation", "embodied", "action", "policy"]
ACRONYM_EXACT_PHRASES = {
    "vla": ["vision-language-action", "vision language action"],
    "wam": ["world-action model", "world action model"],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_options(config: PaperDailyConfig) -> tuple[dict[str, str], float | None]:
    commands: dict[str, str] = {}
    max_budget: float | None = None
    for name in ("codex", "claude"):
        settings = config.providers.settings.get(name, {})
        if settings.get("command"):
            commands[name] = str(settings["command"])
        if name == "claude" and settings.get("max_budget_usd") is not None:
            max_budget = float(settings["max_budget_usd"])
    return commands, max_budget


def _paper_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded paper fields that the browser actually renders."""

    paper = dict(record.get("paper") or {})
    metadata = dict(record.get("metadata") or {})
    canonical = canonicalize_arxiv_id(
        str(record.get("canonical_id") or paper.get("arxiv_id") or "")
    )
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    return {
        "arxiv_id": canonical or str(paper.get("arxiv_id") or ""),
        "rank": record.get("rank"),
        "score": record.get("score"),
        "title": str(paper.get("title") or ""),
        "authors": list(paper.get("authors") or [])[:12],
        "categories": list(paper.get("categories") or [])[:12],
        "published": str(paper.get("published") or ""),
        "abstract": str(paper.get("abstract") or "")[:8_000],
        "url": f"https://arxiv.org/abs/{canonical}" if canonical else "",
        "pdf_url": f"https://arxiv.org/pdf/{canonical}" if canonical else "",
        "matched_topics": list(metadata.get("matched_topics") or []),
        "matched_terms": list(metadata.get("matched_terms") or [])[:16],
        "recommendation_reason": str(metadata.get("recommendation_reason") or ""),
        "summary": {
            "title_zh": str(summary.get("title_zh") or ""),
            "one_sentence_summary": str(summary.get("one_sentence_summary") or ""),
            "method": str(summary.get("method") or ""),
            "contributions": list(summary.get("contributions") or [])[:4],
            "status": str(summary.get("status") or ""),
        },
    }


def _outcome_payload(outcome: RunOutcome) -> dict[str, Any]:
    digest = outcome.digest
    return {
        "dry_run": outcome.dry_run,
        "run_id": digest.run_id,
        "window_start": digest.window_start.isoformat(),
        "window_end": digest.window_end.isoformat(),
        "choice": digest.catchup_mode,
        "stats": dict(digest.stats),
        "warnings": list(outcome.warnings),
        "output_path": str(digest.output_path or ""),
        "recommendations": [
            _paper_payload(
                {
                    "canonical_id": item.canonical_id,
                    "rank": item.rank,
                    "score": item.score,
                    "paper": item.paper,
                    "metadata": {
                        "matched_topics": item.matched_topics,
                        "matched_terms": item.matched_terms,
                        "recommendation_reason": item.recommendation_reason,
                        "summary": item.summary.as_dict() if item.summary else {},
                    },
                }
            )
            for item in digest.recommendations
        ],
    }


class PaperDailyGui:
    """A small local facade over PaperDaily config, storage, and background jobs."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve(strict=False)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._active_keys: dict[str, str] = {}

    @staticmethod
    def _user_id(value: str | None) -> str:
        user_id = str(value or "").strip()
        if not USER_ID_RE.fullmatch(user_id):
            raise ValueError("用户 ID 只能包含字母、数字、连字符和下划线，长度最多 64 位")
        return user_id

    @property
    def _users_dir(self) -> Path:
        return self.config_path.parent / "users"

    def _base_config(self) -> PaperDailyConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"PaperDaily 配置不存在：{self.config_path}。请先运行 `paperdaily init`。"
            )
        return load_config(self.config_path)

    def _user_config_path(self, user_id: str | None, *, create: bool = True) -> Path:
        base = self._base_config()
        normalized = self._user_id(user_id or base.user_id)
        path = self._users_dir / f"{normalized}.yaml"
        if path.exists() or not create:
            return path

        # The existing top-level config becomes the original user's workspace.
        # New local users start with their own empty topic list and output area,
        # while feedback remains isolated by user_id in the shared SQLite DB.
        config = deepcopy(base)
        config.user_id = normalized
        if normalized != base.user_id:
            config.topics = []
            config.output_dir = base.output_dir / "users" / normalized
        save_config(config, path)
        return path

    def _context(self, user_id: str | None = None) -> tuple[PaperDailyConfig, PaperDailyService]:
        config = load_config(self._user_config_path(user_id))
        store = PaperDailyStore(config.database)
        store.initialize()
        return config, PaperDailyService(config, store=store)

    def list_users(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"users": []}
        base = self._base_config()
        users: dict[str, dict[str, str]] = {base.user_id: {"user_id": base.user_id, "label": base.user_id}}
        if self._users_dir.exists():
            for path in sorted(self._users_dir.glob("*.yaml")):
                try:
                    config = load_config(path)
                except Exception:
                    continue
                users[config.user_id] = {"user_id": config.user_id, "label": config.user_id}
        return {"users": list(users.values()), "default_user_id": base.user_id}

    def create_user(self, user_id: str) -> dict[str, Any]:
        normalized = self._user_id(user_id)
        config, _service = self._context(normalized)
        return {"user": {"user_id": config.user_id, "label": config.user_id}, **self.list_users()}

    @staticmethod
    def _task_payload(task: dict[str, Any], *, reused: bool = False) -> dict[str, Any]:
        payload = {
            "task_id": task["task_id"],
            "kind": task["kind"],
            "status": task["status"],
            "started_at": task["started_at"],
            "completed_at": task.get("completed_at"),
            "error": task.get("error", ""),
            "result": task.get("result"),
        }
        if reused:
            payload["reused"] = True
        return payload

    def _active_task(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            task_id = self._active_keys.get(key)
            task = self._tasks.get(task_id) if task_id else None
            if not task or task.get("status") != "running":
                return None
            return self._task_payload(task)

    def _start_task(self, kind: str, key: str, worker: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            existing_id = self._active_keys.get(key)
            if existing_id:
                existing = self._tasks.get(existing_id)
                if existing and existing.get("status") == "running":
                    return self._task_payload(existing, reused=True)

            task_id = str(uuid4())
            task = {
                "task_id": task_id,
                "kind": kind,
                "status": "running",
                "started_at": _utc_now(),
                "completed_at": None,
                "error": "",
                "result": None,
            }
            self._tasks[task_id] = task
            self._active_keys[key] = task_id

        def execute() -> None:
            try:
                result = worker()
                with self._lock:
                    task["status"] = "completed"
                    task["result"] = result
            except Exception as exc:  # The UI receives a bounded operational error, never a traceback.
                with self._lock:
                    task["status"] = "failed"
                    task["error"] = str(exc)[:MAX_TASK_ERROR_CHARS]
            finally:
                with self._lock:
                    task["completed_at"] = _utc_now()
                    if self._active_keys.get(key) == task_id:
                        self._active_keys.pop(key, None)

        threading.Thread(target=execute, name=f"paperdaily-{kind}", daemon=True).start()
        return self._task_payload(task)

    def status(self, user_id: str | None = None) -> dict[str, Any]:
        if not self.config_path.exists():
            return {
                "configured": False,
                "config_path": str(self.config_path),
                "message": "请先运行 `paperdaily init` 创建本地配置。",
                "topics": [],
                "catchup": None,
                "latest_run": None,
            }

        config, service = self._context(user_id)
        commands, _ = _provider_options(config)
        runs = service.store.list_runs(config.user_id, limit=20)
        latest_completed = next((item for item in runs if item.get("status") == "completed"), None)
        latest_populated = self._latest_populated_run(service, config.user_id)
        state = service.store.get_state(config.user_id) or {}
        return {
            "configured": True,
            "config_path": str(self._user_config_path(config.user_id)),
            "user_id": config.user_id,
            "timezone": config.timezone,
            "output_dir": str(config.output_dir),
            "topics": [topic.to_dict() for topic in config.topics],
            "enabled_topic_count": len(service.enabled_topics),
            "categories": service.categories,
            "catchup": service.catchup_plan().to_dict(),
            "state": state,
            "latest_run": self._run_payload(latest_populated or latest_completed),
            "latest_processed_run": self._run_payload(latest_completed) if latest_completed else None,
            "providers": {
                "llm": self._provider_name(build_llm_provider()),
                "embedding": self._provider_name(build_embedding_provider()),
                "agents": diagnose_agent_providers(
                    commands=commands,
                    isolated_home=config.deep_read.isolated_home,
                ),
            },
            "active_digest_task": self._active_task(f"paperdaily-digest:{config.user_id}"),
        }

    @staticmethod
    def _provider_name(provider: Any) -> dict[str, str]:
        return {
            "name": str(getattr(provider, "name", "unknown")),
            "model": str(getattr(provider, "model", "unknown")),
        }

    @staticmethod
    def _run_payload(run: dict[str, Any] | None) -> dict[str, Any] | None:
        if run is None:
            return None
        run_summary = dict((run.get("metadata") or {}).get("run_summary") or {})
        return {
            "run_id": str(run.get("run_id") or ""),
            "status": str(run.get("status") or ""),
            "mode": str(run.get("mode") or ""),
            "window_start": str(run.get("window_start") or ""),
            "window_end": str(run.get("window_end") or ""),
            "started_at": str(run.get("started_at") or ""),
            "completed_at": str(run.get("completed_at") or ""),
            "fetched_count": int(run.get("fetched_count") or 0),
            "candidate_count": int(run.get("candidate_count") or 0),
            "recommendation_count": int(run.get("recommendation_count") or 0),
            "summary_count": int(run.get("summary_count") or 0),
            "matched_count": int(run_summary.get("matched_count") or 0),
            "handled_count": int(run_summary.get("handled_count") or 0),
            "new_recommendation_count": int(
                run_summary.get("new_recommendation_count", run.get("recommendation_count") or 0)
            ),
            "output_path": str((run.get("metadata") or {}).get("output_path") or ""),
            "error": str(run.get("error_message") or ""),
        }

    @staticmethod
    def _latest_populated_run(
        service: PaperDailyService,
        user_id: str,
        *,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> dict[str, Any] | None:
        for run in service.store.list_runs(user_id, limit=100):
            if run.get("status") != "completed" or int(run.get("recommendation_count") or 0) <= 0:
                continue
            if window_start is not None and str(run.get("window_start") or "") != window_start:
                continue
            if window_end is not None and str(run.get("window_end") or "") != window_end:
                continue
            return run
        return None

    def _digest_payload(self, service: PaperDailyService, run: dict[str, Any]) -> dict[str, Any]:
        return {
            "run": self._run_payload(run),
            "recommendations": [
                _paper_payload(item)
                for item in service.store.get_recommendations(str(run["run_id"]))
            ],
        }

    def latest_digest(self, run_id: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        config, service = self._context(user_id)
        run = service.store.get_run(run_id) if run_id else None
        if run is None:
            run = self._latest_populated_run(service, config.user_id)
            if run is None:
                runs = service.store.list_runs(config.user_id, limit=50)
                run = next((item for item in runs if item.get("status") == "completed"), None)
        if run is None or run.get("status") != "completed":
            return {"digest": None}
        if str(run.get("user_id") or "") != config.user_id:
            raise ValueError("该日报不属于当前 PaperDaily 用户")
        return {"digest": self._digest_payload(service, run)}

    def list_digests(self, *, limit: int = 30, user_id: str | None = None) -> dict[str, Any]:
        """List completed digests belonging to the configured local user."""

        config, service = self._context(user_id)
        runs = service.store.list_runs(config.user_id, limit=max(1, min(100, int(limit))))
        return {
            "runs": [
                self._run_payload(run)
                for run in runs
                if run.get("status") == "completed" and int(run.get("recommendation_count") or 0) > 0
            ]
        }

    @staticmethod
    def _quick_topic(raw_topic: dict[str, Any]) -> Topic:
        raw_keywords = raw_topic.get("keywords") or raw_topic.get("keywords_input") or []
        keywords = [str(item).strip() for item in raw_keywords] if isinstance(raw_keywords, list) else [
            item.strip() for item in re.split(r"[,\n]", str(raw_keywords))
        ]
        keywords = list(dict.fromkeys(item for item in keywords if item))
        if not keywords:
            raise ValueError("请至少输入一个研究关键词")

        normalized_keywords = {item.casefold() for item in keywords}
        acronym_context = list(raw_topic.get("context_keywords") or [])
        if {"vla", "wam"} & normalized_keywords and not acronym_context:
            acronym_context = list(DEFAULT_ACRONYM_CONTEXT)
        negatives = list(raw_topic.get("negative_keywords") or [])
        if "wam" in normalized_keywords and not negatives:
            negatives = ["wireless access management", "web application monitoring"]
        def dedupe(values: list[str]) -> list[str]:
            seen: set[str] = set()
            result: list[str] = []
            for value in values:
                text = str(value).strip()
                key = text.casefold()
                if text and key not in seen:
                    seen.add(key)
                    result.append(text)
            return result

        auto_phrases = [
            phrase
            for keyword in normalized_keywords
            for phrase in ACRONYM_EXACT_PHRASES.get(keyword, [])
        ]
        exact_phrases = dedupe([
            *list(raw_topic.get("exact_phrases") or [
                item for item in keywords if " " in item or "-" in item
            ]),
            *auto_phrases,
        ])
        name = str(raw_topic.get("name") or "").strip() or " / ".join(keywords[:3])
        topic_id = str(raw_topic.get("id") or "").strip() or f"topic-{uuid4().hex[:10]}"
        return Topic(
            id=topic_id,
            name=name[:100],
            description=str(raw_topic.get("description") or f"Keywords: {', '.join(keywords)}").strip(),
            enabled=bool(raw_topic.get("enabled", True)),
            arxiv_categories=list(raw_topic.get("arxiv_categories") or DEFAULT_TOPIC_CATEGORIES),
            exact_phrases=exact_phrases,
            keywords=keywords,
            context_keywords=acronym_context,
            negative_keywords=negatives,
            daily_limit=int(raw_topic.get("daily_limit") or 12),
            minimum_score=float(raw_topic.get("minimum_score") or 0.25),
        )

    def save_topic(self, raw_topic: dict[str, Any], *, user_id: str | None = None) -> dict[str, Any]:
        config, _service = self._context(user_id)
        topic = self._quick_topic(raw_topic) if raw_topic.get("quick") else Topic.from_dict(raw_topic)
        for index, current in enumerate(config.topics):
            if current.id == topic.id:
                config.topics[index] = topic
                break
        else:
            config.topics.append(topic)
        save_config(config, self._user_config_path(config.user_id))
        return {"topic": topic.to_dict(), "topics": [item.to_dict() for item in config.topics]}

    def delete_topic(self, topic_id: str, *, user_id: str | None = None) -> dict[str, Any]:
        config, _service = self._context(user_id)
        normalized = str(topic_id or "").strip()
        remaining = [topic for topic in config.topics if topic.id != normalized]
        if len(remaining) == len(config.topics):
            raise ValueError("未找到该话题")
        config.topics = remaining
        save_config(config, self._user_config_path(config.user_id))
        return {"deleted": normalized, "topics": [item.to_dict() for item in config.topics]}

    def set_topic_enabled(self, topic_id: str, enabled: Any, *, user_id: str | None = None) -> dict[str, Any]:
        config, _service = self._context(user_id)
        normalized = str(topic_id or "").strip()
        topic = next((item for item in config.topics if item.id == normalized), None)
        if topic is None:
            raise ValueError("未找到该话题")
        topic.enabled = bool(enabled)
        save_config(config, self._user_config_path(config.user_id))
        return {"topic": topic.to_dict(), "topics": [item.to_dict() for item in config.topics]}

    def update_topic(self, action: str, body: dict[str, Any], *, user_id: str | None = None) -> dict[str, Any]:
        normalized = str(action or "save").strip().lower()
        if normalized == "save":
            raw = body.get("topic")
            if not isinstance(raw, dict):
                raise ValueError("topic 必须是对象")
            return self.save_topic(raw, user_id=user_id)
        if normalized == "delete":
            return self.delete_topic(str(body.get("topic_id") or ""), user_id=user_id)
        if normalized == "enabled":
            return self.set_topic_enabled(str(body.get("topic_id") or ""), body.get("enabled"), user_id=user_id)
        raise ValueError("action 必须是 save、delete 或 enabled")

    def start_digest_task(
        self,
        *,
        user_id: str | None = None,
        choice: str,
        dry_run: bool,
        limit: int | None = None,
        custom_start: str | None = None,
        custom_end: str | None = None,
        include_handled: bool = False,
    ) -> dict[str, Any]:
        normalized_choice = str(choice or "recommended").strip().lower()
        config, service = self._context(user_id)

        def worker() -> dict[str, Any]:
            plan = service.catchup_plan()
            window = service.select_window(
                plan,
                normalized_choice,
                custom_start=custom_start,
                custom_end=custom_end,
            )
            if window.is_empty:
                raise ValueError("当前没有待处理论文；请选择明确的历史日期窗口。")
            normalized_limit = max(1, int(limit)) if limit is not None else None
            if not dry_run and not include_handled:
                preview = service.run(
                    window,
                    dry_run=True,
                    limit=normalized_limit,
                    channels=["markdown"],
                )
                preview_stats = dict(preview.digest.stats)
                all_already_sent = (
                    not preview.digest.recommendations
                    and int(preview_stats.get("matched_count") or 0) > 0
                    and int(preview_stats.get("handled_count") or 0) > 0
                )
                existing_run = self._latest_populated_run(
                    service,
                    config.user_id,
                    window_start=window.start_date.isoformat(),
                    window_end=window.end_date.isoformat(),
                )
                if all_already_sent and existing_run is not None:
                    existing_digest = self._digest_payload(service, existing_run)
                    return {
                        "dry_run": False,
                        "run_id": str(existing_run["run_id"]),
                        "window_start": window.start_date.isoformat(),
                        "window_end": window.end_date.isoformat(),
                        "choice": window.choice,
                        "stats": preview_stats,
                        "warnings": [],
                        "output_path": str((existing_run.get("metadata") or {}).get("output_path") or ""),
                        "recommendations": existing_digest["recommendations"],
                        "reused_existing_digest": True,
                        "reused_run_id": str(existing_run["run_id"]),
                    }
            outcome = service.run(
                window,
                dry_run=dry_run,
                limit=normalized_limit,
                channels=["markdown"],
                include_handled=bool(include_handled),
            )
            return _outcome_payload(outcome)

        return self._start_task(
            "preview" if dry_run else "digest",
            f"paperdaily-digest:{config.user_id}",
            worker,
        )

    def start_codex_read(
        self,
        arxiv_id: str,
        *,
        user_id: str | None = None,
        force_parse: bool = False,
    ) -> dict[str, Any]:
        canonical = canonicalize_arxiv_id(arxiv_id)
        if not canonical:
            raise ValueError("无效的 arXiv ID")

        def worker() -> dict[str, Any]:
            config, service = self._context(user_id)
            commands, max_budget = _provider_options(config)
            provider = build_agent_provider(
                "codex",
                preferred_order=config.providers.fallback_order,
                commands=commands,
                max_budget_usd=max_budget,
                isolated_home=config.deep_read.isolated_home,
            )
            context = "\n\n".join(
                f"## {topic.name}\n{topic.description}" for topic in service.enabled_topics
            )
            reader = DeepReadService(
                workspace_root=config.output_dir.parent / "workspaces",
                notes_dir=config.output_dir / "notes",
                store=service.store,
                max_pdf_pages=config.deep_read.max_pdf_pages,
                max_extracted_text_chars=config.deep_read.max_extracted_text_chars,
            )
            agent_run_id = service.store.start_agent_run(
                provider.name,
                "gui-managed",
                "reading_note",
                canonical_id=canonical,
            )
            try:
                result = reader.run(
                    canonical,
                    provider=provider,
                    topic_context=context,
                    force_parse=bool(force_parse),
                )
                service.store.complete_agent_run(
                    agent_run_id,
                    result.note,
                    metadata={"markdown_path": str(result.markdown_path)},
                )
                service.record_feedback(canonical, "reading_note")
            except Exception as exc:
                with suppress(Exception):
                    service.store.fail_agent_run(agent_run_id, exc)
                raise
            return {
                "arxiv_id": canonical,
                "agent_run_id": agent_run_id,
                "provider": provider.name,
                "note_path": str(result.markdown_path),
            }

        return self._start_task("codex_read", f"paperdaily-codex-read:{user_id or 'default'}", worker)

    def start_summary_retry(self, run_id: str, *, user_id: str | None = None) -> dict[str, Any]:
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            raise ValueError("run_id is required")

        def worker() -> dict[str, Any]:
            config, service = self._context(user_id)
            run = service.store.get_run(normalized_run_id)
            if run is None or run.get("status") != "completed":
                raise ValueError("未找到已完成的日报")
            if str(run.get("user_id") or "") != config.user_id:
                raise ValueError("该日报不属于当前研究者空间")

            records = service.store.get_recommendations(normalized_run_id)
            if not records:
                raise ValueError("该日报没有可重试的论文")
            summary_service = ChineseSummaryService(
                store=service.store,
                provider=service.summary_provider,
                language=config.daily.summary_language,
            )
            recommendations: list[Recommendation] = []
            retried = 0
            for record in records:
                metadata = dict(record.get("metadata") or {})
                previous = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
                summary = summary_service.summarize(dict(record.get("paper") or {}))
                if str(previous.get("status") or "") != "completed":
                    retried += 1
                recommendations.append(
                    Recommendation(
                        rank=int(record.get("rank") or 0),
                        score=float(record.get("score") or 0.0),
                        paper=dict(record.get("paper") or {}),
                        matched_topics=list(metadata.get("matched_topics") or []),
                        matched_terms=list(metadata.get("matched_terms") or []),
                        recommendation_reason=str(metadata.get("recommendation_reason") or ""),
                        component_scores=dict(metadata.get("component_scores") or {}),
                        rerank=dict(metadata.get("rerank") or {}),
                        summary=summary,
                    )
                )

            service.store.save_recommendations(
                normalized_run_id,
                service._store_recommendations(recommendations),
            )
            window_start = datetime.fromisoformat(str(run["window_start"])).date()
            window_end = datetime.fromisoformat(str(run["window_end"])).date()
            digest = Digest(
                run_id=normalized_run_id,
                user_id=config.user_id,
                window_start=window_start,
                window_end=window_end,
                generated_at=datetime.now(timezone.utc),
                recommendations=recommendations,
                catchup_mode=str(run.get("mode") or "daily"),
            )
            stored_path = str((run.get("metadata") or {}).get("output_path") or "").strip()
            output_path = Path(stored_path) if stored_path else None
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(render_digest_markdown(digest), encoding="utf-8")
            service.store.complete_run(
                normalized_run_id,
                summary_count=sum(item.summary.status == "completed" for item in recommendations if item.summary),
                metadata={"output_path": str(output_path or "")},
            )
            return {
                "run_id": normalized_run_id,
                "retried": retried,
                "completed": sum(item.summary.status == "completed" for item in recommendations if item.summary),
                "total": len(recommendations),
            }

        return self._start_task(
            "summary_retry",
            f"paperdaily-summary:{user_id or 'default'}:{normalized_run_id}",
            worker,
        )

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            return self._task_payload(task) if task else None

    def record_feedback(self, arxiv_id: str, action: str, *, user_id: str | None = None) -> dict[str, Any]:
        _config, service = self._context(user_id)
        return {"feedback": service.record_feedback(arxiv_id, action)}

    def read_note(self, arxiv_id: str, *, user_id: str | None = None) -> dict[str, Any]:
        config, _service = self._context(user_id)
        canonical = canonicalize_arxiv_id(arxiv_id)
        if not canonical:
            raise ValueError("无效的 arXiv ID")
        notes_dir = (config.output_dir / "notes").resolve()
        path = (notes_dir / f"{canonical}.md").resolve()
        try:
            path.relative_to(notes_dir)
        except ValueError as exc:  # Defensive: canonical arXiv IDs cannot traverse paths.
            raise ValueError("无效的阅读笔记路径") from exc
        if not path.exists():
            return {"note": None}
        return {
            "note": {
                "arxiv_id": canonical,
                "path": str(path),
                "content": path.read_text(encoding="utf-8")[:MAX_NOTE_CHARS],
            }
        }


__all__ = ["PaperDailyGui"]
