"""Local GUI adapters for the PaperDaily workflow.

The desktop server deliberately keeps PaperDaily separate from the legacy
PaperFlow daily-push pipeline.  It reads the configured PaperDaily user and
SQLite state directly, while long-running arXiv and Codex work runs in daemon
threads so a browser request never owns a model or network operation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from paperdaily.cli import DEFAULT_CONFIG_PATH
from paperdaily.config import PaperDailyConfig, load_config, save_config
from paperdaily.deep_read import DeepReadService
from paperdaily.identifiers import canonicalize_arxiv_id
from paperdaily.providers import build_agent_provider, diagnose_agent_providers
from paperdaily.service import PaperDailyService, RunOutcome
from paperdaily.storage import PaperDailyStore
from paperdaily.topics import Topic
from paperflow.providers import build_embedding_provider, build_llm_provider

MAX_NOTE_CHARS = 180_000
MAX_TASK_ERROR_CHARS = 1_600


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

    def _context(self) -> tuple[PaperDailyConfig, PaperDailyService]:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"PaperDaily 配置不存在：{self.config_path}。请先运行 `paperdaily init`。"
            )
        config = load_config(self.config_path)
        store = PaperDailyStore(config.database)
        store.initialize()
        return config, PaperDailyService(config, store=store)

    @staticmethod
    def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task["task_id"],
            "kind": task["kind"],
            "status": task["status"],
            "started_at": task["started_at"],
            "completed_at": task.get("completed_at"),
            "error": task.get("error", ""),
            "result": task.get("result"),
        }

    def _start_task(self, kind: str, key: str, worker: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            existing_id = self._active_keys.get(key)
            if existing_id:
                existing = self._tasks.get(existing_id)
                if existing and existing.get("status") == "running":
                    return self._task_payload(existing)

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

    def status(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {
                "configured": False,
                "config_path": str(self.config_path),
                "message": "请先运行 `paperdaily init` 创建本地配置。",
                "topics": [],
                "catchup": None,
                "latest_run": None,
            }

        config, service = self._context()
        commands, _ = _provider_options(config)
        runs = service.store.list_runs(config.user_id, limit=20)
        latest_completed = next((item for item in runs if item.get("status") == "completed"), None)
        state = service.store.get_state(config.user_id) or {}
        return {
            "configured": True,
            "config_path": str(self.config_path),
            "user_id": config.user_id,
            "timezone": config.timezone,
            "output_dir": str(config.output_dir),
            "topics": [topic.to_dict() for topic in config.topics],
            "enabled_topic_count": len(service.enabled_topics),
            "categories": service.categories,
            "catchup": service.catchup_plan().to_dict(),
            "state": state,
            "latest_run": self._run_payload(latest_completed) if latest_completed else None,
            "providers": {
                "llm": self._provider_name(build_llm_provider()),
                "embedding": self._provider_name(build_embedding_provider()),
                "agents": diagnose_agent_providers(
                    commands=commands,
                    isolated_home=config.deep_read.isolated_home,
                ),
            },
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
            "output_path": str((run.get("metadata") or {}).get("output_path") or ""),
            "error": str(run.get("error_message") or ""),
        }

    def latest_digest(self, run_id: str | None = None) -> dict[str, Any]:
        config, service = self._context()
        run = service.store.get_run(run_id) if run_id else None
        if run is None:
            runs = service.store.list_runs(config.user_id, limit=50)
            run = next((item for item in runs if item.get("status") == "completed"), None)
        if run is None or run.get("status") != "completed":
            return {"digest": None}
        if str(run.get("user_id") or "") != config.user_id:
            raise ValueError("该日报不属于当前 PaperDaily 用户")
        records = service.store.get_recommendations(str(run["run_id"]))
        return {
            "digest": {
                "run": self._run_payload(run),
                "recommendations": [_paper_payload(item) for item in records],
            }
        }

    def list_digests(self, *, limit: int = 30) -> dict[str, Any]:
        """List completed digests belonging to the configured local user."""

        config, service = self._context()
        runs = service.store.list_runs(config.user_id, limit=max(1, min(100, int(limit))))
        return {
            "runs": [
                self._run_payload(run)
                for run in runs
                if run.get("status") == "completed"
            ]
        }

    def save_topic(self, raw_topic: dict[str, Any]) -> dict[str, Any]:
        config, _service = self._context()
        topic = Topic.from_dict(raw_topic)
        for index, current in enumerate(config.topics):
            if current.id == topic.id:
                config.topics[index] = topic
                break
        else:
            config.topics.append(topic)
        save_config(config, self.config_path)
        return {"topic": topic.to_dict(), "topics": [item.to_dict() for item in config.topics]}

    def delete_topic(self, topic_id: str) -> dict[str, Any]:
        config, _service = self._context()
        normalized = str(topic_id or "").strip()
        remaining = [topic for topic in config.topics if topic.id != normalized]
        if len(remaining) == len(config.topics):
            raise ValueError("未找到该话题")
        config.topics = remaining
        save_config(config, self.config_path)
        return {"deleted": normalized, "topics": [item.to_dict() for item in config.topics]}

    def set_topic_enabled(self, topic_id: str, enabled: Any) -> dict[str, Any]:
        config, _service = self._context()
        normalized = str(topic_id or "").strip()
        topic = next((item for item in config.topics if item.id == normalized), None)
        if topic is None:
            raise ValueError("未找到该话题")
        topic.enabled = bool(enabled)
        save_config(config, self.config_path)
        return {"topic": topic.to_dict(), "topics": [item.to_dict() for item in config.topics]}

    def update_topic(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        normalized = str(action or "save").strip().lower()
        if normalized == "save":
            raw = body.get("topic")
            if not isinstance(raw, dict):
                raise ValueError("topic 必须是对象")
            return self.save_topic(raw)
        if normalized == "delete":
            return self.delete_topic(str(body.get("topic_id") or ""))
        if normalized == "enabled":
            return self.set_topic_enabled(str(body.get("topic_id") or ""), body.get("enabled"))
        raise ValueError("action 必须是 save、delete 或 enabled")

    def start_digest_task(
        self,
        *,
        choice: str,
        dry_run: bool,
        limit: int | None = None,
        custom_start: str | None = None,
        custom_end: str | None = None,
    ) -> dict[str, Any]:
        normalized_choice = str(choice or "recommended").strip().lower()

        def worker() -> dict[str, Any]:
            _config, service = self._context()
            plan = service.catchup_plan()
            window = service.select_window(
                plan,
                normalized_choice,
                custom_start=custom_start,
                custom_end=custom_end,
            )
            if window.is_empty:
                raise ValueError("当前没有待处理论文；请选择明确的历史日期窗口。")
            outcome = service.run(
                window,
                dry_run=dry_run,
                limit=max(1, int(limit)) if limit is not None else None,
                channels=["markdown"],
            )
            return _outcome_payload(outcome)

        return self._start_task(
            "preview" if dry_run else "digest",
            "paperdaily-digest",
            worker,
        )

    def start_codex_read(self, arxiv_id: str, *, force_parse: bool = False) -> dict[str, Any]:
        canonical = canonicalize_arxiv_id(arxiv_id)
        if not canonical:
            raise ValueError("无效的 arXiv ID")

        def worker() -> dict[str, Any]:
            config, service = self._context()
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

        return self._start_task("codex_read", "paperdaily-codex-read", worker)

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(str(task_id or "").strip())
            return self._task_payload(task) if task else None

    def record_feedback(self, arxiv_id: str, action: str) -> dict[str, Any]:
        _config, service = self._context()
        return {"feedback": service.record_feedback(arxiv_id, action)}

    def read_note(self, arxiv_id: str) -> dict[str, Any]:
        config, _service = self._context()
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
