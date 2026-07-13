from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from paperdaily.catchup import DateWindow
from paperdaily.collector import ArxivCollectorError, ArxivFetchResult
from paperdaily.config import DailyConfig, PaperDailyConfig
from paperdaily.models import DeliveryResult
from paperdaily.service import PaperDailyService
from paperdaily.storage import PaperDailyStore
from paperdaily.topics import Topic


class _HashEmbedding:
    name = "hash"
    model = "unit-test"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


class _Collector:
    def __init__(self, papers: list[dict[str, Any]]) -> None:
        self.papers = papers
        self.calls: list[tuple[date, date, list[str]]] = []

    def fetch_window(self, start: date, end: date, categories: list[str]) -> ArxivFetchResult:
        self.calls.append((start, end, list(categories)))
        return ArxivFetchResult(
            papers=list(self.papers),
            window_start=start,
            window_end=end,
            total_available=len(self.papers),
            network_requests=0,
            cache_hits=1,
            truncated=False,
        )


class _TruncatedCollector(_Collector):
    def fetch_window(self, start: date, end: date, categories: list[str]) -> ArxivFetchResult:
        result = super().fetch_window(start, end, categories)
        return ArxivFetchResult(
            papers=result.papers,
            window_start=result.window_start,
            window_end=result.window_end,
            total_available=len(result.papers) + 1,
            network_requests=result.network_requests,
            cache_hits=result.cache_hits,
            truncated=True,
        )


class _AnnouncementCollector(_Collector):
    def __init__(self, papers: list[dict[str, Any]]) -> None:
        super().__init__(papers)
        self.announcement_calls: list[tuple[date, list[str], str, bool]] = []

    def fetch_announcements(
        self,
        target_date: date,
        categories: list[str],
        *,
        timezone_name: str,
        include_cross_list: bool,
    ) -> ArxivFetchResult:
        self.announcement_calls.append(
            (target_date, list(categories), timezone_name, include_cross_list)
        )
        return ArxivFetchResult(
            papers=list(self.papers),
            window_start=target_date,
            window_end=target_date,
            total_available=len(self.papers),
            network_requests=1,
            cache_hits=0,
            truncated=False,
            source="rss_announcements",
        )


def _config(tmp_path: Path, *, channels: list[str] | None = None) -> PaperDailyConfig:
    return PaperDailyConfig(
        user_id="alice",
        timezone="UTC",
        output_dir=tmp_path / "output",
        database=tmp_path / "paperflow.db",
        daily=DailyConfig(
            default_limit=3,
            rerank_limit=3,
            generate_chinese_summary=True,
            channels=channels or ["terminal"],
            arxiv_request_delay_seconds=0,
        ),
        topics=[
            Topic(
                id="embodied-vla",
                name="Embodied VLA",
                arxiv_categories=["cs.RO"],
                exact_phrases=["vision language action"],
                keywords=["VLA"],
                context_keywords=["robot"],
                minimum_score=0.2,
            )
        ],
    )


def _paper() -> dict[str, Any]:
    return {
        "arxiv_id": "2607.00001v1",
        "title": "Vision Language Action for robot manipulation",
        "abstract": "A VLA robot policy.",
        "publish_date": "2026-07-10",
        "authors": ["Ada"],
        "paper_url": "https://arxiv.org/abs/2607.00001",
        "pdf_url": "https://arxiv.org/pdf/2607.00001",
    }


def _papers_for_rerank() -> list[dict[str, Any]]:
    return [
        {
            **_paper(),
            "arxiv_id": f"2607.0000{index}v1",
            "title": f"Vision Language Action robot policy {index}",
            "abstract": f"A VLA robot policy for task {index}.",
        }
        for index in range(1, 5)
    ]


def _table_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_dry_run_does_not_write_run_watermark_summary_delivery_or_output(tmp_path: Path) -> None:
    class ForbiddenSummaryProvider:
        name = "forbidden"
        model = "forbidden"

        def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("dry-run must not call the summary provider")

    config = _config(tmp_path)
    store = PaperDailyStore(config.database)
    service = PaperDailyService(
        config,
        store=store,
        collector=_Collector([_paper()]),
        embedding_provider=_HashEmbedding(),
        summary_provider=ForbiddenSummaryProvider(),
    )
    window = DateWindow(date(2026, 7, 10), date(2026, 7, 10), "yesterday")

    outcome = service.run(window, dry_run=True, channels=["feishu"])

    assert outcome.dry_run is True
    assert len(outcome.digest.recommendations) == 1
    assert outcome.deliveries == []
    assert store.get_state("alice") is None
    assert store.list_runs("alice") == []
    assert _table_count(config.database, "paperdaily_summaries") == 0
    assert _table_count(config.database, "paperdaily_reranks") == 0
    assert _table_count(config.database, "paperdaily_deliveries") == 0
    assert not (config.output_dir / "digests").exists()
    assert outcome.digest.stats["llm_rerank_call_count"] == 0
    assert outcome.digest.stats["llm_rerank_status"] == "skipped_dry_run"


def test_initialize_runtime_uses_only_paperdaily_storage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = PaperDailyService(
        config,
        store=PaperDailyStore(config.database),
        embedding_provider=_HashEmbedding(),
    )

    result = service.initialize_runtime()

    assert result == {
        "database": str(config.database),
        "output_dir": str(config.output_dir),
        "user_id": "alice",
        "topic_count": 1,
    }
    assert config.output_dir.exists()


def test_live_announcement_day_uses_rss_before_api_date_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, channels=["markdown"])
    collector = _AnnouncementCollector([_paper()])
    service = PaperDailyService(
        config,
        store=PaperDailyStore(config.database),
        collector=collector,
        embedding_provider=_HashEmbedding(),
    )
    announcement_day = date(2026, 7, 13)
    monkeypatch.setattr("paperdaily.service.local_today", lambda _timezone: announcement_day)

    outcome = service.run(
        DateWindow(announcement_day, announcement_day, "latest"),
        generate_summary=False,
        channels=["markdown"],
    )

    assert collector.calls == []
    assert collector.announcement_calls == [
        (announcement_day, ["cs.RO"], "UTC", True)
    ]
    assert outcome.digest.stats["arxiv_source"] == "rss_announcements"
    assert service.store.get_state("alice")["last_completed_window_end"] == "2026-07-13"


def test_markdown_success_completes_run_when_optional_feishu_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, channels=["feishu"])
    store = PaperDailyStore(config.database)
    service = PaperDailyService(
        config,
        store=store,
        collector=_Collector([_paper()]),
        embedding_provider=_HashEmbedding(),
    )

    def fail_feishu(_self: Any, _digest: Any) -> DeliveryResult:
        return DeliveryResult(channel="feishu", success=False, error="not configured")

    monkeypatch.setattr("paperdaily.service.FeishuChannel.publish_digest", fail_feishu)
    outcome = service.run(
        DateWindow(date(2026, 7, 10), date(2026, 7, 10), "yesterday"),
        generate_summary=False,
        channels=["feishu"],
    )

    run = store.get_run(outcome.digest.run_id)
    deliveries = {row["channel"]: row for row in store.list_deliveries(outcome.digest.run_id)}
    assert run is not None and run["status"] == "completed"
    assert store.get_state("alice")["last_completed_window_end"] == "2026-07-10"
    assert outcome.digest.output_path is not None and outcome.digest.output_path.exists()
    assert deliveries["markdown"]["status"] == "succeeded"
    assert deliveries["feishu"]["status"] == "failed"
    assert any("feishu" in warning for warning in outcome.warnings)


def test_markdown_failure_fails_run_and_does_not_advance_watermark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, channels=["markdown"])
    store = PaperDailyStore(config.database)
    service = PaperDailyService(
        config,
        store=store,
        collector=_Collector([_paper()]),
        embedding_provider=_HashEmbedding(),
    )

    def fail_markdown(_self: Any, _digest: Any) -> DeliveryResult:
        return DeliveryResult(channel="markdown", success=False, error="disk full")

    monkeypatch.setattr("paperdaily.service.MarkdownChannel.publish_digest", fail_markdown)

    with pytest.raises(RuntimeError, match="Markdown"):
        service.run(
            DateWindow(date(2026, 7, 10), date(2026, 7, 10), "yesterday"),
            generate_summary=False,
            channels=["markdown"],
        )

    runs = store.list_runs("alice")
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert store.get_state("alice")["last_completed_window_end"] is None


def test_truncated_fetch_fails_run_without_advancing_watermark(tmp_path: Path) -> None:
    """A capped source query must never mark an incomplete date window done."""

    config = _config(tmp_path, channels=["markdown"])
    store = PaperDailyStore(config.database)
    service = PaperDailyService(
        config,
        store=store,
        collector=_TruncatedCollector([_paper()]),
        embedding_provider=_HashEmbedding(),
    )

    with pytest.raises(ArxivCollectorError, match="watermark 未推进"):
        service.run(
            DateWindow(date(2026, 7, 10), date(2026, 7, 10), "yesterday"),
            generate_summary=False,
            channels=["markdown"],
        )

    runs = store.list_runs("alice")
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["recommendation_count"] == 0
    assert store.get_state("alice")["last_completed_window_end"] is None
    assert _table_count(config.database, "paperdaily_recommendations") == 0
    assert _table_count(config.database, "paperdaily_summaries") == 0
    assert not (config.output_dir / "digests").exists()


def test_catchup_plan_targets_earliest_gap_before_completed_newer_slice(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = PaperDailyStore(config.database)
    service = PaperDailyService(
        config,
        store=store,
        collector=_Collector([]),
        embedding_provider=_HashEmbedding(),
    )

    baseline = store.start_run("alice", "2026-07-01", "2026-07-01")
    store.complete_run(baseline)
    newer_slice = store.start_run("alice", "2026-07-05", "2026-07-11", mode="catchup")
    store.complete_run(newer_slice)

    plan = service.catchup_plan(today=date(2026, 7, 12))

    assert plan.gap_days == 3
    assert plan.recommended_choice == "7d"
    assert plan.next_gap_end == date(2026, 7, 4)
    assert plan.recommended_window.start_date == date(2026, 7, 2)
    assert plan.recommended_window.end_date == date(2026, 7, 4)

    gap = store.start_run("alice", "2026-07-02", "2026-07-04", mode="catchup")
    store.complete_run(gap)
    assert store.get_state("alice")["last_completed_window_end"] == "2026-07-11"


def test_service_reranks_only_configured_top_n_and_records_usage(tmp_path: Path) -> None:
    class RerankProvider:
        name = "unit-llm"
        model = "rerank-v1"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str, **_kwargs: Any) -> Any:
            self.calls.append(prompt)
            return type(
                "Response",
                (),
                {
                    "text": json.dumps(
                        {
                            "rankings": [
                                {
                                    "arxiv_id": f"2607.0000{index}",
                                    "relevance": 80,
                                    "is_target_domain": True,
                                    "matched_topic_ids": ["embodied-vla"],
                                    "paper_type": "method",
                                    "reason_zh": "摘要讨论机器人策略。",
                                    "uncertainty_zh": "摘要未说明实验细节。",
                                }
                                for index in range(1, 4)
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                },
            )()

    config = _config(tmp_path, channels=["markdown"])
    config.daily.default_limit = 2
    config.daily.rerank_limit = 3
    config.daily.llm_rerank_weight = 0.5
    config.daily.llm_rerank_input_cost_per_million_tokens = 2.0
    config.daily.llm_rerank_output_cost_per_million_tokens = 8.0
    provider = RerankProvider()
    service = PaperDailyService(
        config,
        store=PaperDailyStore(config.database),
        collector=_Collector(_papers_for_rerank()),
        embedding_provider=_HashEmbedding(),
        rerank_provider=provider,
    )

    outcome = service.run(
        DateWindow(date(2026, 7, 10), date(2026, 7, 10), "yesterday"),
        generate_summary=False,
        channels=["markdown"],
    )

    assert len(provider.calls) == 1
    assert "2607.00003" in provider.calls[0]
    assert "2607.00004" not in provider.calls[0]
    assert outcome.digest.stats["llm_rerank_candidate_count"] == 3
    assert outcome.digest.stats["llm_rerank_call_count"] == 1
    assert outcome.digest.stats["llm_rerank_total_tokens"] == 168
    assert outcome.digest.stats["llm_rerank_estimated_cost_usd"] == pytest.approx(0.000606)
    assert all(item.rerank for item in outcome.digest.recommendations)
    stored = service.store.get_recommendations(outcome.digest.run_id)
    assert stored[0]["metadata"]["rerank"]["relevance"] == 80.0
