"""PaperDaily application service: collect, rank, summarize, persist, deliver."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from paperflow.providers import build_embedding_provider

from .catchup import CatchupPlan, CatchupPlanner, DateWindow, default_target_date, local_today
from .channels import FeishuChannel, MarkdownChannel, TerminalChannel
from .collector import ArxivCollector, ArxivCollectorError, ArxivFetchResult
from .config import PaperDailyConfig
from .identifiers import deduplicate_papers
from .models import DeliveryResult, Digest, Recommendation
from .ranking import PaperRanker
from .reranker import LLMReranker
from .storage import PaperDailyStore
from .summaries import ChineseSummaryService


@dataclass
class RunOutcome:
    digest: Digest
    dry_run: bool
    deliveries: list[DeliveryResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PaperDailyService:
    """A local-first orchestration layer that does not depend on Feishu."""

    def __init__(
        self,
        config: PaperDailyConfig,
        *,
        store: PaperDailyStore | None = None,
        collector: ArxivCollector | None = None,
        embedding_provider: Any = None,
        summary_provider: Any = None,
        rerank_provider: Any = None,
    ) -> None:
        self.config = config
        self.store = store or PaperDailyStore(config.database)
        self.embedding_provider = embedding_provider or build_embedding_provider()
        cache_dir = config.output_dir.parent / "cache" / "arxiv"
        self.collector = collector or ArxivCollector(
            page_size=config.daily.arxiv_page_size,
            max_results=config.daily.arxiv_max_results,
            request_delay_seconds=config.daily.arxiv_request_delay_seconds,
            cache_dir=cache_dir,
            cache_ttl_hours=config.daily.arxiv_cache_ttl_hours,
            rss_cache_ttl_minutes=config.daily.arxiv_rss_cache_ttl_minutes,
            api_id_batch_size=config.daily.arxiv_api_id_batch_size,
        )
        self.summary_provider = summary_provider
        self.rerank_provider = rerank_provider if rerank_provider is not None else summary_provider

    @property
    def enabled_topics(self) -> list[Any]:
        return [topic for topic in self.config.topics if topic.enabled]

    @property
    def categories(self) -> list[str]:
        values = {
            category
            for topic in self.enabled_topics
            for category in topic.arxiv_categories
            if category
        }
        return sorted(values)

    def initialize_runtime(self) -> dict[str, Any]:
        """Create the local PaperDaily storage and output directory."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.store.initialize()
        return {
            "database": str(self.config.database),
            "output_dir": str(self.config.output_dir),
            "user_id": self.config.user_id,
            "topic_count": len(self.enabled_topics),
        }

    def catchup_plan(self, *, today: date | None = None) -> CatchupPlan:
        state = self.store.get_state(self.config.user_id)
        last_completed = state.get("last_completed_window_end") if state else None
        planner = CatchupPlanner.from_config(self.config.catchup)
        target = default_target_date(today=today, timezone=self.config.timezone)
        first_gap = (
            self.store.first_uncovered_window(
                self.config.user_id,
                through_date=target,
            )
            if last_completed
            else None
        )
        return planner.plan(
            last_completed,
            target_date=target,
            next_gap_end=first_gap[1] if first_gap else None,
            timezone=self.config.timezone,
        )

    def select_window(
        self,
        plan: CatchupPlan,
        choice: str,
        *,
        custom_start: date | str | None = None,
        custom_end: date | str | None = None,
    ) -> DateWindow:
        return CatchupPlanner.from_config(self.config.catchup).select(
            plan,
            choice,
            custom_start=custom_start,
            custom_end=custom_end,
        )

    def _rank(
        self,
        fetch_result: ArxivFetchResult,
        *,
        limit: int,
        include_handled: bool,
        use_llm_rerank: bool,
    ) -> tuple[list[Recommendation], dict[str, Any]]:
        ranker = PaperRanker(
            self.enabled_topics,
            user_id=self.config.user_id,
            limit=limit,
            store=self.store,
            embedding_provider=self.embedding_provider,
            mmr_lambda=self.config.daily.mmr_lambda,
            include_handled=include_handled,
        )
        candidates = ranker.rank_candidates(fetch_result.papers, window_end=fetch_result.window_end)
        if use_llm_rerank:
            rerank = LLMReranker(
                topics=self.enabled_topics,
                store=self.store,
                provider=self.rerank_provider,
                enabled=self.config.daily.llm_rerank_enabled,
                weight=self.config.daily.llm_rerank_weight,
                max_tokens=self.config.daily.llm_rerank_max_tokens,
                input_cost_per_million_tokens=self.config.daily.llm_rerank_input_cost_per_million_tokens,
                output_cost_per_million_tokens=self.config.daily.llm_rerank_output_cost_per_million_tokens,
            ).rerank(candidates, candidate_limit=self.config.daily.rerank_limit)
            candidates = rerank.recommendations
            rerank_stats = rerank.stats
        else:
            rerank_stats = {
                "llm_rerank_enabled": self.config.daily.llm_rerank_enabled,
                "llm_rerank_status": "skipped_dry_run",
                "llm_rerank_candidate_count": 0,
                "llm_rerank_call_count": 0,
                "llm_rerank_cache_hits": 0,
                "llm_rerank_applied_count": 0,
                "llm_rerank_fallback_count": 0,
                "llm_rerank_prompt_tokens": 0,
                "llm_rerank_completion_tokens": 0,
                "llm_rerank_total_tokens": 0,
                "llm_rerank_estimated_cost_usd": 0.0,
                "llm_rerank_cost_pricing_configured": bool(
                    self.config.daily.llm_rerank_input_cost_per_million_tokens
                    or self.config.daily.llm_rerank_output_cost_per_million_tokens
                ),
            }
        recommendations = ranker.select(candidates, limit=limit)
        return recommendations, {**ranker.last_diagnostics, **rerank_stats}

    def _collect_window(self, window: DateWindow) -> ArxivFetchResult:
        """Use live RSS announcements for today and API date windows for history."""

        assert window.start_date is not None
        announcement_day = local_today(self.config.timezone)
        if not self.config.daily.arxiv_rss_enabled or window.end_date != announcement_day:
            return self.collector.fetch_window(window.start_date, window.end_date, self.categories)

        if window.start_date == announcement_day:
            return self.collector.fetch_announcements(
                announcement_day,
                self.categories,
                timezone_name=self.config.timezone,
                include_cross_list=self.config.daily.arxiv_rss_include_cross_list,
            )

        historical = self.collector.fetch_window(
            window.start_date,
            announcement_day - timedelta(days=1),
            self.categories,
        )
        announcements = self.collector.fetch_announcements(
            announcement_day,
            self.categories,
            timezone_name=self.config.timezone,
            include_cross_list=self.config.daily.arxiv_rss_include_cross_list,
        )
        return ArxivFetchResult(
            papers=deduplicate_papers([*historical.papers, *announcements.papers]),
            window_start=window.start_date,
            window_end=window.end_date,
            total_available=historical.total_available + announcements.total_available,
            network_requests=historical.network_requests + announcements.network_requests,
            cache_hits=historical.cache_hits + announcements.cache_hits,
            truncated=historical.truncated or announcements.truncated,
            source=f"{historical.source}+{announcements.source}",
        )

    @staticmethod
    def _store_recommendations(recommendations: Iterable[Recommendation]) -> list[dict[str, Any]]:
        return [
            {
                "rank": item.rank,
                "score": item.score,
                "topic_id": item.matched_topics[0] if item.matched_topics else None,
                "paper": item.paper,
                "metadata": {
                    "matched_topics": item.matched_topics,
                    "matched_terms": item.matched_terms,
                    "recommendation_reason": item.recommendation_reason,
                    "component_scores": item.component_scores,
                    "rerank": item.rerank,
                    "summary": item.summary.as_dict() if item.summary else None,
                },
            }
            for item in recommendations
        ]

    def _channels(
        self,
        requested: Iterable[str] | None,
        *,
        feishu_chat_id: str | None,
        feishu_user_id: str | None,
    ) -> list[Any]:
        names = [str(name).strip().lower() for name in (requested or self.config.daily.channels)]
        # Markdown is the durable local success boundary and is always present.
        ordered = ["markdown", *names]
        result: list[Any] = []
        seen: set[str] = set()
        for name in ordered:
            if not name or name in seen:
                continue
            seen.add(name)
            if name == "markdown":
                result.append(MarkdownChannel(self.config.output_dir / "digests"))
            elif name == "terminal":
                result.append(TerminalChannel())
            elif name == "feishu":
                result.append(
                    FeishuChannel(
                        chat_id=feishu_chat_id or os.environ.get("FEISHU_CHAT_ID"),
                        user_id=feishu_user_id or os.environ.get("FEISHU_USER_ID"),
                    )
                )
            else:
                raise ValueError(f"unsupported channel: {name}")
        return result

    def run(
        self,
        window: DateWindow,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        generate_summary: bool | None = None,
        channels: Iterable[str] | None = None,
        include_handled: bool = False,
        feishu_chat_id: str | None = None,
        feishu_user_id: str | None = None,
    ) -> RunOutcome:
        if window.is_empty or window.start_date is None:
            raise ValueError("selected window is empty")
        if not self.enabled_topics:
            raise ValueError("没有启用的话题，请先配置 topics")
        if not self.categories:
            raise ValueError("启用的话题没有配置 arXiv 分类")

        default_limit = (
            self.config.daily.default_limit
            if window.days == 1
            else self.config.catchup.max_papers_per_run
        )
        requested_limit = max(1, int(limit or default_limit))
        run_id: str | None = None
        mode = "daily" if window.days == 1 else "catchup"
        if not dry_run:
            run_id = self.store.start_run(
                self.config.user_id,
                window.start_date,
                window.end_date,
                mode=mode,
                metadata={"choice": window.choice, "categories": self.categories},
            )

        try:
            fetched = self._collect_window(window)
            # A capped Atom query is not a complete date window.  Completing
            # the run in that state would advance the watermark past papers
            # that were never considered, making the loss permanent on the
            # next incremental invocation.  Fail closed until the caller can
            # retry a smaller/split window (or a future collector performs the
            # split itself).
            if fetched.truncated:
                raise ArxivCollectorError(
                    "arXiv 查询达到安全上限，结果窗口不完整；"
                    "本次运行已停止，watermark 未推进。"
                    "请缩短 --window/--since 范围或提高 daily.arxiv_max_results 后重试。"
                )
            recommendations, ranking_stats = self._rank(
                fetched,
                limit=requested_limit,
                include_handled=include_handled,
                use_llm_rerank=not dry_run,
            )
            warnings: list[str] = []
            stats = {
                "fetched_count": len(fetched.papers),
                "arxiv_total_available": fetched.total_available,
                "network_requests": fetched.network_requests,
                "cache_hits": fetched.cache_hits,
                "arxiv_source": fetched.source,
                **ranking_stats,
            }
            digest = Digest(
                run_id=run_id or "00000000-0000-0000-0000-000000000000",
                user_id=self.config.user_id,
                window_start=window.start_date,
                window_end=window.end_date,
                recommendations=recommendations,
                generated_at=datetime.now(timezone.utc),
                stats=stats,
                catchup_mode=mode,
            )
            if dry_run:
                return RunOutcome(digest=digest, dry_run=True, warnings=warnings)

            assert run_id is not None
            should_summarize = (
                self.config.daily.generate_chinese_summary
                if generate_summary is None
                else bool(generate_summary)
            )
            summary_count = 0
            if should_summarize:
                summary_service = ChineseSummaryService(
                    store=self.store,
                    provider=self.summary_provider,
                    language=self.config.daily.summary_language,
                )
                summary_count = summary_service.summarize_recommendations(recommendations)

            self.store.save_recommendations(run_id, self._store_recommendations(recommendations))
            deliveries: list[DeliveryResult] = []
            local_markdown_succeeded = False
            for channel in self._channels(
                channels,
                feishu_chat_id=feishu_chat_id,
                feishu_user_id=feishu_user_id,
            ):
                self.store.ensure_delivery(run_id, channel.name)
                current = self.store.get_delivery(run_id, channel.name)
                if current and current.get("status") == "succeeded":
                    deliveries.append(
                        DeliveryResult(channel=channel.name, success=True, target="cached")
                    )
                    local_markdown_succeeded |= channel.name == "markdown"
                    continue
                self.store.mark_delivery_started(run_id, channel.name)
                result = channel.publish_digest(digest)
                deliveries.append(result)
                if result.success:
                    self.store.complete_delivery(
                        run_id,
                        channel.name,
                        external_id=result.target or None,
                    )
                    local_markdown_succeeded |= channel.name == "markdown"
                else:
                    self.store.fail_delivery(run_id, channel.name, result.error)
                    warnings.append(f"{channel.name} 发送失败：{result.error}")

            if not local_markdown_succeeded:
                raise RuntimeError("本地 Markdown 日报写入失败，watermark 未推进")

            self.store.complete_run(
                run_id,
                fetched_count=len(fetched.papers),
                candidate_count=int(ranking_stats.get("candidate_count", 0)),
                recommendation_count=len(recommendations),
                summary_count=summary_count,
                delivery_count=sum(1 for item in deliveries if item.success),
                metadata={
                    "warnings": warnings,
                    "output_path": str(digest.output_path or ""),
                    "run_summary": {
                        "matched_count": int(ranking_stats.get("matched_count", 0)),
                        "handled_count": int(ranking_stats.get("handled_count", 0)),
                        "candidate_count": int(ranking_stats.get("candidate_count", 0)),
                        "new_recommendation_count": len(recommendations),
                        "include_handled": bool(include_handled),
                    },
                },
            )
            return RunOutcome(
                digest=digest,
                dry_run=False,
                deliveries=deliveries,
                warnings=warnings,
            )
        except Exception as exc:
            if run_id is not None:
                with suppress(Exception):
                    self.store.fail_run(run_id, exc)
            raise

    def record_feedback(self, canonical_id: str, action: str) -> dict[str, Any]:
        action = str(action).strip().lower()
        allowed = {"interested", "irrelevant", "later", "saved", "read", "detailed", "reading_note"}
        if action not in allowed:
            raise ValueError(f"action must be one of: {', '.join(sorted(allowed))}")
        recommendation = self.store.find_latest_recommendation(self.config.user_id, canonical_id)
        metadata = dict((recommendation or {}).get("metadata") or {})
        run_id = (recommendation or {}).get("run_id")
        feedback = self.store.record_feedback(
            self.config.user_id,
            canonical_id,
            action,
            run_id=run_id,
            metadata={
                "matched_topics": metadata.get("matched_topics") or [],
                "rank": (recommendation or {}).get("rank"),
            },
        )

        return feedback


__all__ = ["PaperDailyService", "RunOutcome"]
