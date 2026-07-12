from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from paperdaily.ranking import PaperRanker
from paperdaily.topics import Topic


class _HashEmbedding:
    name = "hash"
    model = "unit-test"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1)] for index, _text in enumerate(texts)]


def _topic(*, daily_limit: int = 12) -> Topic:
    return Topic(
        id="embodied-vla",
        name="Embodied VLA",
        description="Vision-language-action robot policies",
        arxiv_categories=["cs.RO"],
        exact_phrases=["vision language action"],
        keywords=["VLA", "robot policy"],
        context_keywords=["robot"],
        daily_limit=daily_limit,
        minimum_score=0.2,
    )


def _paper(identifier: str, title: str) -> dict[str, Any]:
    return {
        "arxiv_id": identifier,
        "title": title,
        "abstract": "A robot policy for manipulation.",
        "publish_date": "2026-07-10",
        "authors": ["Ada"],
    }


def test_rank_excludes_handled_papers_by_default() -> None:
    store = SimpleNamespace(list_handled_canonical_ids=lambda _user_id: {"2607.00001"})
    ranker = PaperRanker(
        [_topic()],
        user_id="alice",
        store=store,
        embedding_provider=_HashEmbedding(),
        profile_loader=lambda _user_id: None,
    )

    recommendations = ranker.rank(
        [
            _paper("2607.00001v2", "Vision Language Action for a robot"),
            _paper("2607.00002v1", "A VLA robot policy"),
        ],
        window_end=date(2026, 7, 11),
    )

    assert [item.canonical_id for item in recommendations] == ["2607.00002"]
    assert ranker.last_diagnostics["handled_count"] == 1


def test_rank_can_explicitly_include_handled_papers() -> None:
    store = SimpleNamespace(list_handled_canonical_ids=lambda _user_id: {"2607.00001"})
    ranker = PaperRanker(
        [_topic()],
        user_id="alice",
        store=store,
        embedding_provider=_HashEmbedding(),
        profile_loader=lambda _user_id: None,
        include_handled=True,
    )

    recommendations = ranker.rank(
        [_paper("2607.00001v2", "Vision Language Action for a robot")],
        window_end=date(2026, 7, 11),
    )

    assert [item.canonical_id for item in recommendations] == ["2607.00001"]
    assert ranker.last_diagnostics["handled_count"] == 0


def test_rank_applies_topic_quota_and_returns_explainable_components() -> None:
    ranker = PaperRanker(
        [_topic(daily_limit=1)],
        user_id="alice",
        limit=3,
        embedding_provider=_HashEmbedding(),
        profile_loader=lambda _user_id: None,
    )
    recommendations = ranker.rank(
        [
            _paper("2607.00001", "Vision Language Action for a robot"),
            _paper("2607.00002", "A VLA robot policy"),
        ],
        window_end=date(2026, 7, 11),
    )

    assert len(recommendations) == 1
    assert recommendations[0].rank == 1
    assert recommendations[0].matched_topics == ["embodied-vla"]
    assert set(recommendations[0].component_scores) == {
        "topic_rule",
        "topic_semantic",
        "profile_semantic",
        "freshness",
        "quality",
        "feedback",
    }
    assert ranker.last_diagnostics["semantic_enabled"] is False
