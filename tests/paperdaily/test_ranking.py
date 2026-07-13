from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from paperdaily.ranking import PaperRanker
from paperdaily.storage import PaperDailyStore
from paperdaily.topics import Topic


class _HashEmbedding:
    name = "hash"
    model = "unit-test"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1)] for index, _text in enumerate(texts)]


class _SemanticEmbedding:
    name = "sentence_transformers"
    model = "semantic-unit-test"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.casefold()
            if normalized.startswith("research topic:") or "vision language action" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "future visual observations" in normalized:
                vectors.append([0.98, 0.02, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class _FailingEmbedding:
    name = "openai"
    model = "unavailable-unit-test"
    dimensions = 3

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")


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
        "freshness",
        "quality",
        "feedback",
    }
    assert ranker.last_diagnostics["semantic_enabled"] is False


def test_zero_limits_keep_every_matching_paper() -> None:
    ranker = PaperRanker(
        [_topic(daily_limit=0)],
        user_id="alice",
        limit=0,
        embedding_provider=_HashEmbedding(),
    )

    recommendations = ranker.rank(
        [
            _paper("2607.00001", "Vision Language Action for a robot"),
            _paper("2607.00002", "A VLA robot policy"),
            _paper("2607.00003", "Vision Language Action for robot learning"),
        ],
        window_end=date(2026, 7, 11),
    )

    assert len(recommendations) == 3
    assert [item.rank for item in recommendations] == [1, 2, 3]


def test_semantic_recall_rescues_an_abstract_without_literal_keywords() -> None:
    provider = _SemanticEmbedding()
    ranker = PaperRanker(
        [_topic(daily_limit=0)],
        user_id="alice",
        embedding_provider=provider,
        semantic_recall_threshold=0.8,
    )
    semantic_paper = {
        **_paper("2607.00002", "Learning Predictive Representations for Control"),
        "abstract": (
            "The policy forecasts future visual observations and uses the predicted "
            "scene evolution to select actions."
        ),
    }
    unrelated = {
        **_paper("2607.00003", "Database Query Optimization"),
        "abstract": "We improve relational query planning and database indexes.",
    }

    recommendations = ranker.rank(
        [
            _paper("2607.00001", "Vision Language Action for robot manipulation"),
            semantic_paper,
            unrelated,
        ],
        window_end=date(2026, 7, 11),
    )

    by_id = {item.canonical_id: item for item in recommendations}
    assert set(by_id) == {"2607.00001", "2607.00002"}
    rescued = by_id["2607.00002"]
    assert rescued.matched_topics == ["embodied-vla"]
    assert rescued.matched_terms == []
    assert rescued.paper["semantic_recall"] is True
    assert rescued.component_scores["topic_rule"] == 0.0
    assert rescued.component_scores["topic_semantic"] > 0.9
    assert "摘要语义命中" in rescued.recommendation_reason
    assert ranker.last_diagnostics["rule_matched_count"] == 1
    assert ranker.last_diagnostics["semantic_recalled_count"] == 1
    assert ranker.last_diagnostics["semantic_recall_considered_count"] == 3


def test_negative_keyword_blocks_semantic_rescue_for_that_topic() -> None:
    provider = _SemanticEmbedding()
    topic = _topic(daily_limit=0)
    topic.negative_keywords = ["wireless access management"]
    ranker = PaperRanker(
        [topic],
        user_id="alice",
        embedding_provider=provider,
        semantic_recall_threshold=0.8,
    )
    paper = {
        **_paper("2607.00004", "Adaptive Network Control"),
        "abstract": (
            "Wireless access management forecasts future visual observations "
            "for network resource allocation."
        ),
    }

    assert ranker.rank([paper], window_end=date(2026, 7, 11)) == []
    assert "wireless access management" not in provider.calls[0][0].casefold()


def test_embedding_failure_falls_back_to_rule_recall() -> None:
    ranker = PaperRanker(
        [_topic(daily_limit=0)],
        user_id="alice",
        embedding_provider=_FailingEmbedding(),
        semantic_recall_threshold=0.8,
    )

    recommendations = ranker.rank(
        [
            _paper("2607.00001", "Vision Language Action for robot manipulation"),
            {
                **_paper("2607.00002", "Learning Predictive Representations for Control"),
                "abstract": "The policy forecasts future visual observations to select actions.",
            },
        ],
        window_end=date(2026, 7, 11),
    )

    assert [item.canonical_id for item in recommendations] == ["2607.00001"]
    assert ranker.last_diagnostics["rule_matched_count"] == 1
    assert ranker.last_diagnostics["semantic_recalled_count"] == 0
    assert ranker.last_diagnostics["semantic_enabled"] is False


def test_semantic_embeddings_are_reused_from_sqlite_cache(tmp_path) -> None:
    provider = _SemanticEmbedding()
    store = PaperDailyStore(tmp_path / "paperflow.db")
    papers = [
        _paper("2607.00001", "Vision Language Action for robot manipulation"),
        {
            **_paper("2607.00002", "Learning Predictive Representations for Control"),
            "abstract": "The policy forecasts future visual observations to select actions.",
        },
    ]

    first = PaperRanker(
        [_topic(daily_limit=0)],
        user_id="alice",
        store=store,
        embedding_provider=provider,
        semantic_recall_threshold=0.8,
    )
    first.rank(papers, window_end=date(2026, 7, 11))
    assert len(provider.calls) == 2
    assert first.last_diagnostics["embedding_call_count"] == 2

    second = PaperRanker(
        [_topic(daily_limit=0)],
        user_id="alice",
        store=store,
        embedding_provider=provider,
        semantic_recall_threshold=0.8,
    )
    second.rank(papers, window_end=date(2026, 7, 11))

    assert len(provider.calls) == 2
    assert second.last_diagnostics["embedding_call_count"] == 0
    assert second.last_diagnostics["embedding_cache_hits"] == 3
