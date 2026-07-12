from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from paperdaily.models import Recommendation
from paperdaily.reranker import LLMReranker
from paperdaily.storage import PaperDailyStore
from paperdaily.topics import Topic


def _topic() -> Topic:
    return Topic(
        id="embodied-vla",
        name="Embodied VLA",
        description="Vision-language-action robot policies",
        arxiv_categories=["cs.RO"],
        exact_phrases=["vision language action"],
        keywords=["VLA", "robot policy"],
        context_keywords=["robot"],
        minimum_score=0.2,
    )


def _recommendation(identifier: str, score: float) -> Recommendation:
    return Recommendation(
        rank=0,
        score=score,
        paper={
            "arxiv_id": identifier,
            "title": f"Paper {identifier}",
            "abstract": f"Abstract {identifier} describes a robot policy.",
            "authors": ["PRIVATE AUTHOR"],
            "secret": "PRIVATE METADATA",
        },
        matched_topics=["embodied-vla"],
        recommendation_reason="rule match",
    )


def _response(*identifiers: str) -> str:
    return json.dumps(
        {
            "rankings": [
                {
                    "arxiv_id": identifier,
                    "relevance": 90 if index == 0 else 40,
                    "is_target_domain": True,
                    "matched_topic_ids": ["embodied-vla"],
                    "paper_type": "method",
                    "reason_zh": "摘要明确讨论机器人策略。",
                    "uncertainty_zh": "摘要未说明实验细节。",
                }
                for index, identifier in enumerate(identifiers)
            ]
        },
        ensure_ascii=False,
    )


class _Provider:
    name = "unit-llm"
    model = "rerank-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(
            text=_response("2607.00001", "2607.00002"),
            prompt_tokens=120,
            completion_tokens=40,
        )


def test_mock_provider_makes_zero_calls_and_preserves_baseline_order() -> None:
    class MockProvider:
        name = "mock"
        model = "mock-llm"

        def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("mock provider must not be called")

    recommendations = [_recommendation("2607.00001", 0.8), _recommendation("2607.00002", 0.6)]
    outcome = LLMReranker(topics=[_topic()], provider=MockProvider()).rerank(
        recommendations,
        candidate_limit=2,
    )

    assert outcome.recommendations == recommendations
    assert [item.score for item in recommendations] == [0.8, 0.6]
    assert all(not item.rerank for item in recommendations)
    assert outcome.stats["llm_rerank_call_count"] == 0
    assert outcome.stats["llm_rerank_status"] == "provider_unconfigured"


def test_reranker_limits_prompt_to_title_abstract_topics_and_uses_cache(tmp_path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    provider = _Provider()
    baseline = [_recommendation("2607.00001", 0.4), _recommendation("2607.00002", 0.8)]

    first = LLMReranker(topics=[_topic()], store=store, provider=provider, weight=0.5).rerank(
        baseline,
        candidate_limit=2,
    )

    assert first.stats["llm_rerank_call_count"] == 1
    assert first.stats["llm_rerank_applied_count"] == 2
    assert first.stats["llm_rerank_prompt_tokens"] == 120
    assert first.stats["llm_rerank_completion_tokens"] == 40
    assert first.stats["llm_rerank_estimated_cost_usd"] == 0.0
    assert baseline[0].score == 0.65
    assert "LLM复核" in baseline[0].recommendation_reason
    assert baseline[0].rerank["cached"] is False
    prompt, kwargs = provider.calls[0]
    assert "Paper 2607.00001" in prompt
    assert "Abstract 2607.00001" in prompt
    assert "Embodied VLA" in prompt
    assert "PRIVATE AUTHOR" not in prompt
    assert "PRIVATE METADATA" not in prompt
    assert kwargs["temperature"] == 0.0

    repeated = [_recommendation("2607.00001", 0.4), _recommendation("2607.00002", 0.8)]
    second = LLMReranker(topics=[_topic()], store=store, provider=provider, weight=0.5).rerank(
        repeated,
        candidate_limit=2,
    )

    assert len(provider.calls) == 1
    assert second.stats["llm_rerank_call_count"] == 0
    assert second.stats["llm_rerank_cache_hits"] == 2
    assert repeated[0].score == baseline[0].score
    assert repeated[0].rerank["cached"] is True


def test_malformed_response_falls_back_without_changing_scores(tmp_path) -> None:
    class BrokenProvider(_Provider):
        def generate(self, prompt: str, **kwargs: Any) -> Any:
            self.calls.append((prompt, kwargs))
            return SimpleNamespace(text="not-json", prompt_tokens=11, completion_tokens=0)

    provider = BrokenProvider()
    recommendations = [_recommendation("2607.00001", 0.8), _recommendation("2607.00002", 0.6)]
    outcome = LLMReranker(
        topics=[_topic()],
        store=PaperDailyStore(tmp_path / "paperflow.db"),
        provider=provider,
    ).rerank(recommendations, candidate_limit=2)

    assert [item.score for item in recommendations] == [0.8, 0.6]
    assert outcome.stats["llm_rerank_status"] == "failed_fallback"
    assert outcome.stats["llm_rerank_fallback_count"] == 2
    assert outcome.stats["llm_rerank_call_count"] == 1
