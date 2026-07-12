from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from paperdaily.storage import PaperDailyStore
from paperdaily.summaries import ChineseSummaryService


class _Provider:
    name = "unit-llm"
    model = "summary-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(
            text="""{
              "title_zh": "中文标题",
              "one_sentence_summary": "一句话摘要",
              "problem": "研究问题",
              "method": "核心方法",
              "contributions": ["贡献一", "贡献二"],
              "limitations_from_abstract": ["摘要未说明规模"],
              "recommendation_reason": "与 VLA 相关"
            }""",
            prompt_tokens=20,
            completion_tokens=30,
        )


def test_summary_uses_only_title_and_abstract_and_hits_cache(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    provider = _Provider()
    service = ChineseSummaryService(store=store, provider=provider)
    paper = {
        "arxiv_id": "2607.00001v2",
        "title": "A VLA paper",
        "abstract": "This abstract describes a robot policy.",
        "authors": ["PRIVATE AUTHOR SHOULD NOT ENTER PROMPT"],
        "secret": "PRIVATE METADATA SHOULD NOT ENTER PROMPT",
        "user_id": "PRIVATE USER SHOULD NOT ENTER PROMPT",
    }
    topic_a = "TOPIC A MUST NOT ENTER PROMPT"
    reason_a = "REASON A MUST NOT ENTER PROMPT"

    first = service.summarize(
        paper,
        matched_topics=[topic_a],
        recommendation_reason=reason_a,
    )
    second = service.summarize(
        paper,
        matched_topics=[topic_a],
        recommendation_reason=reason_a,
    )

    assert first.status == "completed"
    assert first.cached is False
    assert second.cached is True
    assert second.title_zh == "中文标题"
    assert first.recommendation_reason == ""
    assert len(provider.calls) == 1
    prompt, kwargs = provider.calls[0]
    assert paper["title"] in prompt
    assert paper["abstract"] in prompt
    assert paper["secret"] not in prompt
    assert paper["authors"][0] not in prompt
    assert paper["user_id"] not in prompt
    assert topic_a not in prompt
    assert reason_a not in prompt
    assert kwargs["temperature"] == 0.0
    cached = store.get_summary(*service._cache_key(paper))
    assert cached is not None
    assert "recommendation_reason" not in cached["payload"]


def test_summary_cache_is_invalidated_when_title_or_abstract_changes(tmp_path: Path) -> None:
    provider = _Provider()
    service = ChineseSummaryService(
        store=PaperDailyStore(tmp_path / "paperflow.db"),
        provider=provider,
    )
    paper = {"arxiv_id": "2607.00001", "title": "A paper", "abstract": "Version one"}

    service.summarize(paper)
    service.summarize({**paper, "title": "Corrected paper title"})
    service.summarize({**paper, "abstract": "Version two"})

    assert len(provider.calls) == 3


def test_context_free_summary_reuses_cache_without_cross_topic_or_user_leakage(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    provider = _Provider()
    service = ChineseSummaryService(store=store, provider=provider)
    paper = {
        "arxiv_id": "2607.00002",
        "title": "A robot paper",
        "abstract": "This paper studies robot policies.",
        "user_id": "first-user",
    }

    first = service.summarize(
        paper,
        matched_topics=["first user's private topic"],
        recommendation_reason="first user's ranking reason",
    )
    second = service.summarize(
        {**paper, "user_id": "second-user"},
        matched_topics=["second user's unrelated topic"],
        recommendation_reason="second user's ranking reason",
    )

    assert first.cached is False
    assert second.cached is True
    assert first.as_dict() == {**second.as_dict(), "cached": False}
    assert len(provider.calls) == 1
    prompt, _ = provider.calls[0]
    assert "first user's private topic" not in prompt
    assert "first user's ranking reason" not in prompt
    assert "first-user" not in prompt
    assert "second-user" not in prompt


def test_mock_provider_never_calls_generate(tmp_path: Path) -> None:
    class MockProvider:
        name = "mock"
        model = "mock-llm"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **kwargs: Any) -> Any:
            del prompt, kwargs
            self.calls += 1
            raise AssertionError("mock providers must never be invoked")

    provider = MockProvider()
    service = ChineseSummaryService(
        store=PaperDailyStore(tmp_path / "paperflow.db"),
        provider=provider,
    )

    summary = service.summarize(
        {"arxiv_id": "2607.00003", "title": "A paper", "abstract": "An abstract"},
        matched_topics=["VLA"],
        recommendation_reason="a reason",
    )

    assert summary.status == "provider_unconfigured"
    assert provider.calls == 0


def test_malformed_summary_returns_honest_failure_and_caches_no_success(tmp_path: Path) -> None:
    class BrokenProvider(_Provider):
        def generate(self, prompt: str, **kwargs: Any) -> Any:
            self.calls.append((prompt, kwargs))
            return SimpleNamespace(text="not-json")

    store = PaperDailyStore(tmp_path / "paperflow.db")
    provider = BrokenProvider()
    service = ChineseSummaryService(store=store, provider=provider)
    paper = {"arxiv_id": "2607.00001", "title": "A paper", "abstract": "An abstract"}

    first = service.summarize(paper)
    second = service.summarize(paper)

    assert first.status == "failed"
    assert first.contributions == []
    assert first.limitations_from_abstract
    assert second.status == "failed"
    assert len(provider.calls) == 2
