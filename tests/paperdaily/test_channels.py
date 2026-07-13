from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperdaily.channels import FeishuChannel, MarkdownChannel, TerminalChannel
from paperdaily.models import ChineseSummary, Digest, Recommendation


def _digest() -> Digest:
    recommendation = Recommendation(
        rank=1,
        score=0.9,
        paper={
            "arxiv_id": "2607.00001",
            "title": "A VLA paper",
            "abstract": "A robot policy.",
            "paper_url": "https://arxiv.org/abs/2607.00001",
            "pdf_url": "https://arxiv.org/pdf/2607.00001",
        },
        matched_topics=["embodied-vla"],
        recommendation_reason="topic match",
    )
    return Digest(
        run_id="12345678-1234-1234-1234-123456789012",
        user_id="alice",
        window_start=date(2026, 7, 10),
        window_end=date(2026, 7, 10),
        recommendations=[recommendation],
        generated_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
        stats={"fetched_count": 10, "matched_count": 2},
    )


def test_markdown_and_terminal_channels_publish_locally(tmp_path: Path) -> None:
    digest = _digest()
    markdown = MarkdownChannel(tmp_path)
    captured: list[str] = []

    markdown_result = markdown.publish_digest(digest)
    terminal_result = TerminalChannel(writer=captured.append).publish_digest(digest)

    assert markdown_result.success is True
    assert digest.output_path is not None and digest.output_path.exists()
    content = digest.output_path.read_text(encoding="utf-8")
    assert "2607.00001" in content
    assert "https://hjfy.top/" in content
    assert terminal_result.success is True
    assert captured == [content]


def test_digest_keeps_the_original_arxiv_title_when_a_chinese_summary_exists() -> None:
    digest = _digest()
    digest.recommendations[0].summary = ChineseSummary(
        title_zh="Translated title",
        one_sentence_summary="Chinese summary remains available.",
    )

    from paperdaily.channels import render_digest_markdown

    rendered = render_digest_markdown(digest)

    assert "## 1. A VLA paper" in rendered
    assert "Translated title" not in rendered
    assert "Chinese summary remains available." in rendered


def test_feishu_without_target_is_a_non_throwing_optional_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    reporter = SimpleNamespace(
        send_text_to_chat=lambda target, text: calls.append((target, text)),
        send_daily_push=lambda target, text: calls.append((target, text)),
    )
    monkeypatch.setattr("paperdaily.channels.importlib.import_module", lambda _name: reporter)

    result = FeishuChannel().publish_digest(_digest())

    assert result.success is False
    assert result.channel == "feishu"
    assert result.error
    assert calls == []


def test_feishu_import_or_send_error_is_returned_instead_of_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_name: str) -> None:
        raise ImportError("feishu module unavailable")

    monkeypatch.setattr("paperdaily.channels.importlib.import_module", unavailable)

    result = FeishuChannel(chat_id="chat-1").publish_digest(_digest())

    assert result.success is False
    assert "unavailable" in result.error
