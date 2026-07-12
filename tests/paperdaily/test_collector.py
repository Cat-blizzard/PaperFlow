from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests

from paperdaily.collector import ArxivCollector, ArxivCollectorError


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def _feed(entries: list[dict[str, Any]], *, total: int) -> str:
    rendered: list[str] = []
    for entry in entries:
        rendered.append(
            """
            <entry>
              <id>https://arxiv.org/abs/{identifier}</id>
              <updated>{updated}</updated>
              <published>{published}</published>
              <title>{title}</title>
              <summary>{abstract}</summary>
              <author><name>{author}</name></author>
              <category term="{category}" />
              <link rel="alternate" href="https://arxiv.org/abs/{identifier}" />
              <link title="pdf" type="application/pdf"
                    href="https://arxiv.org/pdf/{identifier}" />
            </entry>
            """.format(
                identifier=entry["identifier"],
                updated=entry.get("updated", "2026-07-02T00:00:00Z"),
                published=entry.get("published", "2026-07-01T00:00:00Z"),
                title=entry.get("title", "A paper"),
                abstract=entry.get("abstract", "An abstract"),
                author=entry.get("author", "Ada Researcher"),
                category=entry.get("category", "cs.RO"),
            )
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>{total}</opensearch:totalResults>
      {''.join(rendered)}
    </feed>
    """


def test_fetch_window_paginates_deduplicates_versions_and_reports_requests() -> None:
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def requester(_url: str, **kwargs: Any) -> _Response:
        params = kwargs["params"]
        calls.append(dict(params))
        if params["start"] == 0:
            return _Response(
                _feed(
                    [
                        {"identifier": "2607.00001v1", "title": "Old title"},
                        {"identifier": "2607.00002v1", "title": "Other paper"},
                    ],
                    total=3,
                )
            )
        return _Response(
            _feed(
                [{"identifier": "2607.00001v2", "title": "Revised title"}],
                total=3,
            )
        )

    collector = ArxivCollector(
        page_size=2,
        max_results=10,
        request_delay_seconds=0.25,
        requester=requester,
        sleeper=sleeps.append,
    )
    result = collector.fetch_window(date(2026, 7, 1), date(2026, 7, 2), ["cs.RO", "cs.RO"])

    assert [call["start"] for call in calls] == [0, 2]
    assert "cat:cs.RO" in calls[0]["search_query"]
    assert "submittedDate:[20260701000000 TO 20260702235959]" in calls[0]["search_query"]
    assert result.total_available == 3
    assert result.network_requests == 2
    assert result.cache_hits == 0
    assert result.truncated is False
    assert len(result.papers) == 2
    revised = next(paper for paper in result.papers if paper["arxiv_id"] == "2607.00001")
    assert revised["arxiv_version"] == 2
    assert revised["title"] == "Revised title"
    assert sleeps == [0.25]


def test_fetch_window_uses_disk_cache_without_another_request(tmp_path: Path) -> None:
    calls = 0

    def requester(_url: str, **_kwargs: Any) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(_feed([{"identifier": "2607.00001v1"}], total=1))

    collector = ArxivCollector(
        page_size=10,
        cache_dir=tmp_path / "cache",
        cache_ttl_hours=24,
        request_delay_seconds=0,
        requester=requester,
    )
    first = collector.fetch_window(date(2026, 7, 1), date(2026, 7, 1), ["cs.RO"])
    second = collector.fetch_window(date(2026, 7, 1), date(2026, 7, 1), ["cs.RO"])

    assert calls == 1
    assert first.network_requests == 1
    assert second.network_requests == 0
    assert second.cache_hits == 1


def test_fetch_window_retries_request_errors_without_real_sleep() -> None:
    calls = 0
    sleeps: list[float] = []

    def requester(_url: str, **_kwargs: Any) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.ConnectionError("temporary")
        return _Response(_feed([{"identifier": "2607.00001v1"}], total=1))

    collector = ArxivCollector(
        max_retries=2,
        request_delay_seconds=0,
        requester=requester,
        sleeper=sleeps.append,
    )
    result = collector.fetch_window(date(2026, 7, 1), date(2026, 7, 1), ["cs.RO"])

    assert calls == 2
    assert sleeps == [1]
    assert len(result.papers) == 1


def test_fetch_window_rejects_invalid_window_and_malformed_xml() -> None:
    collector = ArxivCollector(
        request_delay_seconds=0,
        requester=lambda *_args, **_kwargs: _Response("not xml"),
    )

    with pytest.raises(ValueError, match="window start"):
        collector.fetch_window(date(2026, 7, 2), date(2026, 7, 1), ["cs.RO"])
    with pytest.raises(ArxivCollectorError, match="malformed XML"):
        collector.fetch_window(date(2026, 7, 1), date(2026, 7, 1), ["cs.RO"])
