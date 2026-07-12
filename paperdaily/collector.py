"""Explicit-window, paginated arXiv collection for PaperDaily.

The original PaperFlow daily pipeline accepts a relative ``days`` value.  A
catch-up workflow needs a less ambiguous contract, so this module works with
inclusive ``date`` boundaries and reports when a configured safety cap
truncated a result set.
"""

from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .identifiers import canonicalize_arxiv_id, deduplicate_papers, normalize_arxiv_id

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPEN_SEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"


@dataclass(frozen=True)
class ArxivFetchResult:
    """Result and diagnostics for one inclusive arXiv window."""

    papers: list[dict[str, Any]]
    window_start: date
    window_end: date
    total_available: int
    network_requests: int
    cache_hits: int
    truncated: bool


class ArxivCollectorError(RuntimeError):
    """Raised when an arXiv request cannot be completed safely."""


def _element_text(element: ET.Element, path: str, namespaces: dict[str, str]) -> str:
    node = element.find(path, namespaces)
    return " ".join((node.text or "").split()) if node is not None and node.text else ""


def _parse_feed(xml_text: str) -> tuple[list[dict[str, Any]], int]:
    namespaces = {"atom": ATOM_NS, "arxiv": ARXIV_NS, "os": OPEN_SEARCH_NS}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivCollectorError(f"arXiv returned malformed XML: {exc}") from exc

    total_node = root.find("os:totalResults", namespaces)
    try:
        total = int((total_node.text or "0").strip()) if total_node is not None else 0
    except ValueError:
        total = 0

    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", namespaces):
        raw_identifier = _element_text(entry, "atom:id", namespaces)
        raw_identifier = raw_identifier.rsplit("/", 1)[-1]
        canonical_id = canonicalize_arxiv_id(raw_identifier)
        version = normalize_arxiv_id(raw_identifier)[1]
        authors = [
            _element_text(author, "atom:name", namespaces)
            for author in entry.findall("atom:author", namespaces)
        ]
        categories = [
            str(node.attrib.get("term") or "").strip()
            for node in entry.findall("atom:category", namespaces)
            if str(node.attrib.get("term") or "").strip()
        ]
        pdf_url = ""
        paper_url = ""
        for link in entry.findall("atom:link", namespaces):
            href = str(link.attrib.get("href") or "").strip()
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = href
            if link.attrib.get("rel") == "alternate":
                paper_url = href

        published = _element_text(entry, "atom:published", namespaces)
        updated = _element_text(entry, "atom:updated", namespaces)
        papers.append(
            {
                "arxiv_id": canonical_id,
                "raw_arxiv_id": raw_identifier,
                "arxiv_version": version,
                "title": _element_text(entry, "atom:title", namespaces),
                "authors": [author for author in authors if author],
                "abstract": _element_text(entry, "atom:summary", namespaces),
                "categories": categories,
                "publish_date": published[:10],
                "published_at": published,
                "updated_at": updated,
                "pdf_url": pdf_url or (f"https://arxiv.org/pdf/{canonical_id}" if canonical_id else ""),
                "paper_url": paper_url or (f"https://arxiv.org/abs/{canonical_id}" if canonical_id else ""),
                "url": paper_url or (f"https://arxiv.org/abs/{canonical_id}" if canonical_id else ""),
                "doi": _element_text(entry, "arxiv:doi", namespaces),
                "source": "arxiv",
            }
        )
    return papers, total


def _query_for(categories: Iterable[str], start: date, end: date) -> str:
    normalized = sorted({str(category).strip() for category in categories if str(category).strip()})
    category_query = " OR ".join(f"cat:{category}" for category in normalized)
    date_query = f"submittedDate:[{start:%Y%m%d}000000 TO {end:%Y%m%d}235959]"
    return f"({category_query}) AND {date_query}" if category_query else date_query


class ArxivCollector:
    """Fetch arXiv Atom pages with caching, retries, and a polite delay."""

    def __init__(
        self,
        *,
        endpoint: str = ARXIV_API_URL,
        page_size: int = 200,
        max_results: int = 5000,
        request_delay_seconds: float = 3.0,
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 24,
        user_agent: str = "PaperDaily/0.1 (+https://github.com/Cat-blizzard/PaperFlow)",
        requester: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint
        self.page_size = max(1, min(2000, int(page_size)))
        self.max_results = max(1, int(max_results))
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(1, int(max_retries))
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.cache_ttl = timedelta(hours=max(0, int(cache_ttl_hours)))
        self.user_agent = user_agent
        self._requester = requester or requests.get
        self._sleeper = sleeper

    def _cache_path(self, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.xml"

    def _read_cache(self, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - modified > self.cache_ttl:
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, path: Path | None, text: str) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _request_page(self, params: dict[str, Any]) -> tuple[str, bool]:
        cache_path = self._cache_path(params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached, True

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._requester(
                    self.endpoint,
                    params=params,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                text = str(response.text)
                self._write_cache(cache_path, text)
                return text, False
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    self._sleeper(min(2 ** attempt, 8))
        raise ArxivCollectorError(f"arXiv request failed after {self.max_retries} attempts: {last_error}")

    def fetch_window(self, start: date, end: date, categories: Iterable[str]) -> ArxivFetchResult:
        """Fetch all configured pages for an inclusive submitted-date window."""

        if start > end:
            raise ValueError("window start must be on or before window end")

        query = _query_for(categories, start, end)
        offset = 0
        papers: list[dict[str, Any]] = []
        total_available = 0
        network_requests = 0
        cache_hits = 0

        while offset < self.max_results:
            page_size = min(self.page_size, self.max_results - offset)
            params = {
                "search_query": query,
                "start": offset,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            xml_text, from_cache = self._request_page(params)
            cache_hits += int(from_cache)
            network_requests += int(not from_cache)
            page, reported_total = _parse_feed(xml_text)
            total_available = max(total_available, reported_total)
            papers.extend(page)
            offset += len(page)

            if not page or len(page) < page_size or offset >= total_available:
                break
            if not from_cache and self.request_delay_seconds:
                self._sleeper(self.request_delay_seconds)

        deduped = deduplicate_papers(papers)
        truncated = total_available > len(papers) and len(papers) >= self.max_results
        return ArxivFetchResult(
            papers=deduped,
            window_start=start,
            window_end=end,
            total_available=total_available,
            network_requests=network_requests,
            cache_hits=cache_hits,
            truncated=truncated,
        )

    def fetch_by_id(self, arxiv_id: str) -> dict[str, Any] | None:
        """Fetch one paper by canonical arXiv identifier."""

        canonical_id = canonicalize_arxiv_id(arxiv_id)
        if not canonical_id:
            raise ValueError(f"invalid arXiv identifier: {arxiv_id!r}")
        params = {
            "search_query": f"id:{canonical_id}",
            "start": 0,
            "max_results": 1,
        }
        xml_text, _ = self._request_page(params)
        papers, _ = _parse_feed(xml_text)
        return papers[0] if papers else None


__all__ = [
    "ARXIV_API_URL",
    "ArxivCollector",
    "ArxivCollectorError",
    "ArxivFetchResult",
]
