"""arXiv identifier normalization and paper-level deduplication helpers.

The arXiv Atom API commonly returns versioned identifiers while links copied
from browsers often omit the version or include ``/abs`` / ``/pdf`` prefixes.
PaperDaily uses the version-free identifier as its stable identity and keeps
the version as separate metadata.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urlsplit

_MODERN_ID_RE = re.compile(r"^(?P<canonical>\d{4}\.\d{4,5})(?:v(?P<version>\d+))?$", re.IGNORECASE)
_LEGACY_ID_RE = re.compile(
    r"^(?P<archive>[a-z][a-z0-9.-]*)/(?P<number>\d{7})(?:v(?P<version>\d+))?$",
    re.IGNORECASE,
)
_ARXIV_HOST_RE = re.compile(r"^(?:www\.|export\.)?arxiv\.org$", re.IGNORECASE)
_VERSION_RE = re.compile(r"^v?(?P<version>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ArxivIdentity:
    """A parsed arXiv identity.

    ``canonical_id`` never contains a version suffix. ``version`` is ``None``
    when the supplied value did not specify one.
    """

    canonical_id: str
    version: int | None = None

    @property
    def versioned_id(self) -> str:
        if self.version is None:
            return self.canonical_id
        return f"{self.canonical_id}v{self.version}"


def _identifier_candidate(value: Any) -> str:
    if value is None:
        return ""
    text = unquote(str(value)).strip().strip("<>\"'")
    if not text:
        return ""

    # Scheme-less arxiv.org links are common in copied terminal output.
    parsed_text = text
    if re.match(r"^(?:www\.|export\.)?arxiv\.org/", parsed_text, re.IGNORECASE):
        parsed_text = f"https://{parsed_text}"

    parsed = urlsplit(parsed_text)
    if parsed.scheme and _ARXIV_HOST_RE.match(parsed.netloc.split("@")[-1].split(":")[0]):
        text = parsed.path
    else:
        # Query strings and fragments are never part of an arXiv identifier.
        text = text.split("#", 1)[0].split("?", 1)[0]

    text = text.strip()
    text = re.sub(r"^arxiv\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.lstrip("/")
    text = re.sub(r"^(?:abs|pdf|src|format)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    return text.strip().strip("/")


def parse_arxiv_id(value: Any) -> ArxivIdentity | None:
    """Parse an arXiv ID or arxiv.org URL.

    Both modern identifiers (``2607.12345v2``) and legacy identifiers
    (``hep-th/9901001v3``) are supported. Invalid values return ``None``.
    """

    candidate = _identifier_candidate(value)
    if not candidate:
        return None

    modern = _MODERN_ID_RE.fullmatch(candidate)
    if modern:
        version = int(modern.group("version")) if modern.group("version") else None
        if version is not None and version < 1:
            return None
        return ArxivIdentity(modern.group("canonical"), version)

    legacy = _LEGACY_ID_RE.fullmatch(candidate)
    if legacy:
        version = int(legacy.group("version")) if legacy.group("version") else None
        if version is not None and version < 1:
            return None
        canonical = f"{legacy.group('archive').lower()}/{legacy.group('number')}"
        return ArxivIdentity(canonical, version)

    return None


def normalize_arxiv_id(value: Any) -> tuple[str, int | None]:
    """Return ``(canonical_id, version)`` or ``("", None)`` when invalid."""

    identity = parse_arxiv_id(value)
    if identity is None:
        return "", None
    return identity.canonical_id, identity.version


def canonicalize_arxiv_id(value: Any) -> str:
    """Return the stable, version-free arXiv ID or an empty string."""

    return normalize_arxiv_id(value)[0]


def arxiv_version(value: Any) -> int | None:
    """Return the explicit arXiv version from a value, if present."""

    return normalize_arxiv_id(value)[1]


def require_arxiv_identity(value: Any) -> ArxivIdentity:
    """Parse ``value`` and raise ``ValueError`` when it is not an arXiv ID."""

    identity = parse_arxiv_id(value)
    if identity is None:
        raise ValueError(f"Invalid arXiv identifier: {value!r}")
    return identity


def _coerce_version(value: Any) -> int | None:
    if value in (None, ""):
        return None
    match = _VERSION_RE.fullmatch(str(value).strip())
    if not match:
        return None
    version = int(match.group("version"))
    return version if version >= 1 else None


def paper_arxiv_identity(paper: Mapping[str, Any]) -> ArxivIdentity | None:
    """Resolve an arXiv identity from common paper metadata fields."""

    candidates = (
        paper.get("arxiv_id"),
        paper.get("arxiv_url"),
        paper.get("paper_url"),
        paper.get("pdf_url"),
        paper.get("url"),
        paper.get("canonical_arxiv_id"),
        paper.get("canonical_id"),
        paper.get("id") if isinstance(paper.get("id"), str) else None,
    )
    for candidate in candidates:
        identity = parse_arxiv_id(candidate)
        if identity is None:
            continue
        supplied_version = _coerce_version(paper.get("arxiv_version") or paper.get("version"))
        if identity.version is None and supplied_version is not None:
            return ArxivIdentity(identity.canonical_id, supplied_version)
        return identity
    return None


def _updated_timestamp(paper: Mapping[str, Any]) -> float:
    for key in ("updated", "updated_at", "updated_date", "last_updated", "published", "publish_date"):
        raw = paper.get(key)
        if raw in (None, ""):
            continue
        text = str(raw).strip()
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).timestamp()
    return float("-inf")


def _paper_preference_key(paper: Mapping[str, Any], identity: ArxivIdentity) -> tuple[int, float]:
    return identity.version or 0, _updated_timestamp(paper)


def deduplicate_papers_by_arxiv_id(papers: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate papers by canonical arXiv ID.

    The highest explicit version wins. For equal versions, the newest
    ``updated``-style timestamp wins. Papers without a recognizable arXiv ID
    are retained individually. Input mappings are never mutated, and output
    order follows the first occurrence of each canonical identifier.
    """

    result: list[dict[str, Any]] = []
    index_by_canonical: dict[str, int] = {}
    preference_by_canonical: dict[str, tuple[int, float]] = {}

    for source in papers:
        paper = dict(source)
        identity = paper_arxiv_identity(paper)
        if identity is None:
            result.append(paper)
            continue

        paper["canonical_arxiv_id"] = identity.canonical_id
        paper["arxiv_version"] = identity.version
        preference = _paper_preference_key(paper, identity)
        existing_index = index_by_canonical.get(identity.canonical_id)
        if existing_index is None:
            index_by_canonical[identity.canonical_id] = len(result)
            preference_by_canonical[identity.canonical_id] = preference
            result.append(paper)
            continue

        if preference > preference_by_canonical[identity.canonical_id]:
            result[existing_index] = paper
            preference_by_canonical[identity.canonical_id] = preference

    return result


# Short alias for call sites that already operate solely on paper mappings.
deduplicate_papers = deduplicate_papers_by_arxiv_id


__all__ = [
    "ArxivIdentity",
    "arxiv_version",
    "canonicalize_arxiv_id",
    "deduplicate_papers",
    "deduplicate_papers_by_arxiv_id",
    "normalize_arxiv_id",
    "paper_arxiv_identity",
    "parse_arxiv_id",
    "require_arxiv_identity",
]
