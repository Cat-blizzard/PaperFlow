"""Research-topic configuration and explainable paper matching.

The matcher is deliberately deterministic and inexpensive.  It is intended for
the first recall stage, before embeddings or an LLM reranker are invoked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_AMBIGUOUS_ACRONYMS = {"vla", "wam"}
_ACRONYM_EXPANSIONS = {
    "vla": ("vision-language-action", "vision language action"),
    "wam": ("world-action model", "world action model"),
}


class TopicConfigError(ValueError):
    """Raised when a topic definition is incomplete or invalid."""


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TopicConfigError(f"{field_name} must be a string or list of strings")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _normalize_text(value: Any) -> str:
    """Normalize punctuation so hyphenated and spaced phrases are equivalent."""

    return re.sub(r"[^\w]+", " ", str(value or "").casefold(), flags=re.UNICODE).strip()


def _english_word_forms(token: str) -> set[str]:
    """Return conservative singular/plural forms for an English token."""

    forms = {token}
    if len(token) < 4 or not token.isascii() or not token.isalpha():
        return forms
    if token.endswith("y") and token[-2] not in "aeiou":
        forms.add(f"{token[:-1]}ies")
    elif token.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(f"{token}es")
    else:
        forms.add(f"{token}s")
    return forms


def _token_matches(text_token: str, term_token: str) -> bool:
    if text_token == term_token:
        return True
    # Match both directions so a user-entered plural also finds the singular.
    return text_token in _english_word_forms(term_token) or term_token in _english_word_forms(text_token)


def _contains(normalized_text: str, term: str) -> bool:
    normalized_term = _normalize_text(term)
    if not normalized_term or not normalized_text:
        return False
    text_tokens = normalized_text.split()
    term_tokens = normalized_term.split()
    if len(term_tokens) > len(text_tokens):
        return False
    return any(
        all(
            _token_matches(text_tokens[start + offset], term_token)
            for offset, term_token in enumerate(term_tokens)
        )
        for start in range(len(text_tokens) - len(term_tokens) + 1)
    )


@dataclass
class Topic:
    """One user-configurable research subscription."""

    id: str
    name: str
    description: str = ""
    enabled: bool = True
    arxiv_categories: list[str] = field(default_factory=list)
    exact_phrases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    context_keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    # ``0`` means that this topic has no per-digest quota.
    daily_limit: int = 0
    minimum_score: float = 0.25

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.name = str(self.name or "").strip()
        self.description = str(self.description or "").strip()
        if not self.id:
            raise TopicConfigError("topic id is required")
        if not self.name:
            raise TopicConfigError(f"topic {self.id!r} requires a name")

        self.arxiv_categories = _string_list(self.arxiv_categories, field_name="arxiv_categories")
        self.exact_phrases = _string_list(self.exact_phrases, field_name="exact_phrases")
        self.keywords = _string_list(self.keywords, field_name="keywords")
        self.context_keywords = _string_list(self.context_keywords, field_name="context_keywords")
        self.negative_keywords = _string_list(self.negative_keywords, field_name="negative_keywords")

        self.daily_limit = int(self.daily_limit)
        self.minimum_score = float(self.minimum_score)
        if self.daily_limit < 0:
            raise TopicConfigError(f"topic {self.id!r} daily_limit must be zero or positive")
        if not 0.0 <= self.minimum_score <= 1.0:
            raise TopicConfigError(f"topic {self.id!r} minimum_score must be between 0 and 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Topic:
        if not isinstance(data, Mapping):
            raise TopicConfigError("each topic must be a mapping")
        return cls(
            id=data.get("id", ""),
            name=data.get("name") or data.get("id", ""),
            description=data.get("description", ""),
            enabled=bool(data.get("enabled", True)),
            arxiv_categories=data.get("arxiv_categories", data.get("categories", [])),
            exact_phrases=data.get("exact_phrases", data.get("phrases", [])),
            keywords=data.get("keywords", []),
            context_keywords=data.get("context_keywords", data.get("context", [])),
            negative_keywords=data.get("negative_keywords", []),
            daily_limit=data.get("daily_limit", 0),
            minimum_score=data.get("minimum_score", 0.25),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "arxiv_categories": list(self.arxiv_categories),
            "exact_phrases": list(self.exact_phrases),
            "keywords": list(self.keywords),
            "context_keywords": list(self.context_keywords),
            "negative_keywords": list(self.negative_keywords),
            "daily_limit": self.daily_limit,
            "minimum_score": self.minimum_score,
        }


@dataclass
class TopicScore:
    """Per-topic evidence used to build an aggregate match."""

    topic_id: str
    topic_name: str
    score: float
    matched_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "negative_terms": list(self.negative_terms),
            "passed": self.passed,
        }


@dataclass
class TopicMatch:
    """Explainable aggregate result for one paper."""

    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    topic_names: list[str] = field(default_factory=list)
    topic_scores: dict[str, float] = field(default_factory=dict)
    negative_terms: list[str] = field(default_factory=list)
    details: list[TopicScore] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return bool(self.topics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "topics": list(self.topics),
            "topic_names": list(self.topic_names),
            "topic_scores": dict(self.topic_scores),
            "negative_terms": list(self.negative_terms),
            "details": [detail.to_dict() for detail in self.details],
        }


class TopicMatcher:
    """Match titles and abstracts against a collection of :class:`Topic` objects.

    Title evidence is always worth more than the equivalent abstract evidence.
    Full exact phrases are unambiguous and therefore match directly.  The short
    acronyms ``VLA`` and ``WAM`` are accepted only when a configured context
    keyword is also present, unless their expanded phrase is present.
    """

    EXACT_TITLE_WEIGHT = 0.70
    EXACT_ABSTRACT_WEIGHT = 0.50
    KEYWORD_TITLE_WEIGHT = 0.38
    # A clear keyword found in the abstract should meet the default 0.25
    # threshold by itself. Ambiguous acronyms still require topic context.
    KEYWORD_ABSTRACT_WEIGHT = 0.25
    CONTEXT_TITLE_WEIGHT = 0.10
    CONTEXT_ABSTRACT_WEIGHT = 0.05

    def __init__(self, topics: Iterable[Topic | Mapping[str, Any]]) -> None:
        parsed: list[Topic] = []
        seen: set[str] = set()
        for item in topics:
            topic = item if isinstance(item, Topic) else Topic.from_dict(item)
            if topic.id in seen:
                raise TopicConfigError(f"duplicate topic id: {topic.id}")
            seen.add(topic.id)
            parsed.append(topic)
        self.topics = parsed

    @staticmethod
    def _matched_locations(title: str, abstract: str, term: str) -> tuple[bool, bool]:
        return _contains(title, term), _contains(abstract, term)

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = str(value).casefold()
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    @staticmethod
    def _exact_phrases(topic: Topic) -> list[str]:
        """Expand common acronyms so older keyword-only topics still match."""

        expanded = [
            phrase
            for keyword in topic.keywords
            for phrase in _ACRONYM_EXPANSIONS.get(_normalize_text(keyword), ())
        ]
        return TopicMatcher._dedupe([*topic.exact_phrases, *expanded])

    def _score_topic(self, topic: Topic, title: str, abstract: str, threshold: float) -> TopicScore:
        negative_terms = [
            term
            for term in topic.negative_keywords
            if _contains(title, term) or _contains(abstract, term)
        ]
        if negative_terms:
            return TopicScore(
                topic_id=topic.id,
                topic_name=topic.name,
                score=0.0,
                negative_terms=self._dedupe(negative_terms),
                passed=False,
            )

        context_locations: dict[str, tuple[bool, bool]] = {
            term: self._matched_locations(title, abstract, term) for term in topic.context_keywords
        }
        has_context = any(in_title or in_abstract for in_title, in_abstract in context_locations.values())

        contributions: dict[str, tuple[str, float]] = {}

        def add(term: str, score: float) -> None:
            normalized = _normalize_text(term)
            previous = contributions.get(normalized)
            if previous is None or score > previous[1]:
                contributions[normalized] = (term, score)

        for phrase in self._exact_phrases(topic):
            in_title, in_abstract = self._matched_locations(title, abstract, phrase)
            if not (in_title or in_abstract):
                continue
            if _normalize_text(phrase) in _AMBIGUOUS_ACRONYMS and not has_context:
                continue
            add(phrase, self.EXACT_TITLE_WEIGHT if in_title else self.EXACT_ABSTRACT_WEIGHT)

        for keyword in topic.keywords:
            in_title, in_abstract = self._matched_locations(title, abstract, keyword)
            if not (in_title or in_abstract):
                continue
            if _normalize_text(keyword) in _AMBIGUOUS_ACRONYMS and not has_context:
                continue
            add(keyword, self.KEYWORD_TITLE_WEIGHT if in_title else self.KEYWORD_ABSTRACT_WEIGHT)

        if contributions:
            for context, (in_title, in_abstract) in context_locations.items():
                if in_title:
                    add(context, self.CONTEXT_TITLE_WEIGHT)
                elif in_abstract:
                    add(context, self.CONTEXT_ABSTRACT_WEIGHT)

        score = min(1.0, sum(value[1] for value in contributions.values()))
        matched_terms = [value[0] for value in contributions.values()]
        return TopicScore(
            topic_id=topic.id,
            topic_name=topic.name,
            score=round(score, 6),
            matched_terms=matched_terms,
            passed=bool(matched_terms) and score >= threshold,
        )

    def match(self, title: str, abstract: str = "", *, minimum_score: float | None = None) -> TopicMatch:
        normalized_title = _normalize_text(title)
        normalized_abstract = _normalize_text(abstract)
        details: list[TopicScore] = []

        for topic in self.topics:
            if not topic.enabled:
                continue
            threshold = topic.minimum_score if minimum_score is None else float(minimum_score)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("minimum_score must be between 0 and 1")
            details.append(self._score_topic(topic, normalized_title, normalized_abstract, threshold))

        passed = [detail for detail in details if detail.passed]
        matched_terms = self._dedupe(term for detail in passed for term in detail.matched_terms)
        negative_terms = self._dedupe(term for detail in details for term in detail.negative_terms)
        return TopicMatch(
            score=max((detail.score for detail in passed), default=0.0),
            matched_terms=matched_terms,
            topics=[detail.topic_id for detail in passed],
            topic_names=[detail.topic_name for detail in passed],
            topic_scores={detail.topic_id: detail.score for detail in passed},
            negative_terms=negative_terms,
            details=details,
        )

    def match_paper(self, paper: Mapping[str, Any], *, minimum_score: float | None = None) -> TopicMatch:
        return self.match(
            str(paper.get("title") or ""),
            str(paper.get("abstract") or paper.get("summary") or ""),
            minimum_score=minimum_score,
        )

    def annotate(self, paper: Mapping[str, Any], *, minimum_score: float | None = None) -> dict[str, Any]:
        result = dict(paper)
        match = self.match_paper(paper, minimum_score=minimum_score)
        result["topic_match"] = match.to_dict()
        result["topic_score"] = match.score
        result["matched_topics"] = list(match.topics)
        result["matched_terms"] = list(match.matched_terms)
        return result

    def filter(
        self,
        papers: Iterable[Mapping[str, Any]],
        *,
        minimum_score: float | None = None,
    ) -> list[dict[str, Any]]:
        annotated = [self.annotate(paper, minimum_score=minimum_score) for paper in papers]
        return [paper for paper in annotated if paper["topic_match"]["matched"]]


def parse_topics(data: Any) -> list[Topic]:
    """Parse either a raw topic list or a ``{"topics": [...]}`` mapping."""

    raw = data.get("topics", []) if isinstance(data, Mapping) else data
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TopicConfigError("topics must be a list")
    topics = [item if isinstance(item, Topic) else Topic.from_dict(item) for item in raw]
    # Reuse duplicate validation in the matcher without retaining it.
    TopicMatcher(topics)
    return topics


__all__ = [
    "Topic",
    "TopicConfigError",
    "TopicMatch",
    "TopicMatcher",
    "TopicScore",
    "parse_topics",
]
