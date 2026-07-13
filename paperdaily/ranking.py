"""Explainable topic-first ranking built on PaperFlow embedding providers."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from paperflow.providers import build_embedding_provider

from .identifiers import canonicalize_arxiv_id
from .models import Recommendation
from .storage import PaperDailyStore
from .topics import Topic, TopicMatcher


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _tokens(paper: dict[str, Any]) -> set[str]:
    text = " ".join(
        [
            str(paper.get("title") or ""),
            " ".join(map(str, paper.get("matched_topics") or [])),
            " ".join(map(str, paper.get("matched_terms") or [])),
        ]
    ).casefold()
    return {value for value in re.findall(r"[a-z0-9][a-z0-9.+-]+", text) if len(value) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _quality_score(paper: dict[str, Any]) -> float:
    score = 0.2
    abstract = str(paper.get("abstract") or "")
    score += min(0.3, len(abstract) / 5000)
    score += 0.15 if paper.get("authors") else 0.0
    score += 0.10 if paper.get("doi") else 0.0
    combined = f"{paper.get('title', '')} {abstract}".casefold()
    score += 0.15 if any(token in combined for token in ("github.com", "code is available", "open-source")) else 0.0
    score += 0.10 if any(token in combined for token in ("benchmark", "ablation", "real-world", "real robot")) else 0.0
    return min(1.0, score)


class PaperRanker:
    """Rank topic-matched papers and apply MMR plus per-topic quotas."""

    def __init__(
        self,
        topics: Iterable[Topic],
        *,
        user_id: str,
        limit: int | None = None,
        store: PaperDailyStore | None = None,
        embedding_provider: Any = None,
        mmr_lambda: float = 0.75,
        include_handled: bool = False,
    ) -> None:
        self.topics = [topic for topic in topics if topic.enabled]
        self.matcher = TopicMatcher(self.topics)
        self.user_id = user_id
        self.limit = self._normalize_limit(limit)
        self.store = store
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))
        self.include_handled = include_handled
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _normalize_limit(value: int | None) -> int | None:
        if value is None:
            return None
        normalized = int(value)
        if normalized < 0:
            raise ValueError("limit must be zero or positive")
        return normalized or None

    @property
    def semantic_enabled(self) -> bool:
        return str(getattr(self.embedding_provider, "name", "hash")) != "hash"

    def _topic_text(self, topic: Topic) -> str:
        return "\n".join(
            filter(
                None,
                [
                    topic.name,
                    topic.description,
                    " ".join(topic.exact_phrases),
                    " ".join(topic.keywords),
                ],
            )
        )

    def _feedback_topic_adjustments(self) -> dict[str, float]:
        if self.store is None:
            return {}
        action_weights = {
            "interested": 0.035,
            "saved": 0.05,
            "read": 0.07,
            "reading_note": 0.09,
            "detailed": 0.06,
            "later": 0.01,
            "irrelevant": -0.08,
        }
        adjustments: dict[str, float] = defaultdict(float)
        try:
            events = self.store.list_feedback(self.user_id, limit=300)
        except Exception:
            return {}
        for event in events:
            action = str(event.get("action") or "").lower()
            weight = action_weights.get(action, 0.0)
            metadata = event.get("metadata") or {}
            for topic_id in metadata.get("matched_topics") or []:
                adjustments[str(topic_id)] += weight
        return {key: max(-0.20, min(0.20, value)) for key, value in adjustments.items()}

    def rank_candidates(self, papers: Iterable[dict[str, Any]], *, window_end: date) -> list[Recommendation]:
        """Return all deterministically scored candidates before MMR selection.

        This separation lets the service rerank a small, bounded prefix with an
        LLM while retaining the same rule-based recall, quotas, and diversity
        behaviour for offline users.
        """
        source = list(papers)
        matched = self.matcher.filter(source)
        handled: set[str] = set()
        if self.store is not None and not self.include_handled:
            handled = self.store.list_handled_canonical_ids(self.user_id)

        candidates: list[dict[str, Any]] = []
        for paper in matched:
            canonical = canonicalize_arxiv_id(paper.get("arxiv_id") or paper.get("url"))
            if not canonical or canonical in handled:
                continue
            paper["arxiv_id"] = canonical
            candidates.append(paper)

        topic_vectors: dict[str, list[float]] = {}
        semantic_enabled = self.semantic_enabled
        if semantic_enabled and self.topics:
            try:
                vectors = self.embedding_provider.embed_batch([self._topic_text(topic) for topic in self.topics])
                topic_vectors = {
                    topic.id: list(vector)
                    for topic, vector in zip(self.topics, vectors, strict=True)
                }
            except Exception:
                semantic_enabled = False
                topic_vectors = {}

        paper_vectors: list[list[float]] = []
        if candidates:
            texts = [f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}" for paper in candidates]
            try:
                paper_vectors = [list(vector) for vector in self.embedding_provider.embed_batch(texts)]
            except Exception:
                paper_vectors = [[] for _ in candidates]
                semantic_enabled = False

        feedback = self._feedback_topic_adjustments()
        scored: list[Recommendation] = []
        for paper, vector in zip(candidates, paper_vectors, strict=True):
            paper["embedding"] = vector
            paper["embedding_model"] = (
                f"{getattr(self.embedding_provider, 'name', 'unknown')}:{getattr(self.embedding_provider, 'model', '')}"
            )
            rule_score = float(paper.get("topic_score") or 0.0)
            matched_topics = list(paper.get("matched_topics") or [])
            topic_semantic = max(
                (_cosine(vector, topic_vectors[topic_id]) for topic_id in matched_topics if topic_id in topic_vectors),
                default=0.0,
            )
            topic_semantic = max(0.0, topic_semantic) if semantic_enabled else 0.0
            published = _parse_date(paper.get("publish_date"))
            age_days = max(0, (window_end - published).days) if published else 30
            freshness = max(0.0, 1.0 - min(age_days, 30) / 30)
            quality = _quality_score(paper)
            feedback_bonus = max((feedback.get(topic_id, 0.0) for topic_id in matched_topics), default=0.0)
            score = (
                0.52 * rule_score
                + 0.32 * topic_semantic
                + 0.09 * freshness
                + 0.07 * quality
                + feedback_bonus
            )
            score = max(0.0, min(1.0, score))
            match_data = paper.get("topic_match") or {}
            topic_names = list(match_data.get("topic_names") or matched_topics)
            terms = list(paper.get("matched_terms") or [])
            reason_parts = []
            if topic_names:
                reason_parts.append(f"命中 {', '.join(topic_names)}")
            if terms:
                reason_parts.append(f"关键词/短语：{', '.join(terms[:5])}")
            if semantic_enabled and topic_semantic > 0:
                reason_parts.append(f"话题语义相似度 {topic_semantic:.2f}")
            scored.append(
                Recommendation(
                    rank=0,
                    score=score,
                    paper=paper,
                    matched_topics=matched_topics,
                    matched_terms=terms,
                    recommendation_reason="；".join(reason_parts) or "符合订阅规则",
                    component_scores={
                        "topic_rule": round(rule_score, 6),
                        "topic_semantic": round(topic_semantic, 6),
                        "freshness": round(freshness, 6),
                        "quality": round(quality, 6),
                        "feedback": round(feedback_bonus, 6),
                    },
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        self.last_diagnostics = {
            "input_count": len(source),
            "matched_count": len(matched),
            "handled_count": len(matched) - len(candidates),
            "candidate_count": len(candidates),
            "selected_count": 0,
            "semantic_enabled": semantic_enabled,
            "embedding_provider": str(getattr(self.embedding_provider, "name", "unknown")),
        }
        return scored

    def select(self, scored: Iterable[Recommendation], *, limit: int | None = None) -> list[Recommendation]:
        """Apply per-topic quota and MMR diversity to scored candidates."""

        selected: list[Recommendation] = []
        topic_counts: dict[str, int] = defaultdict(int)
        quota_by_topic = {topic.id: topic.daily_limit for topic in self.topics}
        remaining = list(scored)
        selected_limit = self.limit if limit is None else self._normalize_limit(limit)
        semantic_enabled = bool(self.last_diagnostics.get("semantic_enabled", self.semantic_enabled))
        while remaining and (selected_limit is None or len(selected) < selected_limit):
            best_index: int | None = None
            best_value = float("-inf")
            for index, candidate in enumerate(remaining):
                topic_scores = (candidate.paper.get("topic_match") or {}).get("topic_scores") or {}
                ordered_topics = sorted(
                    candidate.matched_topics,
                    key=lambda topic_id: float(topic_scores.get(topic_id, 0.0)),
                    reverse=True,
                )
                available_topics = [
                    topic_id
                    for topic_id in ordered_topics
                    if quota_by_topic.get(topic_id, 0) <= 0
                    or topic_counts[topic_id] < quota_by_topic[topic_id]
                ]
                if ordered_topics and not available_topics:
                    continue
                candidate.paper["_paperdaily_quota_topic"] = available_topics[0] if available_topics else ""
                redundancy = 0.0
                for picked in selected:
                    left_vector = list(candidate.paper.get("embedding") or [])
                    right_vector = list(picked.paper.get("embedding") or [])
                    similarity = (
                        max(0.0, _cosine(left_vector, right_vector))
                        if semantic_enabled
                        else _jaccard(_tokens(candidate.paper), _tokens(picked.paper))
                    )
                    redundancy = max(redundancy, similarity)
                mmr = self.mmr_lambda * candidate.score - (1.0 - self.mmr_lambda) * redundancy
                if mmr > best_value:
                    best_value = mmr
                    best_index = index
            if best_index is None:
                break
            picked = remaining.pop(best_index)
            quota_topic = str(picked.paper.pop("_paperdaily_quota_topic", ""))
            if quota_topic:
                topic_counts[quota_topic] += 1
            selected.append(picked)

        for rank, item in enumerate(selected, start=1):
            item.rank = rank
        self.last_diagnostics["selected_count"] = len(selected)
        return selected

    def rank(self, papers: Iterable[dict[str, Any]], *, window_end: date) -> list[Recommendation]:
        """Compatibility entry point: deterministic candidates followed by MMR."""

        return self.select(self.rank_candidates(papers, window_end=window_end))


__all__ = ["PaperRanker"]
