"""Explainable hybrid recall and ranking for PaperDaily."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from datetime import date, datetime
from typing import Any

from paperflow.providers import build_embedding_provider

from .identifiers import canonicalize_arxiv_id
from .models import Recommendation
from .storage import PaperDailyStore, hash_abstract
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
    """Recall papers with rules and semantics, then rank and diversify them."""

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
        semantic_recall_enabled: bool = True,
        semantic_recall_threshold: float = 0.58,
        semantic_recall_limit: int = 30,
    ) -> None:
        self.topics = [topic for topic in topics if topic.enabled]
        self.matcher = TopicMatcher(self.topics)
        self.user_id = user_id
        self.limit = self._normalize_limit(limit)
        self.store = store
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self.mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))
        self.include_handled = include_handled
        self.semantic_recall_enabled = bool(semantic_recall_enabled)
        self.semantic_recall_threshold = float(semantic_recall_threshold)
        self.semantic_recall_limit = max(1, int(semantic_recall_limit))
        if not 0.0 <= self.semantic_recall_threshold <= 1.0:
            raise ValueError("semantic_recall_threshold must be between 0 and 1")
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
        provider = str(getattr(self.embedding_provider, "name", "hash") or "hash").casefold()
        return self.semantic_recall_enabled and provider != "hash"

    def _topic_text(self, topic: Topic) -> str:
        return "\n".join(
            filter(
                None,
                [
                    f"Research topic: {topic.name}",
                    topic.description,
                    f"Core phrases: {' '.join(topic.exact_phrases)}" if topic.exact_phrases else "",
                    f"Keywords: {' '.join(topic.keywords)}" if topic.keywords else "",
                    f"Context: {' '.join(topic.context_keywords)}" if topic.context_keywords else "",
                ],
            )
        )

    @staticmethod
    def _paper_text(paper: dict[str, Any]) -> str:
        return f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}"

    @property
    def _embedding_identity(self) -> tuple[str, str, int]:
        provider = str(getattr(self.embedding_provider, "name", "unknown") or "unknown").lower()
        model = str(getattr(self.embedding_provider, "model", "unknown") or "unknown")
        dimensions = max(0, int(getattr(self.embedding_provider, "dimensions", 0) or 0))
        return provider, model, dimensions

    def _embed_cached(
        self,
        document_kind: str,
        documents: list[tuple[str, str]],
        *,
        batch_size: int = 64,
    ) -> tuple[list[list[float]], int, int]:
        """Embed documents in bounded batches and reuse exact local cache hits."""

        if not documents:
            return [], 0, 0
        provider, model, dimensions = self._embedding_identity
        vectors: list[list[float] | None] = [None] * len(documents)
        missing: list[tuple[int, str, str, str]] = []
        cache_hits = 0
        for index, (document_id, text) in enumerate(documents):
            content_hash = hash_abstract(text)
            cached = None
            if self.store is not None and dimensions > 0:
                with suppress(Exception):
                    cached = self.store.get_embedding(
                        document_kind,
                        document_id,
                        content_hash,
                        provider,
                        model,
                        dimensions,
                    )
            if cached is not None:
                vectors[index] = list(cached.get("vector") or [])
                cache_hits += 1
            else:
                missing.append((index, document_id, text, content_hash))

        normalized_batch_size = max(1, int(batch_size))
        call_count = 0
        for start in range(0, len(missing), normalized_batch_size):
            batch = missing[start : start + normalized_batch_size]
            generated = [
                list(vector)
                for vector in self.embedding_provider.embed_batch([item[2] for item in batch])
            ]
            call_count += 1
            if len(generated) != len(batch):
                raise ValueError("embedding provider returned an unexpected batch size")
            for (index, document_id, _text, content_hash), vector in zip(batch, generated, strict=True):
                if not vector:
                    raise ValueError("embedding provider returned an empty vector")
                if dimensions > 0 and len(vector) != dimensions:
                    raise ValueError("embedding provider returned an unexpected vector dimension")
                vectors[index] = vector
                if self.store is not None:
                    with suppress(Exception):
                        self.store.save_embedding(
                            document_kind,
                            document_id,
                            content_hash,
                            provider,
                            model,
                            len(vector),
                            vector,
                        )

        if any(vector is None for vector in vectors):  # pragma: no cover
            raise RuntimeError("embedding batch did not produce every requested vector")
        return [list(vector or []) for vector in vectors], cache_hits, call_count

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
        """Recall and score candidates before LLM reranking and MMR selection."""

        source = list(papers)
        annotated = [self.matcher.annotate(paper) for paper in source]
        handled: set[str] = set()
        if self.store is not None and not self.include_handled:
            handled = self.store.list_handled_canonical_ids(self.user_id)

        valid: list[dict[str, Any]] = []
        for paper in annotated:
            canonical = canonicalize_arxiv_id(paper.get("arxiv_id") or paper.get("url"))
            if not canonical:
                continue
            paper["arxiv_id"] = canonical
            valid.append(paper)

        semantic_enabled = self.semantic_enabled and bool(self.topics)
        topic_vectors: dict[str, list[float]] = {}
        paper_vectors: list[list[float]] = [[] for _ in valid]
        embedding_cache_hits = 0
        embedding_call_count = 0
        if semantic_enabled:
            try:
                vectors, cache_hits, call_count = self._embed_cached(
                    "topic",
                    [(topic.id, self._topic_text(topic)) for topic in self.topics],
                )
                topic_vectors = {
                    topic.id: list(vector)
                    for topic, vector in zip(self.topics, vectors, strict=True)
                }
                embedding_cache_hits += cache_hits
                embedding_call_count += call_count
                paper_vectors, cache_hits, call_count = self._embed_cached(
                    "paper",
                    [(str(paper["arxiv_id"]), self._paper_text(paper)) for paper in valid],
                )
                embedding_cache_hits += cache_hits
                embedding_call_count += call_count
            except Exception:
                semantic_enabled = False
                topic_vectors = {}
                paper_vectors = [[] for _ in valid]

        topic_names = {topic.id: topic.name for topic in self.topics}
        matched: list[dict[str, Any]] = []
        semantic_only: list[tuple[float, dict[str, Any]]] = []
        semantic_match_count = 0
        for paper, vector in zip(valid, paper_vectors, strict=True):
            match_data = dict(paper.get("topic_match") or {})
            rule_topics = list(paper.get("matched_topics") or [])
            all_semantic_scores: dict[str, float] = {}
            semantic_recall_scores: dict[str, float] = {}
            if semantic_enabled:
                blocked_topics = {
                    str(detail.get("topic_id") or "")
                    for detail in match_data.get("details") or []
                    if detail.get("negative_terms")
                }
                for topic in self.topics:
                    if topic.id in blocked_topics:
                        continue
                    similarity = round(max(0.0, _cosine(vector, topic_vectors.get(topic.id, []))), 6)
                    all_semantic_scores[topic.id] = similarity
                    if similarity >= self.semantic_recall_threshold:
                        semantic_recall_scores[topic.id] = similarity

            semantic_topics = sorted(
                semantic_recall_scores,
                key=semantic_recall_scores.get,
                reverse=True,
            )
            if semantic_topics:
                semantic_match_count += 1
            combined_topics = list(dict.fromkeys([*rule_topics, *semantic_topics]))
            merged_topic_scores = dict(match_data.get("topic_scores") or {})
            for topic_id, similarity in semantic_recall_scores.items():
                merged_topic_scores[topic_id] = max(
                    float(merged_topic_scores.get(topic_id, 0.0)),
                    similarity,
                )

            paper["rule_matched_topics"] = rule_topics
            paper["semantic_topic_scores"] = all_semantic_scores
            paper["semantic_recall_scores"] = semantic_recall_scores
            paper["semantic_recall"] = bool(semantic_topics and not rule_topics)
            paper["embedding"] = vector
            paper["embedding_model"] = (
                f"{getattr(self.embedding_provider, 'name', 'unknown')}:{getattr(self.embedding_provider, 'model', '')}"
                if semantic_enabled
                else ""
            )
            match_data.update(
                {
                    "matched": bool(combined_topics),
                    "topics": combined_topics,
                    "topic_names": [topic_names.get(topic_id, topic_id) for topic_id in combined_topics],
                    "topic_scores": merged_topic_scores,
                    "semantic_topic_scores": all_semantic_scores,
                    "semantic_recall_scores": semantic_recall_scores,
                }
            )
            paper["topic_match"] = match_data
            paper["matched_topics"] = combined_topics
            if rule_topics:
                matched.append(paper)
            elif semantic_topics:
                semantic_only.append((max(semantic_recall_scores.values()), paper))

        semantic_only.sort(key=lambda item: item[0], reverse=True)
        semantic_recalled = [paper for _score, paper in semantic_only[: self.semantic_recall_limit]]
        matched.extend(semantic_recalled)
        candidates = [paper for paper in matched if paper["arxiv_id"] not in handled]

        feedback = self._feedback_topic_adjustments()
        scored: list[Recommendation] = []
        for paper in candidates:
            rule_score = float(paper.get("topic_score") or 0.0)
            matched_topics = list(paper.get("matched_topics") or [])
            all_semantic_scores = dict(paper.get("semantic_topic_scores") or {})
            topic_semantic = max(
                (float(all_semantic_scores.get(topic_id, 0.0)) for topic_id in matched_topics),
                default=0.0,
            )
            topic_semantic = max(0.0, topic_semantic) if semantic_enabled else 0.0
            published = _parse_date(paper.get("publish_date"))
            age_days = max(0, (window_end - published).days) if published else 30
            freshness = max(0.0, 1.0 - min(age_days, 30) / 30)
            quality = _quality_score(paper)
            feedback_bonus = max(
                (feedback.get(topic_id, 0.0) for topic_id in matched_topics),
                default=0.0,
            )
            score = (
                0.52 * rule_score
                + 0.32 * topic_semantic
                + 0.09 * freshness
                + 0.07 * quality
                + feedback_bonus
            )
            score = max(0.0, min(1.0, score))
            terms = list(paper.get("matched_terms") or [])
            reason_parts: list[str] = []
            rule_topics = list(paper.get("rule_matched_topics") or [])
            if rule_topics:
                reason_parts.append(
                    f"命中 {', '.join(topic_names.get(topic_id, topic_id) for topic_id in rule_topics)}"
                )
            if terms:
                reason_parts.append(f"关键词/短语：{', '.join(terms[:5])}")
            semantic_recall_scores = dict(paper.get("semantic_recall_scores") or {})
            if semantic_enabled and semantic_recall_scores:
                semantic_labels = ", ".join(
                    f"{topic_names.get(topic_id, topic_id)} {similarity:.2f}"
                    for topic_id, similarity in sorted(
                        semantic_recall_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                reason_parts.append(f"摘要语义命中：{semantic_labels}")
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
            "rule_matched_count": sum(bool(paper.get("rule_matched_topics")) for paper in matched),
            "semantic_match_count": semantic_match_count,
            "semantic_recalled_count": len(semantic_recalled),
            "semantic_recall_considered_count": len(valid) if semantic_enabled else 0,
            "semantic_recall_threshold": self.semantic_recall_threshold,
            "semantic_recall_limit": self.semantic_recall_limit,
            "handled_count": len(matched) - len(candidates),
            "candidate_count": len(candidates),
            "selected_count": 0,
            "semantic_enabled": semantic_enabled,
            "embedding_provider": str(getattr(self.embedding_provider, "name", "unknown")),
            "embedding_model": str(getattr(self.embedding_provider, "model", "unknown")),
            "embedding_cache_hits": embedding_cache_hits,
            "embedding_call_count": embedding_call_count,
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
            # Vectors are transient ranking data; the dedicated cache owns
            # persistence so recommendation rows stay small.
            item.paper.pop("embedding", None)
        self.last_diagnostics["selected_count"] = len(selected)
        return selected

    def rank(self, papers: Iterable[dict[str, Any]], *, window_end: date) -> list[Recommendation]:
        """Compatibility entry point: hybrid candidates followed by MMR."""

        return self.select(self.rank_candidates(papers, window_end=window_end))


__all__ = ["PaperRanker"]
