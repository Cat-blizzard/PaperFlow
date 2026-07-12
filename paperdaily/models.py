"""Shared value objects for the PaperDaily service layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass
class ChineseSummary:
    title_zh: str
    one_sentence_summary: str
    problem: str = ""
    method: str = ""
    contributions: list[str] = field(default_factory=list)
    limitations_from_abstract: list[str] = field(default_factory=list)
    recommendation_reason: str = ""
    provider: str = ""
    model: str = ""
    cached: bool = False
    status: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    rank: int
    score: float
    paper: dict[str, Any]
    matched_topics: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    recommendation_reason: str = ""
    component_scores: dict[str, float] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    summary: ChineseSummary | None = None

    @property
    def canonical_id(self) -> str:
        return str(self.paper.get("arxiv_id") or "")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical_id"] = self.canonical_id
        return payload


@dataclass
class Digest:
    run_id: str
    user_id: str
    window_start: date
    window_end: date
    recommendations: list[Recommendation]
    generated_at: datetime
    stats: dict[str, Any] = field(default_factory=dict)
    catchup_mode: str = "daily"
    output_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "catchup_mode": self.catchup_mode,
            "stats": self.stats,
            "recommendations": [item.as_dict() for item in self.recommendations],
            "output_path": str(self.output_path) if self.output_path else None,
        }


@dataclass(frozen=True)
class DeliveryResult:
    channel: str
    success: bool
    target: str = ""
    error: str = ""


__all__ = [
    "ChineseSummary",
    "DeliveryResult",
    "Digest",
    "Recommendation",
]
