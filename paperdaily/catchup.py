"""Date-window planning for normal and catch-up PaperDaily runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DateValue = date | datetime | str


def _coerce_date(value: DateValue | None, *, name: str, allow_none: bool = False) -> date | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format") from exc


def local_today(timezone: str = "Asia/Shanghai") -> date:
    """Return today's date in the configured user timezone."""

    try:
        return datetime.now(ZoneInfo(timezone)).date()
    except ZoneInfoNotFoundError:
        # ``tzdata`` is not guaranteed to be present in a minimal installation
        # on Windows.  Falling back to the host-local day is preferable to
        # making an otherwise local CLI unusable; ``doctor`` can still warn.
        return date.today()


def default_target_date(*, today: DateValue | None = None, timezone: str = "Asia/Shanghai") -> date:
    """The arXiv announcement day currently being tracked in the user's timezone."""

    resolved_today = _coerce_date(today, name="today", allow_none=True) or local_today(timezone)
    return resolved_today


@dataclass(frozen=True)
class DateWindow:
    """Inclusive date window selected for a run."""

    start_date: date | None
    end_date: date
    choice: str

    @property
    def is_empty(self) -> bool:
        return self.start_date is None or self.start_date > self.end_date

    @property
    def days(self) -> int:
        if self.is_empty:
            return 0
        assert self.start_date is not None
        return (self.end_date - self.start_date).days + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat(),
            "days": self.days,
            "choice": self.choice,
            "is_empty": self.is_empty,
        }


@dataclass(frozen=True)
class CatchupPlan:
    """Decision metadata shown by an interactive or unattended CLI."""

    target_date: date
    last_completed_date: date | None
    missing_start: date | None
    gap_days: int
    first_run: bool
    requires_confirmation: bool
    recommended_choice: str
    recommended_window: DateWindow
    reason: str
    watermark_in_future: bool = False
    # When a newer interval has already been completed out of order, this is
    # the end of the earliest unresolved gap.  The public target remains the
    # latest complete day so explicit ``all``/custom choices retain their
    # existing meaning.
    next_gap_end: date | None = None

    @property
    def has_work(self) -> bool:
        return self.gap_days > 0 and self.missing_start is not None

    @property
    def start_date(self) -> date | None:
        return self.recommended_window.start_date

    @property
    def end_date(self) -> date:
        return self.recommended_window.end_date

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_date": self.target_date.isoformat(),
            "last_completed_date": self.last_completed_date.isoformat() if self.last_completed_date else None,
            "missing_start": self.missing_start.isoformat() if self.missing_start else None,
            "gap_days": self.gap_days,
            "first_run": self.first_run,
            "requires_confirmation": self.requires_confirmation,
            "recommended_choice": self.recommended_choice,
            "recommended_window": self.recommended_window.to_dict(),
            "reason": self.reason,
            "watermark_in_future": self.watermark_in_future,
            "next_gap_end": self.next_gap_end.isoformat() if self.next_gap_end else None,
            "has_work": self.has_work,
        }


_CHOICE_ALIASES = {
    "latest": "yesterday",
    "today": "yesterday",
    "announcement": "yesterday",
    "1d": "yesterday",
    "last_day": "yesterday",
    "last_7_days": "7d",
    "week": "7d",
    "last_30_days": "30d",
    "month": "30d",
    "all_missing": "all",
    "recommended": "recommended",
}


class CatchupPlanner:
    """Plan missed arXiv windows without performing any network or database I/O."""

    def __init__(
        self,
        *,
        auto_catchup_days: int = 2,
        first_run_days: int = 7,
        default_window_days: int = 7,
        max_window_days: int = 30,
        interactive: bool = True,
    ) -> None:
        self.auto_catchup_days = int(auto_catchup_days)
        self.first_run_days = int(first_run_days)
        self.default_window_days = int(default_window_days)
        self.max_window_days = int(max_window_days)
        self.interactive = bool(interactive)
        if not 0 <= self.auto_catchup_days <= 2:
            raise ValueError("auto_catchup_days must be between 0 and 2")
        if min(self.first_run_days, self.default_window_days, self.max_window_days) <= 0:
            raise ValueError("catch-up window sizes must be positive")
        if self.default_window_days > self.max_window_days:
            raise ValueError("default_window_days cannot exceed max_window_days")

    @classmethod
    def from_config(cls, config: Any) -> CatchupPlanner:
        """Construct from ``CatchupConfig`` or any object with matching fields."""

        return cls(
            auto_catchup_days=getattr(config, "auto_catchup_days", 2),
            first_run_days=getattr(config, "first_run_days", 7),
            default_window_days=getattr(config, "default_window_days", 7),
            max_window_days=getattr(config, "max_window_days", 30),
            interactive=getattr(config, "interactive", True),
        )

    def plan(
        self,
        last_completed_date: DateValue | None,
        *,
        today: DateValue | None = None,
        target_date: DateValue | None = None,
        next_gap_end: DateValue | None = None,
        timezone: str = "Asia/Shanghai",
    ) -> CatchupPlan:
        target = (
            _coerce_date(target_date, name="target_date")
            if target_date is not None
            else default_target_date(today=today, timezone=timezone)
        )
        assert target is not None
        completed = _coerce_date(
            last_completed_date,
            name="last_completed_date",
            allow_none=True,
        )

        if completed is None:
            start = target - timedelta(days=self.first_run_days - 1)
            window = DateWindow(start, target, "7d" if self.first_run_days == 7 else f"{self.first_run_days}d")
            return CatchupPlan(
                target_date=target,
                last_completed_date=None,
                missing_start=start,
                gap_days=self.first_run_days,
                first_run=True,
                requires_confirmation=self.interactive,
                recommended_choice="7d" if self.first_run_days == 7 else "all",
                recommended_window=window,
                reason="first_run",
            )

        if completed >= target:
            future = completed > target
            return CatchupPlan(
                target_date=target,
                last_completed_date=completed,
                missing_start=None,
                gap_days=0,
                first_run=False,
                requires_confirmation=False,
                recommended_choice="none",
                recommended_window=DateWindow(None, target, "none"),
                reason="future_watermark" if future else "up_to_date",
                watermark_in_future=future,
            )

        missing_start = completed + timedelta(days=1)
        gap_end = _coerce_date(next_gap_end, name="next_gap_end", allow_none=True) or target
        if gap_end < missing_start:
            raise ValueError("next_gap_end cannot be earlier than the missing start")
        gap_end = min(gap_end, target)
        gap_days = (gap_end - missing_start).days + 1
        if gap_days <= self.auto_catchup_days:
            recommended_choice = "all"
            reason = "automatic" if gap_end == target else "earliest_gap"
            requires_confirmation = False
        elif gap_days <= 7:
            recommended_choice = "7d"
            reason = "short_gap" if gap_end == target else "earliest_gap"
            requires_confirmation = self.interactive
        elif gap_days <= 30:
            recommended_choice = "7d"
            reason = "medium_gap" if gap_end == target else "earliest_gap"
            requires_confirmation = self.interactive
        else:
            recommended_choice = "30d"
            reason = "long_gap" if gap_end == target else "earliest_gap"
            requires_confirmation = self.interactive

        provisional = CatchupPlan(
            target_date=target,
            last_completed_date=completed,
            missing_start=missing_start,
            gap_days=gap_days,
            first_run=False,
            requires_confirmation=requires_confirmation,
            recommended_choice=recommended_choice,
            recommended_window=DateWindow(None, target, recommended_choice),
            reason=reason,
            next_gap_end=gap_end if gap_end < target else None,
        )
        window = self.select(provisional, recommended_choice)
        return CatchupPlan(
            target_date=target,
            last_completed_date=completed,
            missing_start=missing_start,
            gap_days=gap_days,
            first_run=False,
            requires_confirmation=requires_confirmation,
            recommended_choice=recommended_choice,
            recommended_window=window,
            reason=reason,
            next_gap_end=gap_end if gap_end < target else None,
        )

    def select(
        self,
        plan: CatchupPlan,
        choice: str,
        *,
        custom_start: DateValue | None = None,
        custom_end: DateValue | None = None,
    ) -> DateWindow:
        normalized_choice = str(choice or "").strip().lower()
        normalized_choice = _CHOICE_ALIASES.get(normalized_choice, normalized_choice)
        if normalized_choice == "recommended":
            normalized_choice = plan.recommended_choice

        target = plan.target_date
        # The default 7d/30d choice should repair the earliest unresolved gap
        # when a newer interval was completed out of order. Explicit ``all``
        # and ``custom`` still retain their full-range semantics.
        window_target = plan.next_gap_end or target
        if normalized_choice == "none":
            return DateWindow(None, target, "none")
        if normalized_choice == "yesterday":
            return DateWindow(target, target, "yesterday")

        if normalized_choice in {"7d", "30d"}:
            days = int(normalized_choice[:-1])
            start = window_target - timedelta(days=days - 1)
            if not plan.first_run and plan.missing_start is not None:
                start = max(start, plan.missing_start)
            return DateWindow(start, window_target, normalized_choice)

        if normalized_choice == "all":
            if plan.missing_start is None:
                return DateWindow(None, target, "all")
            return DateWindow(plan.missing_start, target, "all")

        if normalized_choice == "custom":
            start = _coerce_date(custom_start, name="custom_start")
            end = _coerce_date(custom_end, name="custom_end", allow_none=True) or target
            assert start is not None
            if end > target:
                raise ValueError("custom_end cannot be later than target_date")
            if start > end:
                raise ValueError("custom_start cannot be later than custom_end")
            if not plan.first_run and plan.missing_start is not None:
                start = max(start, plan.missing_start)
            if start > end:
                return DateWindow(None, end, "custom")
            return DateWindow(start, end, "custom")

        raise ValueError("choice must be latest, yesterday, 7d, 30d, all, custom, or recommended")


def resolve_fetch_window(
    last_completed_date: DateValue | None,
    *,
    today: DateValue | None = None,
    target_date: DateValue | None = None,
    timezone: str = "Asia/Shanghai",
    planner: CatchupPlanner | None = None,
) -> CatchupPlan:
    """Convenience wrapper returning a complete catch-up decision."""

    return (planner or CatchupPlanner()).plan(
        last_completed_date,
        today=today,
        target_date=target_date,
        timezone=timezone,
    )


def apply_catchup_choice(
    plan: CatchupPlan,
    choice: str,
    *,
    custom_start: DateValue | None = None,
    custom_end: DateValue | None = None,
    planner: CatchupPlanner | None = None,
) -> DateWindow:
    """Apply one of the supported interactive choices to an existing plan."""

    return (planner or CatchupPlanner()).select(
        plan,
        choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )


__all__ = [
    "CatchupPlan",
    "CatchupPlanner",
    "DateWindow",
    "apply_catchup_choice",
    "default_target_date",
    "local_today",
    "resolve_fetch_window",
]
