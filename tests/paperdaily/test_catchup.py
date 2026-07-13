from __future__ import annotations

from datetime import date

import pytest

from paperdaily.catchup import (
    CatchupPlanner,
    apply_catchup_choice,
    default_target_date,
    resolve_fetch_window,
)

TODAY = date(2026, 7, 12)
TARGET = date(2026, 7, 12)


def test_default_target_is_current_announcement_day() -> None:
    assert default_target_date(today=TODAY) == TARGET


def test_first_run_defaults_to_seven_days_and_confirmation() -> None:
    plan = resolve_fetch_window(None, today=TODAY)

    assert plan.first_run is True
    assert plan.gap_days == 7
    assert plan.missing_start == date(2026, 7, 6)
    assert plan.recommended_choice == "7d"
    assert plan.recommended_window.days == 7
    assert plan.requires_confirmation is True


@pytest.mark.parametrize(
    ("last_completed", "gap", "choice", "confirm", "start"),
    [
        (date(2026, 7, 10), 2, "all", False, date(2026, 7, 11)),
        (date(2026, 7, 9), 3, "7d", True, date(2026, 7, 10)),
        (date(2026, 7, 8), 4, "7d", True, date(2026, 7, 9)),
        (date(2026, 7, 4), 8, "7d", True, date(2026, 7, 6)),
        (date(2026, 6, 23), 19, "7d", True, date(2026, 7, 6)),
        (date(2026, 5, 31), 42, "30d", True, date(2026, 6, 13)),
    ],
)
def test_gap_policy_boundaries(last_completed, gap, choice, confirm, start) -> None:
    plan = resolve_fetch_window(last_completed, today=TODAY)

    assert plan.gap_days == gap
    assert plan.recommended_choice == choice
    assert plan.requires_confirmation is confirm
    assert plan.recommended_window.start_date == start
    assert plan.recommended_window.end_date == TARGET


def test_up_to_date_and_future_watermark_produce_empty_plan() -> None:
    current = resolve_fetch_window(TARGET, today=TODAY)
    future = resolve_fetch_window(date(2026, 7, 20), today=TODAY)

    assert current.gap_days == 0
    assert current.recommended_window.is_empty
    assert current.reason == "up_to_date"
    assert not current.watermark_in_future

    assert future.gap_days == 0
    assert future.recommended_window.is_empty
    assert future.reason == "future_watermark"
    assert future.watermark_in_future


def test_supported_choices_are_inclusive_and_clamped_to_missing_range() -> None:
    planner = CatchupPlanner()
    plan = planner.plan(date(2026, 6, 23), today=TODAY)

    assert planner.select(plan, "yesterday").days == 1
    assert planner.select(plan, "7d").start_date == date(2026, 7, 6)
    assert planner.select(plan, "30d").start_date == date(2026, 6, 24)
    assert planner.select(plan, "all").start_date == date(2026, 6, 24)
    assert planner.select(plan, "recommended") == plan.recommended_window


def test_first_run_can_explicitly_request_thirty_days() -> None:
    planner = CatchupPlanner()
    plan = planner.plan(None, today=TODAY)

    selected = apply_catchup_choice(plan, "30d", planner=planner)

    assert selected.start_date == date(2026, 6, 13)
    assert selected.end_date == TARGET
    assert selected.days == 30


def test_custom_window_validation_and_existing_watermark_clamp() -> None:
    planner = CatchupPlanner()
    plan = planner.plan(date(2026, 7, 1), today=TODAY)

    selected = planner.select(
        plan,
        "custom",
        custom_start=date(2026, 6, 1),
        custom_end=date(2026, 7, 8),
    )
    assert selected.start_date == date(2026, 7, 2)
    assert selected.end_date == date(2026, 7, 8)

    with pytest.raises(ValueError, match="later than target_date"):
        planner.select(plan, "custom", custom_start=date(2026, 7, 2), custom_end=date(2026, 7, 13))
    with pytest.raises(ValueError, match="later than custom_end"):
        planner.select(plan, "custom", custom_start=date(2026, 7, 9), custom_end=date(2026, 7, 8))


def test_non_interactive_planner_keeps_recommendation_without_pausing() -> None:
    planner = CatchupPlanner(interactive=False)
    plan = planner.plan(date(2026, 5, 1), today=TODAY)

    assert plan.recommended_choice == "30d"
    assert plan.requires_confirmation is False
