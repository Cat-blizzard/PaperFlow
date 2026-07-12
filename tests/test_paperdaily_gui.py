"""Focused tests for the PaperDaily desktop GUI bridge."""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path

from deployments.desktop.shared.paperdaily import PaperDailyGui
from paperdaily.catchup import DateWindow
from paperdaily.config import PaperDailyConfig, load_config, save_config
from paperdaily.models import Digest, Recommendation
from paperdaily.service import RunOutcome
from paperdaily.topics import Topic


def _config(path: Path) -> PaperDailyConfig:
    config = PaperDailyConfig(
        user_id="gui_user",
        timezone="Asia/Shanghai",
        database=path.parent / "paperdaily.db",
        output_dir=path.parent / "output",
        topics=[
            Topic(
                id="embodied-vla",
                name="具身智能与 VLA",
                description="Robot policies",
                arxiv_categories=["cs.RO", "cs.AI"],
                exact_phrases=["vision-language-action"],
                keywords=["VLA"],
            )
        ],
    )
    save_config(config, path)
    return config


def test_gui_status_and_topic_changes_use_paperdaily_config(tmp_path: Path) -> None:
    config_path = tmp_path / "paperdaily.yaml"
    _config(config_path)
    gui = PaperDailyGui(config_path)

    status = gui.status()

    assert status["configured"] is True
    assert status["user_id"] == "gui_user"
    assert status["topics"][0]["id"] == "embodied-vla"

    created = gui.save_topic(
        {
            "id": "world-model",
            "name": "机器人世界模型",
            "description": "World models for action prediction",
            "enabled": True,
            "arxiv_categories": ["cs.RO"],
            "exact_phrases": ["world action model"],
            "keywords": ["WAM"],
            "context_keywords": ["robot"],
            "negative_keywords": ["wireless access management"],
            "daily_limit": 8,
            "minimum_score": 0.3,
        }
    )
    assert created["topic"]["id"] == "world-model"

    gui.set_topic_enabled("world-model", False)
    persisted = load_config(config_path)
    topic = next(item for item in persisted.topics if item.id == "world-model")
    assert topic.enabled is False

    deleted = gui.delete_topic("world-model")
    assert deleted["deleted"] == "world-model"
    assert [item.id for item in load_config(config_path).topics] == ["embodied-vla"]


def test_gui_reports_missing_configuration_without_creating_files(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    status = PaperDailyGui(path).status()

    assert status["configured"] is False
    assert "paperdaily init" in status["message"]
    assert not path.exists()


def test_gui_lists_only_completed_digests_for_the_configured_user(tmp_path: Path) -> None:
    config_path = tmp_path / "paperdaily.yaml"
    config = _config(config_path)
    gui = PaperDailyGui(config_path)
    _loaded, service = gui._context()  # noqa: SLF001 - test the GUI storage boundary

    completed_run = service.store.start_run(
        config.user_id,
        date(2026, 7, 1),
        date(2026, 7, 1),
        mode="daily",
    )
    service.store.complete_run(completed_run, recommendation_count=4)
    service.store.start_run(
        config.user_id,
        date(2026, 7, 2),
        date(2026, 7, 2),
        mode="daily",
    )
    other_user_run = service.store.start_run(
        "other_user",
        date(2026, 7, 3),
        date(2026, 7, 3),
        mode="daily",
    )
    service.store.complete_run(other_user_run, recommendation_count=9)

    payload = gui.list_digests()

    assert [run["run_id"] for run in payload["runs"]] == [completed_run]
    assert payload["runs"][0]["recommendation_count"] == 4


def test_gui_preview_task_is_background_and_returns_compact_digest(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "paperdaily.yaml"
    config = _config(config_path)
    gui = PaperDailyGui(config_path)
    paper = {
        "arxiv_id": "2607.00001",
        "title": "Robot VLA",
        "abstract": "A robot policy.",
        "authors": ["Ada"],
        "categories": ["cs.RO"],
    }

    class FakeService:
        def catchup_plan(self):
            return object()

        def select_window(self, _plan, _choice, **_kwargs):
            return DateWindow(date(2026, 7, 1), date(2026, 7, 1), "recommended")

        def run(self, window, **_kwargs):
            digest = Digest(
                run_id="00000000-0000-0000-0000-000000000000",
                user_id=config.user_id,
                window_start=window.start_date,
                window_end=window.end_date,
                generated_at=datetime.now(timezone.utc),
                recommendations=[
                    Recommendation(
                        rank=1,
                        score=0.9,
                        paper=paper,
                        matched_topics=["embodied-vla"],
                        recommendation_reason="Robot policy",
                    )
                ],
            )
            return RunOutcome(digest=digest, dry_run=True)

    monkeypatch.setattr(gui, "_context", lambda: (config, FakeService()))
    started = gui.start_digest_task(choice="recommended", dry_run=True, limit=12)

    task = gui.task(started["task_id"])
    for _ in range(40):
        if task and task["status"] != "running":
            break
        time.sleep(0.01)
        task = gui.task(started["task_id"])

    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["dry_run"] is True
    assert task["result"]["recommendations"][0]["arxiv_id"] == "2607.00001"
    assert task["result"]["recommendations"][0]["url"] == "https://arxiv.org/abs/2607.00001"
