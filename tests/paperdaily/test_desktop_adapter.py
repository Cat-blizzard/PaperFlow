from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from deployments.desktop.shared.paperdaily import PaperDailyGui
from paperdaily.catchup import DateWindow
from paperdaily.config import DailyConfig, PaperDailyConfig, save_config
from paperdaily.models import Digest
from paperdaily.service import RunOutcome
from paperdaily.storage import PaperDailyStore
from paperdaily.topics import Topic


def _config(tmp_path: Path) -> tuple[Path, PaperDailyConfig]:
    config = PaperDailyConfig(
        user_id="alice",
        timezone="UTC",
        output_dir=tmp_path / "output",
        database=tmp_path / "paperflow.db",
        daily=DailyConfig(generate_chinese_summary=False, channels=["markdown"]),
        topics=[
            Topic(
                id="embodied-vla",
                name="Embodied VLA",
                arxiv_categories=["cs.RO"],
                keywords=["VLA"],
                context_keywords=["robot"],
            )
        ],
    )
    path = tmp_path / "paperdaily.yaml"
    save_config(config, path)
    return path, config


def _recommendation() -> dict[str, object]:
    return {
        "rank": 1,
        "score": 0.9,
        "paper": {
            "arxiv_id": "2607.00001",
            "title": "Vision Language Action for robot manipulation",
            "abstract": "A VLA robot policy.",
            "categories": ["cs.RO"],
        },
        "metadata": {
            "matched_topics": ["embodied-vla"],
            "matched_terms": ["VLA", "robot"],
            "summary": {"one_sentence_summary": "中文摘要", "status": "completed"},
        },
    }


def _completed_populated_run(store: PaperDailyStore) -> str:
    run_id = store.start_run("alice", "2026-07-10", "2026-07-10")
    store.save_recommendations(run_id, [_recommendation()])
    store.complete_run(
        run_id,
        fetched_count=10,
        candidate_count=1,
        recommendation_count=1,
        summary_count=1,
        delivery_count=1,
    )
    return run_id


def test_desktop_adapter_hides_empty_runs_and_defaults_to_populated_digest(tmp_path: Path) -> None:
    path, config = _config(tmp_path)
    store = PaperDailyStore(config.database)
    store.initialize()
    populated_run_id = _completed_populated_run(store)
    empty_run_id = store.start_run("alice", "2026-07-11", "2026-07-11")
    store.complete_run(empty_run_id, fetched_count=20)
    gui = PaperDailyGui(path)

    digest = gui.latest_digest(user_id="alice")["digest"]
    runs = gui.list_digests(user_id="alice")["runs"]
    status = gui.status(user_id="alice")

    assert digest is not None
    assert digest["run"]["run_id"] == populated_run_id
    assert len(digest["recommendations"]) == 1
    assert [run["run_id"] for run in runs] == [populated_run_id]
    assert status["providers"]["embedding"]["semantic_recall_enabled"] is True
    assert status["providers"]["embedding"]["semantic_recall_active"] is False
    assert status["providers"]["embedding"]["semantic_recall_threshold"] == 0.58


def test_desktop_adapter_reuses_existing_digest_when_all_matches_are_handled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path, config = _config(tmp_path)
    store = PaperDailyStore(config.database)
    store.initialize()
    populated_run_id = _completed_populated_run(store)

    class PreviewService:
        def __init__(self) -> None:
            self.store = store
            self.calls: list[bool] = []

        def catchup_plan(self):
            return SimpleNamespace()

        def select_window(self, _plan, _choice, **_kwargs):
            return DateWindow(date(2026, 7, 10), date(2026, 7, 10), "yesterday")

        def run(self, window, *, dry_run, **_kwargs):
            self.calls.append(dry_run)
            assert dry_run is True
            return RunOutcome(
                digest=Digest(
                    run_id="00000000-0000-0000-0000-000000000000",
                    user_id="alice",
                    window_start=window.start_date,
                    window_end=window.end_date,
                    recommendations=[],
                    generated_at=datetime.now(timezone.utc),
                    stats={"fetched_count": 10, "matched_count": 1, "handled_count": 1, "candidate_count": 0},
                    catchup_mode="daily",
                ),
                dry_run=True,
            )

    service = PreviewService()
    gui = PaperDailyGui(path)
    monkeypatch.setattr(gui, "_context", lambda _user_id=None: (config, service))

    task = gui.start_digest_task(user_id="alice", choice="yesterday", dry_run=False)
    for _ in range(100):
        result = gui.task(task["task_id"])
        if result and result["status"] != "running":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("PaperDaily reuse task did not finish")

    assert result["status"] == "completed"
    assert result["result"]["reused_existing_digest"] is True
    assert result["result"]["reused_run_id"] == populated_run_id
    assert len(result["result"]["recommendations"]) == 1
    assert service.calls == [True]
    assert len(store.list_runs("alice")) == 1
