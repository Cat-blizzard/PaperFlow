from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from uuid import UUID

import pytest

from paperdaily.storage import PaperDailyStore, hash_abstract


def _table_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }


def test_initialize_is_idempotent_and_preserves_existing_paperflow_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "paperflow.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, arxiv_id TEXT UNIQUE)")
        conn.execute("INSERT INTO papers (arxiv_id) VALUES ('2607.00001v1')")

    PaperDailyStore(db_path)
    PaperDailyStore(db_path)

    expected = {
        "paperdaily_state",
        "paperdaily_runs",
        "paperdaily_recommendations",
        "paperdaily_summaries",
        "paperdaily_reranks",
        "paperdaily_deliveries",
        "paperdaily_feedback",
        "paperdaily_agent_runs",
    }
    assert expected <= _table_names(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT arxiv_id FROM papers").fetchone()[0] == "2607.00001v1"


def test_run_lifecycle_advances_watermark_only_after_success(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    failed_id = store.start_run("alice", "2026-07-01", "2026-07-03", mode="catchup")
    UUID(failed_id)
    assert store.get_state("alice")["last_completed_window_end"] is None

    failed = store.fail_run(failed_id, "network unavailable")
    assert failed["status"] == "failed"
    assert store.get_state("alice")["last_completed_window_end"] is None

    completed_id = store.start_run(
        "alice",
        "2026-07-04",
        "2026-07-07",
        metadata={"choice": "7d"},
    )
    completed = store.complete_run(
        completed_id,
        fetched_count=100,
        candidate_count=20,
        recommendation_count=10,
        summary_count=10,
        delivery_count=1,
        metadata={"digest": "curated"},
    )

    assert completed["status"] == "completed"
    assert completed["window_start"] == "2026-07-04"
    assert completed["window_end"] == "2026-07-07"
    assert completed["fetched_count"] == 100
    assert completed["metadata"] == {"choice": "7d", "digest": "curated"}
    state = store.get_state("alice")
    assert state["last_completed_window_end"] == "2026-07-07"
    assert state["last_run_id"] == completed_id

    # Finishing an older backfill later must not rewind the source watermark.
    older_id = store.start_run("alice", "2026-06-01", "2026-06-30", mode="catchup")
    store.complete_run(older_id)
    assert store.get_state("alice")["last_completed_window_end"] == "2026-07-07"


def test_out_of_order_completed_windows_merge_only_after_earliest_gap_is_filled(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")

    baseline = store.start_run("alice", "2026-07-01", "2026-07-01")
    store.complete_run(baseline)
    newer_slice = store.start_run("alice", "2026-07-05", "2026-07-11", mode="catchup")
    store.complete_run(newer_slice)

    # The newer successful slice is retained, but cannot prove that 7/2..7/4
    # were handled, so the incremental watermark remains at 7/1.
    assert store.get_run(newer_slice)["status"] == "completed"
    assert store.get_state("alice")["last_completed_window_end"] == "2026-07-01"
    assert store.first_uncovered_window("alice", through_date="2026-07-11") == (
        date(2026, 7, 2),
        date(2026, 7, 4),
    )

    gap = store.start_run("alice", "2026-07-02", "2026-07-04", mode="catchup")
    store.complete_run(gap)

    # Completing the gap atomically merges the already-completed 7/5..7/11
    # interval into the contiguous watermark.
    assert store.get_state("alice")["last_completed_window_end"] == "2026-07-11"
    assert store.first_uncovered_window("alice", through_date="2026-07-11") is None


def test_complete_run_is_transactional_when_metric_validation_fails(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    run_id = store.start_run("alice", "2026-07-01", "2026-07-01")

    with pytest.raises(ValueError):
        store.complete_run(run_id, fetched_count="not-an-integer")  # type: ignore[arg-type]

    assert store.get_run(run_id)["status"] == "running"
    assert store.get_state("alice")["last_completed_window_end"] is None


def test_run_window_validation(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    with pytest.raises(ValueError, match="window_start"):
        store.start_run("alice", "2026-07-02", "2026-07-01")
    with pytest.raises(ValueError, match="ISO date"):
        store.start_run("alice", "07/01/2026", "2026-07-02")


def test_recommendations_are_saved_and_upserted_per_run_and_canonical_id(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    run_id = store.start_run("alice", "2026-07-01", "2026-07-01")

    store.save_recommendations(
        run_id,
        [
            {
                "paper": {
                    "arxiv_id": "2607.00001v2",
                    "title": "A VLA Paper",
                    "abstract": "First abstract",
                },
                "rank": 1,
                "score": 0.91,
                "topic_id": "embodied-vla",
                "metadata": {"reason": "exact phrase"},
            },
            {
                "arxiv_id": "2607.00002v1",
                "title": "A Robot Data Paper",
                "rank": 2,
                "score": 0.82,
            },
        ],
    )
    store.save_recommendations(
        run_id,
        [
            {
                "arxiv_id": "https://arxiv.org/abs/2607.00001v3",
                "title": "A VLA Paper, Revised",
                "rank": 1,
                "score": 0.95,
            }
        ],
    )

    rows = store.get_recommendations(run_id)
    assert len(rows) == 2
    assert rows[0]["canonical_id"] == "2607.00001"
    assert rows[0]["arxiv_version"] == 3
    assert rows[0]["paper"]["title"] == "A VLA Paper, Revised"
    assert rows[0]["score"] == pytest.approx(0.95)
    assert rows[1]["canonical_id"] == "2607.00002"


def test_summary_cache_key_is_exact_and_upserts_payload(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    abstract_hash = hash_abstract("  A   robot policy.  ")
    assert abstract_hash == hash_abstract("A robot policy.")

    first = store.save_summary(
        "2607.00001v1",
        abstract_hash,
        "daily-v1",
        "zh-CN",
        "openai",
        "gpt-test",
        {"one_sentence_summary": "第一版"},
        prompt_tokens=12,
        completion_tokens=8,
    )
    second = store.save_summary(
        "https://arxiv.org/abs/2607.00001v2",
        abstract_hash,
        "daily-v1",
        "zh-CN",
        "OPENAI",
        "gpt-test",
        {"one_sentence_summary": "更新版"},
        prompt_tokens=13,
        completion_tokens=9,
    )

    assert second["summary_id"] == first["summary_id"]
    assert second["payload"]["one_sentence_summary"] == "更新版"
    assert second["prompt_tokens"] == 13
    assert (
        store.get_summary(
            "2607.00001",
            abstract_hash,
            "daily-v1",
            "zh-cn",
            "openai",
            "gpt-test",
        )["summary_id"]
        == first["summary_id"]
    )
    assert (
        store.get_summary(
            "2607.00001",
            abstract_hash,
            "daily-v2",
            "zh-cn",
            "openai",
            "gpt-test",
        )
        is None
    )


def test_embedding_cache_is_isolated_by_content_model_and_dimensions(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    first = store.save_embedding(
        "paper",
        "2607.00001",
        "content-v1",
        "OPENAI",
        "bge-m3",
        3,
        [0.1, 0.2, 0.3],
    )
    updated = store.save_embedding(
        "paper",
        "2607.00001",
        "content-v1",
        "openai",
        "bge-m3",
        3,
        [0.4, 0.5, 0.6],
    )

    assert updated["embedding_id"] == first["embedding_id"]
    assert updated["vector"] == pytest.approx([0.4, 0.5, 0.6])
    assert store.get_embedding("paper", "2607.00001", "content-v2", "openai", "bge-m3", 3) is None
    assert store.get_embedding("paper", "2607.00001", "content-v1", "openai", "other", 3) is None
    with pytest.raises(ValueError, match="dimensions"):
        store.save_embedding("paper", "bad", "hash", "openai", "bge-m3", 3, [0.1])


def test_delivery_is_unique_per_run_and_channel_and_success_is_idempotent(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    run_id = store.start_run("alice", "2026-07-01", "2026-07-01")

    first = store.ensure_delivery(run_id, "Feishu", payload_hash="abc")
    duplicate = store.ensure_delivery(run_id, "feishu", payload_hash="different")
    assert duplicate["delivery_id"] == first["delivery_id"]
    assert duplicate["payload_hash"] == "abc"

    started = store.mark_delivery_started(run_id, "feishu")
    assert started["attempts"] == 1
    assert started["status"] == "sending"
    failed = store.fail_delivery(run_id, "feishu", "timeout")
    assert failed["attempts"] == 1
    assert failed["status"] == "failed"
    assert store.mark_delivery_started(run_id, "feishu")["attempts"] == 2

    succeeded = store.complete_delivery(run_id, "feishu", external_id="om_123")
    repeated = store.complete_delivery(run_id, "feishu", external_id="om_other")
    assert succeeded["status"] == "succeeded"
    assert repeated["delivery_id"] == succeeded["delivery_id"]
    assert repeated["attempts"] == 2
    assert repeated["external_id"] == "om_123"
    assert len(store.list_deliveries(run_id)) == 1


def test_feedback_supports_idempotency_and_canonical_lookup(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    run_id = store.start_run("alice", "2026-07-01", "2026-07-01")

    first = store.record_feedback(
        "alice",
        "2607.00001v2",
        "interested",
        run_id=run_id,
        weight=2,
        idempotency_key="click-123",
        metadata={"source": "terminal"},
    )
    duplicate = store.record_feedback(
        "alice",
        "2607.00001v3",
        "irrelevant",
        run_id=run_id,
        idempotency_key="click-123",
    )

    assert duplicate["feedback_id"] == first["feedback_id"]
    assert duplicate["action"] == "interested"
    assert first["canonical_id"] == "2607.00001"
    rows = store.list_feedback("alice", canonical_id="https://arxiv.org/abs/2607.00001")
    assert len(rows) == 1
    assert rows[0]["metadata"] == {"source": "terminal"}


def test_handled_ids_cover_completed_recommendations_feedback_and_legacy_data(tmp_path: Path) -> None:
    db_path = tmp_path / "paperflow.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, arxiv_id TEXT)")
        conn.execute(
            "CREATE TABLE behavior_logs (id INTEGER PRIMARY KEY, user_id TEXT, paper_id INTEGER, action TEXT)"
        )
        paper_id = conn.execute("INSERT INTO papers (arxiv_id) VALUES ('2607.00003v4')").lastrowid
        conn.execute(
            "INSERT INTO behavior_logs (user_id, paper_id, action) VALUES ('alice', ?, 'pushed')",
            (paper_id,),
        )

    store = PaperDailyStore(db_path)
    run_id = store.start_run("alice", "2026-07-01", "2026-07-01")
    store.save_recommendations(run_id, [{"arxiv_id": "2607.00001v2", "title": "Recommended"}])
    store.record_feedback("alice", "2607.00002v1", "interested")

    # Recommendations only become handled after their run succeeds. Explicit
    # feedback and legacy PaperFlow behavior are available immediately.
    assert store.list_handled_canonical_ids("alice") == {"2607.00002", "2607.00003"}
    store.complete_run(run_id)
    assert store.list_handled_canonical_ids("alice") == {
        "2607.00001",
        "2607.00002",
        "2607.00003",
    }


def test_agent_run_lifecycle(tmp_path: Path) -> None:
    store = PaperDailyStore(tmp_path / "paperflow.db")
    run_id = store.start_run("alice", "2026-07-01", "2026-07-01")
    agent_run_id = store.start_agent_run(
        "codex",
        "gpt-test",
        "reading_note",
        run_id=run_id,
        canonical_id="2607.00001v2",
        input_hash="input-sha",
    )
    UUID(agent_run_id)

    completed = store.complete_agent_run(
        agent_run_id,
        {"note": "结构化阅读笔记"},
        usage={"input_tokens": 100},
    )
    assert completed["status"] == "completed"
    assert completed["canonical_id"] == "2607.00001"
    assert completed["output"] == {"note": "结构化阅读笔记"}
    assert completed["usage"] == {"input_tokens": 100}
    assert store.get_agent_run(agent_run_id)["agent_run_id"] == agent_run_id
