"""SQLite persistence for the PaperDaily extension.

The store deliberately uses side tables prefixed with ``paperdaily_`` so it
can be initialized against an existing PaperFlow database without changing
the meaning or schema of PaperFlow's own tables.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from paperdaily.identifiers import canonicalize_arxiv_id, paper_arxiv_identity

SCHEMA_VERSION = 2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paperdaily_schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paperdaily_state (
    user_id TEXT PRIMARY KEY,
    last_completed_window_end TEXT,
    last_successful_run_at TEXT,
    last_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paperdaily_runs (
    run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'daily',
    status TEXT NOT NULL,
    fetched_count INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    recommendation_count INTEGER NOT NULL DEFAULT 0,
    summary_count INTEGER NOT NULL DEFAULT 0,
    delivery_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (window_start <= window_end)
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_runs_user_window
    ON paperdaily_runs(user_id, window_end DESC, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_paperdaily_runs_status
    ON paperdaily_runs(status, started_at DESC);

CREATE TABLE IF NOT EXISTS paperdaily_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    raw_arxiv_id TEXT,
    arxiv_version INTEGER,
    rank INTEGER,
    score REAL,
    topic_id TEXT,
    title TEXT,
    paper_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, canonical_id),
    FOREIGN KEY (run_id) REFERENCES paperdaily_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_recommendations_run_rank
    ON paperdaily_recommendations(run_id, rank, recommendation_id);
CREATE INDEX IF NOT EXISTS idx_paperdaily_recommendations_canonical
    ON paperdaily_recommendations(canonical_id);

CREATE TABLE IF NOT EXISTS paperdaily_summaries (
    summary_id TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    abstract_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    language TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (canonical_id, abstract_hash, prompt_version, language, provider, model)
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_summaries_canonical
    ON paperdaily_summaries(canonical_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS paperdaily_reranks (
    rerank_id TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    abstract_hash TEXT NOT NULL,
    topic_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (canonical_id, abstract_hash, topic_hash, prompt_version, provider, model)
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_reranks_canonical
    ON paperdaily_reranks(canonical_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS paperdaily_deliveries (
    delivery_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    payload_hash TEXT,
    external_id TEXT,
    error_message TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, channel),
    FOREIGN KEY (run_id) REFERENCES paperdaily_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_deliveries_status
    ON paperdaily_deliveries(status, updated_at);

CREATE TABLE IF NOT EXISTS paperdaily_feedback (
    feedback_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id TEXT,
    canonical_id TEXT NOT NULL,
    action TEXT NOT NULL,
    weight REAL,
    idempotency_key TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (user_id, idempotency_key),
    FOREIGN KEY (run_id) REFERENCES paperdaily_runs(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_feedback_user_time
    ON paperdaily_feedback(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_paperdaily_feedback_canonical
    ON paperdaily_feedback(canonical_id, created_at DESC);

CREATE TABLE IF NOT EXISTS paperdaily_agent_runs (
    agent_run_id TEXT PRIMARY KEY,
    run_id TEXT,
    canonical_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash TEXT,
    output_json TEXT NOT NULL DEFAULT '{}',
    usage_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (run_id) REFERENCES paperdaily_runs(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_paperdaily_agent_runs_lookup
    ON paperdaily_agent_runs(canonical_id, task_type, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_paperdaily_agent_runs_status
    ON paperdaily_agent_runs(status, started_at DESC);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return {} if default is None else default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {} if default is None else default


def _merge_json(existing: Any, incoming: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = _json_loads(existing, {})
    if not isinstance(merged, dict):
        merged = {}
    if incoming:
        merged.update(dict(incoming))
    return merged


def _normalize_date(value: date | datetime | str, field_name: str) -> str:
    if isinstance(value, datetime):
        normalized = value.date().isoformat()
    elif isinstance(value, date):
        normalized = value.isoformat()
    else:
        normalized = str(value or "").strip()
    try:
        return date.fromisoformat(normalized).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from exc


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _uuid_text(value: str | None = None) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Expected a UUID, got {value!r}") from exc


def hash_abstract(abstract: Any) -> str:
    """Return a stable hash for summary-cache invalidation."""

    normalized = " ".join(str(abstract or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class PaperDailyStore:
    """Transactional access to PaperDaily's SQLite side tables."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Idempotently create side tables without modifying PaperFlow tables."""

        now = _utc_now()
        with self._transaction() as conn:
            # ``sqlite3.Connection.executescript`` commits an already-open
            # transaction before running the script. Execute these simple DDL
            # statements individually so schema creation and the version row
            # remain atomic.
            for statement in _SCHEMA_SQL.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute(
                """
                INSERT INTO paperdaily_schema_meta (key, value, updated_at)
                VALUES ('schema_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (str(SCHEMA_VERSION), now),
            )

    # ------------------------------------------------------------------
    # Run state and watermark
    # ------------------------------------------------------------------

    def start_run(
        self,
        user_id: str,
        window_start: date | datetime | str,
        window_end: date | datetime | str,
        *,
        mode: str = "daily",
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        normalized_user = _required_text(user_id, "user_id")
        start = _normalize_date(window_start, "window_start")
        end = _normalize_date(window_end, "window_end")
        if start > end:
            raise ValueError("window_start must not be after window_end")
        normalized_mode = _required_text(mode, "mode")
        normalized_run_id = _uuid_text(run_id)
        now = _utc_now()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO paperdaily_state
                    (user_id, last_completed_window_end, last_successful_run_at, last_run_id, created_at, updated_at)
                VALUES (?, NULL, NULL, NULL, ?, ?)
                """,
                (normalized_user, now, now),
            )
            conn.execute(
                """
                INSERT INTO paperdaily_runs
                    (run_id, user_id, window_start, window_end, mode, status, metadata_json, started_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    normalized_run_id,
                    normalized_user,
                    start,
                    end,
                    normalized_mode,
                    _json_dumps(dict(metadata or {})),
                    now,
                ),
            )
        return normalized_run_id

    @staticmethod
    def _date_text(value: date) -> str:
        """Store dates in the same sortable ISO representation as run rows."""

        return value.isoformat()

    def _contiguous_completed_end(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        watermark: str,
    ) -> str:
        """Extend a known contiguous watermark through completed run intervals.

        ``paperdaily_state.last_completed_window_end`` deliberately means
        *contiguous* coverage, not simply the end of the most recently finished
        run.  This lets a user inspect a newer seven-day slice without silently
        discarding an earlier gap.
        """

        cursor = date.fromisoformat(watermark)
        rows = conn.execute(
            """
            SELECT window_start, window_end
            FROM paperdaily_runs
            WHERE user_id = ? AND status = 'completed' AND window_end > ?
            ORDER BY window_start ASC, window_end ASC, started_at ASC, run_id ASC
            """,
            (user_id, watermark),
        ).fetchall()
        for interval in rows:
            start = date.fromisoformat(interval["window_start"])
            end = date.fromisoformat(interval["window_end"])
            if end <= cursor:
                continue
            if start > cursor + timedelta(days=1):
                # Rows are ordered by start date, so no later interval can
                # close this earliest missing date.
                break
            cursor = max(cursor, end)
        return self._date_text(cursor)

    def complete_run(
        self,
        run_id: str,
        *,
        fetched_count: int | None = None,
        candidate_count: int | None = None,
        recommendation_count: int | None = None,
        summary_count: int | None = None,
        delivery_count: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_run_id = _uuid_text(run_id)
        now = _utc_now()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_runs WHERE run_id = ?", (normalized_run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run_id: {normalized_run_id}")
            if row["status"] == "completed":
                return self._run_record(row)
            if row["status"] != "running":
                raise ValueError(f"Cannot complete run in status {row['status']!r}")

            counts = {
                "fetched_count": row["fetched_count"]
                if fetched_count is None
                else max(0, int(fetched_count)),
                "candidate_count": row["candidate_count"]
                if candidate_count is None
                else max(0, int(candidate_count)),
                "recommendation_count": (
                    row["recommendation_count"]
                    if recommendation_count is None
                    else max(0, int(recommendation_count))
                ),
                "summary_count": row["summary_count"]
                if summary_count is None
                else max(0, int(summary_count)),
                "delivery_count": row["delivery_count"]
                if delivery_count is None
                else max(0, int(delivery_count)),
            }
            merged_metadata = _merge_json(row["metadata_json"], metadata)
            conn.execute(
                """
                UPDATE paperdaily_runs
                SET status = 'completed', fetched_count = ?, candidate_count = ?, recommendation_count = ?,
                    summary_count = ?, delivery_count = ?, metadata_json = ?, error_message = NULL,
                    completed_at = ?
                WHERE run_id = ?
                """,
                (
                    counts["fetched_count"],
                    counts["candidate_count"],
                    counts["recommendation_count"],
                    counts["summary_count"],
                    counts["delivery_count"],
                    _json_dumps(merged_metadata),
                    now,
                    normalized_run_id,
                ),
            )

            state = conn.execute(
                "SELECT last_completed_window_end FROM paperdaily_state WHERE user_id = ?",
                (row["user_id"],),
            ).fetchone()
            previous_watermark = state["last_completed_window_end"] if state else None
            if previous_watermark:
                # The current run has already been made visible as completed,
                # so it participates in this atomic interval merge.
                next_watermark = self._contiguous_completed_end(
                    conn,
                    user_id=row["user_id"],
                    watermark=str(previous_watermark),
                )
            else:
                # The first successful run establishes the initial historical
                # baseline selected during setup/first-run catchup.
                next_watermark = row["window_end"]

            # Completion and contiguous-watermark movement are atomic.  A
            # successful non-contiguous backfill records its run, but leaves
            # the watermark at the earliest unresolved date.
            conn.execute(
                """
                INSERT INTO paperdaily_state
                    (user_id, last_completed_window_end, last_successful_run_at, last_run_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_completed_window_end = excluded.last_completed_window_end,
                    last_successful_run_at = excluded.last_successful_run_at,
                    last_run_id = excluded.last_run_id,
                    updated_at = excluded.updated_at
                """,
                (row["user_id"], next_watermark, now, normalized_run_id, now, now),
            )
            completed = conn.execute(
                "SELECT * FROM paperdaily_runs WHERE run_id = ?", (normalized_run_id,)
            ).fetchone()
            return self._run_record(completed)

    def fail_run(
        self,
        run_id: str,
        error_message: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_run_id = _uuid_text(run_id)
        now = _utc_now()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_runs WHERE run_id = ?", (normalized_run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run_id: {normalized_run_id}")
            if row["status"] == "completed":
                raise ValueError("A completed run cannot be failed")
            merged_metadata = _merge_json(row["metadata_json"], metadata)
            conn.execute(
                """
                UPDATE paperdaily_runs
                SET status = 'failed', metadata_json = ?, error_message = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (_json_dumps(merged_metadata), str(error_message or "").strip(), now, normalized_run_id),
            )
            failed = conn.execute(
                "SELECT * FROM paperdaily_runs WHERE run_id = ?", (normalized_run_id,)
            ).fetchone()
            return self._run_record(failed)

    def get_state(self, user_id: str) -> dict[str, Any] | None:
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_state WHERE user_id = ?", (_required_text(user_id, "user_id"),)
            ).fetchone()
        return dict(row) if row else None

    def first_uncovered_window(
        self,
        user_id: str,
        *,
        through_date: date | datetime | str,
    ) -> tuple[date, date] | None:
        """Return the earliest contiguous gap through ``through_date``.

        The method only considers completed runs and starts at the persisted
        contiguous watermark.  It is used by catch-up planning after an
        out-of-order slice succeeds: the next automatic run fills the earliest
        gap rather than continually re-running the newest slice.
        """

        normalized_user = _required_text(user_id, "user_id")
        through = date.fromisoformat(_normalize_date(through_date, "through_date"))
        with self._reader() as conn:
            state = conn.execute(
                "SELECT last_completed_window_end FROM paperdaily_state WHERE user_id = ?",
                (normalized_user,),
            ).fetchone()
            if state is None or not state["last_completed_window_end"]:
                return None
            watermark = date.fromisoformat(state["last_completed_window_end"])
            if watermark >= through:
                return None
            rows = conn.execute(
                """
                SELECT window_start, window_end
                FROM paperdaily_runs
                WHERE user_id = ? AND status = 'completed' AND window_end > ?
                ORDER BY window_start ASC, window_end ASC, started_at ASC, run_id ASC
                """,
                (normalized_user, watermark.isoformat()),
            ).fetchall()

        cursor = watermark
        for interval in rows:
            start = date.fromisoformat(interval["window_start"])
            end = date.fromisoformat(interval["window_end"])
            if end <= cursor:
                continue
            if start > cursor + timedelta(days=1):
                return cursor + timedelta(days=1), min(start - timedelta(days=1), through)
            cursor = max(cursor, end)
            if cursor >= through:
                return None
        return cursor + timedelta(days=1), through

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_runs WHERE run_id = ?", (_uuid_text(run_id),)
            ).fetchone()
        return self._run_record(row) if row else None

    def list_runs(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._reader() as conn:
            rows = conn.execute(
                """
                SELECT * FROM paperdaily_runs
                WHERE user_id = ?
                ORDER BY started_at DESC, run_id DESC
                LIMIT ?
                """,
                (_required_text(user_id, "user_id"), max(1, int(limit))),
            ).fetchall()
        return [self._run_record(row) for row in rows]

    @staticmethod
    def _run_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["metadata"] = _json_loads(record.pop("metadata_json", "{}"), {})
        return record

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def save_recommendations(
        self,
        run_id: str,
        recommendations: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_run_id = _uuid_text(run_id)
        now = _utc_now()
        saved_canonical_ids: list[str] = []
        with self._transaction() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM paperdaily_runs WHERE run_id = ?", (normalized_run_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"Unknown run_id: {normalized_run_id}")

            for position, item_source in enumerate(recommendations, start=1):
                item = dict(item_source)
                nested_paper = item.get("paper")
                paper = dict(nested_paper) if isinstance(nested_paper, Mapping) else dict(item)
                identity = paper_arxiv_identity(item) or paper_arxiv_identity(paper)
                if identity is None:
                    raise ValueError(f"Recommendation has no valid arXiv ID: {item!r}")

                canonical_id = identity.canonical_id
                raw_arxiv_id = str(paper.get("arxiv_id") or item.get("arxiv_id") or "").strip() or None
                rank = item.get("rank", position)
                score = item.get("score")
                topic_id = item.get("topic_id") or item.get("matched_topic")
                title = paper.get("title") or item.get("title")
                metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
                conn.execute(
                    """
                    INSERT INTO paperdaily_recommendations
                        (run_id, canonical_id, raw_arxiv_id, arxiv_version, rank, score, topic_id, title,
                         paper_json, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, canonical_id) DO UPDATE SET
                        raw_arxiv_id = excluded.raw_arxiv_id,
                        arxiv_version = excluded.arxiv_version,
                        rank = excluded.rank,
                        score = excluded.score,
                        topic_id = excluded.topic_id,
                        title = excluded.title,
                        paper_json = excluded.paper_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        normalized_run_id,
                        canonical_id,
                        raw_arxiv_id,
                        identity.version,
                        int(rank) if rank is not None else None,
                        float(score) if score is not None else None,
                        str(topic_id).strip() if topic_id else None,
                        str(title).strip() if title else None,
                        _json_dumps(paper),
                        _json_dumps(dict(metadata)),
                        now,
                        now,
                    ),
                )
                if canonical_id not in saved_canonical_ids:
                    saved_canonical_ids.append(canonical_id)

        records = self.get_recommendations(normalized_run_id)
        by_id = {record["canonical_id"]: record for record in records}
        return [by_id[canonical_id] for canonical_id in saved_canonical_ids if canonical_id in by_id]

    def get_recommendations(self, run_id: str) -> list[dict[str, Any]]:
        with self._reader() as conn:
            rows = conn.execute(
                """
                SELECT * FROM paperdaily_recommendations
                WHERE run_id = ?
                ORDER BY CASE WHEN rank IS NULL THEN 1 ELSE 0 END, rank, recommendation_id
                """,
                (_uuid_text(run_id),),
            ).fetchall()
        return [self._recommendation_record(row) for row in rows]

    def find_latest_recommendation(
        self,
        user_id: str,
        canonical_id: str,
    ) -> dict[str, Any] | None:
        """Find the newest saved recommendation for feedback or deep reading."""

        canonical = canonicalize_arxiv_id(canonical_id)
        if not canonical:
            return None
        with self._reader() as conn:
            row = conn.execute(
                """
                SELECT recommendation.*
                FROM paperdaily_recommendations AS recommendation
                JOIN paperdaily_runs AS run ON run.run_id = recommendation.run_id
                WHERE run.user_id = ? AND recommendation.canonical_id = ?
                ORDER BY run.started_at DESC, recommendation.recommendation_id DESC
                LIMIT 1
                """,
                (_required_text(user_id, "user_id"), canonical),
            ).fetchone()
        return self._recommendation_record(row) if row else None

    @staticmethod
    def _recommendation_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["paper"] = _json_loads(record.pop("paper_json", "{}"), {})
        record["metadata"] = _json_loads(record.pop("metadata_json", "{}"), {})
        return record

    # ------------------------------------------------------------------
    # Summary cache
    # ------------------------------------------------------------------

    def save_summary(
        self,
        canonical_id: str,
        abstract_hash: str,
        prompt_version: str,
        language: str,
        provider: str,
        model: str,
        payload: Any,
        *,
        status: str = "completed",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        canonical = canonicalize_arxiv_id(canonical_id)
        if not canonical:
            raise ValueError(f"Invalid arXiv identifier: {canonical_id!r}")
        key = (
            canonical,
            _required_text(abstract_hash, "abstract_hash"),
            _required_text(prompt_version, "prompt_version"),
            _required_text(language, "language").lower(),
            _required_text(provider, "provider").lower(),
            _required_text(model, "model"),
        )
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO paperdaily_summaries
                    (summary_id, canonical_id, abstract_hash, prompt_version, language, provider, model,
                     status, payload_json, prompt_tokens, completion_tokens, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_id, abstract_hash, prompt_version, language, provider, model) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    str(uuid4()),
                    *key,
                    _required_text(status, "status").lower(),
                    _json_dumps(payload),
                    max(0, int(prompt_tokens)),
                    max(0, int(completion_tokens)),
                    str(error_message).strip() if error_message else None,
                    now,
                    now,
                ),
            )
        summary = self.get_summary(*key)
        if summary is None:  # pragma: no cover - defensive consistency check
            raise RuntimeError("Summary cache write could not be read back")
        return summary

    def get_summary(
        self,
        canonical_id: str,
        abstract_hash: str,
        prompt_version: str,
        language: str,
        provider: str,
        model: str,
    ) -> dict[str, Any] | None:
        canonical = canonicalize_arxiv_id(canonical_id)
        if not canonical:
            return None
        with self._reader() as conn:
            row = conn.execute(
                """
                SELECT * FROM paperdaily_summaries
                WHERE canonical_id = ? AND abstract_hash = ? AND prompt_version = ?
                  AND language = ? AND provider = ? AND model = ?
                """,
                (
                    canonical,
                    str(abstract_hash).strip(),
                    str(prompt_version).strip(),
                    str(language).strip().lower(),
                    str(provider).strip().lower(),
                    str(model).strip(),
                ),
            ).fetchone()
        return self._summary_record(row) if row else None

    @staticmethod
    def _summary_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["payload"] = _json_loads(record.pop("payload_json", "{}"), {})
        return record

    # ------------------------------------------------------------------
    # LLM reranking cache
    # ------------------------------------------------------------------

    def save_rerank(
        self,
        canonical_id: str,
        abstract_hash: str,
        topic_hash: str,
        prompt_version: str,
        provider: str,
        model: str,
        payload: Any,
        *,
        status: str = "completed",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Persist one structured rerank decision without touching legacy tables."""

        canonical = canonicalize_arxiv_id(canonical_id)
        if not canonical:
            raise ValueError(f"Invalid arXiv identifier: {canonical_id!r}")
        key = (
            canonical,
            _required_text(abstract_hash, "abstract_hash"),
            _required_text(topic_hash, "topic_hash"),
            _required_text(prompt_version, "prompt_version"),
            _required_text(provider, "provider").lower(),
            _required_text(model, "model"),
        )
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO paperdaily_reranks
                    (rerank_id, canonical_id, abstract_hash, topic_hash, prompt_version, provider, model,
                     status, payload_json, prompt_tokens, completion_tokens, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_id, abstract_hash, topic_hash, prompt_version, provider, model) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    str(uuid4()),
                    *key,
                    _required_text(status, "status").lower(),
                    _json_dumps(payload),
                    max(0, int(prompt_tokens)),
                    max(0, int(completion_tokens)),
                    str(error_message).strip() if error_message else None,
                    now,
                    now,
                ),
            )
        rerank = self.get_rerank(*key)
        if rerank is None:  # pragma: no cover - defensive consistency check
            raise RuntimeError("Rerank cache write could not be read back")
        return rerank

    def get_rerank(
        self,
        canonical_id: str,
        abstract_hash: str,
        topic_hash: str,
        prompt_version: str,
        provider: str,
        model: str,
    ) -> dict[str, Any] | None:
        canonical = canonicalize_arxiv_id(canonical_id)
        if not canonical:
            return None
        with self._reader() as conn:
            row = conn.execute(
                """
                SELECT * FROM paperdaily_reranks
                WHERE canonical_id = ? AND abstract_hash = ? AND topic_hash = ? AND prompt_version = ?
                  AND provider = ? AND model = ?
                """,
                (
                    canonical,
                    str(abstract_hash).strip(),
                    str(topic_hash).strip(),
                    str(prompt_version).strip(),
                    str(provider).strip().lower(),
                    str(model).strip(),
                ),
            ).fetchone()
        return self._rerank_record(row) if row else None

    @staticmethod
    def _rerank_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["payload"] = _json_loads(record.pop("payload_json", "{}"), {})
        return record

    # ------------------------------------------------------------------
    # Channel deliveries
    # ------------------------------------------------------------------

    def ensure_delivery(
        self,
        run_id: str,
        channel: str,
        *,
        payload_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_run_id = _uuid_text(run_id)
        normalized_channel = _required_text(channel, "channel").lower()
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO paperdaily_deliveries
                    (delivery_id, run_id, channel, status, attempts, payload_hash, metadata_json,
                     created_at, updated_at)
                VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                ON CONFLICT(run_id, channel) DO NOTHING
                """,
                (
                    str(uuid4()),
                    normalized_run_id,
                    normalized_channel,
                    str(payload_hash).strip() if payload_hash else None,
                    _json_dumps(dict(metadata or {})),
                    now,
                    now,
                ),
            )
        delivery = self.get_delivery(normalized_run_id, normalized_channel)
        if delivery is None:  # pragma: no cover
            raise RuntimeError("Delivery write could not be read back")
        return delivery

    def mark_delivery_started(self, run_id: str, channel: str) -> dict[str, Any]:
        delivery = self.ensure_delivery(run_id, channel)
        if delivery["status"] == "succeeded":
            return delivery
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE paperdaily_deliveries
                SET status = 'sending', attempts = attempts + 1, started_at = ?, completed_at = NULL,
                    error_message = NULL, updated_at = ?
                WHERE run_id = ? AND channel = ? AND status != 'succeeded'
                """,
                (now, now, _uuid_text(run_id), _required_text(channel, "channel").lower()),
            )
        return self.get_delivery(run_id, channel) or delivery

    def complete_delivery(
        self,
        run_id: str,
        channel: str,
        *,
        external_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        delivery = self.ensure_delivery(run_id, channel)
        if delivery["status"] == "succeeded":
            return delivery
        now = _utc_now()
        merged_metadata = _merge_json(delivery.get("metadata"), metadata)
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE paperdaily_deliveries
                SET status = 'succeeded', attempts = CASE WHEN attempts = 0 THEN 1 ELSE attempts END,
                    external_id = COALESCE(?, external_id), metadata_json = ?, error_message = NULL,
                    completed_at = ?, updated_at = ?
                WHERE run_id = ? AND channel = ? AND status != 'succeeded'
                """,
                (
                    str(external_id).strip() if external_id else None,
                    _json_dumps(merged_metadata),
                    now,
                    now,
                    _uuid_text(run_id),
                    _required_text(channel, "channel").lower(),
                ),
            )
        return self.get_delivery(run_id, channel) or delivery

    def fail_delivery(self, run_id: str, channel: str, error_message: Any) -> dict[str, Any]:
        delivery = self.ensure_delivery(run_id, channel)
        if delivery["status"] == "succeeded":
            return delivery
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE paperdaily_deliveries
                SET status = 'failed', attempts = CASE WHEN attempts = 0 THEN 1 ELSE attempts END,
                    error_message = ?, completed_at = ?, updated_at = ?
                WHERE run_id = ? AND channel = ? AND status != 'succeeded'
                """,
                (
                    str(error_message or "").strip(),
                    now,
                    now,
                    _uuid_text(run_id),
                    _required_text(channel, "channel").lower(),
                ),
            )
        return self.get_delivery(run_id, channel) or delivery

    def get_delivery(self, run_id: str, channel: str) -> dict[str, Any] | None:
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_deliveries WHERE run_id = ? AND channel = ?",
                (_uuid_text(run_id), _required_text(channel, "channel").lower()),
            ).fetchone()
        return self._delivery_record(row) if row else None

    def list_deliveries(self, run_id: str) -> list[dict[str, Any]]:
        with self._reader() as conn:
            rows = conn.execute(
                "SELECT * FROM paperdaily_deliveries WHERE run_id = ? ORDER BY channel",
                (_uuid_text(run_id),),
            ).fetchall()
        return [self._delivery_record(row) for row in rows]

    @staticmethod
    def _delivery_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["metadata"] = _json_loads(record.pop("metadata_json", "{}"), {})
        return record

    # ------------------------------------------------------------------
    # Feedback events
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        user_id: str,
        canonical_id: str,
        action: str,
        *,
        run_id: str | None = None,
        weight: float | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_user = _required_text(user_id, "user_id")
        canonical = canonicalize_arxiv_id(canonical_id)
        if not canonical:
            raise ValueError(f"Invalid arXiv identifier: {canonical_id!r}")
        normalized_run_id = _uuid_text(run_id) if run_id else None
        normalized_key = str(idempotency_key).strip() if idempotency_key else None
        feedback_id = str(uuid4())
        now = _utc_now()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO paperdaily_feedback
                    (feedback_id, user_id, run_id, canonical_id, action, weight,
                     idempotency_key, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, idempotency_key) DO NOTHING
                """,
                (
                    feedback_id,
                    normalized_user,
                    normalized_run_id,
                    canonical,
                    _required_text(action, "action").lower(),
                    float(weight) if weight is not None else None,
                    normalized_key,
                    _json_dumps(dict(metadata or {})),
                    now,
                ),
            )
            if normalized_key:
                row = conn.execute(
                    "SELECT * FROM paperdaily_feedback WHERE user_id = ? AND idempotency_key = ?",
                    (normalized_user, normalized_key),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM paperdaily_feedback WHERE feedback_id = ?", (feedback_id,)
                ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("Feedback write could not be read back")
        return self._feedback_record(row)

    def list_feedback(
        self,
        user_id: str,
        *,
        canonical_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [_required_text(user_id, "user_id")]
        sql = "SELECT * FROM paperdaily_feedback WHERE user_id = ?"
        if canonical_id is not None:
            canonical = canonicalize_arxiv_id(canonical_id)
            if not canonical:
                return []
            sql += " AND canonical_id = ?"
            params.append(canonical)
        sql += " ORDER BY created_at DESC, feedback_id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._reader() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._feedback_record(row) for row in rows]

    def list_handled_canonical_ids(
        self,
        user_id: str,
        *,
        include_recommendations: bool = True,
        include_legacy_paperflow: bool = True,
    ) -> set[str]:
        """Return canonical IDs that should not be recommended as new again.

        Completed PaperDaily recommendations and explicit PaperDaily feedback
        are always considered handled. When the store was added to an existing
        PaperFlow database, legacy push/selection/report behavior is included
        as well so an upgrade does not immediately repeat old papers.
        """

        normalized_user = _required_text(user_id, "user_id")
        handled: set[str] = set()
        with self._reader() as conn:
            feedback_rows = conn.execute(
                "SELECT canonical_id FROM paperdaily_feedback WHERE user_id = ?",
                (normalized_user,),
            ).fetchall()
            handled.update(str(row["canonical_id"]) for row in feedback_rows if row["canonical_id"])

            if include_recommendations:
                recommendation_rows = conn.execute(
                    """
                    SELECT DISTINCT rec.canonical_id
                    FROM paperdaily_recommendations rec
                    JOIN paperdaily_runs run ON run.run_id = rec.run_id
                    WHERE run.user_id = ? AND run.status = 'completed'
                    """,
                    (normalized_user,),
                ).fetchall()
                handled.update(str(row["canonical_id"]) for row in recommendation_rows if row["canonical_id"])

            if include_legacy_paperflow:
                table_names = {
                    str(row["name"])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('papers', 'behavior_logs')"
                    ).fetchall()
                }
                if {"papers", "behavior_logs"} <= table_names:
                    legacy_rows = conn.execute(
                        """
                        SELECT p.arxiv_id
                        FROM behavior_logs bl
                        JOIN papers p ON p.id = bl.paper_id
                        WHERE bl.user_id = ?
                          AND bl.action IN ('pushed', 'selected', 'skipped', 'created_report')
                        """,
                        (normalized_user,),
                    ).fetchall()
                    for row in legacy_rows:
                        canonical = canonicalize_arxiv_id(row["arxiv_id"])
                        if canonical:
                            handled.add(canonical)

        return handled

    @staticmethod
    def _feedback_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["metadata"] = _json_loads(record.pop("metadata_json", "{}"), {})
        return record

    # ------------------------------------------------------------------
    # Agent runs
    # ------------------------------------------------------------------

    def start_agent_run(
        self,
        provider: str,
        model: str,
        task_type: str,
        *,
        run_id: str | None = None,
        canonical_id: str | None = None,
        input_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        normalized_run_id = _uuid_text(run_id) if run_id else None
        canonical = None
        if canonical_id is not None:
            canonical = canonicalize_arxiv_id(canonical_id)
            if not canonical:
                raise ValueError(f"Invalid arXiv identifier: {canonical_id!r}")
        agent_run_id = str(uuid4())
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO paperdaily_agent_runs
                    (agent_run_id, run_id, canonical_id, provider, model, task_type, status,
                     input_hash, output_json, usage_json, metadata_json, started_at)
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?, '{}', '{}', ?, ?)
                """,
                (
                    agent_run_id,
                    normalized_run_id,
                    canonical,
                    _required_text(provider, "provider").lower(),
                    _required_text(model, "model"),
                    _required_text(task_type, "task_type").lower(),
                    str(input_hash).strip() if input_hash else None,
                    _json_dumps(dict(metadata or {})),
                    now,
                ),
            )
        return agent_run_id

    def complete_agent_run(
        self,
        agent_run_id: str,
        output: Any,
        *,
        usage: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_id = _uuid_text(agent_run_id)
        now = _utc_now()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_agent_runs WHERE agent_run_id = ?", (normalized_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown agent_run_id: {normalized_id}")
            if row["status"] == "completed":
                return self._agent_run_record(row)
            if row["status"] != "running":
                raise ValueError(f"Cannot complete agent run in status {row['status']!r}")
            merged_metadata = _merge_json(row["metadata_json"], metadata)
            conn.execute(
                """
                UPDATE paperdaily_agent_runs
                SET status = 'completed', output_json = ?, usage_json = ?, metadata_json = ?,
                    error_message = NULL, completed_at = ?
                WHERE agent_run_id = ?
                """,
                (
                    _json_dumps(output),
                    _json_dumps(dict(usage or {})),
                    _json_dumps(merged_metadata),
                    now,
                    normalized_id,
                ),
            )
            completed = conn.execute(
                "SELECT * FROM paperdaily_agent_runs WHERE agent_run_id = ?", (normalized_id,)
            ).fetchone()
            return self._agent_run_record(completed)

    def fail_agent_run(self, agent_run_id: str, error_message: Any) -> dict[str, Any]:
        normalized_id = _uuid_text(agent_run_id)
        now = _utc_now()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_agent_runs WHERE agent_run_id = ?", (normalized_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown agent_run_id: {normalized_id}")
            if row["status"] == "completed":
                raise ValueError("A completed agent run cannot be failed")
            conn.execute(
                """
                UPDATE paperdaily_agent_runs
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE agent_run_id = ?
                """,
                (str(error_message or "").strip(), now, normalized_id),
            )
            failed = conn.execute(
                "SELECT * FROM paperdaily_agent_runs WHERE agent_run_id = ?", (normalized_id,)
            ).fetchone()
            return self._agent_run_record(failed)

    def get_agent_run(self, agent_run_id: str) -> dict[str, Any] | None:
        with self._reader() as conn:
            row = conn.execute(
                "SELECT * FROM paperdaily_agent_runs WHERE agent_run_id = ?", (_uuid_text(agent_run_id),)
            ).fetchone()
        return self._agent_run_record(row) if row else None

    @staticmethod
    def _agent_run_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["output"] = _json_loads(record.pop("output_json", "{}"), {})
        record["usage"] = _json_loads(record.pop("usage_json", "{}"), {})
        record["metadata"] = _json_loads(record.pop("metadata_json", "{}"), {})
        return record


__all__ = ["PaperDailyStore", "SCHEMA_VERSION", "hash_abstract"]
