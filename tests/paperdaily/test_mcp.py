"""Offline tests for the deliberately narrow PaperDaily MCP surface."""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

pytest.importorskip("mcp")

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from paperdaily.cli import app
from paperdaily.config import PaperDailyConfig, save_config
from paperdaily.mcp import create_server
from paperdaily.storage import PaperDailyStore
from paperdaily.topics import Topic


def _config(tmp_path: Path) -> tuple[Path, PaperDailyConfig]:
    config = PaperDailyConfig(
        user_id="mcp-user",
        timezone="Asia/Shanghai",
        database=tmp_path / "paperflow.db",
        output_dir=tmp_path / "output",
        topics=[
            Topic(
                id="embodied-vla",
                name="Embodied VLA",
                description="Robotic vision-language-action systems.",
                arxiv_categories=["cs.RO"],
                exact_phrases=["vision-language-action"],
                keywords=["VLA"],
                context_keywords=["robot"],
            ),
            Topic(
                id="disabled-topic",
                name="Disabled",
                description="Not currently active.",
                enabled=False,
            ),
        ],
    )
    path = save_config(config, tmp_path / "paperdaily.yaml")
    return path, config


def _completed_run(config: PaperDailyConfig) -> str:
    store = PaperDailyStore(config.database)
    run_id = store.start_run(config.user_id, date(2026, 7, 10), date(2026, 7, 10))
    store.save_recommendations(
        run_id,
        [
            {
                "rank": 1,
                "score": 0.91,
                "topic_id": "embodied-vla",
                "paper": {
                    "arxiv_id": "2607.08182v2",
                    "title": "A VLA robot paper",
                    "abstract": "A vision-language-action robot policy for manipulation.",
                    "authors": ["A. Researcher"],
                    "categories": ["cs.RO"],
                    "published_at": "2026-07-10T00:00:00Z",
                    # This must not be returned as an arbitrary URL.
                    "pdf_url": "https://untrusted.example/paper.pdf",
                },
                "metadata": {
                    "matched_topics": ["embodied-vla"],
                    "matched_terms": ["VLA", "robot"],
                    "recommendation_reason": "Strong VLA match",
                    "component_scores": {"topic_rule": 1.0},
                    "summary": {"title_zh": "中文标题", "one_sentence_summary": "中文摘要"},
                },
            }
        ],
    )
    store.complete_run(
        run_id,
        fetched_count=1,
        candidate_count=1,
        recommendation_count=1,
        summary_count=1,
        delivery_count=1,
        metadata={"output_path": "C:/must-not-be-exposed.md"},
    )
    return run_id


def _structured_call(server: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = asyncio.run(server.call_tool(name, arguments))
    # FastMCP returns (content, structured_data) when structured_output=True.
    assert isinstance(response, tuple)
    return response[1]


def test_server_exposes_only_bounded_business_tools_and_json_schemas(tmp_path: Path) -> None:
    config_path, _ = _config(tmp_path)
    server = create_server(config_path)

    tools = asyncio.run(server.list_tools())
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {
        "get_status",
        "list_topics",
        "get_daily_digest",
        "search_recommendations",
        "record_feedback",
    }
    assert "generate_reading_note" not in by_name
    assert "query" in by_name["search_recommendations"].inputSchema["properties"]
    assert "action" in by_name["record_feedback"].inputSchema["properties"]
    assert by_name["get_status"].outputSchema["type"] == "object"
    assert "WRITE OPERATION" in (by_name["record_feedback"].description or "")


def test_digest_and_search_are_local_sanitized_reads(tmp_path: Path) -> None:
    config_path, config = _config(tmp_path)
    run_id = _completed_run(config)
    server = create_server(config_path)

    digest = _structured_call(server, "get_daily_digest", {"run_id": run_id, "limit": 5})
    assert digest["ok"] is True
    item = digest["data"]["recommendations"][0]
    assert item["canonical_id"] == "2607.08182"
    assert item["pdf_url"] == "https://arxiv.org/pdf/2607.08182"
    assert "untrusted.example" not in str(item)
    assert "must-not-be-exposed" not in str(digest)

    search = _structured_call(server, "search_recommendations", {"query": "VLA robot"})
    assert search["ok"] is True
    assert search["data"]["returned_count"] == 1
    assert search["data"]["results"][0]["canonical_id"] == "2607.08182"


def test_latest_digest_ignores_a_newer_failed_run(tmp_path: Path) -> None:
    config_path, config = _config(tmp_path)
    completed_run_id = _completed_run(config)
    store = PaperDailyStore(config.database)
    failed_run_id = store.start_run(config.user_id, date(2026, 7, 11), date(2026, 7, 11))
    store.fail_run(failed_run_id, "temporary source failure")
    server = create_server(config_path)

    digest = _structured_call(server, "get_daily_digest", {"limit": 5})

    assert digest["ok"] is True
    assert digest["data"]["run"]["run_id"] == completed_run_id


def test_topics_are_a_bounded_local_read(tmp_path: Path) -> None:
    config_path, _ = _config(tmp_path)
    server = create_server(config_path)

    enabled = _structured_call(server, "list_topics", {"include_disabled": False})
    assert enabled["ok"] is True
    assert [item["id"] for item in enabled["data"]["topics"]] == ["embodied-vla"]

def test_feedback_is_the_explicit_idempotent_write_operation(tmp_path: Path) -> None:
    config_path, config = _config(tmp_path)
    _completed_run(config)
    server = create_server(config_path)

    first = _structured_call(
        server,
        "record_feedback",
        {"arxiv_id": "2607.08182v2", "action": "interested", "idempotency_key": "mcp-confirmed-1"},
    )
    second = _structured_call(
        server,
        "record_feedback",
        {"arxiv_id": "2607.08182", "action": "interested", "idempotency_key": "mcp-confirmed-1"},
    )
    assert first["ok"] is second["ok"] is True
    assert first["data"]["feedback_id"] == second["data"]["feedback_id"]
    assert PaperDailyStore(config.database).list_feedback(config.user_id, limit=10)[0]["action"] == "interested"


def test_cli_mcp_serve_delegates_config_without_starting_a_long_lived_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, _ = _config(tmp_path)
    observed: dict[str, Path] = {}

    def fake_serve(path: Path) -> None:
        observed["path"] = path

    monkeypatch.setattr("paperdaily.mcp.serve", fake_serve)
    result = CliRunner().invoke(app, ["mcp", "serve", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert observed["path"] == config_path.resolve()


def test_stdio_server_completes_mcp_handshake_without_network_or_models(tmp_path: Path) -> None:
    config_path, _ = _config(tmp_path)

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "paperdaily.cli", "mcp", "serve", "--config", str(config_path)],
            cwd=str(Path.cwd()),
        )
        with Path(os.devnull).open("w", encoding="utf-8") as errlog:
            async with stdio_client(parameters, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    tools = await session.list_tools()
                    status = await session.call_tool("get_status", {})
        assert initialized.serverInfo.name == "PaperDaily"
        assert {item.name for item in tools.tools} >= {"get_status", "record_feedback"}
        assert status.structuredContent and status.structuredContent["ok"] is True

    anyio.run(exercise)
