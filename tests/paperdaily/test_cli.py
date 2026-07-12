from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from typer.testing import CliRunner

import paperdaily.cli as cli
from paperdaily.catchup import CatchupPlanner
from paperdaily.config import PaperDailyConfig, load_config, save_config
from paperdaily.providers import AgentProviderError
from paperdaily.topics import Topic

runner = CliRunner()


def test_root_help_and_version_do_not_require_configuration() -> None:
    help_result = runner.invoke(cli.app, ["--help"])
    version_result = runner.invoke(cli.app, ["--version"])

    assert help_result.exit_code == 0
    assert "paperdaily" in help_result.output.casefold()
    assert "run" in help_result.output
    assert version_result.exit_code == 0
    assert version_result.output.startswith("paperdaily ")


def test_core_subcommand_help_does_not_perform_network_or_load_config() -> None:
    for arguments in (
        ["run", "--help"],
        ["auto", "--help"],
        ["topic", "--help"],
        ["provider", "--help"],
        ["notes", "--help"],
        ["read", "--help"],
    ):
        result = runner.invoke(cli.app, arguments)
        assert result.exit_code == 0, (arguments, result.output, result.exception)
        assert "--help" in result.output


def _config(tmp_path) -> tuple[PaperDailyConfig, object]:
    config = PaperDailyConfig(
        user_id="test-user",
        timezone="Asia/Shanghai",
        database=tmp_path / "paperflow.db",
        output_dir=tmp_path / "output",
        topics=[
            Topic(
                id="embodied-vla",
                name="具身智能与 VLA",
                description="初始描述",
                arxiv_categories=["cs.RO"],
                keywords=["VLA"],
                context_keywords=["robot"],
            )
        ],
    )
    path = tmp_path / "paperdaily.yaml"
    save_config(config, path)
    return config, path


def test_topic_edit_enable_and_disable_persist_config(tmp_path) -> None:
    _, config_path = _config(tmp_path)

    edited = runner.invoke(
        cli.app,
        [
            "topic",
            "edit",
            "embodied-vla",
            "--config",
            str(config_path),
            "--name",
            "更新的话题",
            "--description",
            "更新描述",
            "--category",
            "cs.RO",
            "--category",
            "cs.AI",
            "--keyword",
            "vision-language-action",
            "--daily-limit",
            "7",
            "--minimum-score",
            "0.5",
        ],
    )

    assert edited.exit_code == 0, edited.output
    current = load_config(config_path).topics[0]
    assert current.name == "更新的话题"
    assert current.description == "更新描述"
    assert current.arxiv_categories == ["cs.RO", "cs.AI"]
    assert current.keywords == ["vision-language-action"]
    assert current.context_keywords == ["robot"]
    assert current.daily_limit == 7
    assert current.minimum_score == 0.5

    disabled = runner.invoke(
        cli.app,
        ["topic", "disable", "embodied-vla", "--config", str(config_path)],
    )
    assert disabled.exit_code == 0, disabled.output
    assert load_config(config_path).topics[0].enabled is False

    enabled = runner.invoke(
        cli.app,
        ["topic", "enable", "embodied-vla", "--config", str(config_path)],
    )
    assert enabled.exit_code == 0, enabled.output
    assert load_config(config_path).topics[0].enabled is True


def test_topic_edit_can_clear_a_repeated_field(tmp_path) -> None:
    _, config_path = _config(tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "topic",
            "edit",
            "embodied-vla",
            "--config",
            str(config_path),
            "--clear-keywords",
        ],
    )

    assert result.exit_code == 0, result.output
    assert load_config(config_path).topics[0].keywords == []


def test_notes_list_and_show_read_only_local_markdown(tmp_path) -> None:
    config, config_path = _config(tmp_path)
    notes_dir = config.output_dir / "notes"
    notes_dir.mkdir(parents=True)
    note_path = notes_dir / "2607.12345.md"
    note_path.write_text("# 本地阅读笔记\n\n内容\n", encoding="utf-8")

    listed = runner.invoke(cli.app, ["notes", "list", "--config", str(config_path)])
    shown = runner.invoke(
        cli.app,
        ["notes", "show", "2607.12345v2", "--config", str(config_path)],
    )

    assert listed.exit_code == 0, listed.output
    assert "2607.12345" in listed.output
    assert shown.exit_code == 0, shown.output
    assert "# 本地阅读笔记" in shown.output
    assert "内容" in shown.output


def test_auto_uses_noninteractive_recommended_window_and_configured_channels(monkeypatch, tmp_path) -> None:
    config, config_path = _config(tmp_path)
    config.daily.channels = ["terminal", "markdown", "feishu"]
    planner = CatchupPlanner(interactive=True)
    plan = planner.plan(date(2026, 7, 1), target_date=date(2026, 7, 12))
    calls: dict[str, object] = {}

    class FakeService:
        def __init__(self) -> None:
            self.config = config

        def catchup_plan(self):
            return plan

        def select_window(self, incoming_plan, choice: str):
            calls["choice"] = choice
            return planner.select(incoming_plan, choice)

        def run(self, window, **kwargs):
            calls["window"] = window
            calls["channels"] = kwargs["channels"] or self.config.daily.channels
            digest = SimpleNamespace(
                recommendations=[],
                stats={"fetched_count": 0, "matched_count": 0},
                output_path=None,
            )
            return SimpleNamespace(digest=digest, warnings=[])

    fake = FakeService()
    monkeypatch.setattr(cli, "_service", lambda _path: (config_path, config, fake))

    result = runner.invoke(cli.app, ["auto", "--dry-run", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert calls["choice"] == "7d"
    assert calls["window"].choice == "7d"
    # Feishu is only retained because the user explicitly configured it.
    assert calls["channels"] == ["terminal", "markdown", "feishu"]


def test_auto_no_push_removes_feishu_from_explicit_configuration(monkeypatch, tmp_path) -> None:
    config, config_path = _config(tmp_path)
    config.daily.channels = ["terminal", "markdown", "feishu"]
    planner = CatchupPlanner(interactive=True)
    plan = planner.plan(date(2026, 7, 10), target_date=date(2026, 7, 12))
    calls: dict[str, object] = {}

    class FakeService:
        def __init__(self) -> None:
            self.config = config

        def catchup_plan(self):
            return plan

        def select_window(self, incoming_plan, choice: str):
            return planner.select(incoming_plan, choice)

        def run(self, window, **kwargs):
            calls["channels"] = kwargs["channels"]
            digest = SimpleNamespace(
                recommendations=[],
                stats={"fetched_count": 0, "matched_count": 0},
                output_path=None,
            )
            return SimpleNamespace(digest=digest, warnings=[])

    monkeypatch.setattr(cli, "_service", lambda _path: (config_path, config, FakeService()))

    result = runner.invoke(cli.app, ["auto", "--dry-run", "--no-push", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert calls["channels"] == ["terminal", "markdown"]


def test_read_reports_agent_errors_without_a_traceback(monkeypatch, tmp_path) -> None:
    config, config_path = _config(tmp_path)
    failures: list[object] = []

    class FakeStore:
        def start_agent_run(self, *args, **kwargs):
            return "agent-run"

        def fail_agent_run(self, *args):
            failures.append(args)

    class FakeService:
        def __init__(self) -> None:
            self.store = FakeStore()
            self.enabled_topics = config.topics

        def record_feedback(self, *args):
            raise AssertionError("feedback must not be written after a failed read")

    def fail_provider(*args, **kwargs):
        raise AgentProviderError(
            "codex",
            "process_failed",
            "Provider exited with status 134",
            stderr="untrusted provider stderr",
        )

    monkeypatch.setattr(cli, "_service", lambda _path: (config_path, config, FakeService()))
    monkeypatch.setattr(cli, "build_agent_provider", fail_provider)

    result = runner.invoke(
        cli.app,
        ["read", "2607.12345", "--provider", "codex", "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert "精读失败（codex/process_failed）" in result.output
    assert "untrusted provider stderr" not in result.output
    assert "Traceback" not in result.output
    # Provider selection failed before an Agent run existed, so no stale run
    # record should be written either.
    assert not failures


def test_provider_test_accepts_explicit_isolated_home_override(monkeypatch, tmp_path) -> None:
    config, config_path = _config(tmp_path)
    captured: dict[str, object] = {}

    class Provider:
        def diagnose(self):
            return {"provider": "codex", "available": True, "ready": True}

    def build_provider(*_args, **kwargs):
        captured.update(kwargs)
        return Provider()

    monkeypatch.setattr(cli, "build_agent_provider", build_provider)
    result = runner.invoke(
        cli.app,
        [
            "provider",
            "test",
            "codex",
            "--isolated-home",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["isolated_home"] is True

    config.deep_read.isolated_home = True
    save_config(config, config_path)
    captured.clear()
    shared = runner.invoke(
        cli.app,
        [
            "provider",
            "test",
            "codex",
            "--shared-home",
            "--config",
            str(config_path),
        ],
    )
    assert shared.exit_code == 0, shared.output
    assert captured["isolated_home"] is False
