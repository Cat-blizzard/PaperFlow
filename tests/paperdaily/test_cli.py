from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from typer.testing import CliRunner

import paperdaily.cli as cli
from paperdaily.catchup import CatchupPlanner
from paperdaily.config import PaperDailyConfig, load_config, save_config
from paperdaily.topics import Topic

runner = CliRunner()


def test_root_help_and_version_do_not_require_configuration() -> None:
    help_result = runner.invoke(cli.app, ["--help"])
    version_result = runner.invoke(cli.app, ["--version"])

    assert help_result.exit_code == 0
    assert "paperdaily" in help_result.output.casefold()
    assert "run" in help_result.output
    assert "read" not in help_result.output
    assert "provider" not in help_result.output
    assert version_result.exit_code == 0
    assert version_result.output.startswith("paperdaily ")


def test_core_subcommand_help_does_not_perform_network_or_load_config() -> None:
    for arguments in (
        ["run", "--help"],
        ["auto", "--help"],
        ["topic", "--help"],
        ["mcp", "--help"],
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
                name="Embodied VLA",
                description="Initial description",
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
            "Updated VLA",
            "--description",
            "Updated description",
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
    assert current.name == "Updated VLA"
    assert current.description == "Updated description"
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
