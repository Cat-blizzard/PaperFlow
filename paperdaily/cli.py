# ruff: noqa: B008
"""Terminal entrypoint for local-first PaperDaily workflows."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer

from paperflow.providers import build_embedding_provider, build_llm_provider

from . import __version__
from .catchup import DateWindow, default_target_date
from .config import PaperDailyConfig, load_config, save_config
from .service import PaperDailyService
from .storage import PaperDailyStore
from .topics import Topic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "paperdaily" / "config.yaml"
EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent / "resources" / "paperdaily.example.yaml"

app = typer.Typer(
    name="paperdaily",
    help="个性化 arXiv 检索、中文摘要、补推与反馈。",
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)
topic_app = typer.Typer(help="管理研究话题。", no_args_is_help=True)
mcp_app = typer.Typer(help="启动仅暴露本地业务工具的 stdio MCP Server。", no_args_is_help=True)
app.add_typer(topic_app, name="topic")
app.add_typer(mcp_app, name="mcp")


def _config_path(path: Path | None) -> Path:
    return (path or DEFAULT_CONFIG_PATH).expanduser().resolve()


def _load(path: Path | None) -> tuple[Path, PaperDailyConfig]:
    resolved = _config_path(path)
    if not resolved.exists():
        raise typer.BadParameter(f"配置不存在：{resolved}。请先运行 `paperdaily init`。")
    return resolved, load_config(resolved)


def _service(path: Path | None) -> tuple[Path, PaperDailyConfig, PaperDailyService]:
    resolved, config = _load(path)
    return resolved, config, PaperDailyService(config)


def _topic_or_error(config: PaperDailyConfig, topic_id: str) -> Topic:
    topic = next((item for item in config.topics if item.id == topic_id), None)
    if topic is None:
        raise typer.BadParameter(f"未知话题：{topic_id}")
    return topic


def _replace_topic(config: PaperDailyConfig, replacement: Topic) -> None:
    for index, topic in enumerate(config.topics):
        if topic.id == replacement.id:
            config.topics[index] = replacement
            return
    raise typer.BadParameter(f"未知话题：{replacement.id}")


def _edited_topic_values(
    current: list[str],
    supplied: list[str] | None,
    *,
    clear: bool,
    field: str,
) -> list[str]:
    """Apply a repeated CLI option without making an omitted option destructive."""

    if clear and supplied:
        raise typer.BadParameter(f"{field} 不能与对应的 --clear-* 选项同时使用")
    if clear:
        return []
    if supplied is not None:
        return list(supplied)
    return list(current)


@app.callback()
def _callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="显示版本并退出。"),
) -> None:
    if version:
        typer.echo(f"paperdaily {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def init(
    config: Path | None = typer.Option(None, "--config", "-c", help="配置文件路径。"),
    user_id: str | None = typer.Option(None, "--user-id", "-u", help="PaperFlow 用户 ID。"),
    force: bool = typer.Option(False, "--force", help="覆盖已有配置。"),
) -> None:
    """创建本地配置、SQLite side tables 和研究者空间。"""

    target = _config_path(config)
    if target.exists() and not force:
        cfg = load_config(target)
    else:
        cfg = load_config(EXAMPLE_CONFIG_PATH)
        cfg.database = (PROJECT_ROOT / "data" / "paperflow.db").resolve()
        cfg.output_dir = (PROJECT_ROOT / "data" / "output").resolve()
        if user_id:
            cfg.user_id = user_id.strip()
        save_config(cfg, target)
    service = PaperDailyService(cfg)
    result = service.initialize_runtime()
    typer.echo(f"配置：{target}")
    typer.echo(f"数据库：{result['database']}")
    typer.echo(f"研究者空间：{result['user_id']}")
    typer.echo(f"启用话题：{result['topic_count']}")


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """检查配置、数据库、摘要与 Embedding Provider。"""

    resolved, cfg = _load(config)
    store = PaperDailyStore(cfg.database)
    embed = build_embedding_provider()
    llm = build_llm_provider()
    typer.echo(f"PaperDaily {__version__}")
    typer.echo(f"配置：{resolved}")
    typer.echo(f"数据库：{store.db_path}")
    typer.echo(f"话题：{len([topic for topic in cfg.topics if topic.enabled])}")
    typer.echo(f"arXiv 分类：{', '.join(PaperDailyService(cfg, store=store).categories)}")
    typer.echo(f"Embedding：{embed.name}:{embed.model}")
    if embed.name == "hash":
        typer.echo("[警告] hash embedding 不具备语义能力；当前主要依赖话题规则排序。")
    typer.echo(f"中文摘要：{llm.name}:{llm.model}")
    if llm.name == "mock":
        typer.echo("[警告] 未配置真实 LLM，日报会显示原始摘要回退。")


@app.command()
def status(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """显示 watermark、下次窗口和近期运行。"""

    _, cfg, service = _service(config)
    state = service.store.get_state(cfg.user_id)
    plan = service.catchup_plan()
    typer.echo(f"用户：{cfg.user_id}")
    typer.echo(f"上次完成：{(state or {}).get('last_completed_window_end') or '从未运行'}")
    typer.echo(f"待处理天数：{plan.gap_days}")
    typer.echo(f"建议选择：{plan.recommended_choice}")
    if plan.recommended_window.start_date:
        typer.echo(
            f"建议窗口：{plan.recommended_window.start_date.isoformat()} 至 "
            f"{plan.recommended_window.end_date.isoformat()}"
        )
    runs = service.store.list_runs(cfg.user_id, limit=5)
    for run in runs:
        typer.echo(
            f"{run['run_id']}  {run['status']}  {run['window_start']}..{run['window_end']}  "
            f"推荐 {run['recommendation_count']}"
        )


def _resolve_window(
    service: PaperDailyService,
    *,
    choice: str | None,
    since: str | None,
    until: str | None,
    non_interactive: bool,
) -> DateWindow:
    plan = service.catchup_plan()
    if since or until:
        if not since:
            raise typer.BadParameter("使用 --until 时必须同时提供 --since")
        start = date.fromisoformat(since)
        end = date.fromisoformat(until) if until else default_target_date(timezone=service.config.timezone)
        target = default_target_date(timezone=service.config.timezone)
        if end > target:
            raise typer.BadParameter(f"--until 不能晚于最后一个完整日 {target}")
        if start > end:
            raise typer.BadParameter("--since 不能晚于 --until")
        return DateWindow(start, end, "custom")

    selected = choice
    if not selected:
        if not plan.has_work:
            return DateWindow(None, plan.target_date, "none")
        selected = plan.recommended_choice
        if plan.requires_confirmation and not non_interactive:
            typer.echo(f"检测到 {plan.gap_days} 天未处理，建议：{selected}")
            selected = typer.prompt("选择 latest / 7d / 30d / all", default=selected)
    return service.select_window(plan, selected)


def _execute_run(
    *,
    config: Path | None,
    window: str | None,
    since: str | None,
    until: str | None,
    limit: int | None,
    dry_run: bool,
    no_summary: bool,
    no_push: bool,
    channel: list[str],
    non_interactive: bool,
    include_handled: bool,
    feishu_chat_id: str | None,
) -> None:
    _, cfg, service = _service(config)
    selected = _resolve_window(
        service,
        choice=window,
        since=since,
        until=until,
        non_interactive=non_interactive,
    )
    if selected.is_empty:
        typer.echo("已经处理到最新完整日期，没有待处理论文。")
        return
    requested_channels: list[str] | None = channel or None
    if no_push:
        requested_channels = [name for name in (requested_channels or cfg.daily.channels) if name != "feishu"]
    outcome = service.run(
        selected,
        dry_run=dry_run,
        limit=limit,
        generate_summary=False if no_summary else None,
        channels=requested_channels,
        include_handled=include_handled,
        feishu_chat_id=feishu_chat_id,
    )
    typer.echo(
        f"窗口：{selected.start_date} 至 {selected.end_date}；"
        f"抓取 {outcome.digest.stats.get('fetched_count', 0)}；"
        f"召回 {outcome.digest.stats.get('matched_count', 0)}；"
        f"推荐 {len(outcome.digest.recommendations)}。"
    )
    if dry_run:
        typer.echo("dry-run：未生成摘要、未写运行状态、未发送渠道。")
    elif outcome.digest.output_path:
        typer.echo(f"日报：{outcome.digest.output_path}")
    for warning in outcome.warnings:
        typer.echo(f"[警告] {warning}")


@app.command()
def run(
    config: Path | None = typer.Option(None, "--config", "-c"),
    window: str | None = typer.Option(None, "--window", help="latest/7d/30d/all (yesterday is a legacy alias)"),
    since: str | None = typer.Option(None, "--since", help="YYYY-MM-DD"),
    until: str | None = typer.Option(None, "--until", help="YYYY-MM-DD"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    no_push: bool = typer.Option(False, "--no-push", help="禁用飞书等远程推送。"),
    channel: list[str] | None = typer.Option(None, "--channel", help="可重复：terminal/markdown/feishu"),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
    include_handled: bool = typer.Option(False, "--include-handled", help="允许重新推荐历史论文。"),
    feishu_chat_id: str | None = typer.Option(None, "--feishu-chat-id"),
) -> None:
    """处理默认待办窗口；正常情况检索当前 arXiv 公告批次。"""

    _execute_run(
        config=config,
        window=window,
        since=since,
        until=until,
        limit=limit,
        dry_run=dry_run,
        no_summary=no_summary,
        no_push=no_push,
        channel=channel or [],
        non_interactive=non_interactive,
        include_handled=include_handled,
        feishu_chat_id=feishu_chat_id,
    )


@app.command()
def auto(
    config: Path | None = typer.Option(None, "--config", "-c"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    no_push: bool = typer.Option(False, "--no-push"),
    channel: list[str] | None = typer.Option(None, "--channel", help="可重复：terminal/markdown/feishu"),
    include_handled: bool = typer.Option(False, "--include-handled"),
    feishu_chat_id: str | None = typer.Option(None, "--feishu-chat-id"),
) -> None:
    """无人值守运行：按 catchup 策略选择窗口，不等待终端输入。

    未指定 ``--channel`` 时只使用配置中的 ``daily.channels``。默认配置
    仅包含 terminal 和 markdown，因此不会意外发送飞书；若配置明确加入
    feishu，或本次显式传入 ``--channel feishu``，才会尝试该远程渠道。
    """

    _execute_run(
        config=config,
        window=None,
        since=None,
        until=None,
        limit=limit,
        dry_run=dry_run,
        no_summary=no_summary,
        no_push=no_push,
        channel=channel or [],
        non_interactive=True,
        include_handled=include_handled,
        feishu_chat_id=feishu_chat_id,
    )


@app.command()
def catchup(
    config: Path | None = typer.Option(None, "--config", "-c"),
    window: str | None = typer.Option(None, "--window", help="7d/30d/all"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """补推遗漏窗口；数量过多时按综合排名裁剪。"""

    _execute_run(
        config=config,
        window=window,
        since=None,
        until=None,
        limit=limit,
        dry_run=dry_run,
        no_summary=False,
        no_push=True,
        channel=[],
        non_interactive=non_interactive,
        include_handled=False,
        feishu_chat_id=None,
    )


@app.command()
def feedback(
    arxiv_id: str = typer.Argument(...),
    action: str = typer.Argument(..., help="interested/irrelevant/later/saved/read"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """记录显式反馈，供后续话题排序使用。"""

    _, _, service = _service(config)
    event = service.record_feedback(arxiv_id, action)
    typer.echo(f"已记录：{event['canonical_id']} -> {event['action']}")


@app.command()
def history(
    config: Path | None = typer.Option(None, "--config", "-c"),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """查看最近运行历史。"""

    _, cfg, service = _service(config)
    for run_record in service.store.list_runs(cfg.user_id, limit=limit):
        typer.echo(json.dumps(run_record, ensure_ascii=False))


@topic_app.command("list")
def topic_list(config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    _, cfg = _load(config)
    for topic in cfg.topics:
        typer.echo(
            f"{topic.id}\t{'enabled' if topic.enabled else 'disabled'}\t{topic.name}\t"
            f"categories={','.join(topic.arxiv_categories)}"
        )


@topic_app.command("show")
def topic_show(
    topic_id: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    _, cfg = _load(config)
    topic = _topic_or_error(cfg, topic_id)
    typer.echo(json.dumps(topic.to_dict(), ensure_ascii=False, indent=2))


@topic_app.command("add")
def topic_add(
    topic_id: str = typer.Option(..., "--id"),
    name: str = typer.Option(..., "--name"),
    description: str = typer.Option("", "--description"),
    category: list[str] | None = typer.Option(None, "--category"),
    phrase: list[str] | None = typer.Option(None, "--phrase"),
    keyword: list[str] | None = typer.Option(None, "--keyword"),
    context_keyword: list[str] | None = typer.Option(None, "--context-keyword"),
    negative_keyword: list[str] | None = typer.Option(None, "--negative-keyword"),
    daily_limit: int = typer.Option(0, "--daily-limit", help="0 表示不限"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    path, cfg = _load(config)
    if any(item.id == topic_id for item in cfg.topics):
        raise typer.BadParameter(f"话题已存在：{topic_id}")
    cfg.topics.append(
        Topic(
            id=topic_id,
            name=name,
            description=description,
            arxiv_categories=category or [],
            exact_phrases=phrase or [],
            keywords=keyword or [],
            context_keywords=context_keyword or [],
            negative_keywords=negative_keyword or [],
            daily_limit=daily_limit,
        )
    )
    save_config(cfg, path)
    typer.echo(f"已添加话题：{topic_id}")


@topic_app.command("remove")
def topic_remove(
    topic_id: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    path, cfg = _load(config)
    remaining = [item for item in cfg.topics if item.id != topic_id]
    if len(remaining) == len(cfg.topics):
        raise typer.BadParameter(f"未知话题：{topic_id}")
    cfg.topics = remaining
    save_config(cfg, path)
    typer.echo(f"已移除话题：{topic_id}")


@topic_app.command("edit")
def topic_edit(
    topic_id: str,
    name: str | None = typer.Option(None, "--name"),
    description: str | None = typer.Option(None, "--description"),
    category: list[str] | None = typer.Option(None, "--category"),
    phrase: list[str] | None = typer.Option(None, "--phrase"),
    keyword: list[str] | None = typer.Option(None, "--keyword"),
    context_keyword: list[str] | None = typer.Option(None, "--context-keyword"),
    negative_keyword: list[str] | None = typer.Option(None, "--negative-keyword"),
    clear_categories: bool = typer.Option(False, "--clear-categories"),
    clear_phrases: bool = typer.Option(False, "--clear-phrases"),
    clear_keywords: bool = typer.Option(False, "--clear-keywords"),
    clear_context_keywords: bool = typer.Option(False, "--clear-context-keywords"),
    clear_negative_keywords: bool = typer.Option(False, "--clear-negative-keywords"),
    daily_limit: int | None = typer.Option(None, "--daily-limit", help="0 表示不限"),
    minimum_score: float | None = typer.Option(None, "--minimum-score"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """修改话题字段；重复的列表选项会整体替换对应列表。"""

    path, cfg = _load(config)
    current = _topic_or_error(cfg, topic_id)
    replacement = Topic(
        id=current.id,
        name=current.name if name is None else name,
        description=current.description if description is None else description,
        enabled=current.enabled,
        arxiv_categories=_edited_topic_values(
            current.arxiv_categories,
            category,
            clear=clear_categories,
            field="--category",
        ),
        exact_phrases=_edited_topic_values(
            current.exact_phrases,
            phrase,
            clear=clear_phrases,
            field="--phrase",
        ),
        keywords=_edited_topic_values(
            current.keywords,
            keyword,
            clear=clear_keywords,
            field="--keyword",
        ),
        context_keywords=_edited_topic_values(
            current.context_keywords,
            context_keyword,
            clear=clear_context_keywords,
            field="--context-keyword",
        ),
        negative_keywords=_edited_topic_values(
            current.negative_keywords,
            negative_keyword,
            clear=clear_negative_keywords,
            field="--negative-keyword",
        ),
        daily_limit=current.daily_limit if daily_limit is None else daily_limit,
        minimum_score=current.minimum_score if minimum_score is None else minimum_score,
    )
    _replace_topic(cfg, replacement)
    save_config(cfg, path)
    typer.echo(f"已更新话题：{topic_id}")


def _set_topic_enabled(*, topic_id: str, config: Path | None, enabled: bool) -> None:
    path, cfg = _load(config)
    topic = _topic_or_error(cfg, topic_id)
    topic.enabled = enabled
    save_config(cfg, path)
    action = "启用" if enabled else "禁用"
    typer.echo(f"已{action}话题：{topic_id}")


@topic_app.command("enable")
def topic_enable(
    topic_id: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """启用一个研究话题。"""

    _set_topic_enabled(topic_id=topic_id, config=config, enabled=True)


@topic_app.command("disable")
def topic_disable(
    topic_id: str,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """禁用一个研究话题，但保留其配置和历史。"""

    _set_topic_enabled(topic_id=topic_id, config=config, enabled=False)


@mcp_app.command("serve")
def mcp_serve(
    config: Path | None = typer.Option(None, "--config", "-c", help="配置文件路径。"),
) -> None:
    """以 stdio 方式启动受限的本地 PaperDaily MCP Server。"""

    from .mcp import MCPUnavailableError, serve

    try:
        serve(_config_path(config))
    except MCPUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    except (FileNotFoundError, ValueError) as exc:
        # Do not write a traceback to stdout: stdout is reserved for MCP JSON-RPC.
        typer.echo(f"无法启动 PaperDaily MCP Server：{exc}", err=True)
        raise typer.Exit(code=2) from None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
