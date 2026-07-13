"""Optional digest delivery channels.

Terminal and Markdown are the local-first defaults.  Feishu is lazy-imported
and opt-in so missing credentials or SDKs never prevent the core run from
finishing.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import DeliveryResult, Digest, Recommendation


class Channel(Protocol):
    name: str

    def publish_digest(self, digest: Digest) -> DeliveryResult: ...


def _safe(value: object) -> str:
    return str(value or "").strip()


def _paper_links(item: Recommendation) -> str:
    paper = item.paper
    paper_url = _safe(paper.get("paper_url") or paper.get("url"))
    pdf_url = _safe(paper.get("pdf_url"))
    links: list[str] = []
    if paper_url:
        links.append(f"[论文]({paper_url})")
    if pdf_url:
        links.append(f"[PDF]({pdf_url})")
    if item.canonical_id:
        links.append(f"[外部中文阅读](https://hjfy.top/?q=https://arxiv.org/abs/{item.canonical_id})")
    return " · ".join(links)


def render_digest_markdown(digest: Digest) -> str:
    """Render a deterministic, portable Chinese Markdown digest."""

    label = "论文日报" if digest.window_start == digest.window_end else "论文补推精选"
    lines = [
        f"# {label}",
        "",
        f"- 时间范围：{digest.window_start.isoformat()} 至 {digest.window_end.isoformat()}",
        f"- 用户：{digest.user_id}",
        f"- 推荐数量：{len(digest.recommendations)}",
        f"- 运行 ID：`{digest.run_id}`",
        "",
    ]
    if digest.stats:
        fetched = digest.stats.get("fetched_count", 0)
        matched = digest.stats.get("matched_count", 0)
        lines.extend([f"> 抓取 {fetched} 篇，话题召回 {matched} 篇。", ""])
        announcement_notice = _safe(digest.stats.get("announcement_notice"))
        if announcement_notice:
            lines.extend([f"> 提示：{announcement_notice}", ""])

    if not digest.recommendations:
        lines.extend(["本时间段没有达到阈值的论文。", ""])
        return "\n".join(lines)

    for item in digest.recommendations:
        paper = item.paper
        summary = item.summary
        # arXiv titles are source metadata. Keep them verbatim; only the
        # explanatory content is localized.
        title = _safe(paper.get("title"))
        lines.extend(
            [
                f"## {item.rank}. {title}",
                "",
                f"- arXiv：`{item.canonical_id}`",
                f"- 综合分：{item.score:.3f}",
                f"- 命中话题：{', '.join(item.matched_topics) or '未标注'}",
                f"- 推荐原因：{item.recommendation_reason or (summary.recommendation_reason if summary else '与订阅规则相关')}",
            ]
        )
        authors = paper.get("authors") or []
        if authors:
            lines.append(f"- 作者：{', '.join(map(str, authors))}")
        links = _paper_links(item)
        if links:
            lines.append(f"- 链接：{links}")
        lines.append("")

        if summary:
            lines.extend([summary.one_sentence_summary or "摘要未生成。", ""])
            if summary.problem:
                lines.extend([f"**研究问题：** {summary.problem}", ""])
            if summary.method:
                lines.extend([f"**核心方法：** {summary.method}", ""])
            if summary.contributions:
                lines.append("**主要贡献：**")
                lines.append("")
                lines.extend(f"- {value}" for value in summary.contributions)
                lines.append("")
            if summary.status != "completed":
                lines.extend([f"> 摘要状态：{summary.status}", ""])
        else:
            abstract = _safe(paper.get("abstract"))
            if abstract:
                lines.extend([abstract[:800], ""])

    lines.extend(
        [
            "---",
            "",
            "可在终端记录反馈：",
            "",
            "```bash",
            "paperdaily feedback <arxiv-id> interested",
            "paperdaily feedback <arxiv-id> irrelevant",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass
class TerminalChannel:
    writer: Callable[[str], None] = print
    name: str = "terminal"

    def publish_digest(self, digest: Digest) -> DeliveryResult:
        self.writer(render_digest_markdown(digest))
        return DeliveryResult(channel=self.name, success=True, target="stdout")


@dataclass
class MarkdownChannel:
    output_dir: Path
    name: str = "markdown"

    def publish_digest(self, digest: Digest) -> DeliveryResult:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = digest.window_end.isoformat()
            if digest.window_start != digest.window_end:
                suffix = f"{digest.window_start.isoformat()}_{digest.window_end.isoformat()}"
            path = self.output_dir / f"{suffix}-{digest.run_id[:8]}.md"
            path.write_text(render_digest_markdown(digest), encoding="utf-8")
            digest.output_path = path
            return DeliveryResult(channel=self.name, success=True, target=str(path))
        except Exception as exc:  # channel errors must not corrupt the run
            return DeliveryResult(channel=self.name, success=False, error=str(exc))


@dataclass
class FeishuChannel:
    chat_id: str | None = None
    user_id: str | None = None
    name: str = "feishu"

    def publish_digest(self, digest: Digest) -> DeliveryResult:
        try:
            reporter = importlib.import_module(
                "deployments.feishu.feishu-reporter.scripts.feishu_reporter"
            )

            content = render_digest_markdown(digest)
            if self.chat_id:
                reporter.send_text_to_chat(self.chat_id, content)
                return DeliveryResult(channel=self.name, success=True, target=self.chat_id)
            if self.user_id:
                reporter.send_daily_push(self.user_id, content)
                return DeliveryResult(channel=self.name, success=True, target=self.user_id)
            return DeliveryResult(channel=self.name, success=False, error="未配置飞书 chat_id 或 user_id")
        except Exception as exc:
            return DeliveryResult(channel=self.name, success=False, error=str(exc))


__all__ = [
    "Channel",
    "FeishuChannel",
    "MarkdownChannel",
    "TerminalChannel",
    "render_digest_markdown",
]
