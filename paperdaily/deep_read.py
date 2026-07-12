"""Isolated full-paper workspace and evidence-grounded note generation."""

from __future__ import annotations

import importlib
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from jsonschema import Draft202012Validator, ValidationError

from .collector import ArxivCollector
from .identifiers import canonicalize_arxiv_id

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
ALLOWED_PDF_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
PAGE_EVIDENCE_SCHEMA_VERSION = 1
PAGE_MARKER_PREFIX = "<!-- paperdaily:pdf-page="
# Deep-read workspaces are handed to an Agent, so parsing an arbitrarily large
# PDF is both a reliability and a cost problem. These defaults comfortably
# cover ordinary papers with appendices while rejecting document-sized uploads.
DEFAULT_MAX_PDF_PAGES = 100
DEFAULT_MAX_EXTRACTED_TEXT_CHARS = 500_000
REQUIRED_NOTE_KEYS = {
    "basic_info",
    "one_sentence_conclusion",
    "research_problem",
    "core_contributions",
    "method",
    "data",
    "experiments",
    "ablations",
    "relation_to_user_research",
    "limitations",
    "reproduction",
    "executable_ideas",
    "unanswered_questions",
    "evidence",
}
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w.])[+\-−]?(?:\d{1,3}(?:[,\u00a0\u202f ]\d{3})+|\d+(?:\.\d+)?)(?:[eE][+\-]?\d+)?\s*[%％]?"
)


class DeepReadError(RuntimeError):
    """Raised when preparation, provider execution, or validation fails."""


@dataclass(frozen=True)
class DeepReadResult:
    arxiv_id: str
    provider: str
    workspace: Path
    json_path: Path
    markdown_path: Path
    note: dict[str, Any]


def _is_public_ip(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DeepReadError(f"无法解析 PDF 主机 {host}: {exc}") from exc
    if not addresses:
        return False
    for entry in addresses:
        value = ipaddress.ip_address(entry[4][0])
        if value.is_private or value.is_loopback or value.is_link_local or value.is_reserved:
            return False
    return True


def _validate_pdf_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or host not in ALLOWED_PDF_HOSTS:
        raise DeepReadError("只允许从 arxiv.org 的 HTTPS 地址下载论文 PDF")
    if not _is_public_ip(host):
        raise DeepReadError("PDF 主机解析到了非公网地址，已拒绝请求")


def _download_pdf(url: str, output: Path, *, max_bytes: int, timeout_seconds: float) -> None:
    current_url = url
    response: requests.Response | None = None
    for _ in range(4):
        _validate_pdf_url(current_url)
        response = requests.get(
            current_url,
            stream=True,
            timeout=timeout_seconds,
            allow_redirects=False,
            headers={"User-Agent": "PaperDaily/0.1 (+https://github.com/Cat-blizzard/PaperFlow)"},
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise DeepReadError("arXiv PDF 重定向缺少 Location")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        break
    else:
        raise DeepReadError("arXiv PDF 重定向次数过多")

    assert response is not None
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        response.close()
        raise DeepReadError(f"PDF 超过大小限制 {max_bytes // (1024 * 1024)} MiB")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    total = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=128 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise DeepReadError(
                        f"PDF 超过大小限制 {max_bytes // (1024 * 1024)} MiB"
                    )
                handle.write(chunk)
    except Exception:
        # On Windows an open file cannot be unlinked.  Leave the ``with``
        # block first, then remove the incomplete download so the original
        # validation error is not hidden by ``WinError 32``.
        temporary.unlink(missing_ok=True)
        raise
    finally:
        response.close()
    temporary.replace(output)


def _parse_pdf(pdf_path: Path) -> dict[str, Any]:
    try:
        parser = importlib.import_module("skills.pdf-parser.scripts.parse_pdf")
        return dict(parser.parse_pdf(str(pdf_path)))
    except ImportError as exc:
        raise DeepReadError("缺少 PDF 解析依赖，请安装 `pip install -e \".[parsing]\"`") from exc
    except Exception as exc:
        raise DeepReadError(f"PDF 解析失败: {exc}") from exc


def _validate_pdf_resource_limits(*, max_pdf_pages: int, max_extracted_text_chars: int) -> None:
    """Validate resource limits used before parsing untrusted PDF content."""

    for value, name in (
        (max_pdf_pages, "max_pdf_pages"),
        (max_extracted_text_chars, "max_extracted_text_chars"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DeepReadError(f"{name} must be a positive integer")


def _document_page_count(document: Any) -> int:
    raw_page_count = getattr(document, "page_count", None)
    page_count = int(raw_page_count) if raw_page_count is not None else len(document)
    if page_count <= 0:
        raise DeepReadError("PDF has no pages")
    return page_count


def _preflight_pdf_page_count(pdf_path: Path, *, max_pdf_pages: int) -> int:
    """Check page count before invoking the more expensive general parser.

    PyMuPDF is deliberately a hard requirement for a *new* deep-read parse:
    without it we cannot enforce a page-count ceiling before calling the richer
    parser. Cached, already-bounded workspaces remain usable offline.
    """

    _validate_pdf_resource_limits(
        max_pdf_pages=max_pdf_pages,
        max_extracted_text_chars=DEFAULT_MAX_EXTRACTED_TEXT_CHARS,
    )
    try:
        fitz = importlib.import_module("fitz")
    except ImportError as exc:
        raise DeepReadError(
            "PyMuPDF is required to safely inspect PDF page limits; "
            "install it with `pip install -e \".[parsing]\"`."
        ) from exc

    document: Any | None = None
    try:
        document = fitz.open(str(pdf_path))
        page_count = _document_page_count(document)
        if page_count > max_pdf_pages:
            raise DeepReadError(
                f"PDF has {page_count} pages, exceeding the configured "
                f"max_pdf_pages limit of {max_pdf_pages}."
            )
        return page_count
    except DeepReadError:
        raise
    except Exception as exc:
        raise DeepReadError(f"Could not inspect PDF page count safely: {exc}") from exc
    finally:
        if document is not None:
            close = getattr(document, "close", None)
            if callable(close):
                close()


def _ensure_text_within_limit(text: str, *, max_extracted_text_chars: int, source: str) -> None:
    if len(text) > max_extracted_text_chars:
        raise DeepReadError(
            f"{source} extracted text has {len(text)} characters, exceeding the configured "
            f"max_extracted_text_chars limit of {max_extracted_text_chars}."
        )


def _unavailable_page_evidence(reason: str, *, page_count: int = 0) -> dict[str, Any]:
    """Return an explicit no-page-location manifest.

    The regular PaperFlow parser remains useful for extracting sections, but it
    does not preserve physical PDF page boundaries.  Keeping that distinction
    explicit prevents an Agent from turning a section-order guess into a page
    citation.
    """

    return {
        "schema_version": PAGE_EVIDENCE_SCHEMA_VERSION,
        "status": "unavailable",
        "extractor": "pymupdf",
        "page_numbering": "physical PDF pages, one-indexed",
        "page_count": page_count,
        "reason": reason,
        "pages": [],
    }


def _extract_page_evidence(
    pdf_path: Path,
    *,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_extracted_text_chars: int = DEFAULT_MAX_EXTRACTED_TEXT_CHARS,
) -> dict[str, Any]:
    """Extract physical PDF pages with PyMuPDF without weakening parser fallback.

    A failed optional page extractor is deliberately non-fatal: callers still
    use the established PaperFlow parser for general text/section extraction,
    while the manifest makes it unambiguous that page numbers cannot be cited.
    """

    _validate_pdf_resource_limits(
        max_pdf_pages=max_pdf_pages,
        max_extracted_text_chars=max_extracted_text_chars,
    )
    try:
        fitz = importlib.import_module("fitz")
    except ImportError:
        return _unavailable_page_evidence(
            "PyMuPDF 未安装，已降级为无页码的普通 PDF 解析；证据 page 必须为 null。"
        )

    document: Any | None = None
    try:
        document = fitz.open(str(pdf_path))
        page_count = _document_page_count(document)
        if page_count > max_pdf_pages:
            raise DeepReadError(
                f"PDF has {page_count} pages, exceeding the configured "
                f"max_pdf_pages limit of {max_pdf_pages}."
            )
        if page_count <= 0:
            return _unavailable_page_evidence("PyMuPDF 打开的 PDF 没有页面。")

        pages: list[dict[str, Any]] = []
        total_text_chars = 0
        for index in range(page_count):
            page = document.load_page(index)
            try:
                text = page.get_text("text")
            except TypeError:
                # Kept for tiny test doubles and older compatible bindings.
                text = page.get_text()
            page_text = str(text or "").strip()
            total_text_chars += len(page_text)
            if total_text_chars > max_extracted_text_chars:
                raise DeepReadError(
                    "PDF page text exceeds the configured max_extracted_text_chars "
                    f"limit of {max_extracted_text_chars}."
                )
            pages.append({"page": index + 1, "text": page_text})

        if not any(item["text"] for item in pages):
            return _unavailable_page_evidence(
                "PyMuPDF 未提取到可引用的逐页文本；已保留普通解析回退，证据 page 必须为 null。",
                page_count=page_count,
            )
        return {
            "schema_version": PAGE_EVIDENCE_SCHEMA_VERSION,
            "status": "available",
            "extractor": "pymupdf",
            "page_numbering": "physical PDF pages, one-indexed",
            "page_count": page_count,
            "reason": "",
            "pages": pages,
        }
    except DeepReadError:
        raise
    except Exception as exc:
        return _unavailable_page_evidence(
            f"PyMuPDF 逐页提取失败 ({type(exc).__name__})；已保留普通解析回退，证据 page 必须为 null。"
        )
    finally:
        if document is not None:
            close = getattr(document, "close", None)
            if callable(close):
                close()


def _page_marked_text(page_evidence: dict[str, Any], fallback_text: str) -> str:
    """Create deterministic page markers for Agent-visible paper text."""

    pages = page_evidence.get("pages") if isinstance(page_evidence, dict) else None
    if page_evidence.get("status") == "available" and isinstance(pages, list):
        parts: list[str] = []
        for item in pages:
            if not isinstance(item, dict):
                continue
            page = item.get("page")
            if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
                continue
            parts.append(f"{PAGE_MARKER_PREFIX}{page} -->")
            parts.append(str(item.get("text") or "").strip())
        marked = "\n\n".join(parts).strip()
        if marked:
            return f"{marked}\n"

    # This is intentionally not a numeric marker.  It lets the Agent read the
    # fallback text, but communicates that no physical-page citation is valid.
    return f"{PAGE_MARKER_PREFIX}unavailable -->\n\n{fallback_text.strip()}\n"


def _page_evidence_summary(page_evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(page_evidence.get("status") or "unavailable"),
        "page_count": int(page_evidence.get("page_count") or 0),
        "source": "pages.json",
        "marker_prefix": PAGE_MARKER_PREFIX,
    }


def _validate_page_evidence_resource_limits(
    page_evidence: dict[str, Any],
    *,
    max_pdf_pages: int,
    max_extracted_text_chars: int,
) -> None:
    """Re-check cached page evidence when configured limits change."""

    raw_page_count = page_evidence.get("page_count")
    page_count = int(raw_page_count) if raw_page_count is not None else 0
    if page_count > max_pdf_pages:
        raise DeepReadError(
            f"Cached PDF has {page_count} pages, exceeding the configured "
            f"max_pdf_pages limit of {max_pdf_pages}."
        )
    pages = page_evidence.get("pages")
    if not isinstance(pages, list):
        return
    total_text_chars = sum(
        len(str(item.get("text") or "")) for item in pages if isinstance(item, dict)
    )
    if total_text_chars > max_extracted_text_chars:
        raise DeepReadError(
            "Cached PDF page text exceeds the configured max_extracted_text_chars "
            f"limit of {max_extracted_text_chars}."
        )


def _load_page_evidence(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _unavailable_page_evidence(
            "pages.json 不可读取；证据 page 必须为 null。"
        )
    if not isinstance(payload, dict):
        return _unavailable_page_evidence(
            "pages.json 格式无效；证据 page 必须为 null。"
        )
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_paper_workspace(
    arxiv_id: str,
    *,
    workspace_root: Path,
    collector: ArxivCollector | None = None,
    topic_context: str = "",
    force: bool = False,
    max_pdf_bytes: int = 50 * 1024 * 1024,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_extracted_text_chars: int = DEFAULT_MAX_EXTRACTED_TEXT_CHARS,
    timeout_seconds: float = 60.0,
) -> tuple[Path, dict[str, Any]]:
    """Download and deterministically parse one paper into an isolated folder."""

    _validate_pdf_resource_limits(
        max_pdf_pages=max_pdf_pages,
        max_extracted_text_chars=max_extracted_text_chars,
    )
    canonical_id = canonicalize_arxiv_id(arxiv_id)
    if not canonical_id:
        raise ValueError(f"invalid arXiv identifier: {arxiv_id!r}")
    workspace = Path(workspace_root).expanduser().resolve() / canonical_id
    workspace.mkdir(parents=True, exist_ok=True)
    metadata_path = workspace / "metadata.json"
    paper_path = workspace / "paper.md"
    sections_path = workspace / "sections.json"
    pages_path = workspace / "pages.json"

    if not force and metadata_path.exists() and paper_path.exists() and sections_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if pages_path.exists():
            page_evidence = _load_page_evidence(pages_path)
        else:
            # Backfill older workspaces locally.  This path deliberately never
            # calls the collector or downloader, so a cached parse remains
            # usable offline.
            pdf_path = workspace / "paper.pdf"
            page_evidence = (
                _extract_page_evidence(
                    pdf_path,
                    max_pdf_pages=max_pdf_pages,
                    max_extracted_text_chars=max_extracted_text_chars,
                )
                if pdf_path.exists()
                else _unavailable_page_evidence(
                    "旧工作区没有 paper.pdf，无法补建逐页证据；证据 page 必须为 null。"
                )
            )
            _write_json(pages_path, page_evidence)

        _validate_page_evidence_resource_limits(
            page_evidence,
            max_pdf_pages=max_pdf_pages,
            max_extracted_text_chars=max_extracted_text_chars,
        )
        existing_text = paper_path.read_text(encoding="utf-8")
        _ensure_text_within_limit(
            existing_text,
            max_extracted_text_chars=max_extracted_text_chars,
            source="Cached paper.md",
        )
        if PAGE_MARKER_PREFIX not in existing_text:
            paper_path.write_text(
                _page_marked_text(page_evidence, existing_text), encoding="utf-8"
            )
        metadata["page_evidence"] = _page_evidence_summary(page_evidence)
        _write_json(metadata_path, metadata)
    else:
        metadata = (collector or ArxivCollector()).fetch_by_id(canonical_id)
        if not metadata:
            raise DeepReadError(f"arXiv 未返回论文 {canonical_id}")
        metadata["arxiv_id"] = canonical_id
        pdf_url = str(metadata.get("pdf_url") or f"https://arxiv.org/pdf/{canonical_id}")
        pdf_path = workspace / "paper.pdf"
        if force or not pdf_path.exists():
            _download_pdf(
                pdf_url,
                pdf_path,
                max_bytes=max_pdf_bytes,
                timeout_seconds=timeout_seconds,
            )
        # Count pages before invoking the richer parser. Without this preflight
        # a document-sized PDF could consume substantial CPU/RAM during parsing.
        _preflight_pdf_page_count(pdf_path, max_pdf_pages=max_pdf_pages)
        page_evidence = _extract_page_evidence(
            pdf_path,
            max_pdf_pages=max_pdf_pages,
            max_extracted_text_chars=max_extracted_text_chars,
        )
        try:
            parsed = _parse_pdf(pdf_path)
            parser_name = "paperflow-pdf-parser"
        except DeepReadError:
            # PyMuPDF can still provide a safe, page-addressable workspace
            # when the optional rich parser is absent or rejects this PDF.
            if page_evidence.get("status") != "available":
                raise
            parsed = {
                "full_text": "\n\n".join(
                    str(item.get("text") or "")
                    for item in page_evidence.get("pages") or []
                    if isinstance(item, dict)
                ),
                "sections": {},
            }
            parser_name = "pymupdf-page-text"
        full_text = str(parsed.get("full_text") or "").strip()
        if not full_text:
            raise DeepReadError("PDF 未提取出可读正文")
        _ensure_text_within_limit(
            full_text,
            max_extracted_text_chars=max_extracted_text_chars,
            source="PDF parser",
        )
        paper_path.write_text(
            _page_marked_text(page_evidence, full_text), encoding="utf-8"
        )
        _write_json(sections_path, parsed.get("sections") or {})
        _write_json(pages_path, page_evidence)
        metadata["parsed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["parser"] = parser_name
        metadata["page_evidence"] = _page_evidence_summary(page_evidence)
        _write_json(metadata_path, metadata)

    (workspace / "topic_context.md").write_text(
        topic_context.strip() or "用户未提供额外话题上下文。",
        encoding="utf-8",
    )
    (workspace / "note_template.md").write_text(
        (RESOURCE_DIR / "note_template.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return workspace, metadata


def _normalized_quote(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


@lru_cache(maxsize=1)
def _reading_note_validator() -> Draft202012Validator:
    """Load the shipped note schema once for the host-side trust boundary."""

    schema_path = RESOURCE_DIR / "reading_note.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - packaged resource invariant.
        raise DeepReadError(f"无法读取阅读笔记 Schema: {schema_path}") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # pragma: no cover - packaged resource invariant.
        raise DeepReadError("阅读笔记 Schema 无效") from exc
    return Draft202012Validator(schema)


def _validate_note_schema(note: Any) -> dict[str, Any]:
    if not isinstance(note, dict):
        raise DeepReadError("精读输出必须是 JSON 对象")
    try:
        _reading_note_validator().validate(note)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise DeepReadError(f"精读输出未通过 JSON Schema（{location}）：{exc.message}") from exc
    return note


def _required_evidence_text(item: dict[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DeepReadError(f"evidence.{field} 必须是非空字符串")
    return value.strip()


def _validate_evidence_basics(evidence: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(evidence, list):
        raise DeepReadError("evidence 必须是对象数组")
    by_id: dict[str, dict[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise DeepReadError("evidence 必须是对象数组")
        evidence_id = _required_evidence_text(item, "id")
        _required_evidence_text(item, "claim")
        _required_evidence_text(item, "quote")
        _required_evidence_text(item, "section")
        if evidence_id in by_id:
            raise DeepReadError("evidence.id 必须唯一")
        by_id[evidence_id] = item
    return by_id


def _numeric_tokens(value: Any) -> set[str]:
    """Normalize common paper-result numbers for quote cross-checking.

    ``92.10%``, ``92.1 %``, ``+92.1`` and ``92.1`` all map to ``92.1``;
    comma/non-breaking-space thousands separators and scientific notation are
    handled as well. The check is intentionally conservative: it proves that
    at least one reported number is present in the cited evidence, not that the
    entire interpretation is correct.
    """

    values: set[str] = set()
    for match in _NUMERIC_TOKEN_RE.finditer(str(value or "")):
        raw = match.group(0).strip()
        normalized = (
            raw.replace(",", "")
            .replace("\u00a0", "")
            .replace("\u202f", "")
            .replace(" ", "")
            .replace("％", "%")
            .rstrip("%")
            .replace("−", "-")
        )
        if normalized.startswith("+"):
            normalized = normalized[1:]
        try:
            decimal_value = Decimal(normalized)
        except InvalidOperation:
            continue
        values.add(format(decimal_value.normalize(), "f"))
    return values


def _validate_evidence_pages(
    evidence: Any,
    page_evidence: dict[str, Any] | None,
    *,
    paper_text: str | None,
) -> None:
    """Reject evidence quotes or page locations that cannot be grounded locally."""

    items = evidence if isinstance(evidence, list) else []
    if not items:
        return
    normalized_paper = _normalized_quote(paper_text)
    if not normalized_paper:
        raise DeepReadError("evidence 存在时必须提供可验证的 paper.md 正文")

    manifest = page_evidence if isinstance(page_evidence, dict) else {}
    status = str(manifest.get("status") or "unavailable")
    raw_page_count = manifest.get("page_count")
    page_count = int(raw_page_count) if isinstance(raw_page_count, int) else 0
    raw_pages = manifest.get("pages")
    pages = raw_pages if isinstance(raw_pages, list) else []
    page_text: dict[int, str] = {}
    for source_page in pages:
        if not isinstance(source_page, dict):
            continue
        source_number = source_page.get("page")
        if isinstance(source_number, bool) or not isinstance(source_number, int):
            continue
        page_text[source_number] = _normalized_quote(source_page.get("text"))

    for item in items:
        if not isinstance(item, dict):
            raise DeepReadError("evidence 必须是对象数组")
        quote = _normalized_quote(item.get("quote"))
        if quote not in normalized_paper:
            raise DeepReadError("evidence.quote 必须逐字来自 paper.md，不能编造")
        page = item.get("page")
        if page is None:
            continue
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise DeepReadError("evidence.page 必须是正整数或 null")
        if status != "available":
            raise DeepReadError("当前工作区没有可靠逐页文本，evidence.page 必须为 null")
        if page > page_count or page not in page_text:
            raise DeepReadError("evidence.page 不在 pages.json 的可引用范围内")
        if quote not in page_text[page]:
            raise DeepReadError(
                "带页码的 evidence.quote 必须逐字来自该页 pages.json，不能猜测页码"
            )


def validate_reading_note(
    note: dict[str, Any],
    *,
    page_evidence: dict[str, Any] | None = None,
    paper_text: str | None = None,
) -> None:
    note = _validate_note_schema(note)
    missing = sorted(REQUIRED_NOTE_KEYS - set(note))
    if missing:
        raise DeepReadError(f"精读输出缺少字段: {', '.join(missing)}")
    evidence_by_id = _validate_evidence_basics(note.get("evidence"))
    _validate_evidence_pages(
        note.get("evidence"),
        page_evidence,
        paper_text=paper_text,
    )
    for experiment in note.get("experiments") or []:
        if not isinstance(experiment, dict):
            raise DeepReadError("experiments 必须是对象数组")
        numeric_claim_tokens = _numeric_tokens(
            f"{experiment.get('result', '')} {experiment.get('improvement', '')}"
        )
        evidence_ref = str(experiment.get("evidence_ref") or "").strip()
        if numeric_claim_tokens and (not evidence_ref or evidence_ref not in evidence_by_id):
            raise DeepReadError("包含数字的实验结论必须引用有效 evidence_ref")
        if numeric_claim_tokens:
            evidence_tokens = _numeric_tokens(evidence_by_id[evidence_ref]["quote"])
            if not numeric_claim_tokens & evidence_tokens:
                raise DeepReadError("数值实验结论的 evidence.quote 必须包含至少一个对应数值")


def _list_lines(values: Any) -> list[str]:
    items = values if isinstance(values, list) else []
    return [f"- {str(value).strip()}" for value in items if str(value).strip()] or ["- 论文未说明"]


def render_reading_note(note: dict[str, Any]) -> str:
    """Render the structured note into durable Chinese Markdown."""

    info = note.get("basic_info") or {}
    problem = note.get("research_problem") or {}
    method = note.get("method") or {}
    data = note.get("data") or {}
    relation = note.get("relation_to_user_research") or {}
    limitations = note.get("limitations") or {}
    lines = [
        f"# {info.get('title_zh') or info.get('title') or '论文阅读笔记'}",
        "",
        f"- 原题：{info.get('title', '')}",
        f"- 作者：{', '.join(info.get('authors') or [])}",
        f"- arXiv：`{info.get('arxiv_id', '')}`",
        f"- 论文：{info.get('paper_url', '')}",
        f"- 代码：{info.get('code_url') or '论文未说明'}",
        "",
        "## 一句话结论",
        "",
        str(note.get("one_sentence_conclusion") or "论文未说明"),
        "",
        "## 研究问题",
        "",
        f"- 问题：{problem.get('problem') or '论文未说明'}",
        f"- 现有方法不足：{problem.get('why_existing_methods_are_insufficient') or '论文未说明'}",
        f"- 研究价值：{problem.get('research_value') or '论文未说明'}",
        "",
        "## 核心贡献",
        "",
        *_list_lines(note.get("core_contributions")),
        "",
        "## 方法",
        "",
        f"- 总览：{method.get('overview') or '论文未说明'}",
        f"- 输入：{method.get('input') or '论文未说明'}",
        f"- 输出：{method.get('output') or '论文未说明'}",
        f"- 架构：{method.get('architecture') or '论文未说明'}",
        f"- 动作表示：{method.get('action_representation') or '论文未说明'}",
        f"- 训练目标：{method.get('training_objective') or '论文未说明'}",
        f"- 推理：{method.get('inference') or '论文未说明'}",
        "",
        "## 数据",
        "",
        f"- 训练数据：{data.get('training_data') or '论文未说明'}",
        f"- 规模：{data.get('scale') or '论文未说明'}",
        f"- 机器人本体：{data.get('embodiments') or '论文未说明'}",
        f"- 真实数据：{data.get('real_data') or '论文未说明'}",
        f"- 仿真数据：{data.get('simulation_data') or '论文未说明'}",
        f"- 跨本体：{data.get('cross_embodiment') or '论文未说明'}",
        "",
        "## 实验",
        "",
        "| 实验 | 基线 | 结果 | 提升 | 证据 |",
        "|---|---|---|---|---|",
    ]
    for experiment in note.get("experiments") or []:
        lines.append(
            "| {experiment} | {baseline} | {result} | {improvement} | {evidence_ref} |".format(
                **{key: str(experiment.get(key) or "论文未说明").replace("|", "\\|") for key in (
                    "experiment", "baseline", "result", "improvement", "evidence_ref"
                )}
            )
        )
    lines.extend(
        [
            "",
            "## 消融实验",
            "",
            *_list_lines(note.get("ablations")),
            "",
            "## 与我的研究方向的关系",
            "",
            f"- VLA：{relation.get('vla') or '论文未说明'}",
            f"- WAM：{relation.get('world_action_model') or '论文未说明'}",
            f"- 机器人基础模型：{relation.get('robot_foundation_model') or '论文未说明'}",
            "- 可复用模块：",
            *_list_lines(relation.get("reusable_components")),
            "",
            "## 局限性",
            "",
            "### 作者明确说明",
            "",
            *_list_lines(limitations.get("author_stated")),
            "",
            "### 根据证据推断",
            "",
            *_list_lines(limitations.get("inferred")),
            "",
            "## 复现要点",
            "",
            *_list_lines(note.get("reproduction")),
            "",
            "## 可执行想法",
            "",
            *_list_lines(note.get("executable_ideas")),
            "",
            "## 尚未解决的问题",
            "",
            *_list_lines(note.get("unanswered_questions")),
            "",
            "## 证据清单",
            "",
        ]
    )
    for evidence in note.get("evidence") or []:
        location = str(evidence.get("section") or "章节未知")
        if evidence.get("page") is not None:
            location += f"，第 {evidence['page']} 页"
        if evidence.get("table"):
            location += f"，{evidence['table']}"
        lines.extend(
            [
                f"### {evidence.get('id')}: {evidence.get('claim')}",
                "",
                f"> {evidence.get('quote') or '未保留短引文'}",
                "",
                f"位置：{location}；置信度：{float(evidence.get('confidence') or 0):.2f}",
                "",
            ]
        )
    return "\n".join(lines)


class DeepReadService:
    """Prepare a paper and execute one isolated Agent provider."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        notes_dir: Path,
        store: Any = None,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        max_extracted_text_chars: int = DEFAULT_MAX_EXTRACTED_TEXT_CHARS,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.notes_dir = Path(notes_dir)
        self.store = store
        _validate_pdf_resource_limits(
            max_pdf_pages=max_pdf_pages,
            max_extracted_text_chars=max_extracted_text_chars,
        )
        self.max_pdf_pages = max_pdf_pages
        self.max_extracted_text_chars = max_extracted_text_chars

    def run(
        self,
        arxiv_id: str,
        *,
        provider: Any,
        topic_context: str = "",
        timeout_seconds: int = 1800,
        force_parse: bool = False,
    ) -> DeepReadResult:
        workspace, _metadata = prepare_paper_workspace(
            arxiv_id,
            workspace_root=self.workspace_root,
            topic_context=topic_context,
            force=force_parse,
            max_pdf_pages=self.max_pdf_pages,
            max_extracted_text_chars=self.max_extracted_text_chars,
        )
        raw_output = workspace / "reading_note.raw.json"
        prompt_file = RESOURCE_DIR / "deep_read_prompt.md"
        schema_file = RESOURCE_DIR / "reading_note.schema.json"
        note = provider.run_structured(
            workspace=workspace,
            prompt_file=prompt_file,
            schema_file=schema_file,
            output_file=raw_output,
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(note, dict):
            raise DeepReadError("Agent Provider 未返回 JSON 对象")
        try:
            paper_text = (workspace / "paper.md").read_text(encoding="utf-8")
        except OSError as exc:
            raise DeepReadError("无法读取用于证据校验的 paper.md") from exc
        validate_reading_note(
            note,
            page_evidence=_load_page_evidence(workspace / "pages.json"),
            paper_text=paper_text,
        )

        canonical_id = canonicalize_arxiv_id(arxiv_id)
        json_path = workspace / "reading_note.json"
        _write_json(json_path, note)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = self.notes_dir / f"{canonical_id}.md"
        markdown_path.write_text(render_reading_note(note), encoding="utf-8")
        return DeepReadResult(
            arxiv_id=canonical_id,
            provider=str(getattr(provider, "name", provider.__class__.__name__)),
            workspace=workspace,
            json_path=json_path,
            markdown_path=markdown_path,
            note=note,
        )


__all__ = [
    "DeepReadError",
    "DeepReadResult",
    "DeepReadService",
    "prepare_paper_workspace",
    "render_reading_note",
    "validate_reading_note",
]
