from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paperdaily.deep_read import (
    DeepReadError,
    DeepReadService,
    _download_pdf,
    _extract_page_evidence,
    _validate_pdf_url,
    prepare_paper_workspace,
    validate_reading_note,
)


def _note() -> dict[str, Any]:
    return {
        "basic_info": {
            "title": "A paper",
            "title_zh": "一篇论文",
            "authors": ["Ada"],
            "arxiv_id": "2607.00001",
            "paper_url": "https://arxiv.org/abs/2607.00001",
            "code_url": "",
            "topics": ["VLA"],
        },
        "one_sentence_conclusion": "Conclusion",
        "research_problem": {
            "problem": "Problem",
            "why_existing_methods_are_insufficient": "Unknown",
            "research_value": "Value",
        },
        "core_contributions": [],
        "method": {
            "overview": "Overview",
            "input": "Input",
            "output": "Output",
            "architecture": "Architecture",
            "action_representation": "Unknown",
            "training_objective": "Objective",
            "inference": "Inference",
        },
        "data": {
            "training_data": "Unknown",
            "scale": "Unknown",
            "embodiments": "Unknown",
            "real_data": "Unknown",
            "simulation_data": "Unknown",
            "cross_embodiment": "Unknown",
        },
        "experiments": [],
        "ablations": [],
        "relation_to_user_research": {
            "vla": "Unknown",
            "world_action_model": "Unknown",
            "robot_foundation_model": "Unknown",
            "reusable_components": [],
        },
        "limitations": {"author_stated": [], "inferred": []},
        "reproduction": [],
        "executable_ideas": [],
        "unanswered_questions": [],
        "evidence": [],
    }


class _DownloadResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        assert chunk_size > 0
        return self._chunks

    def close(self) -> None:
        self.closed = True


class _FakePdfPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, mode: str = "text") -> str:
        assert mode == "text"
        return self.text


class _FakePdfDocument:
    def __init__(self, pages: list[str]) -> None:
        self._pages = [_FakePdfPage(text) for text in pages]
        self.page_count = len(self._pages)
        self.closed = False

    def load_page(self, index: int) -> _FakePdfPage:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


def test_pdf_url_rejects_non_https_non_arxiv_and_private_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DeepReadError):
        _validate_pdf_url("http://arxiv.org/pdf/2607.00001")
    with pytest.raises(DeepReadError):
        _validate_pdf_url("https://evil.example/pdf/2607.00001")

    monkeypatch.setattr("paperdaily.deep_read._is_public_ip", lambda _host: False)
    with pytest.raises(DeepReadError, match="PDF"):
        _validate_pdf_url("https://arxiv.org/pdf/2607.00001")


def test_download_revalidates_redirect_before_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[str] = []
    response = _DownloadResponse(
        status_code=302,
        headers={"Location": "https://evil.example/stolen.pdf"},
    )

    def validate(url: str) -> None:
        checked.append(url)
        if "evil.example" in url:
            raise DeepReadError("blocked redirect")

    requests: list[str] = []
    monkeypatch.setattr("paperdaily.deep_read._validate_pdf_url", validate)
    monkeypatch.setattr(
        "paperdaily.deep_read.requests.get",
        lambda url, **_kwargs: requests.append(url) or response,
    )

    with pytest.raises(DeepReadError, match="blocked redirect"):
        _download_pdf(
            "https://arxiv.org/pdf/2607.00001",
            tmp_path / "paper.pdf",
            max_bytes=1024,
            timeout_seconds=1,
        )

    assert checked == [
        "https://arxiv.org/pdf/2607.00001",
        "https://evil.example/stolen.pdf",
    ]
    assert requests == ["https://arxiv.org/pdf/2607.00001"]
    assert response.closed is True
    assert not (tmp_path / "paper.pdf").exists()


def test_download_enforces_streaming_size_limit_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _DownloadResponse(chunks=[b"123", b"456"])
    monkeypatch.setattr("paperdaily.deep_read._validate_pdf_url", lambda _url: None)
    monkeypatch.setattr("paperdaily.deep_read.requests.get", lambda *_args, **_kwargs: response)
    output = tmp_path / "paper.pdf"

    with pytest.raises(DeepReadError, match="PDF"):
        _download_pdf(
            "https://arxiv.org/pdf/2607.00001",
            output,
            max_bytes=5,
            timeout_seconds=1,
        )

    assert response.closed is True
    assert not output.exists()
    assert not output.with_suffix(".pdf.part").exists()


def test_pymupdf_page_extraction_preserves_physical_page_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _FakePdfDocument(["First page\nExact result 92.1%", "Second page"])
    fake_fitz = SimpleNamespace(open=lambda _path: document)
    monkeypatch.setattr(
        "paperdaily.deep_read.importlib.import_module",
        lambda name: fake_fitz if name == "fitz" else pytest.fail(f"unexpected import {name}"),
    )

    manifest = _extract_page_evidence(tmp_path / "paper.pdf")

    assert manifest["status"] == "available"
    assert manifest["page_count"] == 2
    assert manifest["pages"] == [
        {"page": 1, "text": "First page\nExact result 92.1%"},
        {"page": 2, "text": "Second page"},
    ]
    assert document.closed is True


def test_pymupdf_page_extraction_rejects_excessive_page_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _FakePdfDocument(["First page", "Second page"])
    fake_fitz = SimpleNamespace(open=lambda _path: document)
    monkeypatch.setattr(
        "paperdaily.deep_read.importlib.import_module",
        lambda name: fake_fitz if name == "fitz" else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(DeepReadError, match="max_pdf_pages"):
        _extract_page_evidence(tmp_path / "paper.pdf", max_pdf_pages=1)

    assert document.closed is True


def test_pymupdf_page_extraction_rejects_excessive_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _FakePdfDocument(["1234", "5678"])
    fake_fitz = SimpleNamespace(open=lambda _path: document)
    monkeypatch.setattr(
        "paperdaily.deep_read.importlib.import_module",
        lambda name: fake_fitz if name == "fitz" else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(DeepReadError, match="max_extracted_text_chars"):
        _extract_page_evidence(
            tmp_path / "paper.pdf",
            max_pdf_pages=3,
            max_extracted_text_chars=5,
        )

    assert document.closed is True


def test_pymupdf_unavailable_is_explicitly_marked_for_page_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_fitz(_name: str) -> Any:
        raise ImportError("no fitz")

    monkeypatch.setattr("paperdaily.deep_read.importlib.import_module", missing_fitz)

    manifest = _extract_page_evidence(tmp_path / "paper.pdf")

    assert manifest["status"] == "unavailable"
    assert manifest["pages"] == []
    assert "page 必须为 null" in manifest["reason"]


def test_prepare_workspace_checks_page_limit_before_general_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _FakePdfDocument(["First page", "Second page"])
    fake_fitz = SimpleNamespace(open=lambda _path: document)
    collector = SimpleNamespace(
        fetch_by_id=lambda _identifier: {
            "title": "Long paper",
            "pdf_url": "https://arxiv.org/pdf/2607.00001",
        }
    )
    parser_called = False

    def unexpected_parser(_path: Path) -> dict[str, Any]:
        nonlocal parser_called
        parser_called = True
        raise AssertionError("general parser must not run after page-limit rejection")

    monkeypatch.setattr(
        "paperdaily.deep_read._download_pdf",
        lambda _url, output, **_kwargs: output.write_bytes(b"fake pdf"),
    )
    monkeypatch.setattr("paperdaily.deep_read._parse_pdf", unexpected_parser)
    monkeypatch.setattr(
        "paperdaily.deep_read.importlib.import_module",
        lambda name: fake_fitz if name == "fitz" else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(DeepReadError, match="max_pdf_pages"):
        prepare_paper_workspace(
            "2607.00001",
            workspace_root=tmp_path,
            collector=collector,
            max_pdf_pages=1,
        )

    assert parser_called is False
    assert document.closed is True


def test_prepare_workspace_rejects_new_parse_when_pymupdf_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = SimpleNamespace(
        fetch_by_id=lambda _identifier: {
            "title": "Test paper",
            "pdf_url": "https://arxiv.org/pdf/2607.00001",
        }
    )
    parser_called = False

    def missing_fitz(_name: str) -> Any:
        raise ImportError("no fitz")

    def unexpected_parser(_path: Path) -> dict[str, Any]:
        nonlocal parser_called
        parser_called = True
        raise AssertionError("general parser must not run without a page preflight")

    monkeypatch.setattr(
        "paperdaily.deep_read._download_pdf",
        lambda _url, output, **_kwargs: output.write_bytes(b"fake pdf"),
    )
    monkeypatch.setattr("paperdaily.deep_read._parse_pdf", unexpected_parser)
    monkeypatch.setattr("paperdaily.deep_read.importlib.import_module", missing_fitz)

    with pytest.raises(DeepReadError, match="PyMuPDF is required"):
        prepare_paper_workspace("2607.00001", workspace_root=tmp_path, collector=collector)

    assert parser_called is False


def test_prepare_workspace_rejects_oversized_general_parser_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = SimpleNamespace(
        fetch_by_id=lambda _identifier: {
            "title": "Test paper",
            "pdf_url": "https://arxiv.org/pdf/2607.00001",
        }
    )
    manifest = {
        "schema_version": 1,
        "status": "available",
        "extractor": "pymupdf",
        "page_numbering": "physical PDF pages, one-indexed",
        "page_count": 1,
        "reason": "",
        "pages": [{"page": 1, "text": "short"}],
    }
    monkeypatch.setattr("paperdaily.deep_read._preflight_pdf_page_count", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr("paperdaily.deep_read._extract_page_evidence", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        "paperdaily.deep_read._download_pdf",
        lambda _url, output, **_kwargs: output.write_bytes(b"fake pdf"),
    )
    monkeypatch.setattr(
        "paperdaily.deep_read._parse_pdf",
        lambda _path: {"full_text": "too much text", "sections": {}},
    )

    with pytest.raises(DeepReadError, match="max_extracted_text_chars"):
        prepare_paper_workspace(
            "2607.00001",
            workspace_root=tmp_path,
            collector=collector,
            max_extracted_text_chars=5,
        )


def test_prepare_workspace_rejects_oversized_cached_text(tmp_path: Path) -> None:
    workspace = tmp_path / "2607.00001"
    workspace.mkdir()
    (workspace / "metadata.json").write_text('{"arxiv_id": "2607.00001"}', encoding="utf-8")
    (workspace / "paper.md").write_text("too much text", encoding="utf-8")
    (workspace / "sections.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DeepReadError, match="max_extracted_text_chars"):
        prepare_paper_workspace(
            "2607.00001",
            workspace_root=tmp_path,
            max_extracted_text_chars=5,
        )


def test_prepare_workspace_reuses_cached_parse_without_network(tmp_path: Path) -> None:
    workspace = tmp_path / "2607.00001"
    workspace.mkdir()
    (workspace / "metadata.json").write_text(
        json.dumps({"arxiv_id": "2607.00001"}),
        encoding="utf-8",
    )
    (workspace / "paper.md").write_text("paper text", encoding="utf-8")
    (workspace / "sections.json").write_text("{}", encoding="utf-8")
    collector = SimpleNamespace(
        fetch_by_id=lambda _identifier: (_ for _ in ()).throw(AssertionError("network called"))
    )

    returned_workspace, metadata = prepare_paper_workspace(
        "2607.00001v2",
        workspace_root=tmp_path,
        collector=collector,
        topic_context="VLA context",
    )

    assert returned_workspace == workspace
    assert metadata["arxiv_id"] == "2607.00001"
    assert json.loads((workspace / "pages.json").read_text(encoding="utf-8"))["status"] == "unavailable"
    assert "<!-- paperdaily:pdf-page=unavailable -->" in (workspace / "paper.md").read_text(
        encoding="utf-8"
    )
    assert (workspace / "topic_context.md").read_text(encoding="utf-8") == "VLA context"
    assert (workspace / "note_template.md").exists()


def test_prepare_workspace_writes_pages_json_and_page_marked_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema_version": 1,
        "status": "available",
        "extractor": "pymupdf",
        "page_numbering": "physical PDF pages, one-indexed",
        "page_count": 2,
        "reason": "",
        "pages": [
            {"page": 1, "text": "Page one result"},
            {"page": 2, "text": "Page two method"},
        ],
    }
    collector = SimpleNamespace(
        fetch_by_id=lambda _identifier: {
            "title": "Test paper",
            "pdf_url": "https://arxiv.org/pdf/2607.00001",
        }
    )
    monkeypatch.setattr(
        "paperdaily.deep_read._download_pdf",
        lambda _url, output, **_kwargs: output.write_bytes(b"fake pdf"),
    )
    monkeypatch.setattr("paperdaily.deep_read._preflight_pdf_page_count", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        "paperdaily.deep_read._extract_page_evidence",
        lambda _path, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        "paperdaily.deep_read._parse_pdf",
        lambda _path: {"full_text": "fallback parser text", "sections": {"method": "..."}},
    )

    workspace, metadata = prepare_paper_workspace(
        "2607.00001",
        workspace_root=tmp_path,
        collector=collector,
    )

    assert json.loads((workspace / "pages.json").read_text(encoding="utf-8")) == manifest
    paper = (workspace / "paper.md").read_text(encoding="utf-8")
    assert "<!-- paperdaily:pdf-page=1 -->" in paper
    assert "<!-- paperdaily:pdf-page=2 -->" in paper
    assert "Page one result" in paper
    assert "fallback parser text" not in paper
    assert metadata["page_evidence"]["status"] == "available"


def test_numeric_experiment_claim_requires_grounded_evidence_reference_and_number() -> None:
    note = _note()
    note["experiments"] = [
        {
            "experiment": "benchmark",
            "baseline": "baseline",
            "result": "92.1%",
            "improvement": "+3.0",
            "evidence_ref": "",
        }
    ]

    with pytest.raises(DeepReadError, match="evidence_ref"):
        validate_reading_note(note, paper_text="Exact result 92.1% with an improvement of 3.0.")

    note["evidence"] = [
        {
            "id": "E1",
            "claim": "result",
            "quote": "Exact result 90.0%",
            "section": "Experiments",
            "page": None,
            "table": "",
            "confidence": 0.9,
        }
    ]
    note["experiments"][0]["evidence_ref"] = "E1"
    with pytest.raises(DeepReadError, match="对应数值"):
        validate_reading_note(note, paper_text="Exact result 90.0%.")

    note["evidence"][0]["quote"] = "Exact result 92.1% with an improvement of 3.0."
    validate_reading_note(note, paper_text="Exact result 92.1% with an improvement of 3.0.")


def test_evidence_requires_nonempty_fields_and_quote_from_paper_text() -> None:
    note = _note()
    note["evidence"] = [
        {
            "id": "E1",
            "claim": "",
            "quote": "Invented evidence",
            "section": "Experiments",
            "page": None,
            "table": "",
            "confidence": 0.9,
        }
    ]
    with pytest.raises(DeepReadError, match="evidence.claim"):
        validate_reading_note(note, paper_text="No such evidence appears here.")

    note["evidence"][0]["claim"] = "result"
    with pytest.raises(DeepReadError, match="来自 paper.md"):
        validate_reading_note(note, paper_text="No such evidence appears here.")


def test_page_citation_requires_quote_from_matching_pymupdf_page() -> None:
    note = _note()
    note["evidence"] = [
        {
                "id": "E1",
                "claim": "result",
                "quote": "Exact result 92.1%",
                "section": "Experiments",
            "page": 1,
            "table": "",
            "confidence": 0.9,
        }
    ]
    manifest = {
        "status": "available",
        "page_count": 2,
        "pages": [
            {"page": 1, "text": "Some text. Exact result 92.1%."},
            {"page": 2, "text": "Other text."},
        ],
    }

    validate_reading_note(
        note,
        page_evidence=manifest,
        paper_text="Some text. Exact result 92.1%. Other text.",
    )

    note["evidence"][0]["page"] = 2
    with pytest.raises(DeepReadError, match="逐字来自该页"):
        validate_reading_note(
            note,
            page_evidence=manifest,
            paper_text="Some text. Exact result 92.1%. Other text.",
        )

    note["evidence"][0]["page"] = 1
    with pytest.raises(DeepReadError, match="必须为 null"):
        validate_reading_note(
            note,
            page_evidence={"status": "unavailable", "page_count": 0, "pages": []},
            paper_text="Some text. Exact result 92.1%.",
        )


def test_deep_read_service_uses_structured_provider_and_writes_validated_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspaces" / "2607.00001"
    workspace.mkdir(parents=True)
    (workspace / "paper.md").write_text("A paper body.", encoding="utf-8")
    note = _note()
    provider_calls: list[dict[str, Any]] = []

    class Provider:
        name = "unit-agent"

        def run_structured(self, **kwargs: Any) -> dict[str, Any]:
            provider_calls.append(kwargs)
            return note

    monkeypatch.setattr(
        "paperdaily.deep_read.prepare_paper_workspace",
        lambda *_args, **_kwargs: (workspace, {"arxiv_id": "2607.00001"}),
    )
    service = DeepReadService(
        workspace_root=tmp_path / "workspaces",
        notes_dir=tmp_path / "notes",
    )

    result = service.run("2607.00001", provider=Provider(), timeout_seconds=9)

    assert result.provider == "unit-agent"
    assert result.json_path.exists()
    assert result.markdown_path.exists()
    assert json.loads(result.json_path.read_text(encoding="utf-8")) == note
    assert provider_calls[0]["workspace"] == workspace
    assert provider_calls[0]["timeout_seconds"] == 9


def test_deep_read_service_rejects_provider_bypass_of_nested_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspaces" / "2607.00001"
    workspace.mkdir(parents=True)
    (workspace / "paper.md").write_text("A paper body.", encoding="utf-8")
    note = _note()
    note["basic_info"]["unexpected"] = "not allowed"

    class Provider:
        name = "unit-agent"

        def run_structured(self, **_kwargs: Any) -> dict[str, Any]:
            return note

    monkeypatch.setattr(
        "paperdaily.deep_read.prepare_paper_workspace",
        lambda *_args, **_kwargs: (workspace, {"arxiv_id": "2607.00001"}),
    )
    service = DeepReadService(
        workspace_root=tmp_path / "workspaces",
        notes_dir=tmp_path / "notes",
    )

    with pytest.raises(DeepReadError, match="JSON Schema"):
        service.run("2607.00001", provider=Provider())
    assert not (workspace / "reading_note.json").exists()
    assert not (tmp_path / "notes" / "2607.00001.md").exists()
