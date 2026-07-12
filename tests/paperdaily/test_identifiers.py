from __future__ import annotations

import pytest

from paperdaily.identifiers import (
    canonicalize_arxiv_id,
    deduplicate_papers_by_arxiv_id,
    normalize_arxiv_id,
    parse_arxiv_id,
    require_arxiv_identity,
)


@pytest.mark.parametrize(
    ("value", "canonical", "version"),
    [
        ("2607.12345", "2607.12345", None),
        ("arXiv:2607.12345v2", "2607.12345", 2),
        ("https://arxiv.org/abs/2607.12345v12", "2607.12345", 12),
        ("https://export.arxiv.org/pdf/2607.12345v3.pdf?download=1", "2607.12345", 3),
        ("arxiv.org/abs/2607.12345v4#section", "2607.12345", 4),
        ("abs/2607.12345v5", "2607.12345", 5),
        ("pdf/hep-th/9901001v3.pdf", "hep-th/9901001", 3),
        ("https://arxiv.org/abs/Math/0301234v2", "math/0301234", 2),
    ],
)
def test_normalize_arxiv_id(value: str, canonical: str, version: int | None) -> None:
    assert normalize_arxiv_id(value) == (canonical, version)
    assert canonicalize_arxiv_id(value) == canonical
    identity = parse_arxiv_id(value)
    assert identity is not None
    assert identity.versioned_id == f"{canonical}{f'v{version}' if version else ''}"


@pytest.mark.parametrize("value", [None, "", "not-an-id", "2607.123v1", "2607.12345v0", 123])
def test_invalid_arxiv_ids_are_rejected(value: object) -> None:
    assert normalize_arxiv_id(value) == ("", None)
    assert parse_arxiv_id(value) is None
    with pytest.raises(ValueError):
        require_arxiv_identity(value)


def test_deduplicate_prefers_highest_version_then_newest_updated() -> None:
    papers = [
        {
            "arxiv_id": "2607.00001v1",
            "title": "Version one",
            "updated": "2026-07-12T10:00:00Z",
        },
        {
            "pdf_url": "https://arxiv.org/pdf/2607.00001v3.pdf",
            "title": "Version three older timestamp",
            "updated": "2026-07-10T10:00:00Z",
        },
        {
            "arxiv_id": "2607.00001v2",
            "title": "Version two newer timestamp",
            "updated": "2026-07-13T10:00:00Z",
        },
        {
            "arxiv_id": "2607.00002v1",
            "title": "Stale metadata",
            "updated_date": "2026-07-10",
        },
        {
            "arxiv_id": "2607.00002v1",
            "title": "Fresh metadata",
            "updated_date": "2026-07-12",
        },
    ]

    deduplicated = deduplicate_papers_by_arxiv_id(papers)

    assert [paper["canonical_arxiv_id"] for paper in deduplicated] == ["2607.00001", "2607.00002"]
    assert deduplicated[0]["title"] == "Version three older timestamp"
    assert deduplicated[0]["arxiv_version"] == 3
    assert deduplicated[1]["title"] == "Fresh metadata"


def test_deduplicate_retains_unidentified_papers_and_does_not_mutate_inputs() -> None:
    original = {"arxiv_id": "2607.00003v2", "title": "Recognized"}
    unknown_one = {"title": "Journal paper one"}
    unknown_two = {"title": "Journal paper two"}

    result = deduplicate_papers_by_arxiv_id([original, unknown_one, unknown_two])

    assert result == [
        {
            "arxiv_id": "2607.00003v2",
            "title": "Recognized",
            "canonical_arxiv_id": "2607.00003",
            "arxiv_version": 2,
        },
        unknown_one,
        unknown_two,
    ]
    assert "canonical_arxiv_id" not in original


def test_separate_version_field_is_used_for_canonical_metadata() -> None:
    result = deduplicate_papers_by_arxiv_id(
        [
            {"canonical_arxiv_id": "2607.00004", "arxiv_version": "v2", "title": "Second"},
            {"arxiv_id": "2607.00004v1", "title": "First"},
        ]
    )

    assert len(result) == 1
    assert result[0]["title"] == "Second"
    assert result[0]["arxiv_version"] == 2
