"""Docling chunker bridge: mocked HybridChunker + ImportError + fallback paths."""

from __future__ import annotations

import pytest

from chonkai import ChunkBudget, DoclingChunker, IngestResult
from chonkai.chunkers import docling as docling_module


class _FakeItem:
    def __init__(self, text: str, headings: tuple[str, ...] = ()) -> None:
        self.text = text
        self.headings = headings


class _FakeChunker:
    def __init__(self, items: list[_FakeItem], **kwargs: object) -> None:
        self.kwargs = kwargs
        self._items = items

    def chunk(self, document: object) -> list[_FakeItem]:
        del document
        return self._items


def _document_result(text: str) -> IngestResult:
    return IngestResult(
        source=_ingest(text).source,
        text=text,
        document=object(),  # a docling document was attached by the Ingestor
    )


def _ingest(text: str) -> IngestResult:
    return IngestResult.from_text(text, path="doc.md", content_type="markdown")


def test_hybrid_chunker_heading_metadata_becomes_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "# Chapter 1\n\nsome text under chapter 1\n"
    items = [_FakeItem("some text under chapter 1", ("Chapter 1",))]
    monkeypatch.setattr(
        docling_module,
        "_load_hybrid_chunker",
        lambda **kwargs: _FakeChunker(items, **kwargs),
    )
    chunks = DoclingChunker().chunk(_document_result(markdown))
    assert len(chunks) == 1
    assert chunks[0].content == "some text under chapter 1"
    assert chunks[0].parent == "Chapter 1"
    assert chunks[0].start_line == 3 and chunks[0].end_line == 3  # 1-based


def test_max_tokens_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    items: list[_FakeItem] = []

    def fake_load(**kwargs: object) -> _FakeChunker:
        captured.update(kwargs)
        return _FakeChunker(items, **kwargs)

    monkeypatch.setattr(docling_module, "_load_hybrid_chunker", fake_load)
    DoclingChunker(max_tokens=64).chunk(_document_result("x"))
    assert captured["max_tokens"] == 64


def test_budget_max_tokens_used_when_no_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load(**kwargs: object) -> _FakeChunker:
        captured.update(kwargs)
        return _FakeChunker([], **kwargs)

    monkeypatch.setattr(docling_module, "_load_hybrid_chunker", fake_load)
    DoclingChunker().chunk(_document_result("x"), ChunkBudget(max_tokens=32))
    assert captured["max_tokens"] == 32


def test_without_document_falls_back_to_markdown_chunker() -> None:
    ingest = _ingest("# T\n\ntext\n")
    chunks = DoclingChunker().chunk(ingest)
    assert [c.chunk_type for c in chunks] == ["heading", "paragraph"]
    assert chunks[1].parent == "T"


def test_unlocatable_text_falls_back_to_document_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_FakeItem("normalized text that is not in the source")]
    monkeypatch.setattr(
        docling_module,
        "_load_hybrid_chunker",
        lambda **kwargs: _FakeChunker(items, **kwargs),
    )
    chunks = DoclingChunker().chunk(_document_result("line one\nline two\n"))
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3  # whole-document range fallback (incl. trailing line)


def test_load_hybrid_chunker_import_error() -> None:
    # no monkeypatching: docling is not installed in the dev env, so the lazy
    # loader raises ImportError on first use
    with pytest.raises(ImportError):
        docling_module._load_hybrid_chunker()


def test_locate_range_unit() -> None:
    source = "a\nb\nc\nd"
    assert docling_module._locate_range(source, "b") == (2, 2)
    assert docling_module._locate_range(source, "b\nc") == (2, 3)
    assert docling_module._locate_range(source, "nope") is None
