"""Ingestor tests: reading, encoding fallback, hashing, docling routing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest

from chonkai import ContentType, IngestError, Ingestor, compute_content_hash
from chonkai.ingest.ingestor import decode_bytes


def test_decode_bytes_utf8() -> None:
    text, warnings = decode_bytes("héllo wörld".encode())
    assert text == "héllo wörld"
    assert warnings == ()


def test_decode_bytes_latin1_fallback_collects_warning() -> None:
    # 0xE9 is invalid as the start of a UTF-8 multi-byte sequence
    raw = b"caf\xe9 au lait"
    text, warnings = decode_bytes(raw)
    assert text == "café au lait"
    assert len(warnings) == 1
    assert "utf-8 decode failed" in warnings[0]
    assert "latin-1" in warnings[0]


def test_decode_bytes_is_total() -> None:
    for raw in (b"\xff\xfe\x00", b"\x80", b"\xc3", bytes(range(256))):
        text, warnings = decode_bytes(raw)
        assert text
        assert len(warnings) == 1  # latin-1 never fails


def test_ingest_text_populates_metadata() -> None:
    ingest = Ingestor().ingest_text("some text", path="mem.txt", content_type="text")
    assert ingest.source.path == "mem.txt"
    assert ingest.source.content_type == "text"
    assert ingest.source.size_bytes == len(b"some text")
    assert ingest.source.content_hash == compute_content_hash(b"some text")
    assert ingest.source.mtime is not None
    assert ingest.warnings == ()


def test_compute_content_hash_matches_sha256() -> None:
    data = b"abc"
    assert compute_content_hash(data) == hashlib.sha256(data).hexdigest()


def test_ingest_file_utf8(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("# hello", encoding="utf-8")
    ingest = Ingestor().ingest_file(p)
    assert ingest.text == "# hello"
    assert ingest.source.path == str(p)
    assert ingest.source.size_bytes == p.stat().st_size
    assert ingest.source.content_type == ContentType.MARKDOWN
    assert ingest.source.content_hash == compute_content_hash(p.read_bytes())
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
    assert ingest.source.mtime == mtime
    assert ingest.warnings == ()


def test_ingest_file_latin1_collects_warning(tmp_path: Path) -> None:
    p = tmp_path / "legacy.txt"
    p.write_bytes(b"caf\xe9")  # latin-1, not valid utf-8
    ingest = Ingestor().ingest_file(p)
    assert ingest.text == "café"
    assert len(ingest.warnings) == 1
    assert ingest.source.content_type == ContentType.TEXT


def test_ingest_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="cannot stat"):
        Ingestor().ingest_file(tmp_path / "nope.py")


def test_ingest_file_unreadable_raises(tmp_path: Path) -> None:
    p = tmp_path / "dir"
    p.mkdir()
    with pytest.raises(IngestError, match="cannot read"):
        Ingestor().ingest_file(p)


class _FakeStatus:
    value = "partial_failure"


class _FakeDocument:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeResult:
    def __init__(self, markdown: str, status: object | None) -> None:
        self.document = _FakeDocument(markdown)
        self.status = status


class _FakeConverter:
    def __init__(self, markdown: str, status: object | None = None) -> None:
        self._markdown = markdown
        self._status = status
        self.sources: list[str] = []

    def convert(self, source: str) -> _FakeResult:
        self.sources.append(source)
        return _FakeResult(self._markdown, self._status)


def test_ingest_file_routes_pdf_to_docling(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    converter = _FakeConverter("# Title\n\ntext")
    ingest = Ingestor(converter=converter).ingest_file(p)
    assert converter.sources == [str(p)]
    assert ingest.text == "# Title\n\ntext"
    assert ingest.document is not None  # DoclingDocument attached for chunkers
    assert ingest.source.content_type == ContentType.PDF
    assert ingest.warnings == ()


def test_ingest_file_collects_docling_status_warning(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    converter = _FakeConverter("text", status=_FakeStatus())
    ingest = Ingestor(converter=converter).ingest_file(p)
    assert ingest.warnings == ("docling conversion status: partial_failure",)


def test_ingest_file_wraps_docling_failure(tmp_path: Path) -> None:
    class _Broken:
        def convert(self, source: str) -> NoReturn:
            del source
            raise RuntimeError("parser exploded")

    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(IngestError, match="docling conversion failed"):
        Ingestor(converter=_Broken()).ingest_file(p)


def test_ingest_file_docling_import_error(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    ing = Ingestor()
    # no converter injected -> the lazy loader must run (and fail) on demand
    import chonkai.ingest.ingestor as ingestor_mod

    def fake_load() -> object:
        raise ImportError("docling extra not installed")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ingestor_mod, "load_document_converter", fake_load)
    try:
        with pytest.raises(ImportError, match="docling extra"):
            ing.ingest_file(p)
    finally:
        monkeypatch.undo()
