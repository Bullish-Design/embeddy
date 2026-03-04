# tests/test_ingest.py
"""Tests for the ingestion layer.

TDD: These tests are written before the implementation.
Covers: content type detection, text/file ingestion, content hashing,
        Docling routing, error handling.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embeddy.exceptions import IngestError
from embeddy.models import ContentType, IngestResult, SourceMetadata


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------


class TestDetectContentType:
    """Tests for file extension → ContentType mapping."""

    def test_python_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("example.py") == ContentType.PYTHON

    def test_javascript_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("app.js") == ContentType.JAVASCRIPT

    def test_typescript_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("app.ts") == ContentType.TYPESCRIPT

    def test_rust_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("main.rs") == ContentType.RUST

    def test_go_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("main.go") == ContentType.GO

    def test_c_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("main.c") == ContentType.C

    def test_cpp_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        for ext in ["main.cpp", "main.cc", "main.cxx"]:
            assert detect_content_type(ext) == ContentType.CPP

    def test_header_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        # .h files → C (reasonable default)
        assert detect_content_type("util.h") == ContentType.C

    def test_java_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("Main.java") == ContentType.JAVA

    def test_ruby_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("app.rb") == ContentType.RUBY

    def test_shell_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        for ext in ["script.sh", "script.bash"]:
            assert detect_content_type(ext) == ContentType.SHELL

    def test_markdown_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("README.md") == ContentType.MARKDOWN

    def test_rst_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("index.rst") == ContentType.RST

    def test_txt_extension(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("notes.txt") == ContentType.GENERIC

    def test_unknown_extension_is_generic(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("data.xyz") == ContentType.GENERIC

    def test_no_extension_is_generic(self):
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("Makefile") == ContentType.GENERIC

    def test_docling_extensions(self):
        """PDF, DOCX, and other rich formats map to DOCLING."""
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("report.pdf") == ContentType.DOCLING
        assert detect_content_type("doc.docx") == ContentType.DOCLING
        assert detect_content_type("slides.pptx") == ContentType.DOCLING
        assert detect_content_type("page.html") == ContentType.DOCLING
        assert detect_content_type("page.htm") == ContentType.DOCLING
        assert detect_content_type("photo.png") == ContentType.DOCLING
        assert detect_content_type("photo.jpg") == ContentType.DOCLING
        assert detect_content_type("photo.jpeg") == ContentType.DOCLING
        assert detect_content_type("scan.tiff") == ContentType.DOCLING
        assert detect_content_type("diagram.bmp") == ContentType.DOCLING

    def test_case_insensitive(self):
        """Extension matching is case-insensitive."""
        from embeddy.ingest.ingestor import detect_content_type

        assert detect_content_type("README.MD") == ContentType.MARKDOWN
        assert detect_content_type("Main.PY") == ContentType.PYTHON
        assert detect_content_type("REPORT.PDF") == ContentType.DOCLING


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


class TestContentHash:
    """Tests for content hash computation."""

    def test_hash_is_sha256(self):
        """Content hash uses SHA-256."""
        from embeddy.ingest.ingestor import compute_content_hash

        text = "Hello, world!"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert compute_content_hash(text) == expected

    def test_hash_deterministic(self):
        """Same content always produces the same hash."""
        from embeddy.ingest.ingestor import compute_content_hash

        assert compute_content_hash("abc") == compute_content_hash("abc")

    def test_hash_different_for_different_content(self):
        """Different content produces different hashes."""
        from embeddy.ingest.ingestor import compute_content_hash

        assert compute_content_hash("abc") != compute_content_hash("def")


# ---------------------------------------------------------------------------
# Ingestor — text ingestion
# ---------------------------------------------------------------------------


class TestIngestText:
    """Tests for ingesting raw text."""

    async def test_ingest_text_returns_ingest_result(self):
        """ingest_text returns an IngestResult."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        result = await ingestor.ingest_text("Hello, world!")
        assert isinstance(result, IngestResult)
        assert result.text == "Hello, world!"

    async def test_ingest_text_default_content_type_generic(self):
        """Without explicit content_type, defaults to GENERIC."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        result = await ingestor.ingest_text("Some text")
        assert result.content_type == ContentType.GENERIC

    async def test_ingest_text_explicit_content_type(self):
        """Caller can specify content_type."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        result = await ingestor.ingest_text("def hello(): pass", content_type=ContentType.PYTHON)
        assert result.content_type == ContentType.PYTHON

    async def test_ingest_text_has_content_hash(self):
        """Ingested text has a content_hash computed."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        result = await ingestor.ingest_text("Hello, world!")
        assert result.source.content_hash is not None
        expected = hashlib.sha256("Hello, world!".encode("utf-8")).hexdigest()
        assert result.source.content_hash == expected

    async def test_ingest_text_source_metadata(self):
        """Source field can be set for text ingestion."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        result = await ingestor.ingest_text("data", source="clipboard")
        assert result.source.file_path == "clipboard"

    async def test_ingest_text_no_docling_document(self):
        """Text ingestion does not set docling_document."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        result = await ingestor.ingest_text("data")
        assert result.docling_document is None

    async def test_ingest_empty_text_raises(self):
        """Ingesting empty text raises IngestError."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        with pytest.raises(IngestError, match="empty"):
            await ingestor.ingest_text("")

    async def test_ingest_whitespace_only_raises(self):
        """Ingesting whitespace-only text raises IngestError."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        with pytest.raises(IngestError, match="empty"):
            await ingestor.ingest_text("   \n\t  ")


# ---------------------------------------------------------------------------
# Ingestor — file ingestion (text path)
# ---------------------------------------------------------------------------


class TestIngestFile:
    """Tests for ingesting files from disk."""

    async def test_ingest_python_file(self, tmp_path: Path):
        """Ingest a .py file."""
        from embeddy.ingest.ingestor import Ingestor

        py_file = tmp_path / "example.py"
        py_file.write_text("def hello():\n    return 'hi'\n")

        ingestor = Ingestor()
        result = await ingestor.ingest_file(py_file)

        assert result.content_type == ContentType.PYTHON
        assert "def hello" in result.text
        assert result.source.file_path == str(py_file)
        assert result.source.size_bytes == py_file.stat().st_size
        assert result.source.content_hash is not None
        assert result.docling_document is None

    async def test_ingest_markdown_file(self, tmp_path: Path):
        """Ingest a .md file."""
        from embeddy.ingest.ingestor import Ingestor

        md_file = tmp_path / "README.md"
        md_file.write_text("# Hello\n\nWorld.\n")

        ingestor = Ingestor()
        result = await ingestor.ingest_file(md_file)

        assert result.content_type == ContentType.MARKDOWN
        assert "# Hello" in result.text

    async def test_ingest_generic_txt_file(self, tmp_path: Path):
        """Ingest a .txt file as generic."""
        from embeddy.ingest.ingestor import Ingestor

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Just some notes.\n")

        ingestor = Ingestor()
        result = await ingestor.ingest_file(txt_file)

        assert result.content_type == ContentType.GENERIC
        assert "Just some notes" in result.text

    async def test_ingest_file_sets_modified_at(self, tmp_path: Path):
        """Ingested file has modified_at set from file mtime."""
        from embeddy.ingest.ingestor import Ingestor

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("content\n")

        ingestor = Ingestor()
        result = await ingestor.ingest_file(txt_file)

        assert result.source.modified_at is not None

    async def test_ingest_nonexistent_file_raises(self):
        """Ingesting a nonexistent file raises IngestError."""
        from embeddy.ingest.ingestor import Ingestor

        ingestor = Ingestor()
        with pytest.raises(IngestError, match="not found|does not exist"):
            await ingestor.ingest_file(Path("/nonexistent/file.py"))

    async def test_ingest_file_explicit_content_type(self, tmp_path: Path):
        """Caller can override auto-detected content type."""
        from embeddy.ingest.ingestor import Ingestor

        txt_file = tmp_path / "data.txt"
        txt_file.write_text("def hello(): pass\n")

        ingestor = Ingestor()
        result = await ingestor.ingest_file(txt_file, content_type=ContentType.PYTHON)
        assert result.content_type == ContentType.PYTHON

    async def test_ingest_file_content_hash_matches(self, tmp_path: Path):
        """Content hash matches SHA-256 of file content."""
        from embeddy.ingest.ingestor import Ingestor

        txt_file = tmp_path / "notes.txt"
        content = "Hello, world!"
        txt_file.write_text(content)

        ingestor = Ingestor()
        result = await ingestor.ingest_file(txt_file)

        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert result.source.content_hash == expected


# ---------------------------------------------------------------------------
# Ingestor — file ingestion (Docling path)
# ---------------------------------------------------------------------------


class TestIngestDoclingFile:
    """Tests for Docling-routed file ingestion.

    Mocks Docling's DocumentConverter since we don't want heavyweight
    document parsing in unit tests.
    """

    async def test_pdf_routes_through_docling(self, tmp_path: Path):
        """A .pdf file is routed through Docling."""
        from embeddy.ingest.ingestor import Ingestor

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_doc = MagicMock()
        mock_doc.export_to_text.return_value = "Extracted text from PDF."

        mock_conv_result = MagicMock()
        mock_conv_result.document = mock_doc

        with patch("embeddy.ingest.ingestor.DocumentConverter") as MockConverter:
            mock_converter_instance = MockConverter.return_value
            mock_converter_instance.convert.return_value = mock_conv_result

            ingestor = Ingestor()
            result = await ingestor.ingest_file(pdf_file)

        assert result.content_type == ContentType.DOCLING
        assert result.text == "Extracted text from PDF."
        assert result.docling_document is mock_doc
        assert result.source.file_path == str(pdf_file)

    async def test_docx_routes_through_docling(self, tmp_path: Path):
        """A .docx file is routed through Docling."""
        from embeddy.ingest.ingestor import Ingestor

        docx_file = tmp_path / "doc.docx"
        docx_file.write_bytes(b"PK fake docx")

        mock_doc = MagicMock()
        mock_doc.export_to_text.return_value = "DOCX content."

        mock_conv_result = MagicMock()
        mock_conv_result.document = mock_doc

        with patch("embeddy.ingest.ingestor.DocumentConverter") as MockConverter:
            mock_converter_instance = MockConverter.return_value
            mock_converter_instance.convert.return_value = mock_conv_result

            ingestor = Ingestor()
            result = await ingestor.ingest_file(docx_file)

        assert result.content_type == ContentType.DOCLING
        assert result.docling_document is mock_doc

    async def test_docling_failure_raises_ingest_error(self, tmp_path: Path):
        """If Docling conversion fails, IngestError is raised."""
        from embeddy.ingest.ingestor import Ingestor

        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"not a real pdf")

        with patch("embeddy.ingest.ingestor.DocumentConverter") as MockConverter:
            mock_converter_instance = MockConverter.return_value
            mock_converter_instance.convert.side_effect = Exception("Docling parse error")

            ingestor = Ingestor()
            with pytest.raises(IngestError, match="Docling|conversion|failed"):
                await ingestor.ingest_file(pdf_file)

    async def test_docling_has_content_hash(self, tmp_path: Path):
        """Even Docling-ingested files have content hash from exported text."""
        from embeddy.ingest.ingestor import Ingestor

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_doc = MagicMock()
        mock_doc.export_to_text.return_value = "Extracted text."

        mock_conv_result = MagicMock()
        mock_conv_result.document = mock_doc

        with patch("embeddy.ingest.ingestor.DocumentConverter") as MockConverter:
            mock_converter_instance = MockConverter.return_value
            mock_converter_instance.convert.return_value = mock_conv_result

            ingestor = Ingestor()
            result = await ingestor.ingest_file(pdf_file)

        expected = hashlib.sha256("Extracted text.".encode("utf-8")).hexdigest()
        assert result.source.content_hash == expected


# ---------------------------------------------------------------------------
# Ingestor — is_docling_path helper
# ---------------------------------------------------------------------------


class TestIsDoclingPath:
    """Tests for the helper that determines if a file should go through Docling."""

    def test_pdf_is_docling(self):
        from embeddy.ingest.ingestor import is_docling_path

        assert is_docling_path("report.pdf") is True

    def test_python_is_not_docling(self):
        from embeddy.ingest.ingestor import is_docling_path

        assert is_docling_path("main.py") is False

    def test_markdown_is_not_docling(self):
        from embeddy.ingest.ingestor import is_docling_path

        assert is_docling_path("README.md") is False

    def test_html_is_docling(self):
        from embeddy.ingest.ingestor import is_docling_path

        assert is_docling_path("page.html") is True

    def test_image_is_docling(self):
        from embeddy.ingest.ingestor import is_docling_path

        assert is_docling_path("photo.png") is True
        assert is_docling_path("photo.jpg") is True
