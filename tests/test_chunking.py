# tests/test_chunking.py
"""Tests for the chunking layer.

TDD: These tests are written before the implementation.
Covers: BaseChunker ABC, PythonChunker, MarkdownChunker, ParagraphChunker,
        TokenWindowChunker, DoclingChunker, get_chunker factory.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from embeddy.config import ChunkConfig
from embeddy.exceptions import ChunkingError
from embeddy.models import Chunk, ContentType, IngestResult, SourceMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ingest_result(
    text: str,
    content_type: ContentType = ContentType.GENERIC,
    file_path: str | None = None,
) -> IngestResult:
    """Create an IngestResult for testing."""
    return IngestResult(
        text=text,
        content_type=content_type,
        source=SourceMetadata(file_path=file_path),
    )


# ---------------------------------------------------------------------------
# BaseChunker ABC
# ---------------------------------------------------------------------------


class TestBaseChunker:
    """Tests for the BaseChunker abstract base class."""

    def test_cannot_instantiate_abc(self):
        """BaseChunker cannot be directly instantiated."""
        from embeddy.chunking.base import BaseChunker

        with pytest.raises(TypeError):
            BaseChunker(config=ChunkConfig())  # type: ignore[abstract]

    def test_subclass_must_implement_chunk(self):
        """Subclass must implement the chunk method."""
        from embeddy.chunking.base import BaseChunker

        class IncompleteChunker(BaseChunker):
            pass

        with pytest.raises(TypeError):
            IncompleteChunker(config=ChunkConfig())  # type: ignore[abstract]

    def test_subclass_with_chunk_can_instantiate(self):
        """A properly implemented subclass can be instantiated."""
        from embeddy.chunking.base import BaseChunker

        class GoodChunker(BaseChunker):
            def chunk(self, ingest_result: IngestResult) -> list[Chunk]:
                return []

        chunker = GoodChunker(config=ChunkConfig())
        assert chunker.config.strategy == "auto"

    def test_config_stored(self):
        """Config is accessible on the chunker instance."""
        from embeddy.chunking.base import BaseChunker

        class GoodChunker(BaseChunker):
            def chunk(self, ingest_result: IngestResult) -> list[Chunk]:
                return []

        config = ChunkConfig(max_tokens=256, overlap_tokens=32)
        chunker = GoodChunker(config=config)
        assert chunker.config.max_tokens == 256
        assert chunker.config.overlap_tokens == 32


# ---------------------------------------------------------------------------
# PythonChunker
# ---------------------------------------------------------------------------


SAMPLE_PYTHON = textwrap.dedent('''\
    """Module docstring."""

    import os

    CONSTANT = 42


    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"


    def farewell(name: str) -> str:
        """Say goodbye."""
        return f"Goodbye, {name}!"


    class Calculator:
        """A simple calculator."""

        def __init__(self):
            self.value = 0

        def add(self, x: int) -> int:
            self.value += x
            return self.value

        def reset(self):
            self.value = 0
''')

SAMPLE_PYTHON_SIMPLE_FUNC = textwrap.dedent("""\
    def hello():
        return "hi"
""")

SAMPLE_PYTHON_SYNTAX_ERROR = textwrap.dedent("""\
    def broken(
        # missing closing paren and colon
        return 42
""")


class TestPythonChunker:
    """Tests for the AST-based Python chunker."""

    def test_chunks_functions(self):
        """Extracts top-level functions as individual chunks."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        names = [c.name for c in func_chunks]
        assert "greet" in names
        assert "farewell" in names

    def test_chunks_classes(self):
        """Extracts classes as chunks."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        names = [c.name for c in class_chunks]
        assert "Calculator" in names

    def test_chunks_module_level(self):
        """Extracts module-level code (imports, constants, docstring) as a chunk."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        module_chunks = [c for c in chunks if c.chunk_type == "module"]
        assert len(module_chunks) >= 1
        # Module-level should contain the imports and constants
        module_text = module_chunks[0].content
        assert "import os" in module_text or "CONSTANT" in module_text

    def test_chunk_has_line_numbers(self):
        """Each chunk has start_line and end_line set."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.start_line is not None
            assert chunk.end_line is not None
            assert chunk.start_line <= chunk.end_line

    def test_chunk_content_type_is_python(self):
        """All chunks have content_type set to PYTHON."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.content_type == ContentType.PYTHON

    def test_chunk_source_metadata_preserved(self):
        """Source metadata flows from IngestResult to Chunk."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.source.file_path == "example.py"

    def test_syntax_error_falls_back_to_paragraph(self):
        """Unparseable Python falls back to paragraph-style chunking."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON_SYNTAX_ERROR, ContentType.PYTHON, "bad.py")
        chunks = chunker.chunk(result)

        # Should still produce at least one chunk (fallback)
        assert len(chunks) >= 1
        # The fallback chunks should not be "function" or "class" type
        for chunk in chunks:
            assert chunk.chunk_type not in ("function", "class")

    def test_single_function_file(self):
        """A file with just one function produces one function chunk."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON_SIMPLE_FUNC, ContentType.PYTHON, "simple.py")
        chunks = chunker.chunk(result)

        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].name == "hello"

    def test_empty_file_produces_no_chunks(self):
        """An empty Python file produces no chunks."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        # IngestResult validates non-empty content, so use whitespace-only workaround
        # Actually Chunk validates non-empty, but IngestResult.text is just str.
        # An empty python file with just a comment:
        result = _make_ingest_result("# empty file\n", ContentType.PYTHON, "empty.py")
        chunks = chunker.chunk(result)

        # Should produce at most a module chunk with the comment
        assert len(chunks) <= 1

    def test_class_granularity_includes_methods(self):
        """At class granularity, the entire class body is one chunk."""
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig(python_granularity="class"))
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1
        # The class chunk should contain method definitions
        calc_chunk = [c for c in class_chunks if c.name == "Calculator"][0]
        assert "add" in calc_chunk.content
        assert "reset" in calc_chunk.content


# ---------------------------------------------------------------------------
# MarkdownChunker
# ---------------------------------------------------------------------------


SAMPLE_MARKDOWN = textwrap.dedent("""\
    # Main Title

    Introduction paragraph.

    ## Section One

    Content of section one.

    More content in section one.

    ## Section Two

    Content of section two.

    ### Subsection 2.1

    Subsection content.

    ## Section Three

    Final section content.
""")

SAMPLE_MARKDOWN_NO_HEADINGS = textwrap.dedent("""\
    Just a plain paragraph.

    Another paragraph with no headings at all.
""")


class TestMarkdownChunker:
    """Tests for the heading-boundary markdown chunker."""

    def test_splits_at_h2_by_default(self):
        """Splits at ## headings by default (markdown_heading_level=2)."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        # Should have chunks for: intro (under #), section one, section two (with subsection), section three
        assert len(chunks) >= 3
        names = [c.name for c in chunks if c.name]
        assert "Section One" in names
        assert "Section Two" in names
        assert "Section Three" in names

    def test_splits_at_h1(self):
        """When markdown_heading_level=1, splits at # headings."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig(markdown_heading_level=1))
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        # With h1 split, the entire document under "# Main Title" is one chunk
        # (since there's only one h1)
        assert len(chunks) >= 1

    def test_chunk_type_is_heading_section(self):
        """Chunks have chunk_type 'heading_section'."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.chunk_type == "heading_section"

    def test_content_type_is_markdown(self):
        """All chunks have content_type MARKDOWN."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.content_type == ContentType.MARKDOWN

    def test_source_metadata_preserved(self):
        """Source metadata is preserved on all chunks."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.source.file_path == "doc.md"

    def test_no_headings_produces_single_chunk(self):
        """Text with no headings produces a single chunk."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_MARKDOWN_NO_HEADINGS, ContentType.MARKDOWN, "plain.md")
        chunks = chunker.chunk(result)

        assert len(chunks) == 1
        assert "Just a plain paragraph" in chunks[0].content

    def test_subsection_included_in_parent(self):
        """At h2 level, h3 subsections are included in the parent h2 chunk."""
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig(markdown_heading_level=2))
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        # Find the "Section Two" chunk — it should contain the h3 subsection
        section_two = [c for c in chunks if c.name == "Section Two"]
        assert len(section_two) == 1
        assert "Subsection 2.1" in section_two[0].content
        assert "Subsection content" in section_two[0].content


# ---------------------------------------------------------------------------
# ParagraphChunker
# ---------------------------------------------------------------------------


SAMPLE_PARAGRAPHS = textwrap.dedent("""\
    This is the first paragraph. It has some content that spans
    multiple lines but is still a single paragraph.

    This is the second paragraph. It's shorter.

    This is the third paragraph with enough words to be meaningful.
    It continues here with more text to make it reasonably sized
    for testing purposes.

    Short.

    Also short.

    This is the sixth paragraph with a decent amount of content
    that should be long enough to not need merging with neighbors.
""")


class TestParagraphChunker:
    """Tests for the double-newline paragraph chunker."""

    def test_splits_on_double_newline(self):
        """Splits text on double-newline boundaries."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        result = _make_ingest_result(SAMPLE_PARAGRAPHS, ContentType.GENERIC, "text.txt")
        chunks = chunker.chunk(result)

        # Should have 6 paragraphs (without merging)
        assert len(chunks) == 6

    def test_merges_short_paragraphs(self):
        """Short paragraphs are merged when merge_short=True."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=True, min_tokens=10))
        result = _make_ingest_result(SAMPLE_PARAGRAPHS, ContentType.GENERIC, "text.txt")
        chunks = chunker.chunk(result)

        # "Short." and "Also short." should be merged together
        # Total chunks should be fewer than 6
        assert len(chunks) < 6

    def test_chunk_type_is_paragraph(self):
        """All chunks have chunk_type 'paragraph'."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        result = _make_ingest_result(SAMPLE_PARAGRAPHS, ContentType.GENERIC, "text.txt")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.chunk_type == "paragraph"

    def test_content_type_preserved(self):
        """Content type from IngestResult flows to chunks."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        result = _make_ingest_result(SAMPLE_PARAGRAPHS, ContentType.GENERIC, "text.txt")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.content_type == ContentType.GENERIC

    def test_source_metadata_preserved(self):
        """Source metadata preserved on chunks."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        result = _make_ingest_result(SAMPLE_PARAGRAPHS, ContentType.GENERIC, "text.txt")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.source.file_path == "text.txt"

    def test_single_paragraph(self):
        """A single paragraph produces one chunk."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        result = _make_ingest_result("Just one paragraph.", ContentType.GENERIC)
        chunks = chunker.chunk(result)

        assert len(chunks) == 1
        assert chunks[0].content == "Just one paragraph."

    def test_whitespace_only_paragraphs_skipped(self):
        """Paragraphs that are only whitespace are skipped."""
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        text = "First paragraph.\n\n   \n\nSecond paragraph."
        result = _make_ingest_result(text, ContentType.GENERIC)
        chunks = chunker.chunk(result)

        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# TokenWindowChunker
# ---------------------------------------------------------------------------


class TestTokenWindowChunker:
    """Tests for the sliding token window chunker."""

    def test_single_window(self):
        """Short text fits in one window."""
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=100, overlap_tokens=10))
        result = _make_ingest_result("Short text that fits easily.", ContentType.GENERIC)
        chunks = chunker.chunk(result)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text that fits easily."

    def test_multiple_windows(self):
        """Long text is split into multiple overlapping windows."""
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        # Create text that's ~50 words
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=20, overlap_tokens=5))
        result = _make_ingest_result(text, ContentType.GENERIC)
        chunks = chunker.chunk(result)

        # Should produce multiple windows
        assert len(chunks) > 1

    def test_overlap_present(self):
        """Adjacent windows share overlapping tokens."""
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        words = [f"word{i}" for i in range(30)]
        text = " ".join(words)

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=10, overlap_tokens=3))
        result = _make_ingest_result(text, ContentType.GENERIC)
        chunks = chunker.chunk(result)

        # Check that consecutive chunks share some words
        if len(chunks) >= 2:
            words_0 = set(chunks[0].content.split())
            words_1 = set(chunks[1].content.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Adjacent windows should share overlapping tokens"

    def test_chunk_type_is_window(self):
        """All chunks have chunk_type 'window'."""
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=100, overlap_tokens=10))
        result = _make_ingest_result("Some text for window chunking.", ContentType.GENERIC)
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.chunk_type == "window"

    def test_content_type_preserved(self):
        """Content type is preserved."""
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=100, overlap_tokens=10))
        result = _make_ingest_result("Some text.", ContentType.RST)
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.content_type == ContentType.RST

    def test_source_metadata_preserved(self):
        """Source metadata flows through."""
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=100, overlap_tokens=10))
        result = _make_ingest_result("Some text.", ContentType.GENERIC, "doc.txt")
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.source.file_path == "doc.txt"


# ---------------------------------------------------------------------------
# DoclingChunker
# ---------------------------------------------------------------------------


class TestDoclingChunker:
    """Tests for the Docling bridge chunker.

    These mock out Docling since we don't want to depend on the actual
    Docling runtime in unit tests.
    """

    def test_requires_docling_document(self):
        """Raises ChunkingError if IngestResult has no docling_document."""
        from embeddy.chunking.docling_chunker import DoclingChunker

        chunker = DoclingChunker(config=ChunkConfig())
        result = _make_ingest_result("plain text", ContentType.DOCLING)
        # No docling_document set

        with pytest.raises(ChunkingError, match="docling_document"):
            chunker.chunk(result)

    def test_produces_chunks_from_docling(self):
        """Produces chunks from a mocked Docling document."""
        from embeddy.chunking.docling_chunker import DoclingChunker

        # Mock the docling document and HybridChunker
        mock_doc = MagicMock()

        # Create mock Docling chunks
        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "First docling chunk content."
        mock_chunk_1.meta = MagicMock()
        mock_chunk_1.meta.headings = ["Chapter 1"]
        mock_chunk_1.meta.origin = None

        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "Second docling chunk content."
        mock_chunk_2.meta = MagicMock()
        mock_chunk_2.meta.headings = ["Chapter 1", "Section 1.1"]
        mock_chunk_2.meta.origin = None

        chunker = DoclingChunker(config=ChunkConfig())
        result = _make_ingest_result("dummy text", ContentType.DOCLING, "doc.pdf")
        result.docling_document = mock_doc

        with patch("embeddy.chunking.docling_chunker.HybridChunker") as MockHybridChunker:
            mock_hc = MockHybridChunker.return_value
            mock_hc.chunk.return_value = [mock_chunk_1, mock_chunk_2]

            chunks = chunker.chunk(result)

        assert len(chunks) == 2
        assert chunks[0].content == "First docling chunk content."
        assert chunks[1].content == "Second docling chunk content."
        assert chunks[0].content_type == ContentType.DOCLING

    def test_chunk_type_is_docling(self):
        """Chunks have chunk_type 'docling'."""
        from embeddy.chunking.docling_chunker import DoclingChunker

        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Some docling content."
        mock_chunk.meta = MagicMock()
        mock_chunk.meta.headings = []
        mock_chunk.meta.origin = None

        chunker = DoclingChunker(config=ChunkConfig())
        result = _make_ingest_result("dummy", ContentType.DOCLING, "doc.pdf")
        result.docling_document = mock_doc

        with patch("embeddy.chunking.docling_chunker.HybridChunker") as MockHybridChunker:
            mock_hc = MockHybridChunker.return_value
            mock_hc.chunk.return_value = [mock_chunk]

            chunks = chunker.chunk(result)

        assert chunks[0].chunk_type == "docling"

    def test_source_metadata_preserved(self):
        """Source metadata from IngestResult preserved."""
        from embeddy.chunking.docling_chunker import DoclingChunker

        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Content."
        mock_chunk.meta = MagicMock()
        mock_chunk.meta.headings = []
        mock_chunk.meta.origin = None

        chunker = DoclingChunker(config=ChunkConfig())
        result = _make_ingest_result("dummy", ContentType.DOCLING, "report.pdf")
        result.docling_document = mock_doc

        with patch("embeddy.chunking.docling_chunker.HybridChunker") as MockHybridChunker:
            mock_hc = MockHybridChunker.return_value
            mock_hc.chunk.return_value = [mock_chunk]

            chunks = chunker.chunk(result)

        assert chunks[0].source.file_path == "report.pdf"


# ---------------------------------------------------------------------------
# get_chunker factory
# ---------------------------------------------------------------------------


class TestGetChunker:
    """Tests for the auto-selection factory function."""

    def test_python_content_type_selects_python_chunker(self):
        """ContentType.PYTHON selects PythonChunker."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = get_chunker(ContentType.PYTHON, ChunkConfig())
        assert isinstance(chunker, PythonChunker)

    def test_markdown_content_type_selects_markdown_chunker(self):
        """ContentType.MARKDOWN selects MarkdownChunker."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = get_chunker(ContentType.MARKDOWN, ChunkConfig())
        assert isinstance(chunker, MarkdownChunker)

    def test_generic_content_type_selects_paragraph_chunker(self):
        """ContentType.GENERIC selects ParagraphChunker."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = get_chunker(ContentType.GENERIC, ChunkConfig())
        assert isinstance(chunker, ParagraphChunker)

    def test_docling_content_type_selects_docling_chunker(self):
        """ContentType.DOCLING selects DoclingChunker."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.docling_chunker import DoclingChunker

        chunker = get_chunker(ContentType.DOCLING, ChunkConfig())
        assert isinstance(chunker, DoclingChunker)

    def test_explicit_strategy_overrides_auto(self):
        """An explicit strategy overrides the auto-selection."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        config = ChunkConfig(strategy="token_window")
        chunker = get_chunker(ContentType.PYTHON, config)
        assert isinstance(chunker, TokenWindowChunker)

    def test_explicit_python_strategy(self):
        """Explicit 'python' strategy selects PythonChunker."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.python_chunker import PythonChunker

        config = ChunkConfig(strategy="python")
        chunker = get_chunker(ContentType.GENERIC, config)
        assert isinstance(chunker, PythonChunker)

    def test_explicit_paragraph_strategy(self):
        """Explicit 'paragraph' strategy selects ParagraphChunker."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        config = ChunkConfig(strategy="paragraph")
        chunker = get_chunker(ContentType.PYTHON, config)
        assert isinstance(chunker, ParagraphChunker)

    def test_auto_strategy_uses_content_type(self):
        """strategy='auto' uses the content_type to pick."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.python_chunker import PythonChunker

        config = ChunkConfig(strategy="auto")
        chunker = get_chunker(ContentType.PYTHON, config)
        assert isinstance(chunker, PythonChunker)

    def test_other_code_types_use_paragraph(self):
        """Non-Python code types (JS, TS, etc.) fall back to ParagraphChunker in auto mode."""
        from embeddy.chunking import get_chunker
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        for ct in [ContentType.JAVASCRIPT, ContentType.TYPESCRIPT, ContentType.RUST]:
            chunker = get_chunker(ct, ChunkConfig())
            assert isinstance(chunker, ParagraphChunker), f"Expected ParagraphChunker for {ct}"


# ---------------------------------------------------------------------------
# Chunk ID uniqueness
# ---------------------------------------------------------------------------


class TestChunkIdUniqueness:
    """All generated chunks should have unique IDs."""

    def test_python_chunk_ids_unique(self):
        from embeddy.chunking.python_chunker import PythonChunker

        chunker = PythonChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_PYTHON, ContentType.PYTHON, "example.py")
        chunks = chunker.chunk(result)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_markdown_chunk_ids_unique(self):
        from embeddy.chunking.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker(config=ChunkConfig())
        result = _make_ingest_result(SAMPLE_MARKDOWN, ContentType.MARKDOWN, "doc.md")
        chunks = chunker.chunk(result)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_paragraph_chunk_ids_unique(self):
        from embeddy.chunking.paragraph_chunker import ParagraphChunker

        chunker = ParagraphChunker(config=ChunkConfig(merge_short=False))
        result = _make_ingest_result(SAMPLE_PARAGRAPHS, ContentType.GENERIC, "text.txt")
        chunks = chunker.chunk(result)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_token_window_chunk_ids_unique(self):
        from embeddy.chunking.token_window_chunker import TokenWindowChunker

        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)

        chunker = TokenWindowChunker(config=ChunkConfig(max_tokens=15, overlap_tokens=3))
        result = _make_ingest_result(text, ContentType.GENERIC)
        chunks = chunker.chunk(result)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
