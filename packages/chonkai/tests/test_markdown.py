"""Markdown chunker: golden test — code-fence-aware headings, parent hierarchy."""

from __future__ import annotations

from conftest import ingest_from_file

from chonkai import IngestResult, MarkdownChunker
from chonkai.models import Chunk


def _chunks(markdown: str) -> list[Chunk]:
    ingest = IngestResult.from_text(markdown, path="a.md", content_type="markdown")
    return MarkdownChunker().chunk(ingest)


def test_golden_markdown_fixture(fixtures) -> None:
    chunks = MarkdownChunker().chunk(ingest_from_file(fixtures / "markdown_sample.md", "markdown"))
    assert [(c.chunk_type, c.start_line, c.end_line, c.parent, c.content) for c in chunks] == [
        ("heading", 1, 1, None, "# Title"),
        ("paragraph", 3, 3, "Title", "Intro paragraph here."),
        ("heading", 5, 5, "Title", "## Section A"),
        ("paragraph", 7, 7, "Section A", "Some text in section A."),
        (
            "code",
            9,
            13,
            "Section A",
            "```python\n# this looks like a heading but is inside a fence\ndef f():\n    pass\n```",
        ),
        ("paragraph", 15, 15, "Section A", "More text after the code block."),
        ("heading", 17, 17, "Section A", "### Subsection"),
        ("paragraph", 19, 19, "Subsection", "Nested paragraph."),
        ("heading", 21, 21, "Title", "## Section B"),
        ("paragraph", 23, 24, "Section B", "- item one\n- item two"),
    ]


def test_hash_inside_fence_is_not_a_heading() -> None:
    chunks = _chunks("# Top\n\n```python\n# comment\nx = 1\n```\n")
    headings = [c for c in chunks if c.chunk_type == "heading"]
    assert [c.content for c in headings] == ["# Top"]
    code = [c for c in chunks if c.chunk_type == "code"]
    assert len(code) == 1
    assert "# comment" in code[0].content


def test_fence_variants() -> None:
    for fence in ("```", "~~~"):
        chunks = _chunks(f"# T\n\n{fence}\n# not a heading\n{fence}\n")
        assert [c.chunk_type for c in chunks] == ["heading", "code"]
        assert "# not a heading" in chunks[1].content


def test_unclosed_fence_consumes_rest() -> None:
    chunks = _chunks("# T\n\n```python\n# still code\n")
    assert chunks[-1].chunk_type == "code"
    assert chunks[-1].content.startswith("```python")


def test_heading_hierarchy_parents() -> None:
    chunks = _chunks("# A\n\n## B\n\ntext\n\n### C\n\nmore\n")
    by_content = {c.content: c for c in chunks}
    assert by_content["## B"].parent == "A"
    assert by_content["### C"].parent == "B"
    assert by_content["text"].parent == "B"
    assert by_content["more"].parent == "C"


def test_sibling_heading_resets_stack() -> None:
    chunks = _chunks("# A\n\n## B\n\ntext\n\n# A2\n\ntext2\n")
    text2 = [c for c in chunks if c.content == "text2"][0]
    assert text2.parent == "A2"


def test_preamble_paragraph_has_no_parent() -> None:
    chunks = _chunks("intro\n\n# T\n\nbody\n")
    intro = chunks[0]
    assert intro.chunk_type == "paragraph"
    assert intro.parent is None
    assert [c.content for c in chunks if c.chunk_type == "heading"] == ["# T"]


def test_empty_input_produces_no_chunks() -> None:
    assert _chunks("") == []
    assert _chunks("\n\n  \n") == []
