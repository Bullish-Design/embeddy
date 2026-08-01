"""Paragraph chunker: blank-line split + short-merge (ChunkBudget.min_tokens)."""

from __future__ import annotations

from chonkai import ChunkBudget, IngestResult, ParagraphChunker, ValidatedChunker


def _counter(s: str) -> int:
    """Deterministic token counter (words)."""
    return len(s.split())


def _ingest(text: str) -> IngestResult:
    return IngestResult.from_text(text, path="t.txt", content_type="text")


def test_basic_split_without_budget() -> None:
    chunks = ParagraphChunker().chunk(_ingest("one two\n\nthree four\n\nfive\n"))
    assert [c.content for c in chunks] == ["one two", "three four", "five"]
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 1), (3, 3), (5, 5)]


def test_trailing_content_without_blank_line() -> None:
    chunks = ParagraphChunker().chunk(_ingest("a\n\nb\nc\n"))
    assert [c.content for c in chunks] == ["a", "b\nc"]
    assert chunks[-1].end_line == 4


def test_empty_and_whitespace_text_produce_no_chunks() -> None:
    assert ParagraphChunker().chunk(_ingest("")) == []
    assert ParagraphChunker().chunk(_ingest("   \n\n  ")) == []


def test_short_merge_reaches_min_tokens() -> None:
    chunker = ValidatedChunker(
        ParagraphChunker(token_counter=_counter),
        budget=ChunkBudget(max_tokens=100, min_tokens=6),
        token_counter=_counter,
    )
    chunks = chunker.chunk(_ingest("a b\n\nc d\n\ne f g\n\nh i j k\n"))
    # first two paragraphs (2+2=4 tokens) don't reach the floor; adding
    # "e f g" (3) gives 7 >= 6 -> merged chunk; "h i j k" alone has 4 < 6
    # but there is nothing left to merge with, so it flushes as-is.
    assert [c.content for c in chunks] == ["a b\n\nc d\n\ne f g", "h i j k"]
    assert chunks[0].start_line == 1 and chunks[0].end_line == 5


def test_merge_never_exceeds_max_tokens() -> None:
    budget = ChunkBudget(max_tokens=10, min_tokens=3)
    chunker = ValidatedChunker(
        ParagraphChunker(token_counter=_counter),
        budget=budget,
        token_counter=_counter,
    )
    text = "\n\n".join(f"w{i} x{i} y{i} z{i}" for i in range(20))  # 4 tokens each
    chunks = chunker.chunk(_ingest(text))
    for c in chunks:
        assert 0 < c.token_count <= budget.max_tokens
    assert all(c.content.strip() for c in chunks)


def test_lone_oversized_paragraph_emitted_whole_then_post_split() -> None:
    # one paragraph already over max_tokens: the merge keeps it whole and the
    # ValidatedChunker post-split cuts it to budget
    budget = ChunkBudget(max_tokens=5, min_tokens=2)
    chunker = ValidatedChunker(
        ParagraphChunker(token_counter=_counter),
        budget=budget,
        token_counter=_counter,
    )
    chunks = chunker.chunk(_ingest("a b c d e f g h\n\nz z\n"))
    assert all(c.token_count <= budget.max_tokens for c in chunks)
    joined = " ".join(c.content for c in chunks)
    assert "a" in joined and "h" in joined


def test_no_merge_without_min_tokens() -> None:
    budget = ChunkBudget(max_tokens=100, min_tokens=0)
    chunks = ParagraphChunker(token_counter=_counter).chunk(_ingest("a\n\nb\n"), budget)
    assert [c.content for c in chunks] == ["a", "b"]
