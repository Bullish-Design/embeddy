"""Semchunk chunker: token accuracy, overlap, char offsets -> line ranges."""

from __future__ import annotations

from chonkai import ChunkBudget, IngestResult, SemchunkChunker


def _counter(s: str) -> int:
    """Deterministic token counter (words)."""
    return len(s.split())


def _ingest(text: str) -> IngestResult:
    return IngestResult.from_text(text, path="s.txt", content_type="text")


def test_chunks_respect_token_budget() -> None:
    text = " ".join(f"w{i}" for i in range(40))
    chunks = SemchunkChunker(token_counter=_counter).chunk(
        _ingest(text), ChunkBudget(max_tokens=10)
    )
    assert len(chunks) == 4
    for c in chunks:
        assert _counter(c.content) <= 10
    joined = " ".join(c.content for c in chunks).split()
    assert len(joined) == 40  # nothing lost


def test_no_budget_emits_whole_text() -> None:
    chunks = SemchunkChunker(token_counter=_counter).chunk(_ingest("a b c"))
    assert len(chunks) == 1
    assert chunks[0].content == "a b c"
    assert chunks[0].start_line == 1 and chunks[0].end_line == 1


def test_empty_input_produces_no_chunks() -> None:
    chunker = SemchunkChunker(token_counter=_counter)
    assert chunker.chunk(_ingest(""), ChunkBudget(max_tokens=10)) == []
    assert chunker.chunk(_ingest("  \n\n "), ChunkBudget(max_tokens=10)) == []


def test_overlap_repeats_boundary_text() -> None:
    text = " ".join(f"w{i}" for i in range(12))
    chunks = SemchunkChunker(token_counter=_counter, overlap=1).chunk(
        _ingest(text), ChunkBudget(max_tokens=5)
    )
    assert len(chunks) > 1
    for a, b in zip(chunks[:-1], chunks[1:], strict=True):
        # one token of overlap between consecutive chunks
        assert a.content.split()[-1] == b.content.split()[0]


def test_line_ranges_from_char_offsets() -> None:
    text = "alpha beta\n\ngamma delta\n\nepsilon\n"
    chunks = SemchunkChunker(token_counter=_counter).chunk(_ingest(text), ChunkBudget(max_tokens=2))
    # each 2-token line is its own chunk; line ranges must be 1-based
    ranges = [(c.start_line, c.end_line) for c in chunks]
    assert ranges == [(1, 1), (3, 3), (5, 5)], ranges


def test_chunk_line_ranges_cover_mid_line_offsets() -> None:
    # a chunk may start mid-line: its range is the line span containing it
    text = "a b c d e f g"
    chunks = SemchunkChunker(token_counter=_counter).chunk(_ingest(text), ChunkBudget(max_tokens=3))
    assert all(c.start_line == 1 and c.end_line == 1 for c in chunks)


def test_chunks_are_contiguous() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    chunks = SemchunkChunker(token_counter=_counter).chunk(_ingest(text), ChunkBudget(max_tokens=7))
    joined = " ".join(c.content for c in chunks).split()
    assert joined == text.split()
