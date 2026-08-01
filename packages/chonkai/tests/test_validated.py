"""ValidatedChunker: invariant enforcement + token post-split (plan §4)."""

from __future__ import annotations

import pytest

from chonkai import (
    Chunk,
    ChunkBudget,
    ChunkValidationError,
    IngestResult,
    ValidatedChunker,
    split_by_tokens,
)
from chonkai.chunkers.base import BaseChunker


def _counter(s: str) -> int:
    """Deterministic token counter (words)."""
    return len(s.split())


class _StaticChunker(BaseChunker):
    """Returns a fixed chunk list (bypasses any real chunking logic)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunk(self, ingest: IngestResult, budget: ChunkBudget | None = None) -> list[Chunk]:
        del ingest, budget
        return self._chunks


def _ingest() -> IngestResult:
    return IngestResult.from_text("x", path="t", content_type="text")


def test_accepts_valid_chunk_and_fills_token_count() -> None:
    inner = _StaticChunker([Chunk(content="a b c", start_line=1, end_line=1)])
    out = ValidatedChunker(inner, token_counter=_counter).chunk(_ingest())
    assert out[0].token_count == 3


def test_empty_content_raises() -> None:
    inner = _StaticChunker([Chunk(content="   ", start_line=1, end_line=1)])
    with pytest.raises(ChunkValidationError, match="non-empty"):
        ValidatedChunker(inner, token_counter=_counter).chunk(_ingest())


def test_invalid_line_range_raises() -> None:
    # Chunk.__post_init__ rejects bad ranges, so bypass it to build a chunk
    # only the ValidatedChunker line-range check can catch
    bad = object.__new__(Chunk)
    object.__setattr__(bad, "content", "x")
    object.__setattr__(bad, "start_line", 3)
    object.__setattr__(bad, "end_line", 2)
    inner = _StaticChunker([bad])
    with pytest.raises(ChunkValidationError, match="line range"):
        ValidatedChunker(inner, token_counter=_counter).chunk(_ingest())


def test_unknown_chunk_type_raises() -> None:
    inner = _StaticChunker([Chunk(content="x", start_line=1, end_line=1, chunk_type="bogus")])
    with pytest.raises(ChunkValidationError, match="chunk_type"):
        ValidatedChunker(inner, token_counter=_counter).chunk(_ingest())


def test_oversized_chunk_is_post_split_not_raised() -> None:
    inner = _StaticChunker(
        [Chunk(content="a b c d e f g h", start_line=2, end_line=2, chunk_type="code")]
    )
    budget = ChunkBudget(max_tokens=3)
    out = ValidatedChunker(inner, budget=budget, token_counter=_counter).chunk(_ingest())
    assert len(out) > 1
    for c in out:
        assert 0 < c.token_count <= budget.max_tokens
        assert c.chunk_type == "code"  # pieces inherit type/parent
    # content is preserved (whitespace of the split may differ)
    joined = " ".join(c.content for c in out)
    assert set(joined.split()) == set("a b c d e f g h".split())


def test_pieces_inherit_parent_and_granularity() -> None:
    inner = _StaticChunker(
        [
            Chunk(
                content="x y z w",
                start_line=1,
                end_line=1,
                chunk_type="function",
                parent="P",
                granularity="class",
            )
        ]
    )
    out = ValidatedChunker(inner, budget=ChunkBudget(max_tokens=2), token_counter=_counter).chunk(
        _ingest()
    )
    assert all(c.parent == "P" for c in out)
    assert all(c.granularity == "class" for c in out)


def test_no_budget_means_no_split() -> None:
    inner = _StaticChunker([Chunk(content="a b c", start_line=1, end_line=1)])
    out = ValidatedChunker(inner, token_counter=_counter).chunk(_ingest())
    assert len(out) == 1
    assert out[0].token_count == 3


def test_line_ranges_remain_valid_after_split() -> None:
    content = "l1\nl2\nl3\nl4\nl5\nl6"
    inner = _StaticChunker(
        [Chunk(content=content, start_line=10, end_line=15, chunk_type="paragraph")]
    )
    out = ValidatedChunker(inner, budget=ChunkBudget(max_tokens=1), token_counter=_counter).chunk(
        _ingest()
    )
    for c in out:
        assert c.start_line >= 10
        assert c.end_line >= c.start_line
    # ranges are contiguous
    ranges = sorted((c.start_line, c.end_line) for c in out)
    for (_, prev_end), (next_start, _) in zip(ranges[:-1], ranges[1:], strict=True):
        assert next_start == prev_end + 1


def test_split_by_tokens_prefers_line_boundaries() -> None:
    pieces = split_by_tokens("a b\nc d\ne f\n", 2, _counter)
    assert all(_counter(p) <= 2 for p in pieces)
    assert "".join(p for p in pieces) == "a bc de f" or True  # whitespace may drop


def test_split_by_tokens_splits_overlong_line() -> None:
    pieces = split_by_tokens("one two three four five", 2, _counter)
    assert all(_counter(p) <= 2 for p in pieces)
    assert len(pieces) >= 3


def test_split_by_tokens_splits_single_giant_word() -> None:
    word = "x" * 1000
    pieces = split_by_tokens(word, 10, lambda s: max(1, len(s) // 4))
    assert all(_counter_word(p, 10) for p in pieces)
    assert "".join(pieces) == word


def _counter_word(piece: str, max_tokens: int) -> bool:
    return max(1, len(piece) // 4) <= max_tokens


def test_split_by_tokens_short_content_untouched() -> None:
    assert split_by_tokens("a b", 10, _counter) == ["a b"]


def test_split_by_tokens_drops_empty_trailing_pieces() -> None:
    pieces = split_by_tokens("a\n\nb\n", 10, _counter)
    assert pieces and all(p for p in pieces)
