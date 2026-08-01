"""Property tests: all chunker invariants hold over an adversarial corpus.

Uses hypothesis to generate text including unicode, markdown-ish structures
(fences/headings), blank-line noise, and pathological single-line inputs.
For every (chunker, budget) combination, the ValidatedChunker guarantees are
asserted: non-empty content, valid 1-based line ranges, chunk_type in the
vocabulary, and token_count <= max_tokens with token_count == counter(text).
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from chonkai import (
    CHUNK_TYPES,
    ChunkBudget,
    IngestResult,
    MarkdownChunker,
    ParagraphChunker,
    SemchunkChunker,
    TreesitterChunker,
    ValidatedChunker,
)


def _counter(s: str) -> int:
    """Deterministic token counter (words)."""
    return len(s.split())


MARKDOWNY_ALPHABET = "abcdefghijklmnopqrstuvwxyz #`~-*_123\n\t"

corpus = st.one_of(
    st.text(),  # full unicode
    st.text(alphabet=MARKDOWNY_ALPHABET, min_size=0, max_size=400),
    st.just(""),
    st.just("\n\n\n  \n"),
    st.just("```python\n# comment\nx = 1\n```"),
    st.just("~~~\n# also code\n~~~"),
    st.just("x" * 5000),  # single giant line
    st.just("\t" * 20),
    st.just("# H1\n\n## H2\n\n### H3\n\nbody\n"),
)

budgets = st.sampled_from(
    [
        ChunkBudget(max_tokens=8),
        ChunkBudget(max_tokens=32, min_tokens=4),
        ChunkBudget(max_tokens=128, min_tokens=64),
    ]
)


def _assert_invariants(chunks, budget: ChunkBudget) -> None:
    for c in chunks:
        assert c.content.strip(), "chunk content must be non-empty"
        assert c.start_line >= 1, "start_line must be 1-based"
        assert c.end_line >= c.start_line, "line range must be valid"
        assert c.chunk_type in CHUNK_TYPES, f"unknown chunk_type {c.chunk_type!r}"
        assert c.token_count == _counter(c.content), "token_count must match the counter"
        assert c.token_count <= budget.max_tokens, "token budget must never be exceeded"


def _validated(inner, budget: ChunkBudget):
    return ValidatedChunker(inner, budget=budget, token_counter=_counter)


@given(text=corpus, budget=budgets)
@settings(max_examples=60)
def test_paragraph_invariants(text: str, budget: ChunkBudget) -> None:
    ingest = IngestResult.from_text(text, path="t", content_type="text")
    chunks = _validated(ParagraphChunker(token_counter=_counter), budget).chunk(ingest)
    _assert_invariants(chunks, budget)


@given(text=corpus, budget=budgets)
@settings(max_examples=60)
def test_markdown_invariants(text: str, budget: ChunkBudget) -> None:
    ingest = IngestResult.from_text(text, path="m.md", content_type="markdown")
    chunks = _validated(MarkdownChunker(), budget).chunk(ingest)
    _assert_invariants(chunks, budget)
    # the H4 fix: a heading line can never live inside a fenced code chunk
    in_fence = False
    for c in chunks:
        if c.chunk_type == "code":
            in_fence = True
        elif in_fence and c.chunk_type == "heading":
            raise AssertionError("heading emitted while inside a code fence")


@given(text=corpus, budget=budgets)
@settings(max_examples=60)
def test_semchunk_invariants(text: str, budget: ChunkBudget) -> None:
    ingest = IngestResult.from_text(text, path="s", content_type="text")
    chunks = _validated(SemchunkChunker(token_counter=_counter), budget).chunk(ingest)
    _assert_invariants(chunks, budget)


@given(text=corpus, budget=budgets)
@settings(max_examples=40)
def test_treesitter_invariants(text: str, budget: ChunkBudget) -> None:
    ingest = IngestResult.from_text(text, path="p.py", content_type="python")
    chunks = _validated(TreesitterChunker(), budget).chunk(ingest)
    _assert_invariants(chunks, budget)


@given(text=corpus, budget=budgets)
@settings(max_examples=40)
def test_treesitter_oversized_invariants(text: str, budget: ChunkBudget) -> None:
    ingest = IngestResult.from_text(text, path="p.py", content_type="python")
    chunks = _validated(TreesitterChunker(chunk_max_size=64), budget).chunk(ingest)
    _assert_invariants(chunks, budget)


@given(text=corpus, budget=budgets)
@settings(max_examples=40)
def test_treesitter_markdown_invariants(text: str, budget: ChunkBudget) -> None:
    ingest = IngestResult.from_text(text, path="m.md", content_type="markdown")
    chunks = _validated(TreesitterChunker(), budget).chunk(ingest)
    _assert_invariants(chunks, budget)


def test_h4_regression_case_is_locked() -> None:
    # exact H4 reproduction from the code review: '#' inside a fence
    text = "# Title\n\n```python\n# comment\n```\n"
    chunks = _validated(MarkdownChunker(), ChunkBudget(max_tokens=128)).chunk(
        IngestResult.from_text(text, path="m.md", content_type="markdown")
    )
    headings = [c for c in chunks if c.chunk_type == "heading"]
    assert [c.content for c in headings] == ["# Title"]
