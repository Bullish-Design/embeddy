"""chonkai chunk-quality harness (plan §9 / §10.10, plan §11 tooling).

Measures chunker throughput on deterministic corpora (markdown with code
fences, decorated Python, generic prose) and asserts the chunk INVARIANTS
(non-empty content, 1-based inclusive line ranges, token budget, known
chunk_type vocabulary) — the "quality" half of the harness.

This directory is OUTSIDE the pytest testpaths (root pyproject.toml), so
`uv run pytest` (the default suite) never collects it. Run it explicitly:

    uv run pytest benchmarks/ --no-cov
    uv run pytest benchmarks/ --no-cov --benchmark-only

`--benchmark-only` skips the invariant/assertion tests and just times; by
default the invariant tests run too (they are cheap and deterministic).
"""

from __future__ import annotations

import pytest

from chonkai import (
    ChunkBudget,
    IngestResult,
    MarkdownChunker,
    ParagraphChunker,
    SemchunkChunker,
    TreesitterChunker,
    ValidatedChunker,
    default_token_counter,
)

PY_CORPUS = '''\
"""docstring"""
import os
from typing import Optional

class BaseRepository:
    """Base."""

    def __init__(self, path: str) -> None:
        self.path = path

    @staticmethod
    def parse(value: str) -> str:
        return value.strip()

    def exists(self) -> bool:
        return os.path.exists(self.path)


def build_repo(path: str) -> BaseRepository:
    repo = BaseRepository(path)
    return repo


async def run_migrations(repos: list[BaseRepository]) -> int:
    total = 0
    for repo in repos:
        total += 1
    return total
'''

MD_CORPUS = """\
# acme platform

The acme platform is a fictional document-retrieval product.

## Token expiry policy

Tokens expire after 30 days. The expiration is enforced by the auth layer.

### Refresh behavior

```python
def refresh(token):
    return token if not expired(token) else mint()
```

> A code fence above must NOT become a heading.

## Rate limits

- 10 requests per minute per key.
- Backoff is exponential with jitter.

## Storage

Documents are chunked, embedded, and stored in a vector index.

| size | dim |
|------|-----|
| 100k | 256 |
| 1M   | 256 |
"""

PROSE_CORPUS = (
    "The acme platform ingests documents, chunks them, embeds them, and serves hybrid search. " * 40
)

BUDGET = ChunkBudget(max_tokens=200)


@pytest.fixture(scope="module")
def token_counter():
    return default_token_counter()


@pytest.fixture(scope="module")
def py_ingest() -> IngestResult:
    return IngestResult.from_text(PY_CORPUS, path="repo.py", content_type="python")


@pytest.fixture(scope="module")
def md_ingest() -> IngestResult:
    return IngestResult.from_text(MD_CORPUS, path="guide.md", content_type="markdown")


@pytest.fixture(scope="module")
def prose_ingest() -> IngestResult:
    return IngestResult.from_text(PROSE_CORPUS, path="prose.txt", content_type="text")


# --------------------------------------------------------------------------- #
# quality: chunk invariants hold on every chunker (the "quality" gate)
# --------------------------------------------------------------------------- #


def _assert_invariants(chunks, budget: ChunkBudget) -> None:
    assert chunks, "no chunks produced"
    for c in chunks:
        assert c.content, "empty chunk content"
        assert c.start_line >= 1 and c.end_line >= c.start_line, "bad line range"
        assert c.chunk_type, "missing chunk_type"
        assert c.token_count <= budget.max_tokens, (
            f"chunk {c.token_count} tokens > budget {budget.max_tokens}"
        )


def test_paragraph_invariants(token_counter, prose_ingest, py_ingest):
    chunker = ValidatedChunker(ParagraphChunker(token_counter=token_counter))
    _assert_invariants(chunker.chunk(prose_ingest, BUDGET), BUDGET)
    _assert_invariants(chunker.chunk(py_ingest, BUDGET), BUDGET)


def test_markdown_invariants(token_counter, md_ingest):
    chunker = ValidatedChunker(MarkdownChunker())
    chunks = chunker.chunk(md_ingest, BUDGET)
    _assert_invariants(chunks, BUDGET)
    # code fences must not become headings, and heading parents must link
    assert not any(c.chunk_type == "heading" and c.content.startswith("```") for c in chunks)


def test_semchunk_invariants(token_counter, prose_ingest):
    chunker = ValidatedChunker(SemchunkChunker(token_counter=token_counter))
    chunks = chunker.chunk(prose_ingest, BUDGET)
    _assert_invariants(chunks, BUDGET)
    assert len(chunks) > 1, "prose corpus should split under a 200-token budget"


def test_treesitter_invariants(token_counter, py_ingest):
    chunker = ValidatedChunker(TreesitterChunker(granularity="function"))
    chunks = chunker.chunk(py_ingest, BUDGET)
    _assert_invariants(chunks, BUDGET)
    # per-definition chunking: the corpus has a class (as its methods) +
    # module-level functions — methods and functions must be extracted
    assert any(c.chunk_type == "method" for c in chunks), "method definition missing"
    assert any(c.chunk_type == "function" for c in chunks), "function missing"


# --------------------------------------------------------------------------- #
# throughput: chunker wall-time per corpus (the "measure" half)
# --------------------------------------------------------------------------- #


def test_bench_paragraph(benchmark, token_counter, prose_ingest):
    chunker = ValidatedChunker(ParagraphChunker(token_counter=token_counter))
    benchmark(lambda: chunker.chunk(prose_ingest, BUDGET))


def test_bench_markdown(benchmark, token_counter, md_ingest):
    chunker = ValidatedChunker(MarkdownChunker())
    benchmark(lambda: chunker.chunk(md_ingest, BUDGET))


def test_bench_semchunk(benchmark, token_counter, prose_ingest):
    chunker = ValidatedChunker(SemchunkChunker(token_counter=token_counter))
    benchmark(lambda: chunker.chunk(prose_ingest, BUDGET))


def test_bench_treesitter(benchmark, token_counter, py_ingest):
    chunker = ValidatedChunker(TreesitterChunker(granularity="function"))
    benchmark(lambda: chunker.chunk(py_ingest, BUDGET))
