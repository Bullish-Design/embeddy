"""Chunker registry base: chunkers are sync, pure document transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chonkai.models import Chunk, ChunkBudget, IngestResult


class BaseChunker(ABC):
    """A chunker turns one IngestResult into a list of Chunk records.

    Chunkers stay simple and strategy-specific; the invariants (non-empty
    content, token budget, line ranges) are enforced once by
    ValidatedChunker, which wraps any chunker (CONCEPT §3.5).
    """

    @abstractmethod
    def chunk(
        self,
        ingest: IngestResult,
        budget: ChunkBudget | None = None,
    ) -> list[Chunk]: ...
