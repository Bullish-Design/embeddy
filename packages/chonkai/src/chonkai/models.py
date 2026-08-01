"""Core chonkai models: the typed records the document layer produces.

chonkai owns these (CONCEPT §4.1) and never imports embeddy. embeddy has its
own storage-side records (`embeddy.protocol.types`); the pipeline converts
between the two at the package boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

# Vocabulary the chunkers may emit and ValidatedChunker accepts. Grows with
# Phase 2 (heading/section/code...). "paragraph" is the Phase-1 baseline.
CHUNK_TYPES: frozenset[str] = frozenset(
    {
        "paragraph",
        "heading",
        "section",
        "function",
        "method",
        "class",
        "module",
        "code",
    }
)


@dataclass(frozen=True, slots=True)
class ChunkBudget:
    """Token budget the caller (embeddy) derives from model context.

    chonkai owns the mechanics, embeddy owns the policy: embeddy passes
    `model.context_length - headroom` (CONCEPT §4.3).
    """

    max_tokens: int
    min_tokens: int = 0  # Phase 2: paragraph short-merge floor

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be within [0, max_tokens={self.max_tokens}]"
            )


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Ingest-side description of one source (file / URL / raw text)."""

    path: str  # canonical path within the collection
    size_bytes: int
    mtime: datetime | None = None
    content_hash: str | None = None  # sha256 hex, computed by the ingestor
    content_type: str | None = None  # extension map / content sniffing (Phase 2)

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got {self.size_bytes}")


@dataclass(frozen=True, slots=True)
class IngestResult:
    """One ingested source: metadata + extracted text, ready to chunk."""

    source: SourceMetadata
    text: str

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        path: str = "<memory>",
        content_type: str | None = None,
        mtime: datetime | None = None,
    ) -> IngestResult:
        """Minimal ingest path used before the Phase-2 Ingestor lands."""
        data = text.encode("utf-8")
        return cls(
            source=SourceMetadata(
                path=path,
                size_bytes=len(data),
                mtime=mtime or datetime.now(UTC),
                content_hash=hashlib.sha256(data).hexdigest(),
                content_type=content_type,
            ),
            text=text,
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One chunk of a source.

    Line ranges are 1-based, inclusive (converted from any 0-based internal
    spans at the public boundary — PRECODE_REPORT). `token_count` is filled by
    ValidatedChunker (the invariant wrapper owns the token counter).
    """

    content: str  # non-empty (ValidatedChunker invariant)
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    chunk_type: str = "paragraph"
    parent: str | None = None  # enclosing heading / definition name
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError(f"invalid line range ({self.start_line}-{self.end_line})")
