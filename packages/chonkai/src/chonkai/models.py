"""Core chonkai models: the typed records the document layer produces.

chonkai owns these (CONCEPT §4.1) and never imports embeddy. embeddy has its
own storage-side records (`embeddy.protocol.types`); the pipeline converts
between the two at the package boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

# Vocabulary the chunkers may emit and ValidatedChunker accepts. Phase 2
# adds the tree-sitter code kinds (struct/enum/trait/interface/impl) and the
# markdown code-block kind.
CHUNK_TYPES: frozenset[str] = frozenset(
    {
        "paragraph",
        "heading",
        "section",
        "function",
        "method",
        "class",
        "struct",
        "enum",
        "trait",
        "interface",
        "impl",
        "module",
        "code",
    }
)


def compute_content_hash(data: bytes) -> str:
    """sha256 hex digest of a source's raw bytes.

    The single hash implementation shared by the ingestor and
    `IngestResult.from_text` (plan §4: "Wire IngestResult.from_text to share
    the hash logic").
    """
    return hashlib.sha256(data).hexdigest()


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
    """One ingested source: metadata + extracted text, ready to chunk.

    `warnings` collects non-fatal ingest issues (encoding fallback, docling
    status) — the old M-utf8 finding is fixed by collecting, not raising.
    `document` optionally carries the docling Document for docling-routed
    sources so the DoclingChunker bridge can use its heading metadata; it is
    typed `object` so chonkai core never imports docling.
    """

    source: SourceMetadata
    text: str
    warnings: tuple[str, ...] = ()
    document: object | None = None

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        path: str = "<memory>",
        content_type: str | None = None,
        mtime: datetime | None = None,
        warnings: tuple[str, ...] = (),
        document: object | None = None,
    ) -> IngestResult:
        """Minimal ingest path for in-memory text (no file I/O)."""
        data = text.encode("utf-8")
        return cls(
            source=SourceMetadata(
                path=path,
                size_bytes=len(data),
                mtime=mtime or datetime.now(UTC),
                content_hash=compute_content_hash(data),
                content_type=content_type,
            ),
            text=text,
            warnings=tuple(warnings),
            document=document,
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One chunk of a source.

    Line ranges are 1-based, inclusive (converted from any 0-based internal
    spans at the public boundary — PRECODE_REPORT). `token_count` is filled by
    ValidatedChunker (the invariant wrapper owns the token counter).
    `granularity` is the tree-sitter granularity ("function" | "class" |
    "module") when the chunk came from a code definition. `has_error_nodes`
    flags chunks from code with broken syntax (tree-sitter still parses it;
    there is no paragraph fallback).
    """

    content: str  # non-empty (ValidatedChunker invariant)
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    chunk_type: str = "paragraph"
    parent: str | None = None  # enclosing heading / definition name
    token_count: int = 0
    granularity: str | None = None  # "function" | "class" | "module"
    has_error_nodes: bool = False

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError(f"invalid line range ({self.start_line}-{self.end_line})")
