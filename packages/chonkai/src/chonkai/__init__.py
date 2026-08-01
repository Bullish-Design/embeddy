"""chonkai — document processing: ingest + parse + chunk, with guaranteed invariants.

Importable with zero extras (IMPLEMENTATION_PLAN §2). This package NEVER
imports embeddy (one-way dependency, enforced in CI). Heavy parsers (docling,
arbitrary-HF tokenizers) are lazy-loaded behind extras.
"""

from chonkai.chunkers import (
    BaseChunker,
    DoclingChunker,
    MarkdownChunker,
    ParagraphChunker,
    SemchunkChunker,
    TreesitterChunker,
    get_chunker,
)
from chonkai.ingest import (
    ContentType,
    IngestError,
    Ingestor,
    decode_bytes,
    detect_content_type,
)
from chonkai.models import (
    CHUNK_TYPES,
    Chunk,
    ChunkBudget,
    IngestResult,
    SourceMetadata,
    compute_content_hash,
)
from chonkai.tokens import default_token_counter, tokenizers_token_counter
from chonkai.validated import ChunkValidationError, ValidatedChunker, split_by_tokens

__version__ = "0.1.0"

__all__ = [
    "CHUNK_TYPES",
    "BaseChunker",
    "Chunk",
    "ChunkBudget",
    "ChunkValidationError",
    "ContentType",
    "DoclingChunker",
    "IngestError",
    "IngestResult",
    "Ingestor",
    "MarkdownChunker",
    "ParagraphChunker",
    "SemchunkChunker",
    "SourceMetadata",
    "TreesitterChunker",
    "ValidatedChunker",
    "compute_content_hash",
    "decode_bytes",
    "default_token_counter",
    "detect_content_type",
    "get_chunker",
    "split_by_tokens",
    "tokenizers_token_counter",
]
