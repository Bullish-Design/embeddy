"""chonkai — document processing: ingest + parse + chunk, with guaranteed invariants.

Importable with zero extras (IMPLEMENTATION_PLAN §2). This package NEVER
imports embeddy (one-way dependency, enforced in CI).
"""

from chonkai.chunkers import BaseChunker, ParagraphChunker
from chonkai.models import CHUNK_TYPES, Chunk, ChunkBudget, IngestResult, SourceMetadata
from chonkai.validated import ChunkValidationError, ValidatedChunker

__version__ = "0.1.0"

__all__ = [
    "CHUNK_TYPES",
    "BaseChunker",
    "Chunk",
    "ChunkBudget",
    "ChunkValidationError",
    "IngestResult",
    "ParagraphChunker",
    "SourceMetadata",
    "ValidatedChunker",
]
