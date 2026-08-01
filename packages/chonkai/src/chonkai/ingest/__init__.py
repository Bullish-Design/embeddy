"""Ingest package: content-type detection + file reading + docling routing."""

from chonkai.ingest.content_types import (
    CODE_CONTENT_TYPES,
    DEFAULT_CONTENT_TYPE,
    DOCLING_ROUTED,
    ContentType,
    detect_content_type,
)
from chonkai.ingest.ingestor import IngestError, Ingestor, decode_bytes

__all__ = [
    "CODE_CONTENT_TYPES",
    "ContentType",
    "DEFAULT_CONTENT_TYPE",
    "DOCLING_ROUTED",
    "IngestError",
    "Ingestor",
    "decode_bytes",
    "detect_content_type",
]
