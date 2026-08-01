"""Ingestor: content-type detection, file reading, docling routing (plan §4).

Reads a file, detects its content type, decodes it (UTF-8 with a latin-1
fallback — the old M-utf8 finding: decode errors are collected into the
result, never raised), computes the sha256 content hash, and populates
`SourceMetadata` (size, mtime, content_hash, content_type). Docling-routed
types go through the docling bridge (lazy, `chonkai[docling]` extra).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from chonkai.ingest.content_types import DOCLING_ROUTED, detect_content_type
from chonkai.ingest.docling import DocumentConverter, load_document_converter
from chonkai.models import IngestResult, SourceMetadata, compute_content_hash


class IngestError(RuntimeError):
    """A file could not be ingested (missing, unreadable, docling failure)."""


def decode_bytes(data: bytes) -> tuple[str, tuple[str, ...]]:
    """Decode file bytes: UTF-8 first, latin-1 fallback, errors collected.

    latin-1 maps every byte so it can never fail; the fallback therefore
    always succeeds and the utf-8 failure is reported as a warning on the
    IngestResult (fixes the old M-utf8 finding).
    """
    try:
        return data.decode("utf-8"), ()
    except UnicodeDecodeError as exc:
        return data.decode("latin-1"), (
            f"utf-8 decode failed at byte {exc.start} ({exc.reason}); fell back to latin-1",
        )


class Ingestor:
    """Ingests files and raw text into `IngestResult` records.

    A docling DocumentConverter is constructed once per Ingestor and reused
    across `ingest_file` calls (per-instance converter reuse). Pass a mock
    `converter` for tests or to pre-warm the docling path.
    """

    def __init__(self, converter: DocumentConverter | None = None) -> None:
        self._converter = converter

    @property
    def _docling(self) -> DocumentConverter:
        if self._converter is None:
            self._converter = load_document_converter()
        return self._converter

    def ingest_text(
        self,
        text: str,
        *,
        path: str = "<memory>",
        content_type: str | None = None,
        mtime: datetime | None = None,
    ) -> IngestResult:
        """Ingest raw text (no file I/O); metadata is derived from the text."""
        return IngestResult.from_text(text, path=path, content_type=content_type, mtime=mtime)

    def ingest_file(self, path: str | Path) -> IngestResult:
        """Ingest a file: stat, read, detect type, decode or route to docling."""
        p = Path(path)
        try:
            st = p.stat()
        except OSError as exc:
            raise IngestError(f"cannot stat {p}: {exc}") from exc
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise IngestError(f"cannot read {p}: {exc}") from exc

        content_type = detect_content_type(p)
        source = SourceMetadata(
            path=str(p),
            size_bytes=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=UTC),
            content_hash=compute_content_hash(data),
            content_type=content_type,
        )
        if content_type in DOCLING_ROUTED:
            text, document, warnings = self._convert_with_docling(p)
            return IngestResult(source=source, text=text, warnings=warnings, document=document)
        text, warnings = decode_bytes(data)
        return IngestResult(source=source, text=text, warnings=warnings)

    def _convert_with_docling(self, path: Path) -> tuple[str, object, tuple[str, ...]]:
        """Run the docling converter; return (markdown text, document, warnings).

        A missing `chonkai[docling]` extra surfaces as ImportError (from the
        lazy loader) — callers can act on it. Other conversion failures are
        wrapped in IngestError.
        """
        converter = self._docling  # ImportError when the extra is missing
        try:
            result = converter.convert(source=str(path))
        except Exception as exc:
            raise IngestError(f"docling conversion failed for {path}: {exc}") from exc
        document = result.document
        text = document.export_to_markdown()
        status = getattr(result, "status", None)
        warnings: tuple[str, ...] = ()
        if status is not None:
            label = getattr(status, "value", str(status))
            if label != "success":
                warnings = (f"docling conversion status: {label}",)
        return text, document, warnings
