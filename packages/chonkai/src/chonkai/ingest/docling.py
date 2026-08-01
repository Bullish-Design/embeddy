"""Docling bridge — the lazy-loading seam behind the `chonkai[docling]` extra.

This is the only chonkai module that loads docling at runtime (acceptance
grep: `grep -r "import docling" packages/chonkai/src` must match only
lazy-loading modules). Docling's DocumentConverter is expensive to construct,
so an Ingestor builds one converter instance and reuses it across
`ingest_file` calls (per-instance converter reuse, plan §4).

The docling surface chonkai consumes is declared as structural Protocols so
the code is typed without importing docling at runtime; the mock-based tests
implement the same shape.
"""

from __future__ import annotations

from typing import Protocol, cast


class _DoclingDocument(Protocol):
    """The slice of docling's DoclingDocument chonkai consumes."""

    def export_to_markdown(self) -> str: ...


class _ConversionResult(Protocol):
    """docling's DocumentConversionResult (duck-typed, read-only)."""

    @property
    def document(self) -> _DoclingDocument: ...

    @property
    def status(self) -> object | None: ...


class DocumentConverter(Protocol):
    """docling's DocumentConverter (duck-typed)."""

    def convert(self, source: str) -> _ConversionResult: ...


def load_document_converter() -> DocumentConverter:
    """Lazily construct a docling DocumentConverter.

    Raises ImportError when the `docling` extra is not installed. The import
    is deliberately inside the function so chonkai core loads cleanly with
    zero extras.
    """
    from docling.document_converter import DocumentConverter as _Converter

    return cast(DocumentConverter, _Converter())
