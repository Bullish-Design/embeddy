"""Shared test helpers for the chonkai unit suite (plan §11: per-package tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chonkai import IngestResult

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    """Path to the golden fixture directory."""
    return FIXTURES


def ingest_from_file(path: Path, content_type: str) -> IngestResult:
    """Read a fixture file into an IngestResult with the given content type."""
    return IngestResult.from_text(
        path.read_text(encoding="utf-8"),
        path=path.name,
        content_type=content_type,
    )
