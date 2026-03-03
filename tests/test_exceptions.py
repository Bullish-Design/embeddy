# tests/test_exceptions.py
"""Tests for the embeddy exception hierarchy."""

from __future__ import annotations

import pytest

from embeddy.exceptions import (
    ChunkingError,
    EmbeddyError,
    EncodingError,
    IngestError,
    ModelLoadError,
    SearchError,
    ServerError,
    StoreError,
    ValidationError,
)


class TestExceptionsHierarchy:
    """All custom exceptions must inherit from EmbeddyError."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        assert issubclass(ModelLoadError, EmbeddyError)
        assert issubclass(EncodingError, EmbeddyError)
        assert issubclass(ValidationError, EmbeddyError)
        assert issubclass(SearchError, EmbeddyError)
        assert issubclass(IngestError, EmbeddyError)
        assert issubclass(StoreError, EmbeddyError)
        assert issubclass(ChunkingError, EmbeddyError)
        assert issubclass(ServerError, EmbeddyError)

    def test_exceptions_are_not_subclasses_of_each_other(self) -> None:
        """Each exception type should be a direct subclass of EmbeddyError, not of each other."""
        concrete = [
            ModelLoadError,
            EncodingError,
            ValidationError,
            SearchError,
            IngestError,
            StoreError,
            ChunkingError,
            ServerError,
        ]
        for i, cls_a in enumerate(concrete):
            for j, cls_b in enumerate(concrete):
                if i != j:
                    assert not issubclass(cls_a, cls_b), f"{cls_a.__name__} should not be subclass of {cls_b.__name__}"

    def test_exceptions_can_be_raised_with_message(self) -> None:
        exceptions = [
            (ModelLoadError, "model issue"),
            (EncodingError, "encoding issue"),
            (ValidationError, "validation issue"),
            (SearchError, "search issue"),
            (IngestError, "ingest issue"),
            (StoreError, "store issue"),
            (ChunkingError, "chunking issue"),
            (ServerError, "server issue"),
        ]
        for exc_class, msg in exceptions:
            with pytest.raises(exc_class, match=msg):
                raise exc_class(msg)

    def test_exception_chaining_preserves_cause(self) -> None:
        original = ValueError("low-level failure")

        with pytest.raises(EncodingError) as exc_info:
            try:
                raise original
            except ValueError as err:
                raise EncodingError("encoding failed") from err

        assert exc_info.value.__cause__ is original

    def test_str_and_repr_include_message(self) -> None:
        msg = "something went wrong"
        error = ModelLoadError(msg)

        assert msg in str(error)
        assert msg in repr(error)

    def test_base_error_is_catchable_for_all_subtypes(self) -> None:
        """Catching EmbeddyError should catch any subtype."""
        for exc_class in [
            ModelLoadError,
            EncodingError,
            ValidationError,
            SearchError,
            IngestError,
            StoreError,
            ChunkingError,
            ServerError,
        ]:
            with pytest.raises(EmbeddyError):
                raise exc_class("test")
