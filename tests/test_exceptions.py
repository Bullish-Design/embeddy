from __future__ import annotations

import pytest

from embeddify.exceptions import (
    EmbeddifyError,
    ModelLoadError,
    EncodingError,
    ValidationError,
    SearchError,
)


class TestExceptionsHierarchy:
    def test_all_exceptions_inherit_from_base(self) -> None:
        assert issubclass(ModelLoadError, EmbeddifyError)
        assert issubclass(EncodingError, EmbeddifyError)
        assert issubclass(ValidationError, EmbeddifyError)
        assert issubclass(SearchError, EmbeddifyError)

    def test_exceptions_can_be_raised_with_message(self) -> None:
        with pytest.raises(ModelLoadError, match="model issue"):
            raise ModelLoadError("model issue")

        with pytest.raises(EncodingError, match="encoding issue"):
            raise EncodingError("encoding issue")

        with pytest.raises(ValidationError, match="validation issue"):
            raise ValidationError("validation issue")

        with pytest.raises(SearchError, match="search issue"):
            raise SearchError("search issue")

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

