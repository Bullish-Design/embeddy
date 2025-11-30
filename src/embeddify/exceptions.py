# src/embeddify/exceptions.py
from __future__ import annotations


class EmbeddifyError(Exception):
    """Base exception for all Embeddify errors.

    All custom exceptions in Embeddify inherit from this type so that users can
    catch a single base class for any library-specific error.
    """


class ModelLoadError(EmbeddifyError):
    """Raised when a SentenceTransformer model fails to load.

    Typical causes include an invalid model path, missing model files, or
    incompatible model formats.
    """


class EncodingError(EmbeddifyError):
    """Raised when text encoding fails.

    Used for failures that occur while converting input text into embeddings,
    including invalid input values and underlying model errors.
    """


class ValidationError(EmbeddifyError):
    """Raised when Embeddify-specific validation fails.

    This is distinct from :class:`pydantic.ValidationError` and is used for
    domain-level validation errors such as dimension mismatches.
    """


class SearchError(EmbeddifyError):
    """Raised when a semantic search operation fails.

    Wraps errors that occur during similarity computation or search, while
    preserving the original exception via exception chaining.
    """
