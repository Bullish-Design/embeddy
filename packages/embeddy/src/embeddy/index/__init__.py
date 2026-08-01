"""Index backends. Default: sqlite-vec + FTS5 (SqliteStore)."""

from embeddy.index.base import Searchable, SearchFilters
from embeddy.index.filters import FilterCompileError
from embeddy.index.sqlite import SchemaError, SqliteStore, StoreError

__all__ = [
    "FilterCompileError",
    "SchemaError",
    "SearchFilters",
    "Searchable",
    "SqliteStore",
    "StoreError",
]
