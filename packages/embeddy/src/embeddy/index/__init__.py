"""Index backends. Default: sqlite-vec + FTS5 (SqliteStore). Scale path
(Phase 8): QdrantStore behind the SAME frozen `Searchable` protocol, plus
the `build_store` factory that maps a store DSN (`store.url`) to a backend.
"""

from embeddy.index.base import CollectionInfo, Searchable, SearchFilters, StoreError
from embeddy.index.factory import StoreSpec, build_store, parse_store_url
from embeddy.index.filters import FilterCompileError
from embeddy.index.qdrant import QdrantStore
from embeddy.index.sqlite import SchemaError, SqliteStore

__all__ = [
    "CollectionInfo",
    "FilterCompileError",
    "QdrantStore",
    "SchemaError",
    "SearchFilters",
    "Searchable",
    "SqliteStore",
    "StoreError",
    "StoreSpec",
    "build_store",
    "parse_store_url",
]
