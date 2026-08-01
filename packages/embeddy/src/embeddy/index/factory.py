"""Store factory — the config-time construction seam for Searchable backends
(Phase 8 / plan §10: the scale path). Mirrors `providers/factory.py`:

  * `parse_store_url(url)` — pure, unit-testable DSN parsing. Supported
    forms (CONCEPT §5.4's "one config line", `store.url`):
      - sqlite:  `sqlite://<path>` | `sqlite://:memory:` | `sqlite:<path>`
                 | a bare path (`embeddy.db`) | `:memory:`
      - qdrant:  `qdrant://:memory:` (offline, hermetic — the default-suite
                 test mode) | `qdrant://host` | `qdrant://host:port`
                 | `qdrant+https://host:port`
                 | `?quantization=int8|binary|none` (collection-level
                 quantization, applied at create_collection time)
  * `build_store(url)` — async, mirrors `build_provider`: returns an OPEN
    `SqliteStore` or `QdrantStore`. For qdrant the store is connected at
    open time (a reachability check — the honest-health contract: an
    unreachable qdrant makes the server report not-ready, never a
    half-open store).

The default (no `store.url` configured) is the server's `store_path`
sqlite DSN — the bare `app = create_app()` path is unchanged (Phase 8
work item 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from embeddy.index.base import Searchable, StoreError
from embeddy.index.sqlite import SqliteStore

# Quantization options accepted in the qdrant DSN (`?quantization=...`).
_QDRANT_QUANTIZATION = ("none", "int8", "binary")


@dataclass(frozen=True, slots=True)
class StoreSpec:
    """The parsed store DSN. `backend` selects the Searchable implementation;
    the remaining fields are backend-specific. Unit-tested (mock-free)."""

    backend: str  # "sqlite" | "qdrant"
    url: str  # the original DSN (for messages)
    # sqlite
    sqlite_path: str = ":memory:"
    # qdrant
    host: str | None = None
    port: int = 6333
    https: bool = False
    memory: bool = False
    quantization: str | None = None  # None | "none" | "int8" | "binary"


def parse_store_url(url: str) -> StoreSpec:
    """Parse a store DSN into a StoreSpec. Unknown schemes and malformed
    qdrant URLs raise StoreError at config time (the build_provider
    pattern: config errors surface before any request)."""
    if not url or not url.strip():
        raise StoreError("store URL must be a non-empty string")
    url = url.strip()
    lower = url.lower()

    if lower.startswith("qdrant") or lower.startswith("qdrant+"):
        return _parse_qdrant_url(url)
    if lower.startswith("sqlite"):
        return StoreSpec(
            backend="sqlite",
            url=url,
            sqlite_path=_sqlite_path(url),
        )
    # bare path (no scheme): a sqlite file path or ":memory:".
    return StoreSpec(backend="sqlite", url=url, sqlite_path=url)


def _parse_qdrant_url(url: str) -> StoreSpec:
    # urlsplit needs a real scheme: keep "qdrant://" as-is (scheme qdrant),
    # or "qdrant+https://" -> "https://" so the transport scheme parses.
    if url.lower().startswith("qdrant+"):
        parsed = urlsplit(url[len("qdrant") + 1 :])
    else:
        parsed = urlsplit(url)
    netloc = parsed.netloc
    # "qdrant://:memory:" — urlsplit gives netloc ":memory:".
    if netloc == ":memory:" or netloc.startswith(":memory:"):
        return StoreSpec(
            backend="qdrant",
            url=url,
            memory=True,
            quantization=_qdrant_quantization(parsed),
        )
    if not netloc:
        raise StoreError(f"invalid qdrant store URL {url!r}: missing host")
    host = parsed.hostname
    if not host:
        raise StoreError(f"invalid qdrant store URL {url!r}: missing host")
    port = parsed.port or 6333
    https = parsed.scheme == "https"
    return StoreSpec(
        backend="qdrant",
        url=url,
        host=host,
        port=port,
        https=https,
        quantization=_qdrant_quantization(parsed),
    )


def _qdrant_quantization(parsed: object) -> str | None:
    query = parse_qs(getattr(parsed, "query", ""))
    values = query.get("quantization")
    if not values:
        return None
    value = values[0].strip().lower()
    if value not in _QDRANT_QUANTIZATION:
        raise StoreError(
            f"invalid quantization {values[0]!r}; expected one of {_QDRANT_QUANTIZATION}"
        )
    return None if value == "none" else value


def _sqlite_path(url: str) -> str:
    """sqlite://<path> | sqlite:///abs/path | sqlite:<path> -> path."""
    rest = url[len("sqlite") :]
    if rest.startswith("://"):
        rest = rest[3:]
    elif rest.startswith(":"):
        rest = rest[1:]
    # "sqlite://" with nothing after -> :memory: (the sqlite convention).
    if not rest:
        return ":memory:"
    return rest


async def build_store(url: str) -> Searchable:
    """Open a Searchable backend for a store DSN (async: sqlite connects
    the DB, qdrant verifies reachability — both report failures to the
    server's honest-health contract BEFORE the app claims ready)."""
    spec = parse_store_url(url)
    if spec.backend == "qdrant":
        from embeddy.index.qdrant import QdrantStore

        return await QdrantStore.open(spec)
    return await SqliteStore.open(spec.sqlite_path)
