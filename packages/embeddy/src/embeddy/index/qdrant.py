"""QdrantStore — the scale-path `Searchable` backend (plan §10, CONCEPT
§5.4 "Scale: Qdrant adapter"). Dense + sparse named vectors, payload
filters, collection-level quantization. Implements the FROZEN M4 protocol
(index/base.py) exactly; no protocol or wire-surface reshape (Phase 8
work item 1/2).

Verified API facts (probes against qdrant-client 1.18.0, 2026-08-01):
  * `QdrantClient(":memory:")` / `QdrantClient(path=":memory:")` are
    hermetic and offline (no docker, no network) — the default test suite
    exercises the real adapter on the in-memory backend.
  * Point ids MUST be UUIDs or integers — `s1:0` raises. Chunk ids are
    therefore hashed to stable uuid5 ids; the ORIGINAL chunk id is stored
    in the payload and restored on every read (the protocol's
    ScoredDocument.chunk_id / StoredChunk.id are never the uuid).
  * Upserting a point with only ONE named vector REPLACES the others (the
    sparse vector is provided TOGETHER with dense in add()/reindex_source,
    never in a later partial upsert).
  * `FieldCondition` has NO startswith/prefix match in 1.18.0; the honest
    prefix translation stores every ancestor path prefix in a
    `path_prefixes` array payload and filters with MatchAny (exact prefix
    semantics — probed).
  * Delete-by-filter with must/must_not works (the reindex stale-cleanup
    primitive), as do filtered scroll pagination, exact count with filter,
    and score_threshold on query_points (cosine: similarity in [-1, 1],
    higher = better — identical min_score direction to SqliteStore).
  * Quantization config is ACCEPTED at collection creation in local mode
    (searches work) but NOT read back via get_collection — a local-mode
    artifact; on a server it applies. The adapter passes it through and
    never reads it back.
  * `client.search()` is REMOVED in 1.18 — the query_points API is the
    only path.

DELIBERATE SEMANTICS (documented in docs/decisions/0004):
  * search_fts is a pure-Python BM25 scan over the stored payloads (Qdrant
    has no BM25/FTS engine). Filters are TRUE pre-filters (the scan only
    reads matching points — the M3 recall contract holds). Scores mirror
    FTS5's negative-rank convention (best hit ~0, higher/less-negative =
    better) so `min_score` means the same thing as on SqliteStore. The
    `raw` option is accepted for protocol compatibility and is a no-op
    (there is no FTS5 query syntax to bypass).
  * reindex_source has NO transactional atomicity (Qdrant has no
    transactions): new chunks upsert FIRST, stale old ids delete second,
    source metadata last. On failure the OLD chunks remain intact and
    queryable — a weaker guarantee than SqliteStore's one-transaction
    swap, documented in 0004.
  * The qdrant-client is synchronous; store methods are async (protocol)
    and call it inline. For the in-memory (tested) mode calls are instant;
    a remote server blocks the event loop during a call — an accepted
    single-user-scale tradeoff (documented in 0004).

EXTRA-only, LAZY: qdrant_client is imported only inside methods — zero
extras `import embeddy` stays clean (the providers/* pattern). The
beyond-protocol extras (create_collection / get_chunk / list_chunks /
list_collections — the SqliteStore extras) are implemented here; the
server's 501 path covers anything missing (count_fts is intentionally
absent).
"""

from __future__ import annotations

import math
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from embeddy.index.base import CollectionInfo, SearchFilters, StoreError
from embeddy.index.factory import StoreSpec
from embeddy.protocol.types import (
    CollectionStats,
    Metric,
    ScoredDocument,
    SourceId,
    SourceMetadata,
    StoredChunk,
    Vector,
    assert_unit_vector,
)

if TYPE_CHECKING:  # pragma: no cover - type-time only (the audit allows it)
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

# Controlled collection ids only: they are interpolated into collection
# names below, so anything exotic is rejected up front (matches sqlite).
_COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")

# The companion collection holding first-class source rows (CONCEPT §3.3 —
# sources are part of the protocol, every backend implements them). Named
# with a suffix that list_collections filters out.
_SOURCES_SUFFIX = "__sources"

# Point-id namespace: chunk ids + source ids hash to stable uuid5 ids
# (probed: qdrant rejects non-UUID string ids).
_NS = uuid.NAMESPACE_URL

# Scroll page size for the payload scans (search_fts / list_chunks /
# list_sources / stale-id discovery).
_SCROLL_LIMIT = 100

# BM25 defaults (classic Robertson/Sparck Jones parameters).
_BM25_K1 = 1.2
_BM25_B = 0.75

# The payload fields every chunk point carries (decode targets).
_CHUNK_PAYLOAD_KEYS = (
    "chunk_id",
    "collection_id",
    "source_id",
    "source_path",
    "content",
    "chunk_type",
    "start_line",
    "end_line",
    "parent",
    "granularity",
    "token_count",
)


@dataclass(frozen=True, slots=True)
class _PointId:
    """Internal point-id helpers (deterministic, collision-free across
    collections). Chunk ids are f"{source_id}:{seq}" strings that qdrant
    rejects as point ids, so the adapter hashes them; the originals are
    always restored from the payload on the way out."""

    @staticmethod
    def chunk(collection: str, chunk_id: str) -> str:
        return str(uuid.uuid5(_NS, f"chunk:{collection}:{chunk_id}"))

    @staticmethod
    def source(source_id: SourceId) -> str:
        return str(uuid.uuid5(_NS, f"source:{source_id}"))


def _sources_collection(collection: str) -> str:
    return f"{collection}{_SOURCES_SUFFIX}"


def compile_qdrant_filter(filters: SearchFilters) -> models.Filter | None:
    """Translate the frozen SearchFilters into a Qdrant payload filter
    (probed against 1.18.0). Pure function — unit-tested mock-free.

    * content_types / chunk_types  -> MatchAny on the stored keyword field
    * source_path_prefix          -> MatchAny over the point's path_prefixes
      array (exact prefix semantics — 1.18 has no string-prefix condition)
    * metadata_match              -> MatchValue per (field, value) pair
    Empty filters -> None (no filter, the qdrant convention).
    """
    from qdrant_client.http import models

    must: list[models.FieldCondition] = []
    if filters.content_types:
        must.append(
            models.FieldCondition(
                key="content_type",
                match=models.MatchAny(any=list(filters.content_types)),
            )
        )
    if filters.source_path_prefix:
        must.append(
            models.FieldCondition(
                key="path_prefixes",
                match=models.MatchAny(any=[filters.source_path_prefix]),
            )
        )
    if filters.chunk_types:
        must.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchAny(any=list(filters.chunk_types)),
            )
        )
    for field, value in filters.metadata_match:
        must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if not must:
        return None
    return models.Filter(must=must)


def _tokenize(text: str) -> list[str]:
    """Query/content tokenizer for the pure-Python BM25 path. Lowercases and
    splits on non-alphanumerics; NO Porter stemming (a documented difference
    from the FTS5 backend — docs/decisions/0004)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(documents: list[tuple[str, str]], query: str) -> list[float]:
    """Pure BM25 (k1=1.2, b=0.75) over (chunk_id, content) pairs. Returns
    one raw BM25 score per document (unbounded, >= 0, higher = better)."""
    n = len(documents)
    if n == 0:
        return []
    term_freqs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df: dict[str, int] = {}
    for _, content in documents:
        tokens = _tokenize(content)
        doc_lens.append(len(tokens))
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in set(tf):
            df[token] = df.get(token, 0) + 1
        term_freqs.append(tf)
    avgdl = sum(doc_lens) / n if n else 0.0
    query_tokens = set(_tokenize(query))
    scores: list[float] = []
    for tf, dl in zip(term_freqs, doc_lens, strict=True):
        score = 0.0
        for token in query_tokens:
            f = tf.get(token, 0)
            if f == 0:
                continue
            idf = math.log(1.0 + (n - df[token] + 0.5) / (df[token] + 0.5))
            denom = f + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / avgdl) if avgdl else f + _BM25_K1
            score += idf * (f * (_BM25_K1 + 1.0)) / denom
        scores.append(score)
    return scores


class QdrantStore:
    """Qdrant backend. Open with `await QdrantStore.open(spec)` (async;
    `spec` comes from `parse_store_url` via `build_store`)."""

    def __init__(self, spec: StoreSpec) -> None:
        self._spec = spec
        self._client: QdrantClient | None = None

    @classmethod
    async def open(cls, spec: StoreSpec) -> QdrantStore:
        """Build + connect the client. For a REMOTE url this VERIFIES
        reachability (cheap get_collections) so the server's honest-health
        contract reports not-ready when qdrant is unreachable — a store is
        never half-open. In-memory mode skips the check (nothing to reach)."""
        store = cls(spec)
        client = store._build_client()
        if not spec.memory:
            try:
                client.get_collections()
            except Exception as exc:  # noqa: BLE001 - wrap ANY transport failure
                client.close()
                raise StoreError(f"cannot reach qdrant at {spec.url!r}: {exc}") from exc
        store._client = client
        return store

    def _build_client(self) -> QdrantClient:
        from qdrant_client import QdrantClient

        spec = self._spec
        if spec.memory:
            return QdrantClient(":memory:")
        # check_compatibility=False: the client's background version-compat
        # check warns (and spawns a thread) when it cannot reach the server;
        # the adapter's own reachability check (open()) raises a proper
        # StoreError for that case instead. The version check adds nothing.
        return QdrantClient(
            host=spec.host, port=spec.port, https=spec.https, check_compatibility=False
        )

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def __aenter__(self) -> QdrantStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # client plumbing
    # ------------------------------------------------------------------ #

    def _c(self) -> QdrantClient:
        if self._client is None:
            raise StoreError("store is not open; call QdrantStore.open() first")
        return self._client

    def _require_collection(self, collection: str) -> None:
        self._validate_collection_id(collection)
        if not self._c().collection_exists(collection):
            raise StoreError(f"unknown collection: {collection!r}")

    @staticmethod
    def _validate_collection_id(collection: str) -> None:
        if not _COLLECTION_ID_RE.match(collection):
            raise StoreError(
                f"invalid collection id {collection!r}; must match {_COLLECTION_ID_RE.pattern}"
            )

    def _collection_dimension(self, collection: str) -> int:
        info = self._c().get_collection(collection)
        params = info.config.params
        vectors = params.vectors if params is not None else None
        if isinstance(vectors, dict):
            # named-vector collections (created with vectors_config={"dense": ...})
            dense = vectors.get("dense")
            if dense is not None:
                return int(dense.size)
        elif vectors is not None:
            return int(vectors.size)
        raise StoreError(f"collection {collection!r} has no dense vector config")

    @staticmethod
    def _validate_vectors(vectors: list[Vector], dim: int) -> None:
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != dim:
                raise StoreError(
                    f"vector shape {arr.shape} does not match collection dimension {dim}"
                )
            assert_unit_vector(arr)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k < 1:
            raise StoreError(f"top_k must be >= 1, got {top_k}")

    @staticmethod
    def _validate_min_score(min_score: float) -> None:
        if not math.isfinite(min_score):
            raise StoreError(f"min_score must be finite, got {min_score!r}")

    # ------------------------------------------------------------------ #
    # collections
    # ------------------------------------------------------------------ #

    async def create_collection(
        self,
        collection: str,
        dimension: int,
        *,
        quantization: str | None = None,
    ) -> None:
        """Create the main collection (named dense vector + the sparse named
        vector + optional quantization) and its companion sources
        collection. Re-creating an existing collection with the same
        dimension is a no-op; a dimension mismatch raises (the C2 bug class
        is impossible — the collection records the resolved dimension).

        `quantization` is "int8" | "binary" | None (beyond-protocol kwarg;
        the store's default comes from the `store.url` DSN query param —
        docs/decisions/0004)."""
        from qdrant_client.http import models

        self._validate_collection_id(collection)
        if dimension < 1:
            raise StoreError(f"dimension must be >= 1, got {dimension}")
        client = self._c()
        if client.collection_exists(collection):
            existing = self._collection_dimension(collection)
            if existing != dimension:
                raise StoreError(
                    f"collection {collection!r} already exists with dimension "
                    f"{existing}, got {dimension}"
                )
        else:
            config: dict[str, Any] = {
                "dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            }
            client.create_collection(
                collection,
                vectors_config=cast(dict, config),
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
                quantization_config=self._quantization_config(quantization),
            )
        await self._ensure_sources_collection(collection)

    def _quantization_config(self, quantization: str | None) -> Any:
        from qdrant_client.http import models

        value = quantization if quantization is not None else self._spec.quantization
        if value is None:
            return None
        if value == "int8":
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            )
        if value == "binary":
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )
        raise StoreError(f"invalid quantization {value!r}; expected one of int8/binary/None")

    async def _ensure_sources_collection(self, collection: str) -> None:
        """The companion collection is a sqlite-sources-table analogue: one
        point per source with a dummy size-1 vector (qdrant requires a
        vector config). Payload indexes are created for the fields the
        adapter filters on (no-op in local mode, effective on a server)."""
        from qdrant_client.http import models

        client = self._c()
        name = _sources_collection(collection)
        if not client.collection_exists(name):
            client.create_collection(
                name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for key in ("path", "content_hash", "content_type"):
                    client.create_payload_index(
                        name, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD
                    )
        if client.collection_exists(collection):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for key in (
                    "chunk_id",
                    "source_id",
                    "chunk_type",
                    "content_type",
                    "source_path",
                    "path_prefixes",
                ):
                    client.create_payload_index(
                        collection, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD
                    )

    async def list_collections(self) -> list[CollectionInfo]:
        """All collections in id order (the server's GET /api/v1/collections).
        Companion source collections are filtered out."""
        client = self._c()
        names = sorted(c.name for c in client.get_collections().collections)
        result: list[CollectionInfo] = []
        for name in names:
            if name.endswith(_SOURCES_SUFFIX):
                continue
            result.append(
                CollectionInfo(
                    collection_id=name,
                    vector_dimension=self._collection_dimension(name),
                )
            )
        return result

    async def stats(self, collection: str) -> CollectionStats:
        self._require_collection(collection)
        client = self._c()
        dimension = self._collection_dimension(collection)
        chunk_count = int(client.count(collection, exact=True).count)
        source_count = int(client.count(_sources_collection(collection), exact=True).count)
        size_bytes = 0
        for _, payload in self._scroll_payloads(_sources_collection(collection)):
            size_bytes += int(payload.get("size_bytes") or 0)
        return CollectionStats(
            collection_id=collection,
            chunk_count=chunk_count,
            source_count=source_count,
            vector_dimension=dimension,
            size_bytes=size_bytes,
            last_updated=None,  # mirrors SqliteStore (no timestamp column)
        )

    # ------------------------------------------------------------------ #
    # sources (first-class — CONCEPT §3.3)
    # ------------------------------------------------------------------ #

    async def upsert_source(self, collection: str, source: SourceMetadata) -> SourceId:
        self._require_collection(collection)
        client = self._c()
        client.upsert(
            _sources_collection(collection),
            points=[
                self._source_point(source),
            ],
        )
        return source.id

    async def get_source(self, collection: str, path: str) -> SourceMetadata | None:
        from qdrant_client.http import models

        self._require_collection(collection)
        client = self._c()
        points, _ = client.scroll(
            _sources_collection(collection),
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="path", match=models.MatchValue(value=path)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        return self._payload_to_source(self._payload_of(points[0]))

    async def list_sources(self, collection: str) -> list[SourceMetadata]:
        self._require_collection(collection)
        sources = [
            self._payload_to_source(payload)
            for _, payload in self._scroll_payloads(_sources_collection(collection))
        ]
        return sorted(sources, key=lambda s: s.path)  # stable diff order (sqlite parity)

    async def delete_source(self, collection: str, source_id: SourceId) -> None:
        from qdrant_client.http import models

        self._require_collection(collection)
        client = self._c()
        client.delete(
            collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id", match=models.MatchValue(value=source_id)
                        ),
                    ]
                )
            ),
        )
        client.delete(_sources_collection(collection), points_selector=[_PointId.source(source_id)])

    def _source_point(self, source: SourceMetadata) -> models.PointStruct:
        from qdrant_client.http import models

        return models.PointStruct(
            id=_PointId.source(source.id),
            vector=[1.0],  # the companion collection's single unnamed size-1 vector
            payload={
                "id": source.id,
                "collection_id": source.collection_id,
                "path": source.path,
                "content_hash": source.content_hash,
                "size_bytes": source.size_bytes,
                "mtime": source.mtime.isoformat() if source.mtime else None,
                "content_type": source.content_type,
            },
        )

    @staticmethod
    def _payload_of(point: Any) -> dict[str, Any]:
        """Record payloads are `dict[str, Any] | None` (a point without a
        payload is legal); normalize to an empty dict."""
        return dict(point.payload or {})

    @staticmethod
    def _payload_to_source(payload: dict[str, Any]) -> SourceMetadata:
        mtime = payload.get("mtime")
        return SourceMetadata(
            id=str(payload["id"]),
            collection_id=str(payload["collection_id"]),
            path=str(payload["path"]),
            content_hash=str(payload["content_hash"]),
            size_bytes=int(payload["size_bytes"]),
            mtime=datetime.fromisoformat(str(mtime)) if mtime else None,
            content_type=payload.get("content_type"),
        )

    # ------------------------------------------------------------------ #
    # chunks
    # ------------------------------------------------------------------ #

    async def add(
        self,
        collection: str,
        chunks: list[StoredChunk],
        vectors: list[Vector],
        *,
        sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
    ) -> None:
        """Store chunks + vectors in one upsert batch. All chunks must
        reference sources that already exist (call upsert_source first).
        Vectors must be unit-norm float32 of the collection's resolved
        dimension.

        `sparse_vectors` (beyond-protocol, Phase 8): aligned (indices,
        values) pairs stored in the point's `sparse` named vector — MUST be
        provided together with dense (a later partial upsert would REPLACE
        the dense vector — probed). Exercised by tests with synthetic
        sparse vectors; the real learned-sparse encoder is deferred
        (docs/decisions/0004)."""
        self._require_collection(collection)
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")
        if not chunks:
            return
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise StoreError(
                f"sparse_vectors length {len(sparse_vectors)} does not match chunks {len(chunks)}"
            )
        dim = self._collection_dimension(collection)
        self._validate_vectors(vectors, dim)
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise StoreError("duplicate chunk id in add() payload")
        missing = self._missing_sources(collection, {c.source_id for c in chunks})
        if missing:
            raise StoreError(
                f"chunks reference unknown source_id(s) {sorted(missing)}; call upsert_source first"
            )
        source_lookup = self._source_lookup(collection, {c.source_id for c in chunks})
        client = self._c()
        client.upsert(
            collection,
            points=[
                self._chunk_point(
                    collection,
                    chunk,
                    vec,
                    source_lookup[chunk.source_id],
                    sparse=sparse_vectors[i] if sparse_vectors is not None else None,
                )
                for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )

    async def delete(self, collection: str, chunk_ids: list[str]) -> None:
        self._require_collection(collection)
        if not chunk_ids:
            return
        self._c().delete(
            collection,
            points_selector=[_PointId.chunk(collection, cid) for cid in chunk_ids],
        )

    def _chunk_point(
        self,
        collection: str,
        chunk: StoredChunk,
        vector: Vector,
        source_info: tuple[str | None, str],
        *,
        sparse: tuple[list[int], list[float]] | None,
    ) -> models.PointStruct:
        from qdrant_client.http import models

        content_type, path = source_info
        prefixes = self._path_prefixes(path)
        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "collection_id": chunk.collection_id,
            "source_id": chunk.source_id,
            "source_path": path,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "parent": chunk.parent,
            "granularity": chunk.granularity,
            "token_count": chunk.token_count,
            "content_type": content_type,
            "path_prefixes": prefixes,
        }
        vector_config: dict[str, Any] = {"dense": np.asarray(vector, dtype=np.float32).tolist()}
        if sparse is not None:
            vector_config["sparse"] = models.SparseVector(indices=sparse[0], values=list(sparse[1]))
        return models.PointStruct(
            id=_PointId.chunk(collection, chunk.id),
            vector=vector_config,
            payload=payload,
        )

    @staticmethod
    def _path_prefixes(path: str) -> list[str]:
        """All ancestor prefixes of a source path, stored per point so the
        source_path_prefix filter is an EXACT prefix match (probed: qdrant
        1.18 FieldCondition has no string-prefix condition)."""
        parts = path.split("/")
        return ["/".join(parts[: i + 1]) for i in range(len(parts)) if parts[i]]

    def _missing_sources(self, collection: str, source_ids: set[SourceId]) -> set[SourceId]:
        if not source_ids:
            return set()
        client = self._c()
        points = client.retrieve(
            _sources_collection(collection),
            ids=[_PointId.source(sid) for sid in sorted(source_ids)],
            with_payload=False,
        )
        found = {str(p.id) for p in points}
        return {sid for sid in source_ids if _PointId.source(sid) not in found}

    def _source_lookup(
        self, collection: str, source_ids: set[SourceId]
    ) -> dict[SourceId, tuple[str | None, str]]:
        """source_id -> (content_type, path) — the payload metadata copied
        onto each chunk point (the vec0-aux-column analogue)."""
        if not source_ids:
            return {}
        client = self._c()
        points = client.retrieve(
            _sources_collection(collection),
            ids=[_PointId.source(sid) for sid in sorted(source_ids)],
            with_payload=True,
        )
        result: dict[SourceId, tuple[str | None, str]] = {}
        for point in points:
            payload = self._payload_of(point)
            sid = str(payload["id"])
            result[sid] = (payload.get("content_type"), str(payload["path"]))
        return result

    async def reindex_source(
        self,
        collection: str,
        source: SourceMetadata,
        chunks: list[StoredChunk],
        vectors: list[Vector],
        *,
        sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
    ) -> None:
        """Swap a source's chunk set WITHOUT delete-then-reingest (fixes
        H7; docs/decisions/0004 documents the achievable guarantee).

        Order (no transactions exist): 1) upsert the new chunk set (new
        ids never collide with the old), 2) delete the STALE old ids by
        filter, 3) refresh the source metadata point. On failure during
        (1) the old chunks remain intact and queryable — the H7 data-loss
        failure mode cannot happen; a failure during (2)/(3) may briefly
        expose BOTH old and new chunks for the source, which the next
        successful reindex of the same source self-heals."""
        self._require_collection(collection)
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")
        dim = self._collection_dimension(collection)
        self._validate_vectors(vectors, dim)
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise StoreError("duplicate chunk id in reindex_source payload")
        foreign = {c.source_id for c in chunks} - {source.id}
        if foreign:
            raise StoreError(
                f"reindex_source chunks reference source_id(s) {sorted(foreign)}; "
                f"expected only {source.id!r} (one source at a time)"
            )
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise StoreError(
                f"sparse_vectors length {len(sparse_vectors)} does not match chunks {len(chunks)}"
            )
        client = self._c()
        old_ids = self._chunk_ids_for_source(collection, source.id)
        new_ids = set(ids)
        client.upsert(
            collection,
            points=[
                self._chunk_point(
                    collection,
                    chunk,
                    vec,
                    (source.content_type, source.path),
                    sparse=sparse_vectors[i] if sparse_vectors is not None else None,
                )
                for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        stale = sorted(old_ids - new_ids)
        if stale:
            self._delete_stale_chunks(collection, source.id, new_ids)
        client.upsert(
            _sources_collection(collection),
            points=[self._source_point(source)],
        )

    def _chunk_ids_for_source(self, collection: str, source_id: SourceId) -> set[str]:
        from qdrant_client.http import models

        ids: set[str] = set()
        for _, payload in self._scroll_payloads(
            collection,
            filter_=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id", match=models.MatchValue(value=source_id)
                    ),
                ]
            ),
        ):
            ids.add(str(payload["chunk_id"]))
        return ids

    def _delete_stale_chunks(self, collection: str, source_id: SourceId, new_ids: set[str]) -> None:
        from qdrant_client.http import models

        client = self._c()
        client.delete(
            collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id", match=models.MatchValue(value=source_id)
                        ),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="chunk_id", match=models.MatchAny(any=sorted(new_ids))
                        ),
                    ],
                )
            ),
        )

    async def get_chunk(self, collection: str, chunk_id: str) -> StoredChunk | None:
        """Fetch ONE stored chunk by id (the server's /api/v1/similar). Store
        EXTRA beyond the Searchable protocol (like SqliteStore's)."""
        self._require_collection(collection)
        points = self._c().retrieve(
            collection, ids=[_PointId.chunk(collection, chunk_id)], with_payload=True
        )
        if not points:
            return None
        return self._payload_to_chunk(collection, self._payload_of(points[0]))

    async def list_chunks(
        self,
        collection: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredChunk]:
        """List stored chunks in id order (the server's GET /api/v1/chunks).
        Store EXTRA (like SqliteStore's). Qdrant scroll order is
        unspecified, so the page is materialized then sorted — honest, but
        O(n) per page (documented; fine at this scale)."""
        self._require_collection(collection)
        if limit < 1:
            raise StoreError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise StoreError(f"offset must be >= 0, got {offset}")
        chunks = [
            self._payload_to_chunk(collection, payload)
            for _, payload in self._scroll_payloads(collection)
        ]
        chunks.sort(key=lambda c: c.id)
        return chunks[offset : offset + limit]

    @staticmethod
    def _payload_to_chunk(collection: str, payload: dict[str, Any]) -> StoredChunk:
        return StoredChunk(
            id=str(payload["chunk_id"]),
            collection_id=str(payload["collection_id"]),
            source_id=str(payload["source_id"]),
            content=str(payload["content"]),
            chunk_type=str(payload["chunk_type"]),
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
            parent=payload.get("parent"),
            granularity=payload.get("granularity"),
            token_count=int(payload.get("token_count") or 0),
        )

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    async def search_vector(
        self,
        collection: str,
        query_vector: Vector,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[ScoredDocument]:
        """Cosine KNN with payload PRE-filters (a restrictive filter still
        returns full top_k — the M3 recall contract; qdrant applies the
        filter before the k-bound scan). Cosine scores are similarities in
        [-1, 1], higher = better — the exact min_score direction of
        SqliteStore's 1 - cosine_distance."""
        from qdrant_client.http import models

        self._require_collection(collection)
        self._validate_top_k(top_k)
        dim = self._collection_dimension(collection)
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != dim:
            raise StoreError(
                f"query vector shape {q.shape} does not match collection dimension {dim}"
            )
        assert_unit_vector(q)
        if min_score is not None:
            self._validate_min_score(min_score)
        response = self._c().query_points(
            collection,
            query=models.NearestQuery(nearest=q.tolist()),
            using="dense",
            query_filter=compile_qdrant_filter(filters),
            score_threshold=min_score,
            limit=top_k,
            with_payload=True,
        )
        return [
            self._point_to_scored(self._payload_of(p), score=float(p.score), metric=Metric.COSINE)
            for p in response.points
        ]

    async def search_fts(
        self,
        collection: str,
        query: str,
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
        raw: bool = False,
    ) -> list[ScoredDocument]:
        """BM25 over the stored payloads (pure-Python — Qdrant has no
        BM25/FTS engine; docs/decisions/0004). Filters are true
        pre-filters: the scan only reads matching points, so restrictive
        filters keep full recall within the filtered set.

        Scores mirror FTS5's negative-rank convention (<= 0, higher /
        less-negative = better) so `min_score` means the same thing as on
        SqliteStore. `raw` is accepted for protocol compatibility and is a
        NO-OP: there is no FTS5 query syntax on this path."""
        del raw  # accepted for protocol compatibility; no FTS5 syntax exists
        self._require_collection(collection)
        self._validate_top_k(top_k)
        if min_score is not None:
            self._validate_min_score(min_score)
        if not query.strip():
            return []
        points = self._scroll_payloads(collection, filter_=compile_qdrant_filter(filters))
        documents = [(str(p["chunk_id"]), str(p["content"])) for _, p in points]
        scores = _bm25_scores(documents, query)
        ranked: list[ScoredDocument] = []
        for (_, payload), raw_score in zip(points, scores, strict=True):
            if raw_score <= 0.0:
                # shares no query term — NOT a match (OR-semantics: docs
                # overlapping any query token rank by BM25; zero-overlap
                # docs are excluded, mirroring FTS5's non-matching set).
                continue
            if min_score is not None and -raw_score < min_score:
                continue
            ranked.append(self._point_to_scored(payload, score=-raw_score, metric=Metric.BM25))
        ranked.sort(key=lambda h: h.score, reverse=True)
        return ranked[:top_k]

    async def search_sparse(
        self,
        collection: str,
        query_indices: list[int],
        query_values: list[float],
        filters: SearchFilters,
        top_k: int,
        *,
        min_score: float | None = None,
    ) -> list[ScoredDocument]:
        """Sparse-vector search over the `sparse` named vector (Phase 8
        EXTENSION, beyond the frozen protocol — the protocol has no sparse
        channel; exercised by tests with synthetic sparse vectors, real
        learned-sparse is deferred — docs/decisions/0004). Dot-product
        similarity, metric SPARSE_DOT, higher = better."""
        from qdrant_client.http import models

        self._require_collection(collection)
        self._validate_top_k(top_k)
        if min_score is not None:
            self._validate_min_score(min_score)
        response = self._c().query_points(
            collection,
            query=models.NearestQuery(
                nearest=models.SparseVector(indices=query_indices, values=query_values)
            ),
            using="sparse",
            query_filter=compile_qdrant_filter(filters),
            score_threshold=min_score,
            limit=top_k,
            with_payload=True,
        )
        return [
            self._point_to_scored(
                self._payload_of(p), score=float(p.score), metric=Metric.SPARSE_DOT
            )
            for p in response.points
        ]

    @staticmethod
    def _point_to_scored(
        payload: dict[str, Any], *, score: float, metric: Metric
    ) -> ScoredDocument:
        return ScoredDocument(
            chunk_id=str(payload["chunk_id"]),
            collection_id=str(payload["collection_id"]),
            source_id=str(payload["source_id"]),
            source_path=str(payload["source_path"]),
            content=str(payload["content"]),
            score=score,
            metric=metric,
        )

    # ------------------------------------------------------------------ #
    # scroll helper (payload scans)
    # ------------------------------------------------------------------ #

    def _scroll_payloads(
        self, collection: str, *, filter_: models.Filter | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        """All (point_id, payload) pairs matching `filter_`, paginated.
        Used by the O(n) paths (BM25, list_chunks, list_sources, stats,
        stale-id discovery) — honest linear scans, documented."""
        client = self._c()
        result: list[tuple[str, dict[str, Any]]] = []
        offset: object = None
        while True:
            points, next_offset = client.scroll(
                collection,
                scroll_filter=filter_,
                limit=_SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            result.extend((str(p.id), self._payload_of(p)) for p in points)
            if next_offset is None:
                return result
            offset = next_offset
