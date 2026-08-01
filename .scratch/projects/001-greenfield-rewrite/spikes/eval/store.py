"""In-memory store implementing the Searchable subset the eval needs.

Pure numpy, zero native extensions — the eval harness must run anywhere,
including CI, with no sqlite-vec. The runner is store-agnostic (duck-typed
against the Searchable protocol from protocols.py); swapping in the real
sqlite backend later must not change the eval code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core_types import Metric, ScoredDocument, Vector, normalize_l2


@dataclass
class _Entry:
    chunk_id: str
    source_path: str
    content: str
    vector: Vector


@dataclass
class InMemoryStore:
    """Minimal Searchable (vector only) for the eval harness."""

    _entries: dict[str, dict[str, _Entry]] = field(default_factory=dict)

    async def add(
        self,
        collection: str,
        chunks: list,
        vectors: list[Vector],
    ) -> None:
        col = self._entries.setdefault(collection, {})
        for chunk, vec in zip(chunks, vectors):
            col[chunk.id] = _Entry(
                chunk_id=chunk.id,
                source_path=getattr(chunk, "source_path", ""),
                content=chunk.content,
                vector=normalize_l2(vec),
            )

    async def search_vector(
        self,
        collection: str,
        query_vector: Vector,
        filters=None,
        top_k: int = 10,
    ) -> list[ScoredDocument]:
        col = self._entries.get(collection, {})
        q = normalize_l2(query_vector)
        scored = sorted(
            (
                (e, float(np.dot(q, e.vector)))
                for e in col.values()
            ),
            key=lambda t: -t[1],
        )
        return [
            ScoredDocument(
                chunk_id=e.chunk_id,
                collection_id=collection,
                source_id="",
                source_path=e.source_path,
                content=e.content,
                score=score,
                metric=Metric.COSINE,
            )
            for e, score in scored[:top_k]
        ]
