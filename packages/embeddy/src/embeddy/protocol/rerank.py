"""RerankerProvider protocol + RerankHit — ported from the Phase-0 spike
(`spikes/protocols.py`). CONCEPT §5.3 / plan §3.

Optional post-fusion stage (retrieve ~50, rerank to top_k — the single
largest retrieval-quality lever). The wire shape is TEI/Jina, NOT a
non-existent OpenAI /v1/rerank standard (CONCEPT §7): the remote provider
POSTs `{"query": ..., "texts": [...], "top_n": N}` to the rerank endpoint.

Drafted at M1, FROZEN at M4 (IMPLEMENTATION_PLAN §12): rerank may still be
reshaped by Phases 4-5 (as a search.py stage); after the M4 freeze, changes
need a docs/decisions/ record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embeddy.protocol.types import Metric


@dataclass(frozen=True, slots=True)
class RerankHit:
    """One reranked hit: `index` into the input document list + score.

    `metric` is always Metric.RERANK — cross-encoder scores are not
    comparable to cosine or BM25 scores (CONCEPT §3.4).
    """

    index: int
    score: float
    metric: Metric = Metric.RERANK

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")


@runtime_checkable
class RerankerProvider(Protocol):
    """Optional post-fusion stage. Caller resolves role -> instruction via
    the registry and passes the resolved string (same rule as
    EmbeddingProvider, CONCEPT §5.1); local CrossEncoders typically ignore
    it."""

    model_name: str

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        instruction: str | None = None,  # already resolved by the caller
    ) -> list[RerankHit]:
        """Score `documents` against `query`, returning hits ranked by
        descending score. `top_k` limits the returned hits (None = all)."""
        ...
