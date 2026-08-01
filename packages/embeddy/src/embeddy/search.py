"""Search — pure fusion + the search_hybrid orchestration (CONCEPT §5.5).

RRF (k=60) is the default fusion; weighted (min-max normalized) is the
alternative. `fuse_rrf` / `fuse_weighted` are pure: no I/O, independently
testable, metric-agnostic in input (RRF uses only ranks; weighted
min-max-normalizes each list before blending). Output carries its own
metric (RRF / WEIGHTED) so scores are never compared across metrics
(CONCEPT §3.4).

`search_hybrid` is the ONE async orchestration entry point the Phase-5
pipeline and Phase-6 server consume. It deliberately lives here as a
function over the `Searchable` protocol, NOT as a store method: any backend
implements the protocol and inherits hybrid search for free (the Qdrant
adapter gets it without duplicating fusion logic), the fusion functions stay
pure and independently testable, and the store stays dumb.
"""

from __future__ import annotations

from collections.abc import Sequence

from embeddy.index.base import Searchable, SearchFilters
from embeddy.protocol.rerank import RerankerProvider
from embeddy.protocol.types import (
    Metric,
    ScoredDocument,
    SearchResult,
    Vector,
)

RRF_DEFAULT_K = 60

_HYBRID_MODES = ("rrf", "weighted")

# B008: no function-call defaults — a frozen empty-filters singleton.
_EMPTY_FILTERS = SearchFilters()


async def search_hybrid(
    store: Searchable,
    *,
    collection: str,
    query_text: str,
    query_vector: Vector,
    filters: SearchFilters = _EMPTY_FILTERS,
    top_k: int = 10,
    mode: str = "rrf",
    weights: Sequence[float] | None = None,
    retrieve_k: int = 50,
    min_score: float | None = None,
    raw: bool = False,
    reranker: RerankerProvider | None = None,
    rerank_top_k: int | None = None,
    instruction: str | None = None,
) -> SearchResult:
    """Retrieve by vector + FTS in parallel legs, fuse, optionally rerank.

    Both legs fetch `retrieve_k` candidates (default 50 — "retrieve ~50,
    rerank to top_k"); fusion combines them; the result is truncated to
    `top_k`. `min_score` is passed to each leg and interpreted in THAT
    leg's metric semantics (cosine [0,1] / BM25 rank <= 0) — never as a
    cross-metric comparison. `raw` opts into verbatim FTS5 query syntax.

    Rerank stage: when `reranker` is given, ALL fused candidates (bounded
    by 2*retrieve_k) are scored against `query_text` and the top
    `rerank_top_k` (default: top_k) are returned with metric RERANK.
    `instruction` is an already-resolved instruction string (registry
    resolve_instruction, query role).

    `total_results` is the number of unique candidate chunks across both
    legs before fusion truncation (a cheap in-memory count, documented in
    SearchResult).
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if retrieve_k < top_k:
        raise ValueError(f"retrieve_k ({retrieve_k}) must be >= top_k ({top_k})")
    if mode not in _HYBRID_MODES:
        raise ValueError(f"mode must be one of {_HYBRID_MODES}, got {mode!r}")

    vec_hits = await store.search_vector(
        collection, query_vector, filters, retrieve_k, min_score=min_score
    )
    fts_hits = await store.search_fts(
        collection, query_text, filters, retrieve_k, min_score=min_score, raw=raw
    )

    if mode == "rrf":
        fused = fuse_rrf([vec_hits, fts_hits])
        metric = Metric.RRF
    else:
        fused = fuse_weighted([vec_hits, fts_hits], weights=weights)
        metric = Metric.WEIGHTED

    total_results = len({hit.chunk_id for hit in vec_hits} | {hit.chunk_id for hit in fts_hits})

    if reranker is not None and fused:
        documents = [hit.content for hit in fused]
        hits = await reranker.rerank(
            query_text,
            documents,
            top_k=rerank_top_k if rerank_top_k is not None else top_k,
            instruction=instruction,
        )
        reordered: list[ScoredDocument] = []
        for hit in hits:
            if 0 <= hit.index < len(fused):  # defensive: provider contract says valid
                reordered.append(
                    _with_metric(fused[hit.index], score=hit.score, metric=Metric.RERANK)
                )
        fused = reordered
        metric = Metric.RERANK

    return SearchResult(
        results=fused[:top_k],
        total_results=total_results,
        metric=metric,
        mode=mode,
        query=query_text,
        collection=collection,
    )


def fuse_rrf(
    result_lists: Sequence[Sequence[ScoredDocument]],
    k: int = RRF_DEFAULT_K,
) -> list[ScoredDocument]:
    """Reciprocal-rank fusion over ranked result lists, keyed by chunk_id.

    score(chunk) = sum over lists of 1 / (k + rank_in_list).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores: dict[str, float] = {}
    docs: dict[str, ScoredDocument] = {}
    for ranked in result_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        _with_metric(docs[chunk_id], score=score, metric=Metric.RRF) for chunk_id, score in ordered
    ]


def fuse_weighted(
    result_lists: Sequence[Sequence[ScoredDocument]],
    weights: Sequence[float] | None = None,
) -> list[ScoredDocument]:
    """Min-max normalize each list to [0, 1], then blend with `weights`.

    Weights default to equal; len(weights) must equal len(result_lists).
    Each list's scores are normalized independently, so mixed metrics (e.g.
    cosine + BM25) become comparable before blending.
    """
    lists = [list(ranked) for ranked in result_lists]
    if not lists:
        return []
    if weights is None:
        weights = (1.0,) * len(lists)
    elif len(weights) != len(lists):
        raise ValueError(f"expected {len(lists)} weights, got {len(weights)}")

    normalized: list[list[tuple[float, ScoredDocument]]] = []
    for ranked in lists:
        if not ranked:
            normalized.append([])
            continue
        lo = min(hit.score for hit in ranked)
        hi = max(hit.score for hit in ranked)
        span = hi - lo
        if span == 0.0:
            # no signal in this list; neutral 0.5 keeps it from dominating
            normalized.append([(0.5, hit) for hit in ranked])
        else:
            normalized.append([((hit.score - lo) / span, hit) for hit in ranked])

    acc: dict[str, float] = {}
    docs: dict[str, ScoredDocument] = {}
    for weight, norm_ranked in zip(weights, normalized, strict=True):
        for norm_score, norm_hit in norm_ranked:
            acc[norm_hit.chunk_id] = acc.get(norm_hit.chunk_id, 0.0) + weight * norm_score
            docs.setdefault(norm_hit.chunk_id, norm_hit)
    ordered = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
    return [
        _with_metric(docs[chunk_id], score=score, metric=Metric.WEIGHTED)
        for chunk_id, score in ordered
    ]


def _with_metric(hit: ScoredDocument, *, score: float, metric: Metric) -> ScoredDocument:
    from dataclasses import replace

    return replace(hit, score=score, metric=metric)
