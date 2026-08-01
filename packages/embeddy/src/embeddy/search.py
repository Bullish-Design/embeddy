"""Search fusion — pure functions over typed ranked results (CONCEPT §5.5).

RRF (k=60) is the default; weighted (min-max normalized) is the alternative.
Both are pure: no I/O, independently testable, metric-agnostic in input (RRF
uses only ranks; weighted min-max-normalizes each list before blending).
Output carries its own metric (RRF / WEIGHTED) so scores are never compared
across metrics (CONCEPT §3.4).
"""

from __future__ import annotations

from collections.abc import Sequence

from embeddy.protocol.types import Metric, ScoredDocument

RRF_DEFAULT_K = 60


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
