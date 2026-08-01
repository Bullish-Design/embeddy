"""Eval runner: nDCG@k / recall@k over a provider + store (CONCEPT §9.6).

Store-agnostic: `evaluate()` accepts any provider with `encode()` and any
store with `add()` / `search_vector()` (duck-typed against the frozen
Searchable protocol). This is the M4 regression gate (plan §7/§12): model,
chunker, or fusion changes must not regress mean nDCG@10 / recall@10 below
the thresholds set in eval/test_eval.py.

Ported from the Phase-0 spike (spikes/eval/runner.py); the provider is now
async (the EmbeddingProvider protocol shape) and `evaluate()` accepts an
optional injectable `search` hook so tests can prove the gate DETECTS
retrieval regressions (broken ranking / garbage retrieval).
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

from embeddy.protocol.types import StoredChunk


def _dcg(rels: list[float], k: int) -> float:
    total = 0.0
    for rank, rel in enumerate(rels[:k], start=1):
        total += rel / math.log2(rank + 1)
    return total


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """nDCG@k with binary relevance (rel=1 for relevant, 0 otherwise)."""
    rels = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    dcg = _dcg(rels, k)
    ideal = _dcg([1.0] * min(k, len(relevant)), k)
    return dcg / ideal if ideal > 0.0 else 0.0


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """recall@k = |retrieved[:k] ∩ relevant| / |relevant|."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


SearchFn = Callable[..., Awaitable[list[str]]]


async def _vector_search(
    store,
    collection: str,
    query_text: str,
    query_vector,
    k: int,
) -> list[str]:
    """Default retrieval: top-k by cosine over the store's vector leg."""
    del query_text
    hits = await store.search_vector(collection, query_vector, top_k=k)
    return [h.chunk_id for h in hits]


async def evaluate(
    provider,
    store,
    *,
    collection: str,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    k: int = 10,
    search: SearchFn | None = None,
) -> dict:
    """Index `corpus` with `provider`, run `queries`, return metrics.

    `search` is the per-query retrieval callable, called as
    `await search(store, collection, query_text, query_vector, k) -> list[chunk_id]`.
    Defaults to a plain vector top-k (the spike behavior). Tests inject a
    broken search to prove the gate detects retrieval regressions.

    Returns {"per_query": {qid: {"ndcg", "recall", "top"}}, "mean_ndcg":
    float, "mean_recall": float} where the mean values are macro-averaged
    over the 8 queries.
    """
    # one chunk per doc, one vector per chunk — using the real typed record
    # (embeddy StoredChunk) so the harness exercises the frozen types.
    chunks = [
        StoredChunk(
            id=doc_id,
            collection_id=collection,
            source_id=f"src-{doc_id}",
            content=text,
            chunk_type="paragraph",
            start_line=1,
            end_line=1,
        )
        for doc_id, text in corpus.items()
    ]
    vectors = await provider.encode(list(corpus.values()))
    await store.add(collection, chunks, vectors)

    if search is None:
        search = _vector_search

    per_query: dict[str, dict] = {}
    for qid, qtext in queries.items():
        qvec = (await provider.encode([qtext]))[0]
        retrieved = await search(store, collection, qtext, qvec, k)
        relevant = qrels[qid]
        per_query[qid] = {
            "ndcg": ndcg_at_k(retrieved, relevant, k),
            "recall": recall_at_k(retrieved, relevant, k),
            "top": retrieved[:5],
        }

    return {
        "per_query": per_query,
        "mean_ndcg": sum(v["ndcg"] for v in per_query.values()) / len(per_query),
        "mean_recall": sum(v["recall"] for v in per_query.values()) / len(per_query),
    }
