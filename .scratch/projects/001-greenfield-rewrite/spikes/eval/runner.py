"""Eval runner: nDCG@k / recall@k over a provider + store (CONCEPT §9.6).

Store-agnostic: `evaluate()` accepts any provider with `encode()` and any
store with `add()` / `search_vector()` (duck-typed against the M1 protocols).
This becomes the M4 regression gate (plan §7/§12): model, chunker, or fusion
changes must not regress mean nDCG@10 / recall@10 below the thresholds set
in test_eval.py.
"""

from __future__ import annotations

import math

from core_types import StoredChunk


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


async def evaluate(provider, store, *, collection: str, corpus, queries,
                   qrels, k: int = 10) -> dict:
    """Index `corpus` with `provider`, run `queries`, return metrics.

    Returns {"per_query": {qid: {...}}, "mean_ndcg": float,
             "mean_recall": float} where mean values are macro-averaged
    over the 8 queries.
    """
    # one chunk per doc, one vector per chunk — using the real typed record
    # (StoredChunk from core_types) so the harness exercises the M1 types.
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
    vectors = provider.encode(list(corpus.values()))
    await store.add(collection, chunks, vectors)

    per_query: dict[str, dict] = {}
    for qid, qtext in queries.items():
        qvec = provider.encode([qtext])[0]
        hits = await store.search_vector(collection, qvec, top_k=k)
        retrieved = [h.chunk_id for h in hits]
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
