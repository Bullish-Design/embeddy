"""Chunk-budget plumbing — the cross-package contract (plan §3, §13).

chonkai owns the mechanics (ChunkBudget validation), embeddy owns the
policy (CONCEPT §4.3): embeddy passes `model.context_length - headroom` as
the chunk budget. The provider's `context_length` comes from the registry
(ModelSpec), so the budget can never drift from the model facts. The plan
§13 risk ("cross-package invariant drift — chunk size vs context") is
covered by the contract test in packages/embeddy/tests/test_budget.py.
"""

from __future__ import annotations

from chonkai import ChunkBudget

DEFAULT_HEADROOM_TOKENS = 512
"""Reserved tokens per chunk for prompt/instruction overhead. The prompt is
appended inside the model context, so the chunk budget must leave room for
it plus tokenizer slack."""


def chunk_budget(
    context_length: int,
    *,
    headroom: int = DEFAULT_HEADROOM_TOKENS,
) -> ChunkBudget:
    """Derive the chunk budget from a model's context length.

    `context_length` comes from the provider (registry ModelSpec), never
    from config. Raises ValueError when the headroom would leave no room
    for content (ChunkBudget requires max_tokens >= 1).
    """
    if headroom < 0:
        raise ValueError(f"headroom must be >= 0, got {headroom}")
    max_tokens = context_length - headroom
    if max_tokens < 1:
        raise ValueError(
            f"context_length {context_length} with headroom {headroom} leaves "
            f"no room for chunk content"
        )
    return ChunkBudget(max_tokens=max_tokens)
