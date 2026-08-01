"""Chunk-budget plumbing tests — the cross-package contract (plan §3, §13).

CONCEPT §4.3: chonkai owns the mechanics (ChunkBudget validation), embeddy
owns the policy (chunk budget = model context - headroom). The provider's
`context_length` comes from the REGISTRY, so the budget can never drift
from the model facts — the plan §13 risk ("cross-package invariant drift —
chunk size vs context") is the contract this file proves.
"""

from __future__ import annotations

import pytest

from chonkai import ChunkBudget
from embeddy import DEFAULT_HEADROOM_TOKENS, chunk_budget
from embeddy.providers.fake import FakeProvider
from embeddy.providers.local import LocalProvider


def test_default_headroom() -> None:
    budget = chunk_budget(32768)
    assert isinstance(budget, ChunkBudget)
    assert budget.max_tokens == 32768 - DEFAULT_HEADROOM_TOKENS
    assert budget.max_tokens == 32256


def test_custom_headroom() -> None:
    assert chunk_budget(4096, headroom=64).max_tokens == 4032


def test_zero_headroom() -> None:
    assert chunk_budget(512, headroom=0).max_tokens == 512


def test_negative_headroom_raises() -> None:
    with pytest.raises(ValueError, match="headroom"):
        chunk_budget(512, headroom=-1)


def test_context_smaller_than_headroom_raises() -> None:
    with pytest.raises(ValueError, match="no room"):
        chunk_budget(100, headroom=512)


def test_provider_context_drives_budget() -> None:
    """Cross-package contract: provider.context_length (registry fact) ->
    ChunkBudget (chonkai model). Any drift between the provider's registry
    context and the chunk budget would break this test."""
    provider = FakeProvider()
    budget = chunk_budget(provider.context_length)
    assert budget.max_tokens == provider.context_length - DEFAULT_HEADROOM_TOKENS
    # ChunkBudget's own invariants hold (chonkai validates the mechanics)
    assert budget.min_tokens == 0
    assert 1 <= budget.max_tokens <= provider.context_length


def test_registry_provider_context_drives_budget() -> None:
    """The real registry path: LocalProvider exposes ModelSpec.context_length
    (32768 for Qwen3) and the budget derives from it."""
    provider = LocalProvider("Qwen/Qwen3-Embedding-0.6B", 256)
    budget = chunk_budget(provider.context_length)
    assert budget.max_tokens == 32256
