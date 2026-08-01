"""EmbeddingProvider protocol — the keystone contract.

Drafted at M1, FROZEN at M4 (IMPLEMENTATION_PLAN §12): rerank, multimodal
and filter paths may still reshape it during Phases 2-4.

Two rules encoded here (CONCEPT §5.1):
  * Role -> instruction resolution happens in the CALLER (registry look-up);
    providers only ever receive a RESOLVED instruction string. This makes the
    H2 bug class (document content embedded with the query instruction) a
    typed, testable contract.
  * `dimension` is the RESOLVED dimension (native or MRL-truncated) — a fact
    from the model registry, never a free config knob (fixes C2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embeddy.protocol.types import EmbedInput, Vector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A model adapter. Caller resolves role -> instruction and passes the
    resolved string; the provider is deliberately dumb about roles."""

    dimension: int  # RESOLVED dimension (native or MRL-truncated)
    context_length: int  # drives the chunk budget (chonkai ChunkBudget)
    model_name: str

    async def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[Vector]:
        """Embed `inputs`, returning unit-norm float32 vectors of
        `self.dimension` dims (assert_unit_vector/assert_dim guards live in
        the pipeline wrapper, not in each adapter)."""
        ...
