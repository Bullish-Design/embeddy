"""[slow] integration tests — real sentence-transformers model (plan §5/§11).

Skipped when `embeddy[local]` is not installed (`pytest.importorskip`), so
the default suite stays fast and offline; CI runs these opt-in with
`-m slow`. The absolute retrieval-quality gate for real models lives here
(CONCEPT §9.6): the fake-provider eval gate is mechanical.

Facts verified 2026-08-01 against sentence-transformers 5.6.1: default
encode returns unit-norm float32 vectors; `truncate_dim` alone does NOT
re-normalize (norm ~0.41 at dim 64); `prompt_name`/`prompt` are applied by
prefixing the input.
"""

from __future__ import annotations

import numpy as np
import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")  # embeddy[local]

from embeddy import ModelSpec, assert_unit_vector  # noqa: E402
from embeddy.providers.local import LocalProvider  # noqa: E402

MINILM_SPEC = ModelSpec(
    id="sentence-transformers/all-MiniLM-L6-v2",
    native_dimension=384,
    mrl_range=None,  # MiniLM has no MRL; the MRL test below injects facts
    context_length=256,
    instructions={"query": "", "document": ""},
    license="Apache-2.0",
)

pytestmark = pytest.mark.slow


def test_local_provider_end_to_end() -> None:
    """Tiny ST model end-to-end: load by id, encode text, unit-norm float32
    vectors of the registry dimension, protocol-conformant."""
    provider = LocalProvider(
        MINILM_SPEC.id,
        None,
        model=None,  # real load (no injection)
        spec=MINILM_SPEC,
    )
    assert provider.dimension == 384
    assert provider.context_length == 256
    vectors = None

    async def run() -> None:
        nonlocal vectors
        vectors = await provider.encode(
            ["the quick brown fox jumps", "embedding vectors for retrieval"]
        )

    import asyncio

    asyncio.run(run())
    assert vectors is not None and len(vectors) == 2
    for v in vectors:
        assert v.shape == (384,)
        assert v.dtype == np.float32
        assert_unit_vector(v)


def test_mrl_truncation_consistent_ranking_across_dims() -> None:
    """MRL policy over a real model: truncating to 64 dims (via
    truncate_and_renormalize) yields correct dims, unit norms, and ranking
    agreement with the native dimension on well-separated clusters."""
    spec = ModelSpec(
        id="sentence-transformers/all-MiniLM-L6-v2",
        native_dimension=384,
        mrl_range=range(32, 385),
        context_length=256,
        instructions={},
        license="Apache-2.0",
    )
    docs = [
        "authentication issues short-lived access tokens for the api gateway",
        "token refresh rotates keys and revokes expired sessions",
        "billing records usage in five minute buckets and invoices monthly",
        "refunds are processed within three business days",
        "the search index stores embeddings in sqlite",
        "hybrid search combines dense vectors with full text ranking",
    ]
    query = "how does the authentication service issue api access tokens"

    async def run() -> tuple[list, list]:
        provider_native = LocalProvider(spec.id, None, spec=spec)
        provider_64 = LocalProvider(spec.id, 64, spec=spec)
        native = await provider_native.encode(docs + [query])
        cut = await provider_64.encode(docs + [query])
        return native, cut

    import asyncio

    native, cut = asyncio.run(run())
    q_native, q_cut = native[-1], cut[-1]
    doc_native, doc_cut = native[:-1], cut[:-1]

    for v in cut:
        assert v.shape == (64,), "MRL slice must have the resolved dimension"
        assert_unit_vector(v)

    def rank(q: np.ndarray, ds: list[np.ndarray]) -> list[int]:
        return sorted(range(len(ds)), key=lambda i: -float(np.dot(q, ds[i])))

    full = rank(q_native, doc_native)
    truncated = rank(q_cut, doc_cut)
    # both dims surface the authentication/token cluster first (docs 0-1)
    assert full[0] in (0, 1)
    assert truncated[0] in (0, 1)
    assert set(full[:2]) == set(truncated[:2]), "top-2 sets agree across dims"
