"""HTTPProvider tests — httpx MockTransport (plan §11: contract layer).

Cases: success, 4xx (no retry), 5xx (retry then raise), timeout (retry
then raise), wrong-dimension response, instruction extra_body,
honors_instructions declaration, multimodal rejection, normalization of
non-normalized upstream vectors, malformed responses, batching.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast

import httpx
import numpy as np
import pytest

from embeddy import ModelSpec
from embeddy.errors import HTTPProviderError, ProviderInputError, WrongDimensionError
from embeddy.protocol.embedding import EmbeddingProvider
from embeddy.protocol.types import ImageInput, assert_unit_vector, normalize_l2
from embeddy.providers.http import HTTPProvider, build_embeddings_request

SPEC = ModelSpec(
    id="test/http-model",
    native_dimension=4,
    mrl_range=None,
    context_length=100,
    instructions={},
)


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> tuple[HTTPProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HTTPProvider(
        "test/http-model", 4, base_url="http://upstream", http_client=client, spec=SPEC, **kwargs
    )
    return provider, client


# --- wire shape ----------------------------------------------------------------


def test_build_request_shape() -> None:
    req = build_embeddings_request(
        base_url="http://upstream/", model="m", inputs=["a", "b"], instruction="instr"
    )
    assert req.method == "POST"
    assert req.url.path == "/v1/embeddings"
    body = json.loads(req.content)
    assert body == {"input": ["a", "b"], "model": "m", "instruction": "instr"}
    # no api_key -> no Authorization header
    assert "Authorization" not in req.headers


def test_build_request_api_key_header() -> None:
    req = build_embeddings_request(base_url="http://u", model="m", inputs=["a"], api_key="k")
    assert req.headers["Authorization"] == "Bearer k"


def test_build_request_no_instruction_when_empty() -> None:
    req = build_embeddings_request(base_url="http://u", model="m", inputs=["a"], instruction="")
    body = json.loads(req.content)
    assert "instruction" not in body


# --- success -------------------------------------------------------------------


async def test_success_returns_unit_norm_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test/http-model"
        assert body["input"] == ["hello", "world"]
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [2.0, 0.0, 0.0, 0.0]}, {"embedding": [0.0, 3.0, 0.0, 0.0]}]
            },
        )

    provider, client = _provider(handler)
    try:
        vectors = await provider.encode(["hello", "world"])
        assert len(vectors) == 2
        for v in vectors:
            assert v.shape == (4,)
            assert v.dtype == np.float32
            assert_unit_vector(v)  # non-normalized upstream -> normalized here
        # ranking preserved by normalization: 2.0 vs 3.0 both -> [1,0,0,0]
        assert np.allclose(vectors[0], [1.0, 0, 0, 0])
    finally:
        await client.aclose()


async def test_success_sends_instruction_extra_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["instruction"] == "some-resolved-instruction"
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0, 0, 0]}]})

    provider, client = _provider(handler)
    try:
        await provider.encode(["x"], instruction="some-resolved-instruction")
    finally:
        await client.aclose()


async def test_empty_inputs_no_request() -> None:
    requests_made: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(request)
        return httpx.Response(200, json={"data": []})

    provider, client = _provider(handler)
    try:
        assert await provider.encode([]) == []
        assert requests_made == []
    finally:
        await client.aclose()


async def test_batching_splits_inputs() -> None:
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        batch = json.loads(request.content)["input"]
        batches.append(batch)
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0, 0, 0]} for _ in batch]})

    provider, client = _provider(handler, batch_size=2)
    try:
        vectors = await provider.encode(["a", "b", "c", "d", "e"])
        assert len(vectors) == 5
        assert batches == [["a", "b"], ["c", "d"], ["e"]]
    finally:
        await client.aclose()


# --- errors --------------------------------------------------------------------


async def test_4xx_raises_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad input"}})

    provider, client = _provider(handler)
    try:
        with pytest.raises(HTTPProviderError, match="400.*bad input"):
            await provider.encode(["x"])
        assert calls == 1, "4xx must not be retried"
    finally:
        await client.aclose()


async def test_5xx_retries_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    provider, client = _provider(handler, max_retries=2)
    try:
        with pytest.raises(HTTPProviderError):
            await provider.encode(["x"])
        assert calls == 3, "503 must be retried max_retries times"
    finally:
        await client.aclose()


async def test_timeout_retries_then_raises() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.TimeoutException("timed out")

    provider, client = _provider(handler, max_retries=1)
    try:
        with pytest.raises(HTTPProviderError, match="attempts"):
            await provider.encode(["x"])
        assert calls == 2
    finally:
        await client.aclose()


async def test_retry_then_success() -> None:
    """A transient 503 followed by a healthy response succeeds."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0, 0, 0]}]})

    provider, client = _provider(handler, max_retries=2)
    try:
        vectors = await provider.encode(["x"])
        assert len(vectors) == 1
        assert calls == 2
    finally:
        await client.aclose()


async def test_wrong_dimension_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0, 0.0]}]})  # 3 != 4

    provider, client = _provider(handler)
    try:
        with pytest.raises(WrongDimensionError, match="dimension 3 for resolved dimension 4"):
            await provider.encode(["x"])
    finally:
        await client.aclose()


async def test_mismatched_item_count_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})  # 0 items for 1 input

    provider, client = _provider(handler)
    try:
        with pytest.raises(HTTPProviderError, match="0 items for 1 inputs"):
            await provider.encode(["x"])
    finally:
        await client.aclose()


async def test_malformed_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    provider, client = _provider(handler)
    try:
        with pytest.raises(HTTPProviderError, match="non-JSON"):
            await provider.encode(["x"])
    finally:
        await client.aclose()


async def test_non_json_embedding_field_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": "oops"}]})

    provider, client = _provider(handler)
    try:
        with pytest.raises(HTTPProviderError, match="no list"):
            await provider.encode(["x"])
    finally:
        await client.aclose()


# --- scope declarations --------------------------------------------------------


async def test_provider_declares_instruction_support() -> None:
    provider, client = _provider(
        lambda r: httpx.Response(200, json={"data": []}), honors_instructions=True
    )
    assert provider.honors_instructions is True
    assert provider.supports_instructions is True
    await client.aclose()
    provider2, client2 = _provider(lambda r: httpx.Response(200, json={"data": []}))
    assert provider2.honors_instructions is False
    assert provider2.supports_instructions is False
    await client2.aclose()


async def test_multimodal_rejected() -> None:
    provider, client = _provider(lambda r: httpx.Response(200, json={"data": []}))
    try:
        with pytest.raises(ProviderInputError, match="text-dense"):
            await provider.encode([ImageInput(data=b"x", mime="image/png")])
    finally:
        await client.aclose()


async def test_unsupported_input_rejected() -> None:
    provider, client = _provider(lambda r: httpx.Response(200, json={"data": []}))
    try:
        with pytest.raises(ProviderInputError, match="unsupported"):
            # dynamic list: the point of the test is the invalid input type
            await provider.encode(cast(Any, [123]))
    finally:
        await client.aclose()


# --- protocol conformance ------------------------------------------------------


async def test_conforms_to_embedding_protocol() -> None:
    provider, client = _provider(lambda r: httpx.Response(200, json={"data": []}))
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 4
    assert provider.context_length == 100
    assert provider.model_name == "test/http-model"
    await client.aclose()


def test_non_mrl_wrong_dimension_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="not MRL-capable"):
        HTTPProvider("test/http-model", 8, base_url="http://upstream", spec=SPEC)  # 8 != native 4


async def test_ranking_parity_local_vs_remote_within_tolerance() -> None:
    """Plan §5 acceptance: local and remote providers produce RANKING-
    PARITY within tolerance for the same model+instruction (NOT bitwise
    equality — ST vs a remote TEI never match exactly). Here the remote
    mock returns the local vectors perturbed by float-level noise; the
    cosine ranking of a small corpus must be identical."""
    rng = np.random.default_rng(3)

    class DeterministicLocal:
        """Protocol-shaped local adapter: deterministic unit vectors."""

        dimension = 4
        context_length = 100
        model_name = "test/http-model"

        async def encode(
            self, inputs: list[str], instruction: str | None = None
        ) -> list[np.ndarray]:
            del instruction
            out = []
            for text in inputs:
                seed = int.from_bytes(str(text).encode()[:8].ljust(8, b"0"), "little")
                vec = normalize_l2(np.array([seed % 7, (seed // 7) % 5, 1, 2], dtype=np.float32))
                out.append(vec)
            return out

    corpus = ["alpha bravo", "alpha charlie", "bravo delta", "echo foxtrot", "golf hotel"]
    local = DeterministicLocal()

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        # remote re-embeds the same texts; float-level drift vs the local
        # vectors (the ST-vs-TEI reality), not bitwise equality
        vectors = []
        for text in texts:
            seed = int.from_bytes(text.encode()[:8].ljust(8, b"0"), "little")
            vec = normalize_l2(np.array([seed % 7, (seed // 7) % 5, 1, 2], dtype=np.float32))
            vectors.append(list(vec + rng.normal(0, 1e-4, 4)))
        return httpx.Response(200, json={"data": [{"embedding": v} for v in vectors]})

    provider, client = _provider(handler, honors_instructions=True)
    try:
        query = "alpha"
        instr = "Instruct: retrieve the relevant passages"
        local_vecs = await local.encode(corpus, instr)
        remote_vecs = await provider.encode(cast(Any, corpus), instr)
        assert not np.array_equal(local_vecs, remote_vecs), (
            "ST vs a remote TEI are never bitwise equal"
        )

        local_q = (await local.encode([query], instr))[0]
        remote_q = (await provider.encode([query], instr))[0]

        def rank(q: np.ndarray, vecs: list[np.ndarray]) -> list[str]:
            return [
                corpus[i]
                for i in sorted(range(len(vecs)), key=lambda i: -float(np.dot(q, vecs[i])))
            ]

        assert rank(local_q, local_vecs) == rank(remote_q, remote_vecs), (
            "ranking parity within tolerance: the same model+instruction must "
            "rank identically through local and remote paths"
        )
    finally:
        await client.aclose()


async def test_backoff_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries sleep with exponential backoff (0.5 * 2**attempt)."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.TransportError("boom")

    provider, client = _provider(handler, max_retries=2)
    try:
        with pytest.raises(HTTPProviderError):
            await provider.encode(["x"])
        assert calls == 3
        assert sleeps == [0.5, 1.0]
    finally:
        await client.aclose()
