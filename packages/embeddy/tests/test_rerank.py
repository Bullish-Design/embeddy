"""Reranker tests — protocol shape (spike port), local CrossEncoder with an
injected fake, remote TEI/Jina wire shape over httpx MockTransport (plan
§11). The wire is TEI/Jina, NOT an OpenAI /v1/rerank standard (CONCEPT §7).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import numpy as np
import pytest

from embeddy import Metric, RerankError
from embeddy.protocol.rerank import RerankerProvider, RerankHit
from embeddy.providers.rerank import CrossEncoderReranker, HTTPReranker


class FakeCrossEncoder:
    """Duck-typed ST CrossEncoder: records pairs, returns raw scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
        del kwargs
        self.calls.append(pairs)
        return np.asarray(self._scores, dtype=np.float32)


# --- protocol shape ------------------------------------------------------------


def test_rerank_hit_validation() -> None:
    hit = RerankHit(index=0, score=1.5)
    assert hit.metric == Metric.RERANK
    with pytest.raises(ValueError, match="index"):
        RerankHit(index=-1, score=1.0)
    with pytest.raises(ValueError, match="finite"):
        RerankHit(index=0, score=float("nan"))


def test_cross_encoder_conforms_to_protocol() -> None:
    r = CrossEncoderReranker("cross/encoder", model=FakeCrossEncoder([1.0]))
    assert isinstance(r, RerankerProvider)
    assert r.model_name == "cross/encoder"


def test_http_reranker_conforms_to_protocol() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    )
    r = HTTPReranker("tei/rerank", base_url="http://upstream", http_client=client)
    assert isinstance(r, RerankerProvider)
    assert r.model_name == "tei/rerank"


# --- local CrossEncoder --------------------------------------------------------


async def test_local_rerank_ranks_descending() -> None:
    model = FakeCrossEncoder([0.3, 0.9, 0.1])
    r = CrossEncoderReranker("cross/encoder", model=model)
    hits = await r.rerank("query", ["doc-a", "doc-b", "doc-c"])
    assert model.calls == [[("query", "doc-a"), ("query", "doc-b"), ("query", "doc-c")]]
    assert [h.index for h in hits] == [1, 0, 2]
    assert hits[0].score == pytest.approx(0.9)  # float32 rounding
    assert hits[1].score == pytest.approx(0.3)
    assert all(h.metric == Metric.RERANK for h in hits)


async def test_local_rerank_top_k() -> None:
    model = FakeCrossEncoder([0.3, 0.9, 0.1])
    r = CrossEncoderReranker("cross/encoder", model=model)
    hits = await r.rerank("query", ["a", "b", "c"], top_k=2)
    assert [h.index for h in hits] == [1, 0]


async def test_local_rerank_empty_documents() -> None:
    r = CrossEncoderReranker("cross/encoder", model=FakeCrossEncoder([]))
    assert await r.rerank("query", []) == []


async def test_local_rerank_ignores_instruction() -> None:
    model = FakeCrossEncoder([1.0, 0.0])
    r = CrossEncoderReranker("cross/encoder", model=model)
    await r.rerank("query", ["a", "b"], instruction="some-resolved-instruction")
    assert model.calls == [[("query", "a"), ("query", "b")]]  # prompt not applied


async def test_cross_encoder_load_without_extra_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embeddy[local] not installed -> RerankError with an actionable
    message (the zero-extras import stays clean)."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    r = CrossEncoderReranker("cross/encoder")  # no injected model
    with pytest.raises(RerankError, match="embeddy\\[local\\]"):
        await r.rerank("query", ["a"])


async def test_http_reranker_close_releases_client() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))
    )
    r = HTTPReranker("tei/rerank", base_url="http://upstream", http_client=client)
    await r.rerank("q", ["a"])
    await r.close()  # no error on a live injected client


# --- remote TEI/Jina wire shape ------------------------------------------------


def _remote(handler: Any, **kwargs: Any) -> tuple[HTTPReranker, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = HTTPReranker("tei/rerank", base_url="http://upstream", http_client=client, **kwargs)
    return r, client


async def test_remote_sends_tei_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/rerank"
        assert body == {"query": "q", "texts": ["a", "b"], "top_n": 1}
        return httpx.Response(200, json={"results": [{"index": 1, "score": 0.9}]})

    r, client = _remote(handler)
    try:
        hits = await r.rerank("q", ["a", "b"], top_k=1)
        assert [(h.index, h.score) for h in hits] == [(1, 0.9)]
    finally:
        await client.aclose()


async def test_remote_no_top_n_when_unset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "top_n" not in body
        return httpx.Response(200, json={"results": [{"index": 0, "score": 1.0}]})

    r, client = _remote(handler)
    try:
        hits = await r.rerank("q", ["a"])
        assert [(h.index, h.score) for h in hits] == [(0, 1.0)]
    finally:
        await client.aclose()


async def test_remote_jina_relevance_score() -> None:
    """Jina names the score field `relevance_score`; both are accepted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})

    r, client = _remote(handler)
    try:
        hits = await r.rerank("q", ["a"])
        assert hits[0].score == pytest.approx(0.7)
    finally:
        await client.aclose()


async def test_remote_custom_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        return httpx.Response(200, json={"results": []})

    r, client = _remote(handler, rerank_path="/v1/rerank")
    try:
        assert await r.rerank("q", ["a"]) == []
    finally:
        await client.aclose()


async def test_remote_4xx_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    r, client = _remote(handler)
    try:
        with pytest.raises(RerankError, match="400"):
            await r.rerank("q", ["a"])
    finally:
        await client.aclose()


async def test_remote_out_of_range_index_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 5, "score": 0.9}]})

    r, client = _remote(handler)
    try:
        with pytest.raises(RerankError, match="out of range"):
            await r.rerank("q", ["a", "b"])
    finally:
        await client.aclose()


async def test_remote_malformed_result_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": "x", "score": 1.0}]})

    r, client = _remote(handler)
    try:
        with pytest.raises(RerankError, match="malformed"):
            await r.rerank("q", ["a"])
    finally:
        await client.aclose()


async def test_remote_empty_documents_no_request() -> None:
    requests_made: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(request)
        return httpx.Response(200, json={"results": []})

    r, client = _remote(handler)
    try:
        assert await r.rerank("q", []) == []
        assert requests_made == []
    finally:
        await client.aclose()


async def test_remote_5xx_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="busy")

    r, client = _remote(handler, max_retries=2)
    try:
        with pytest.raises(RerankError, match="503"):
            await r.rerank("q", ["a"])
        assert calls == 3
    finally:
        await client.aclose()


async def test_remote_transport_error_no_retries() -> None:
    """max_retries=0: a transport error raises immediately."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TransportError("connection refused")

    r, client = _remote(handler, max_retries=0)
    try:
        with pytest.raises(RerankError, match="after 1 attempts"):
            await r.rerank("q", ["a"])
    finally:
        await client.aclose()


async def test_remote_transport_error_retries_then_raises() -> None:
    """A transport error with retries left backs off, then raises after the
    retry budget is exhausted."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.TransportError("connection reset")

    r, client = _remote(handler, max_retries=2)
    try:
        with pytest.raises(RerankError, match="after 3 attempts"):
            await r.rerank("q", ["a"])
        assert calls == 3
    finally:
        await client.aclose()


async def test_remote_non_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>nope</html>")

    r, client = _remote(handler)
    try:
        with pytest.raises(RerankError, match="non-JSON"):
            await r.rerank("q", ["a"])
    finally:
        await client.aclose()


async def test_remote_payload_not_object_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    r, client = _remote(handler)
    try:
        with pytest.raises(RerankError, match="response shape"):
            await r.rerank("q", ["a"])
    finally:
        await client.aclose()


async def test_remote_results_not_a_list_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": "oops"})

    r, client = _remote(handler)
    try:
        with pytest.raises(RerankError, match="no `results` list"):
            await r.rerank("q", ["a"])
    finally:
        await client.aclose()
