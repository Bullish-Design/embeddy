# tests/test_embedding.py
"""Tests for the embedding layer (Phase 2).

Covers: EmbedderBackend ABC, LocalBackend, RemoteBackend, Embedder facade.
All tests use mocked backends — no actual model loading or GPU required.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from embeddy.config import EmbedderConfig
from embeddy.exceptions import EncodingError, ModelLoadError
from embeddy.models import EmbedInput, Embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector(dim: int = 2048, seed: int = 42) -> list[float]:
    """Create a deterministic fake embedding vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _make_raw_vectors(n: int, dim: int = 2048) -> list[list[float]]:
    """Create N distinct fake vectors."""
    return [_make_vector(dim, seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# EmbedderBackend ABC
# ---------------------------------------------------------------------------


class TestEmbedderBackendABC:
    """Test that EmbedderBackend defines the required interface."""

    def test_cannot_instantiate_abc(self) -> None:
        from embeddy.embedding.backend import EmbedderBackend

        with pytest.raises(TypeError):
            EmbedderBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_encode(self) -> None:
        from embeddy.embedding.backend import EmbedderBackend

        class IncompleteBackend(EmbedderBackend):
            async def load(self) -> None:
                pass

            async def health(self) -> bool:
                return True

            @property
            def model_name(self) -> str:
                return "test"

            @property
            def dimension(self) -> int:
                return 2048

        with pytest.raises(TypeError):
            IncompleteBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# RemoteBackend
# ---------------------------------------------------------------------------


class TestRemoteBackend:
    """Test the HTTP client backend for remote embedding server."""

    @pytest.fixture
    def remote_config(self) -> EmbedderConfig:
        return EmbedderConfig(
            mode="remote",
            remote_url="http://100.64.0.1:8586",
            remote_timeout=30.0,
            embedding_dimension=2048,
        )

    def test_construction(self, remote_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import RemoteBackend

        backend = RemoteBackend(remote_config)
        assert backend.model_name == remote_config.model_name

    async def test_encode_sends_http_request(self, remote_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import RemoteBackend

        fake_vectors = _make_raw_vectors(2)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": fake_vectors,
            "dimension": 2048,
            "model": "Qwen/Qwen3-VL-Embedding-2B",
        }
        mock_response.raise_for_status = MagicMock()

        backend = RemoteBackend(remote_config)

        with patch.object(backend, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)

            inputs = [
                EmbedInput(text="hello world"),
                EmbedInput(text="test input"),
            ]
            result = await backend.encode(inputs, instruction="test instruction")

            assert len(result) == 2
            assert len(result[0]) == 2048
            mock_client.post.assert_called_once()

    async def test_encode_error_raises_encoding_error(self, remote_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import RemoteBackend

        import httpx

        backend = RemoteBackend(remote_config)

        with patch.object(backend, "_client") as mock_client:
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )

            with pytest.raises(EncodingError):
                await backend.encode([EmbedInput(text="test")])

    async def test_health_check(self, remote_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import RemoteBackend

        backend = RemoteBackend(remote_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        with patch.object(backend, "_client") as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await backend.health()
            assert result is True

    async def test_health_check_failure(self, remote_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import RemoteBackend

        backend = RemoteBackend(remote_config)

        with patch.object(backend, "_client") as mock_client:
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

            result = await backend.health()
            assert result is False

    async def test_connection_error_raises_encoding_error(self, remote_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import RemoteBackend

        import httpx

        backend = RemoteBackend(remote_config)

        with patch.object(backend, "_client") as mock_client:
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

            with pytest.raises(EncodingError, match="[Cc]onnect"):
                await backend.encode([EmbedInput(text="test")])


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class TestLocalBackend:
    """Test the in-process model backend (with mocked model)."""

    @pytest.fixture
    def local_config(self) -> EmbedderConfig:
        return EmbedderConfig(
            mode="local",
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            embedding_dimension=2048,
        )

    def test_construction(self, local_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import LocalBackend

        backend = LocalBackend(local_config)
        assert backend.model_name == local_config.model_name
        assert backend.dimension == 2048

    async def test_load_model(self, local_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import LocalBackend

        backend = LocalBackend(local_config)

        with patch.object(backend, "_load_model_sync") as mock_load:
            mock_load.return_value = None
            await backend.load()
            mock_load.assert_called_once()

    async def test_load_model_failure_raises_model_load_error(self, local_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import LocalBackend

        backend = LocalBackend(local_config)

        with patch.object(backend, "_load_model_sync", side_effect=RuntimeError("CUDA OOM")):
            with pytest.raises(ModelLoadError, match="CUDA OOM"):
                await backend.load()

    async def test_encode_calls_sync_in_thread(self, local_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import LocalBackend

        fake_vectors = _make_raw_vectors(1)
        backend = LocalBackend(local_config)
        backend._model = MagicMock()  # Pretend model is loaded

        with patch.object(backend, "_encode_sync", return_value=fake_vectors):
            result = await backend.encode([EmbedInput(text="test")])
            assert len(result) == 1
            assert len(result[0]) == 2048

    async def test_encode_without_load_raises(self, local_config: EmbedderConfig) -> None:
        from embeddy.embedding.backend import LocalBackend

        backend = LocalBackend(local_config)
        # Model not loaded — _model is None

        with pytest.raises(ModelLoadError, match="[Nn]ot loaded"):
            await backend.encode([EmbedInput(text="test")])


# ---------------------------------------------------------------------------
# Embedder Facade
# ---------------------------------------------------------------------------


class TestEmbedderFacade:
    """Test the public Embedder class that wraps a backend."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend that returns fake embeddings."""
        backend = MagicMock()
        backend.model_name = "Qwen/Qwen3-VL-Embedding-2B"
        backend.dimension = 2048
        backend.encode = AsyncMock(return_value=_make_raw_vectors(1))
        backend.load = AsyncMock()
        backend.health = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def embedder_config(self) -> EmbedderConfig:
        return EmbedderConfig(
            embedding_dimension=2048,
            normalize=True,
            lru_cache_size=16,
        )

    @pytest.fixture
    def embedder(self, mock_backend: MagicMock, embedder_config: EmbedderConfig) -> "Embedder":
        from embeddy.embedding.embedder import Embedder

        emb = Embedder(embedder_config)
        emb._backend = mock_backend
        return emb

    async def test_encode_text(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        result = await embedder.encode("hello world")
        assert len(result) == 1
        assert isinstance(result[0], Embedding)
        assert result[0].dimension == 2048
        assert result[0].model_name == "Qwen/Qwen3-VL-Embedding-2B"

    async def test_encode_embed_input(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        result = await embedder.encode(EmbedInput(text="test"))
        assert len(result) == 1
        assert isinstance(result[0], Embedding)

    async def test_encode_batch(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        mock_backend.encode = AsyncMock(return_value=_make_raw_vectors(3))
        result = await embedder.encode(["one", "two", "three"])
        assert len(result) == 3
        for emb in result:
            assert isinstance(emb, Embedding)
            assert emb.dimension == 2048

    async def test_encode_query(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        result = await embedder.encode_query("what is embeddy?")
        assert isinstance(result, Embedding)
        assert result.dimension == 2048
        # Should have been called with query instruction
        call_args = mock_backend.encode.call_args
        assert call_args is not None
        # instruction kwarg should contain the query instruction
        _, kwargs = call_args
        assert "instruction" in kwargs
        assert kwargs["instruction"] == "Retrieve relevant documents, images, or text for the user's query."

    async def test_encode_document(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        result = await embedder.encode_document("This is a document about Python.")
        assert isinstance(result, Embedding)
        call_args = mock_backend.encode.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["instruction"] == "Represent the user's input."

    async def test_encode_with_custom_instruction(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        result = await embedder.encode("test", instruction="Custom instruction")
        assert len(result) == 1
        call_args = mock_backend.encode.call_args
        _, kwargs = call_args
        assert kwargs["instruction"] == "Custom instruction"

    async def test_normalization(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        """Embedder should L2-normalize vectors when config.normalize=True."""
        raw = [[1.0, 2.0, 3.0] + [0.0] * 2045]
        mock_backend.encode = AsyncMock(return_value=raw)
        mock_backend.dimension = 2048

        result = await embedder.encode("test")
        vec = result[0].to_list()
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    async def test_mrl_truncation(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        """When embedding_dimension < model dimension, vectors are truncated."""
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(embedding_dimension=512, normalize=True, lru_cache_size=0)
        emb = Embedder(config)
        emb._backend = mock_backend
        mock_backend.dimension = 2048
        mock_backend.encode = AsyncMock(return_value=_make_raw_vectors(1, dim=2048))

        result = await emb.encode("test")
        assert result[0].dimension == 512

    async def test_no_truncation_when_full_dimension(self, embedder: "Embedder", mock_backend: MagicMock) -> None:
        """When embedding_dimension == model dimension, no truncation."""
        result = await embedder.encode("test")
        assert result[0].dimension == 2048


class TestEmbedderLRUCache:
    """Test the in-memory LRU cache on the Embedder facade."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        backend = MagicMock()
        backend.model_name = "test-model"
        backend.dimension = 2048
        backend.encode = AsyncMock(return_value=_make_raw_vectors(1))
        backend.load = AsyncMock()
        backend.health = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def cached_embedder(self, mock_backend: MagicMock) -> "Embedder":
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(lru_cache_size=4, normalize=False, embedding_dimension=2048)
        emb = Embedder(config)
        emb._backend = mock_backend
        return emb

    async def test_cache_hit(self, cached_embedder: "Embedder", mock_backend: MagicMock) -> None:
        """Second call with same input should not hit backend."""
        await cached_embedder.encode("hello")
        await cached_embedder.encode("hello")

        assert mock_backend.encode.call_count == 1

    async def test_cache_miss_different_inputs(self, cached_embedder: "Embedder", mock_backend: MagicMock) -> None:
        """Different inputs should each hit the backend."""
        mock_backend.encode = AsyncMock(
            side_effect=[
                _make_raw_vectors(1, dim=2048),
                _make_raw_vectors(1, dim=2048),
            ]
        )
        await cached_embedder.encode("hello")
        await cached_embedder.encode("world")

        assert mock_backend.encode.call_count == 2

    async def test_cache_miss_different_instructions(
        self, cached_embedder: "Embedder", mock_backend: MagicMock
    ) -> None:
        """Same input with different instruction should be a cache miss."""
        mock_backend.encode = AsyncMock(
            side_effect=[
                _make_raw_vectors(1, dim=2048),
                _make_raw_vectors(1, dim=2048),
            ]
        )
        await cached_embedder.encode("hello", instruction="A")
        await cached_embedder.encode("hello", instruction="B")

        assert mock_backend.encode.call_count == 2

    async def test_cache_eviction(self, cached_embedder: "Embedder", mock_backend: MagicMock) -> None:
        """Cache should evict least-recently-used when full (size=4)."""
        # Use a single mock with side_effect to return distinct vectors
        mock_backend.encode = AsyncMock(side_effect=[_make_raw_vectors(1, dim=2048) for _ in range(6)])

        # Fill cache with 4 entries
        for i in range(4):
            await cached_embedder.encode(f"input-{i}")

        assert mock_backend.encode.call_count == 4

        # Add a 5th — should evict "input-0"
        await cached_embedder.encode("input-4")
        assert mock_backend.encode.call_count == 5

        # "input-0" should now be a cache miss (was evicted)
        await cached_embedder.encode("input-0")
        assert mock_backend.encode.call_count == 6  # Had to re-encode

    async def test_cache_disabled_when_size_zero(self, mock_backend: MagicMock) -> None:
        """When lru_cache_size=0, caching is disabled."""
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(lru_cache_size=0, normalize=False, embedding_dimension=2048)
        emb = Embedder(config)
        emb._backend = mock_backend

        await emb.encode("hello")
        await emb.encode("hello")

        assert mock_backend.encode.call_count == 2

    async def test_batch_inputs_not_cached(self, cached_embedder: "Embedder", mock_backend: MagicMock) -> None:
        """Batch inputs (list of >1) should not use the single-item cache."""
        mock_backend.encode = AsyncMock(return_value=_make_raw_vectors(2, dim=2048))
        await cached_embedder.encode(["a", "b"])
        await cached_embedder.encode(["a", "b"])

        # Both calls should hit the backend since batch bypass cache
        assert mock_backend.encode.call_count == 2


class TestEmbedderFactory:
    """Test that the Embedder creates the correct backend based on config."""

    async def test_remote_mode_creates_remote_backend(self) -> None:
        from embeddy.embedding.backend import RemoteBackend
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(
            mode="remote",
            remote_url="http://100.64.0.1:8586",
        )
        embedder = Embedder(config)
        assert isinstance(embedder._backend, RemoteBackend)

    async def test_local_mode_creates_local_backend(self) -> None:
        from embeddy.embedding.backend import LocalBackend
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(mode="local")
        embedder = Embedder(config)
        assert isinstance(embedder._backend, LocalBackend)

    def test_dimension_property(self) -> None:
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(embedding_dimension=512)
        embedder = Embedder(config)
        assert embedder.dimension == 512

    def test_model_name_property(self) -> None:
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(model_name="custom/model")
        embedder = Embedder(config)
        assert embedder.model_name == "custom/model"


class TestEmbedderErrorWrapping:
    """Test that backend errors are properly wrapped in embeddy exceptions."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        backend = MagicMock()
        backend.model_name = "test-model"
        backend.dimension = 2048
        backend.load = AsyncMock()
        backend.health = AsyncMock(return_value=True)
        return backend

    async def test_encoding_error_wrapping(self, mock_backend: MagicMock) -> None:
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(lru_cache_size=0, embedding_dimension=2048)
        emb = Embedder(config)
        emb._backend = mock_backend
        mock_backend.encode = AsyncMock(side_effect=RuntimeError("CUDA error"))

        with pytest.raises(EncodingError):
            await emb.encode("test")

    async def test_empty_input_raises_encoding_error(self) -> None:
        from embeddy.embedding.embedder import Embedder

        config = EmbedderConfig(lru_cache_size=0, embedding_dimension=2048)
        emb = Embedder(config)
        emb._backend = MagicMock()

        with pytest.raises(EncodingError, match="[Ee]mpty"):
            await emb.encode([])
