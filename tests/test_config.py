# tests/test_config.py
"""Tests for all embeddy configuration classes (Phase 1).

Covers: EmbedderConfig, StoreConfig, ChunkConfig, PipelineConfig,
ServerConfig, EmbeddyConfig, load_config_file, EmbedderConfig.from_env.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from embeddy.config import (
    ChunkConfig,
    EmbedderConfig,
    EmbeddyConfig,
    PipelineConfig,
    ServerConfig,
    StoreConfig,
    load_config_file,
)
from embeddy.exceptions import ValidationError as EmbeddyValidationError


# ---------------------------------------------------------------------------
# EmbedderConfig
# ---------------------------------------------------------------------------


class TestEmbedderConfigValidation:
    def test_defaults(self) -> None:
        config = EmbedderConfig()
        assert config.model_name == "Qwen/Qwen3-VL-Embedding-2B"
        assert config.device is None
        assert config.torch_dtype == "bfloat16"
        assert config.embedding_dimension == 2048
        assert config.max_length == 8192
        assert config.batch_size == 8
        assert config.normalize is True
        assert config.trust_remote_code is True
        assert config.lru_cache_size == 1024

    def test_custom_values(self) -> None:
        config = EmbedderConfig(
            model_name="custom/model",
            device="cpu",
            torch_dtype="float32",
            embedding_dimension=512,
            max_length=4096,
            batch_size=16,
            normalize=False,
            trust_remote_code=False,
            lru_cache_size=0,
        )
        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.torch_dtype == "float32"
        assert config.embedding_dimension == 512
        assert config.normalize is False

    def test_empty_model_name_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_name="")

    def test_whitespace_model_name_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(model_name="   ")

    def test_invalid_torch_dtype_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(torch_dtype="int8")

    def test_invalid_attn_implementation_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(attn_implementation="invalid")

    def test_valid_attn_implementations(self) -> None:
        for impl in ["flash_attention_2", "sdpa", "eager"]:
            config = EmbedderConfig(attn_implementation=impl)
            assert config.attn_implementation == impl

    def test_none_attn_implementation(self) -> None:
        config = EmbedderConfig(attn_implementation=None)
        assert config.attn_implementation is None

    def test_embedding_dimension_too_high_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(embedding_dimension=4096)

    def test_embedding_dimension_zero_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(embedding_dimension=0)

    def test_batch_size_zero_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(batch_size=0)

    def test_max_length_zero_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(max_length=0)

    def test_negative_lru_cache_size_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            EmbedderConfig(lru_cache_size=-1)


class TestEmbedderConfigFromEnv:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no env vars are set, from_env should return default config."""
        # Clear any potentially set env vars
        for key in [
            "EMBEDDY_MODEL_NAME",
            "EMBEDDY_DEVICE",
            "EMBEDDY_TORCH_DTYPE",
            "EMBEDDY_EMBEDDING_DIMENSION",
            "EMBEDDY_MAX_LENGTH",
            "EMBEDDY_BATCH_SIZE",
            "EMBEDDY_NORMALIZE",
            "EMBEDDY_CACHE_DIR",
            "EMBEDDY_TRUST_REMOTE_CODE",
            "EMBEDDY_LRU_CACHE_SIZE",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = EmbedderConfig.from_env()
        assert config.model_name == "Qwen/Qwen3-VL-Embedding-2B"
        assert config.batch_size == 8

    def test_from_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDY_MODEL_NAME", "custom/model")
        monkeypatch.setenv("EMBEDDY_DEVICE", "cuda")
        monkeypatch.setenv("EMBEDDY_TORCH_DTYPE", "float16")
        monkeypatch.setenv("EMBEDDY_EMBEDDING_DIMENSION", "512")
        monkeypatch.setenv("EMBEDDY_MAX_LENGTH", "4096")
        monkeypatch.setenv("EMBEDDY_BATCH_SIZE", "32")
        monkeypatch.setenv("EMBEDDY_NORMALIZE", "false")
        monkeypatch.setenv("EMBEDDY_TRUST_REMOTE_CODE", "0")
        monkeypatch.setenv("EMBEDDY_LRU_CACHE_SIZE", "256")

        config = EmbedderConfig.from_env()

        assert config.model_name == "custom/model"
        assert config.device == "cuda"
        assert config.torch_dtype == "float16"
        assert config.embedding_dimension == 512
        assert config.max_length == 4096
        assert config.batch_size == 32
        assert config.normalize is False
        assert config.trust_remote_code is False
        assert config.lru_cache_size == 256

    def test_from_env_invalid_boolean_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDY_NORMALIZE", "not-a-bool")
        with pytest.raises(EmbeddyValidationError):
            EmbedderConfig.from_env()

    def test_from_env_invalid_int_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDY_EMBEDDING_DIMENSION", "abc")
        with pytest.raises(EmbeddyValidationError):
            EmbedderConfig.from_env()

    def test_from_env_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDY_CACHE_DIR", "/tmp/models")
        config = EmbedderConfig.from_env()
        assert config.cache_dir == "/tmp/models"


# ---------------------------------------------------------------------------
# StoreConfig
# ---------------------------------------------------------------------------


class TestStoreConfigValidation:
    def test_defaults(self) -> None:
        config = StoreConfig()
        assert config.db_path == "embeddy.db"
        assert config.wal_mode is True

    def test_custom_values(self) -> None:
        config = StoreConfig(db_path="/data/my.db", wal_mode=False)
        assert config.db_path == "/data/my.db"
        assert config.wal_mode is False

    def test_empty_db_path_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            StoreConfig(db_path="")


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------


class TestChunkConfigValidation:
    def test_defaults(self) -> None:
        config = ChunkConfig()
        assert config.strategy == "auto"
        assert config.max_tokens == 512
        assert config.overlap_tokens == 64
        assert config.merge_short is True
        assert config.min_tokens == 64

    def test_valid_strategies(self) -> None:
        for strategy in ["auto", "python", "markdown", "paragraph", "token_window", "docling"]:
            config = ChunkConfig(strategy=strategy)
            assert config.strategy == strategy

    def test_invalid_strategy_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ChunkConfig(strategy="unknown")

    def test_max_tokens_zero_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ChunkConfig(max_tokens=0)

    def test_negative_overlap_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ChunkConfig(overlap_tokens=-1)

    def test_overlap_exceeds_max_tokens_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ChunkConfig(max_tokens=100, overlap_tokens=100)

    def test_invalid_python_granularity_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ChunkConfig(python_granularity="statement")

    def test_markdown_heading_level_out_of_range(self) -> None:
        with pytest.raises(PydanticValidationError):
            ChunkConfig(markdown_heading_level=0)
        with pytest.raises(PydanticValidationError):
            ChunkConfig(markdown_heading_level=7)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfigValidation:
    def test_defaults(self) -> None:
        config = PipelineConfig()
        assert config.collection == "default"
        assert config.concurrency == 4
        assert isinstance(config.exclude_patterns, list)

    def test_empty_collection_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            PipelineConfig(collection="")

    def test_zero_concurrency_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            PipelineConfig(concurrency=0)

    def test_custom_patterns(self) -> None:
        config = PipelineConfig(
            include_patterns=["*.py", "*.md"],
            exclude_patterns=["__pycache__"],
        )
        assert config.include_patterns == ["*.py", "*.md"]
        assert config.exclude_patterns == ["__pycache__"]


# ---------------------------------------------------------------------------
# ServerConfig
# ---------------------------------------------------------------------------


class TestServerConfigValidation:
    def test_defaults(self) -> None:
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8585
        assert config.workers == 1
        assert config.log_level == "info"

    def test_invalid_port_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ServerConfig(port=0)
        with pytest.raises(PydanticValidationError):
            ServerConfig(port=70000)

    def test_zero_workers_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ServerConfig(workers=0)

    def test_invalid_log_level_raises_validation_error(self) -> None:
        with pytest.raises(PydanticValidationError):
            ServerConfig(log_level="verbose")

    def test_log_level_case_insensitive(self) -> None:
        config = ServerConfig(log_level="DEBUG")
        assert config.log_level == "debug"


# ---------------------------------------------------------------------------
# EmbeddyConfig (top-level)
# ---------------------------------------------------------------------------


class TestEmbeddyConfig:
    def test_defaults(self) -> None:
        config = EmbeddyConfig()
        assert isinstance(config.embedder, EmbedderConfig)
        assert isinstance(config.store, StoreConfig)
        assert isinstance(config.chunk, ChunkConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.server, ServerConfig)

    def test_nested_override(self) -> None:
        config = EmbeddyConfig(
            embedder=EmbedderConfig(model_name="custom/model"),
            store=StoreConfig(db_path="/custom.db"),
        )
        assert config.embedder.model_name == "custom/model"
        assert config.store.db_path == "/custom.db"


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestConfigFileLoading:
    def _fixture_path(self, filename: str) -> Path:
        return Path(__file__).parent / "fixtures" / "configs" / filename

    def test_load_yaml_config(self) -> None:
        config = load_config_file(str(self._fixture_path("valid.yaml")))

        assert isinstance(config, EmbeddyConfig)
        assert config.embedder.model_name == "Qwen/Qwen3-VL-Embedding-2B"
        assert config.embedder.torch_dtype == "float32"
        assert config.embedder.batch_size == 16
        assert config.store.db_path == "test.db"
        assert config.chunk.strategy == "auto"
        assert config.pipeline.collection == "test-collection"
        assert config.server.port == 8585

    def test_load_json_config(self) -> None:
        config = load_config_file(str(self._fixture_path("valid.json")))

        assert isinstance(config, EmbeddyConfig)
        assert config.embedder.model_name == "Qwen/Qwen3-VL-Embedding-2B"
        assert config.embedder.batch_size == 16
        assert config.store.db_path == "test.db"

    def test_missing_config_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config_file(str(self._fixture_path("does-not-exist.yaml")))

    def test_malformed_config_raises_validation_error(self) -> None:
        with pytest.raises(EmbeddyValidationError):
            load_config_file(str(self._fixture_path("invalid.yaml")))

    def test_no_path_and_no_env_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EMBEDDY_CONFIG_PATH", raising=False)
        with pytest.raises(EmbeddyValidationError):
            load_config_file()

    def test_uses_env_config_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path = self._fixture_path("valid.yaml")
        monkeypatch.setenv("EMBEDDY_CONFIG_PATH", str(path))
        config = load_config_file()
        assert config.embedder.model_name == "Qwen/Qwen3-VL-Embedding-2B"

    def test_json_file_with_partial_config(self, tmp_path: Path) -> None:
        """Loading a config with only some sections should use defaults for the rest."""
        partial = {"embedder": {"model_name": "custom/model"}}
        config_path = tmp_path / "partial.json"
        config_path.write_text(json.dumps(partial))

        config = load_config_file(str(config_path))
        assert config.embedder.model_name == "custom/model"
        assert config.store.db_path == "embeddy.db"  # default
        assert config.chunk.strategy == "auto"  # default
