# tests/conftest.py
"""Shared fixtures for embeddy test suite.

Phase 1 tests cover models, config, exceptions, and package init.
No embedding model is loaded — that comes in Phase 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the local ``src`` tree is importable as a package when running tests
# without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embeddy.config import (
    ChunkConfig,
    EmbedderConfig,
    EmbeddyConfig,
    PipelineConfig,
    ServerConfig,
    StoreConfig,
)


@pytest.fixture
def embedder_config() -> EmbedderConfig:
    """Provide a minimal valid :class:`EmbedderConfig` for tests."""
    return EmbedderConfig()


@pytest.fixture
def store_config() -> StoreConfig:
    """Provide a default :class:`StoreConfig` for tests."""
    return StoreConfig()


@pytest.fixture
def chunk_config() -> ChunkConfig:
    """Provide a default :class:`ChunkConfig` for tests."""
    return ChunkConfig()


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Provide a default :class:`PipelineConfig` for tests."""
    return PipelineConfig()


@pytest.fixture
def server_config() -> ServerConfig:
    """Provide a default :class:`ServerConfig` for tests."""
    return ServerConfig()


@pytest.fixture
def embeddy_config() -> EmbeddyConfig:
    """Provide a default :class:`EmbeddyConfig` with all sub-configs."""
    return EmbeddyConfig()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def configs_dir(fixtures_dir: Path) -> Path:
    """Return the path to the config fixtures directory."""
    return fixtures_dir / "configs"
