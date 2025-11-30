# tests/conftest.py
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Generator


import numpy as np
import pytest

# Ensure the local ``src`` tree is importable as a package when running tests
# without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embeddify.config import EmbedderConfig, RuntimeConfig
from embeddify.embedder import Embedder


@pytest.fixture(autouse=True)
def _install_dummy_sentence_transformers() -> Generator[None, None, None]:
    """Ensure a minimal ``sentence_transformers`` module is available for tests.

    The real dependency may not be installed in the test environment. To keep
    unit tests fast and hermetic, we provide a lightweight stand-in that
    mimics the small portion of the API used by :class:`Embedder`.

    If a real ``sentence_transformers`` module is already present it is left
    untouched.
    """
    if "sentence_transformers" not in sys.modules:
        module = types.ModuleType("sentence_transformers")

        class DummySentenceTransformer:
            """Very small stand-in for :class:`SentenceTransformer`.

            The dummy records the supplied arguments so tests can assert on
            them, but it does not perform any real model loading.
            """

            def __init__(
                self,
                model_name_or_path: str,
                device: str | None = None,
                trust_remote_code: bool | None = None,
            ) -> None:
                self.model_name_or_path = model_name_or_path
                self.device = device or "cpu"
                self.trust_remote_code = bool(trust_remote_code)
            def encode(
                self,
                sentences,
                batch_size: int = 32,
                show_progress_bar: bool = False,
                normalize_embeddings: bool = True,
                convert_to_numpy: bool = False,
                **_: object,
            ):
                """Return simple deterministic vectors for the provided sentences.

                The exact numeric values are unimportant; tests assert on shape and
                basic properties rather than specific magnitudes.
                """
                if isinstance(sentences, str):
                    texts = [sentences]
                else:
                    texts = list(sentences)

                vectors: list[object] = []
                dim = self.get_sentence_embedding_dimension()
                for index, _text in enumerate(texts):
                    base = float(index)
                    data = [base + float(i) for i in range(dim)]
                    if convert_to_numpy:
                        vectors.append(np.array(data, dtype=float))
                    else:
                        vectors.append(data)
                return vectors

            def get_sentence_embedding_dimension(self) -> int:
                """Return the dimensionality of embeddings produced by this dummy model."""
                return 4

        module.SentenceTransformer = DummySentenceTransformer  # type: ignore[attr-defined]
        sys.modules["sentence_transformers"] = module

    yield


@pytest.fixture
def mock_model_path(tmp_path: Path) -> Path:
    """Return a temporary path that represents a local model directory."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def embedder_config(mock_model_path: Path) -> EmbedderConfig:
    """Provide a minimal, valid :class:`EmbedderConfig` for tests."""
    return EmbedderConfig(model_path=str(mock_model_path))


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    """Provide a default :class:`RuntimeConfig` instance for tests."""
    return RuntimeConfig()


@pytest.fixture
def embedder(embedder_config: EmbedderConfig, runtime_config: RuntimeConfig) -> Embedder:
    """Return an initialised :class:`Embedder` using the dummy model."""
    return Embedder(config=embedder_config, runtime_config=runtime_config)