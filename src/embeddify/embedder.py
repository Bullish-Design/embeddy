# src/embeddify/embedder.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from embeddify.config import EmbedderConfig, RuntimeConfig, load_config_file
from embeddify.exceptions import ModelLoadError
from embeddify.models import Embedding


class Embedder(BaseModel):
    """Main interface for working with SentenceTransformer models.

    At this stage the :class:`Embedder` is responsible only for loading and
    exposing the underlying model instance. Higher level operations such as
    ``encode`` and semantic search are introduced in later steps of the
    implementation plan.

    The class is implemented as a Pydantic model so that configuration is
    validated eagerly and so that consumers benefit from rich type hints.
    """

    config: EmbedderConfig = Field(
        description="Validated configuration used to load the underlying model."
    )
    runtime_config: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime configuration controlling execution behaviour.",
    )

    # Internal state that should not be included in the serialised model.
    _model: Any = PrivateAttr()
    _cache: dict[str, Embedding] = PrivateAttr(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - exercised via tests
        """Load the SentenceTransformer model after Pydantic validation.

        The actual import of :mod:`sentence_transformers` happens lazily here so
        that unit tests can provide a lightweight stand-in implementation and
        so that import errors are surfaced as :class:`ModelLoadError` with
        helpful context.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import failure is rare
            raise ModelLoadError(
                "Failed to import SentenceTransformer; is 'sentence-transformers' installed?"
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.config.model_path,
                device=self.config.device,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load SentenceTransformer model from {self.config.model_path!r}: {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        """Return a human-readable identifier for the loaded model.

        When available this uses the underlying model's ``model_name_or_path``
        attribute, falling back to the configured ``model_path``.
        """
        model = getattr(self, "_model", None)
        name = getattr(model, "model_name_or_path", None)
        if isinstance(name, str) and name:
            return name
        return self.config.model_path

    @property
    def device(self) -> str:
        """Return the compute device the model is associated with.

        The dummy model used in tests exposes a ``device`` attribute; real
        SentenceTransformer instances typically expose ``device`` or
        ``target_device``. If neither is present we fall back to the configured
        device string.
        """
        model = getattr(self, "_model", None)
        if model is None:
            return self.config.device

        for attr in ("device", "target_device"):
            value = getattr(model, attr, None)
            if value is not None:
                return str(value)

        return self.config.device

    @classmethod
    def from_config_file(cls, path: str | None = None) -> Embedder:
        """Construct an :class:`Embedder` from a YAML or JSON config file.

        The heavy lifting is delegated to :func:`load_config_file`, which
        validates the configuration and returns a pair of
        (:class:`EmbedderConfig`, :class:`RuntimeConfig`) instances.
        """
        model_config, runtime_config = load_config_file(path)
        return cls(config=model_config, runtime_config=runtime_config)

    def _model_path_name(self) -> str:
        """Return the final path component of the configured model path.

        This helper is primarily used by tests and logging and is separated
        from :attr:`model_name` so that future steps can change how the model
        name is derived without affecting callers that care specifically about
        the filesystem path.
        """
        return Path(self.config.model_path).name
