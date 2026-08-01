"""LocalProvider — sentence-transformers adapter (embeddy[local] extra, LAZY).

CONCEPT §5.1 / plan §3:
  * Load by model id via `get_model()`; `dimension` and `context_length`
    come from the ModelSpec (model facts come from the registry, never
    config — fixes C2). `dimension` is the RESOLVED dimension: native, or
    MRL-truncated via `resolve_dimension`.
  * The caller resolves role -> instruction via the registry and passes a
    RESOLVED string; this provider never sees a role (fixes H2).
  * Prompts: the registry stores CARD-EXACT strings. The provider maps the
    resolved string back to an ST registered prompt name by VALUE
    (`prompt_name=`, the harrier convention: web_search_query/...), and
    falls back to `prompt=<string>` (the Qwen3 convention) when no
    registered prompt matches. Empty string / None means no prompt.
  * MRL: when `mrl_range` is set and the resolved dimension < native, the
    provider post-processes every vector through
    `truncate_and_renormalize` — truncation MUST be followed by L2
    re-normalization (a sliced vector is not unit-norm; cosine assumes unit
    vectors). Verified 2026-08-01: ST's `truncate_dim` alone does NOT
    re-normalize (norm ~0.41 at dim 64), so the registry truncation point is
    load-bearing, not cosmetic.
  * Batching / dtype / device are constructor options; output is always
    float32 unit-norm `Vector` (the protocol invariant).

`sentence_transformers` is imported ONLY inside `load()` — zero-extras
`import embeddy` stays clean (the docling/tokenizers [tool.ty] pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from embeddy.errors import ModelNotLoadedError, ProviderInputError
from embeddy.protocol.types import EmbedInput, ImageInput, Vector
from embeddy.registry import ModelSpec, get_model, resolve_dimension, truncate_and_renormalize


class _STModel(Protocol):
    """The slice of sentence-transformers' SentenceTransformer LocalProvider
    consumes (duck-typed, never imported at runtime — the docling pattern,
    packages/chonkai/src/chonkai/ingest/docling.py)."""

    prompts: dict[str, str]

    def encode(self, inputs: list[object], **kwargs: Any) -> np.ndarray: ...


if TYPE_CHECKING:  # pragma: no cover - import-time type info only
    pass


class LocalProvider:
    """Sentence-transformers adapter. Conforms to the EmbeddingProvider
    protocol (dimension / context_length / model_name / async encode)."""

    def __init__(
        self,
        model_id: str,
        dimension: int | None = None,
        *,
        device: str | None = None,
        batch_size: int = 32,
        model: object | None = None,
        spec: ModelSpec | None = None,
    ) -> None:
        """`model` injects a loaded ST model (tests); `spec` injects model
        facts directly (tests, tiny non-registry models in [slow] tests).
        Production path: model_id -> registry -> ModelSpec."""
        self._spec = spec if spec is not None else get_model(model_id)
        self.model_name = self._spec.id
        self.dimension = resolve_dimension(self._spec, dimension)
        self.context_length = self._spec.context_length
        self._device = device
        self._batch_size = batch_size
        self._model: _STModel | None = cast(_STModel | None, model)

    # ------------------------------------------------------------------ #
    # facts from the registry (never from config / model internals)
    # ------------------------------------------------------------------ #

    @property
    def native_dimension(self) -> int:
        return self._spec.native_dimension

    @property
    def mrl_range(self) -> range | None:
        return self._spec.mrl_range

    @property
    def supports_instructions(self) -> bool:
        """Local ST honors prompts (the prompt conventions are the point of
        the instruction-aware models in the registry)."""
        return True

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #

    def load(self) -> _STModel:
        """Lazy-load the ST model. Raises ModelNotLoadedError with a clear
        message when the `local` extra is not installed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # embeddy[local] not installed
                raise ModelNotLoadedError(
                    "LocalProvider requires the `embeddy[local]` extra "
                    "(sentence-transformers); install it to use local models"
                ) from exc
            self._model = cast(_STModel, SentenceTransformer(self.model_name, device=self._device))
        return self._model

    def close(self) -> None:
        """Release the loaded model reference (the provider lifecycle seam)."""
        self._model = None

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #

    async def encode(
        self,
        inputs: list[EmbedInput],
        instruction: str | None = None,
    ) -> list[Vector]:
        model = self.load()
        st_inputs = [self._to_st_input(item) for item in inputs]
        prompt_kwargs = self._prompt_kwargs(model, instruction)
        if not st_inputs:
            return []
        vectors = model.encode(
            st_inputs,
            batch_size=self._batch_size,
            device=self._device,
            convert_to_numpy=True,
            **prompt_kwargs,
        )
        return [
            truncate_and_renormalize(np.asarray(v, dtype=np.float32), self.dimension)
            for v in vectors
        ]

    @staticmethod
    def _to_st_input(item: EmbedInput) -> object:
        """str -> str; ImageInput -> the ST multimodal dict {"image": RGB
        ndarray}. Image decoding is lazy (PIL, a sentence-transformers
        dependency). Multimodal is local-provider only in v1 (CONCEPT §7)."""
        if isinstance(item, str):
            return item
        if isinstance(item, ImageInput):
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover - ST always brings PIL
                raise ProviderInputError(
                    "PIL is required to decode image inputs for the local "
                    "provider (install `embeddy[local]`)"
                ) from exc
            import io

            try:
                image = Image.open(io.BytesIO(item.data))
                return {"image": np.asarray(image.convert("RGB"), dtype=np.uint8)}
            except Exception as exc:
                raise ProviderInputError(f"cannot decode image input: {exc}") from exc
        raise ProviderInputError(f"unsupported input type: {type(item).__name__!r}")

    @staticmethod
    def _prompt_kwargs(model: _STModel, instruction: str | None) -> dict[str, str]:
        """Map a RESOLVED instruction string to ST prompt kwargs.

        Value-match against the model's registered prompts first (the
        `prompt_name` convention — harrier's web_search_query/..., Qwen3's
        query/document); otherwise pass the string directly (`prompt`
        convention — bge-m3, which registers no ST prompts). Empty/None
        means no prompt (Qwen3/harrier document role).
        """
        if not instruction:
            return {}
        prompts = getattr(model, "prompts", None) or {}
        for name, value in prompts.items():
            if value == instruction:
                return {"prompt_name": name}
        return {"prompt": instruction}
