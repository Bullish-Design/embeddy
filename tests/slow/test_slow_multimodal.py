"""[gpu] multimodal smoke test — Qwen3-VL-Embedding-2B text + image (plan §5).

Optional in CI (marker `gpu`): downloads the ~2B multimodal model and
requires CUDA. Skips cleanly without torch/PIL/CUDA so the default suite
stays fast and offline. The multimodal path is local-provider only in v1
(CONCEPT §7): ImageInput is decoded by LocalProvider to ST's {"image": RGB
ndarray} dict shape.

Verified 2026-08-01 (HF API): Qwen3-VL-Embedding-2B ST config is dim 2048,
MRL 64-2048, last-token pooling + Normalize module, single "default" prompt
("Represent the user's input."). The full text+image encode below needs a
GPU machine to run (this repo's CI skips it).
"""

from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")
sentence_transformers = pytest.importorskip("sentence_transformers")  # noqa: F401

if not torch.cuda.is_available():
    # ty's bundled pytest stub types skip() with no params; the real API
    # accepts (reason, allow_module_level)
    pytest.skip(reason="multimodal smoke requires CUDA", allow_module_level=True)  # ty: ignore[unknown-argument]

pytest.importorskip("PIL")

import numpy as np  # noqa: E402

from embeddy import ModelSpec, assert_unit_vector  # noqa: E402
from embeddy.protocol.types import ImageInput  # noqa: E402
from embeddy.providers.local import LocalProvider  # noqa: E402

VL_SPEC = ModelSpec(
    id="Qwen/Qwen3-VL-Embedding-2B",
    native_dimension=2048,
    mrl_range=range(64, 2049),
    context_length=32768,
    instructions={
        "query": "Represent the user's input.",
        "document": "Represent the user's input.",
    },
    license="Apache-2.0",
)

pytestmark = pytest.mark.gpu


def _tiny_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_qwen3_vl_text_and_image() -> None:
    provider = LocalProvider(VL_SPEC.id, 256, spec=VL_SPEC)  # MRL-truncated
    assert provider.dimension == 256

    vectors = None

    async def run() -> None:
        nonlocal vectors
        vectors = await provider.encode(
            [
                "A red square on a white background",
                ImageInput(data=_tiny_png(), mime="image/png"),
            ],
            instruction="Represent the user's input.",
        )

    import asyncio

    asyncio.run(run())
    assert vectors is not None and len(vectors) == 2
    for v in vectors:
        assert v.shape == (256,), "MRL-resolved dimension for the VL model"
        assert v.dtype == np.float32
        assert_unit_vector(v)


def test_qwen3_vl_native_dimension() -> None:
    provider = LocalProvider(VL_SPEC.id, None, spec=VL_SPEC)
    assert provider.dimension == 2048

    vectors = None

    async def run() -> None:
        nonlocal vectors
        vectors = await provider.encode(
            ["a text-only prompt"], instruction="Represent the user's input."
        )

    import asyncio

    asyncio.run(run())
    assert vectors is not None
    assert vectors[0].shape == (2048,)
    assert_unit_vector(vectors[0])
