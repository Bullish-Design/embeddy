# tests/test_integration.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from embeddify import (
    Embedder,
    EmbedderConfig,
    RuntimeConfig,
    EmbeddingResult,
    SearchResults,
    load_config_file,
    ModelLoadError,
    EncodingError,
    ValidationError,
    SearchError,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CONFIGS_DIR = FIXTURES_DIR / "configs"


class TestEncodingAndSimilarityIntegration:
    """End-to-end tests for encoding and similarity workflows."""

    def test_full_encoding_pipeline_from_config_file(self) -> None:
        """Configuration file + embedder should produce consistent embeddings.

        This exercises configuration loading, model initialisation and the
        :meth:`Embedder.encode` pipeline together.
        """
        config_path = CONFIGS_DIR / "valid.yaml"
        model_config, runtime_config = load_config_file(str(config_path))

        embedder = Embedder(config=model_config, runtime_config=runtime_config)

        texts = ["alpha", "beta", "gamma"]
        result = embedder.encode(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == len(texts)
        assert result.dimensions > 0
        assert result.model_name == embedder.model_name

        for text, embedding in zip(texts, result.embeddings):
            assert embedding.text == text
            assert embedding.model_name == embedder.model_name
            assert embedding.dimensions == result.dimensions

    def test_similarity_between_encoded_texts(self, embedder: Embedder) -> None:
        """Similarity computed between encoded texts should be coherent.

        This verifies that :meth:`Embedder.encode` and
        :meth:`Embedder.similarity` work together using the real dummy model.
        """
        texts = ["one", "one"]
        result = embedder.encode(texts)

        # Using the same text twice should yield identical embeddings and a
        # cosine similarity of 1.0.
        emb1, emb2 = result.embeddings
        score = embedder.similarity(emb1, emb2, metric="cosine")

        assert score.metric == "cosine"
        assert score.score == pytest.approx(1.0, rel=1e-6)


class TestSearchIntegration:
    """Integration tests for semantic search behaviour."""

    def test_search_with_corpus_and_precomputed_embeddings_consistent(
        self,
        embedder: Embedder,
    ) -> None:
        """Search over text corpus and precomputed embeddings must agree.

        This mirrors the Step 12 contract that search with ``corpus`` and
        search with ``corpus_embeddings`` produce the same ranking and scores
        (ignoring the ``text`` field). The behaviour for the ``text`` field
        itself is also asserted here.
        """
        corpus = ["zero", "one", "two", "three"]
        queries = ["alpha", "beta"]
        top_k = 2

        corpus_embeddings = embedder.encode(corpus)

        corpus_results = embedder.search(
            queries=queries,
            corpus=corpus,
            top_k=top_k,
            score_function="cosine",
        )
        precomputed_results = embedder.search(
            queries=queries,
            corpus_embeddings=corpus_embeddings,
            top_k=top_k,
            score_function="cosine",
        )

        assert isinstance(corpus_results, SearchResults)
        assert isinstance(precomputed_results, SearchResults)

        assert corpus_results.query_texts == queries
        assert precomputed_results.query_texts == queries
        assert corpus_results.num_queries == len(queries)
        assert precomputed_results.num_queries == len(queries)

        for hits_pre, hits_corpus in zip(precomputed_results.results, corpus_results.results):
            assert len(hits_pre) == len(hits_corpus) == top_k

            # Rankings and scores must match exactly.
            for hit_pre, hit_corpus in zip(hits_pre, hits_corpus):
                assert hit_pre.corpus_id == hit_corpus.corpus_id
                assert hit_pre.score == pytest.approx(hit_corpus.score)

                # When using precomputed embeddings, corpus text is not available.
                assert hit_pre.text is None
                assert hit_corpus.text == corpus[hit_corpus.corpus_id]

    def test_search_validation_errors_for_invalid_arguments(self, embedder: Embedder) -> None:
        """Invalid search arguments should raise ValidationError, not SearchError."""
        corpus = ["only document"]

        # top_k must be at least 1.
        with pytest.raises(ValidationError):
            embedder.search(queries=["q"], corpus=corpus, top_k=0)

        # top_k cannot exceed corpus size when the corpus is non-empty.
        with pytest.raises(ValidationError):
            embedder.search(queries=["q"], corpus=corpus, top_k=10)

        # Unsupported score function should be rejected.
        with pytest.raises(ValidationError):
            embedder.search(queries=["q"], corpus=corpus, top_k=1, score_function="euclidean")


class TestCachingIntegration:
    """Integration tests for embedding cache behaviour."""

    def test_encode_uses_cache_and_clear_cache_forces_reencode(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated encodes should reuse cached embeddings until cache is cleared."""
        runtime_config.enable_cache = True
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        # Patch the underlying model's encode method to count how often it is called.
        call_count: dict[str, int] = {"count": 0}
        original_encode = embedder._model.encode  # type: ignore[attr-defined]

        def counting_encode(sentences: list[str], *args: Any, **kwargs: Any) -> Any:
            call_count["count"] += 1
            return original_encode(sentences, *args, **kwargs)

        monkeypatch.setattr(embedder._model, "encode", counting_encode)  # type: ignore[attr-defined]

        texts = ["alpha", "beta", "alpha"]

        first_result = embedder.encode(texts)
        first_call_count = call_count["count"]

        # A second encode with identical texts should hit the cache completely and
        # therefore not call the underlying model again.
        second_result = embedder.encode(texts)
        second_call_count = call_count["count"]

        assert len(embedder._cache) == 2  # type: ignore[attr-defined]
        assert first_result.embeddings[0] is second_result.embeddings[0]
        assert second_call_count == first_call_count

        # Clearing the cache must force the model to re-encode.
        embedder.clear_cache()
        assert embedder._cache == {}  # type: ignore[attr-defined]

        embedder.encode(texts)
        assert call_count["count"] == first_call_count + 1


class TestConfigFileWorkflowIntegration:
    """Integration tests for configuration file and environment overrides."""

    def test_embedder_from_config_file_with_env_overrides(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Environment variables should override values from configuration files."""
        config_path = tmp_path / "embeddify-config.yaml"
        config_path.write_text(
            """model:
  path: "/models/from-config"
  device: "cpu"
  normalize_embeddings: true
  trust_remote_code: false
runtime:
  batch_size: 8
  show_progress_bar: false
  enable_cache: false
  convert_to_numpy: false
""",
            encoding="utf-8",
        )

        monkeypatch.setenv("EMBEDDIFY_CONFIG_PATH", str(config_path))
        monkeypatch.setenv("EMBEDDIFY_MODEL_PATH", "/models/from-env")
        monkeypatch.setenv("EMBEDDIFY_BATCH_SIZE", "4")
        monkeypatch.setenv("EMBEDDIFY_ENABLE_CACHE", "true")

        embedder = Embedder.from_config_file()

        # Model path and batch size should respect environment overrides.
        assert embedder.config.model_path == "/models/from-env"
        assert embedder.runtime_config.batch_size == 4
        assert embedder.runtime_config.enable_cache is True


class TestErrorHandlingIntegration:
    """Integration tests for error flow and exception wrapping."""

    def test_invalid_model_initialisation_raises_model_load_error(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failures during underlying model creation must surface as ModelLoadError."""
        import sentence_transformers  # type: ignore[import-not-found]

        def failing_constructor(*_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", failing_constructor)

        with pytest.raises(ModelLoadError):
            Embedder(config=embedder_config, runtime_config=runtime_config)

    def test_encode_invalid_text_raises_encoding_error(self, embedder: Embedder) -> None:
        """Invalid text inputs to encode should raise EncodingError directly."""
        with pytest.raises(EncodingError):
            embedder.encode(["valid text", ""])

    def test_search_runtime_failure_wrapped_in_search_error(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unexpected failures during search should be wrapped in SearchError.

        This test deliberately causes the underlying model's encode to fail so that
        the :class:`SearchError` wrapping logic in :meth:`Embedder.search` is
        exercised end-to-end.
        """

        def failing_encode(sentences: list[str], *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("unexpected failure")

        # Patch the underlying model's encode method, not the Embedder's encode method,
        # since Embedder is a Pydantic BaseModel that doesn't allow arbitrary attributes.
        monkeypatch.setattr(embedder._model, "encode", failing_encode)  # type: ignore[attr-defined]

        # Pass top_k=1 to avoid validation error before encoding is attempted
        with pytest.raises(SearchError):
            embedder.search(queries=["query"], corpus=["document"], top_k=1)
