# tests/test_embedder.py
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from embeddify import (
    Embedder,
    EmbedderConfig,
    RuntimeConfig,
    Embedding,
    EmbeddingResult,
    SearchResults,
    SimilarityScore,
    EncodingError,
    ValidationError,
)


class TestEmbedderInitialisation:
    """Tests for Embedder construction and model loading."""

    def test_embedder_loads_model_with_valid_config(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Creating an Embedder with valid configuration should succeed."""
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        assert embedder.config is embedder_config
        assert embedder.runtime_config is runtime_config
        assert embedder._model is not None

    def test_model_name_property_uses_underlying_model(self, embedder: Embedder) -> None:
        """The model_name property should reflect the underlying model's identifier."""
        name = embedder.model_name

        assert isinstance(name, str)
        assert name

    def test_device_property_reflects_configured_device(self, embedder: Embedder) -> None:
        """The device property should return the configured device string."""
        device = embedder.device

        assert device == "cpu"

    def test_from_config_file_constructs_embedder(self, tmp_path: Any) -> None:
        """from_config_file should load config and construct an Embedder."""
        config_path = tmp_path / "test-config.yaml"
        config_path.write_text(
            f"""
model:
  path: "{tmp_path / "model"}"
  device: "cpu"
  normalize_embeddings: true
  trust_remote_code: false
runtime:
  batch_size: 16
  show_progress_bar: false
  enable_cache: false
  convert_to_numpy: false
""",
            encoding="utf-8",
        )

        embedder = Embedder.from_config_file(str(config_path))

        assert embedder.config.device == "cpu"
        assert embedder.runtime_config.batch_size == 16

    def test_invalid_model_load_raises_model_load_error(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the underlying model fails to load, a ModelLoadError should be raised."""
        import sentence_transformers

        def broken_init(*_: Any, **__: Any) -> None:
            raise RuntimeError("model file not found")

        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            broken_init,
        )

        from embeddify.exceptions import ModelLoadError

        with pytest.raises(ModelLoadError):
            Embedder(config=embedder_config, runtime_config=runtime_config)

    def test_cache_starts_empty(self, embedder: Embedder) -> None:
        """A newly created Embedder should have an empty cache."""
        assert embedder._cache == {}


class TestEmbedderEncode:
    """Tests for the encode method."""

    def test_encode_single_text_returns_embedding_result(self, embedder: Embedder) -> None:
        """Encoding a single text should return an EmbeddingResult with one embedding."""
        text = "hello world"
        result = embedder.encode(text)

        assert isinstance(result, EmbeddingResult)
        assert result.count == 1
        assert result.dimensions > 0
        assert result.embeddings[0].text == text
        assert result.embeddings[0].model_name == embedder.model_name

    def test_encode_list_of_texts_produces_matching_embeddings(self, embedder: Embedder) -> None:
        """Encoding a list of texts should produce embeddings in the same order."""
        texts = ["first", "second", "third"]
        result = embedder.encode(texts)

        assert result.count == len(texts)
        for text, embedding in zip(texts, result.embeddings):
            assert embedding.text == text

    def test_encode_empty_list_returns_empty_result(self, embedder: Embedder) -> None:
        """Encoding an empty list should return an EmbeddingResult with no embeddings."""
        result = embedder.encode([])

        assert result.count == 0
        assert result.dimensions == 0

    def test_encode_with_none_entry_raises_encoding_error(self, embedder: Embedder) -> None:
        """Encoding a list containing None should raise EncodingError."""
        with pytest.raises(EncodingError):
            embedder.encode([None])  # type: ignore[list-item]

    def test_encode_with_empty_string_raises_encoding_error(self, embedder: Embedder) -> None:
        """Encoding a list containing an empty string should raise EncodingError."""
        with pytest.raises(EncodingError):
            embedder.encode(["valid", ""])

    def test_encode_respects_convert_to_numpy_runtime_flag(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Setting convert_to_numpy=True should produce numpy arrays."""
        runtime_config.convert_to_numpy = True
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        result = embedder.encode("test")
        embedding = result.embeddings[0]

        assert isinstance(embedding.vector, np.ndarray)


class TestEncodeCaching:
    """Tests for the embedding cache functionality."""

    def test_cache_disabled_by_default(self, embedder: Embedder) -> None:
        """By default, enable_cache is False and no embeddings are cached."""
        texts = ["one", "two", "one"]
        embedder.encode(texts)

        assert embedder._cache == {}

    def test_cache_enabled_reuses_embeddings(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """With enable_cache=True, repeated texts should reuse cached embeddings."""
        runtime_config.enable_cache = True
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        texts = ["alpha", "beta", "alpha"]
        result = embedder.encode(texts)

        # "alpha" appears twice in the input; both should reference the same object.
        assert result.embeddings[0] is result.embeddings[2]
        assert len(embedder._cache) == 2

    def test_cache_bypassed_when_convert_to_numpy_enabled(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Caching is disabled if convert_to_numpy=True."""
        runtime_config.enable_cache = True
        runtime_config.convert_to_numpy = True
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        embedder.encode(["one", "two"])

        assert embedder._cache == {}

    def test_clear_cache_empties_internal_cache(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Calling clear_cache should remove all cached embeddings."""
        runtime_config.enable_cache = True
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        embedder.encode(["one", "two"])
        assert len(embedder._cache) == 2

        embedder.clear_cache()
        assert embedder._cache == {}

    def test_cache_handles_mixed_cached_and_uncached_texts_in_order(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Subsequent encodes should reuse cached embeddings and preserve order."""
        runtime_config.enable_cache = True
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        first_result = embedder.encode(["a", "b"])
        second_result = embedder.encode(["c", "a", "b"])

        # "a" and "b" from second_result should be the same objects as in first_result.
        assert second_result.embeddings[1] is first_result.embeddings[0]
        assert second_result.embeddings[2] is first_result.embeddings[1]

    def test_cache_is_isolated_per_embedder_instance(
        self,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Each Embedder instance should have its own cache."""
        runtime_config.enable_cache = True
        embedder1 = Embedder(config=embedder_config, runtime_config=runtime_config)
        embedder2 = Embedder(config=embedder_config, runtime_config=runtime_config)

        embedder1.encode(["shared"])
        embedder2.encode(["shared"])

        # Despite encoding the same text, the two embedders should have distinct caches.
        # assert embedder1._cache != embedder2._cache
        assert embedder1._cache is not embedder2._cache
        assert embedder1._cache["shared"] is not embedder2._cache["shared"]


class TestSimilarity:
    """Tests for the similarity method."""

    def test_similarity_cosine_for_identical_embeddings_is_one(self, embedder: Embedder) -> None:
        """The cosine similarity of an embedding with itself should be 1.0."""
        result = embedder.encode("test")
        emb = result.embeddings[0]

        score = embedder.similarity(emb, emb, metric="cosine")

        assert score.metric == "cosine"
        assert score.score == pytest.approx(1.0)

    def test_similarity_dot_product_uses_raw_vectors(self, embedder: Embedder) -> None:
        """Dot product metric should compute the raw vector dot product."""
        result = embedder.encode(["one", "two"])
        emb_one, emb_two = result.embeddings

        score = embedder.similarity(emb_one, emb_two, metric="dot")

        # Compute expected dot product dynamically based on the actual vectors
        vec_one = np.array(emb_one.vector, dtype=float)
        vec_two = np.array(emb_two.vector, dtype=float)
        expected_dot = float(np.dot(vec_one, vec_two))

        assert score.metric == "dot"
        assert score.score == pytest.approx(expected_dot)

    def test_similarity_raises_for_dimension_mismatch(self, embedder: Embedder) -> None:
        """Embeddings with different dimensions should raise a ValidationError."""
        emb_small = Embedding(vector=[1.0, 2.0, 3.0], model_name="test", normalized=False)
        emb_large = Embedding(vector=[1.0, 2.0, 3.0, 4.0], model_name="test", normalized=False)

        with pytest.raises(ValidationError):
            embedder.similarity(emb_small, emb_large)

    def test_similarity_raises_for_invalid_metric(self, embedder: Embedder) -> None:
        """Requesting an unsupported similarity metric should raise ValidationError."""
        result = embedder.encode("test")
        emb = result.embeddings[0]

        with pytest.raises(ValidationError):
            embedder.similarity(emb, emb, metric="euclidean")

    def test_similarity_batch_returns_score_per_pair(self, embedder: Embedder) -> None:
        """similarity_batch should compute pairwise similarities."""
        result = embedder.encode(["a", "b", "c"])
        result2 = embedder.encode(["x", "y", "z"])

        scores = embedder.similarity_batch(result.embeddings, result2.embeddings, metric="cosine")

        assert len(scores) == 3
        for score in scores:
            assert isinstance(score, SimilarityScore)
            assert score.metric == "cosine"

    def test_similarity_batch_length_mismatch_raises(self, embedder: Embedder) -> None:
        """similarity_batch should raise ValidationError if lists have different lengths."""
        result_a = embedder.encode(["a", "b"])
        result_b = embedder.encode(["x", "y", "z"])

        with pytest.raises(ValidationError):
            embedder.similarity_batch(result_a.embeddings, result_b.embeddings)

    def test_similarity_supports_numpy_vectors(self, embedder: Embedder) -> None:
        """The similarity method should accept embeddings with numpy vectors."""
        vec_a = np.array([1.0, 2.0, 3.0], dtype=float)
        vec_b = np.array([4.0, 5.0, 6.0], dtype=float)
        emb_a = Embedding(vector=vec_a, model_name="test", normalized=False)
        emb_b = Embedding(vector=vec_b, model_name="test", normalized=False)

        score = embedder.similarity(emb_a, emb_b, metric="dot")

        expected = float(np.dot(vec_a, vec_b))
        assert score.score == pytest.approx(expected)


class TestEmbedderSearch:
    """Tests for the search method."""

    def test_basic_search_returns_results_for_each_query(self, embedder: Embedder) -> None:
        """Search should return a list of hits for each query."""
        queries = ["q1", "q2"]
        corpus = ["doc1", "doc2", "doc3"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=2)

        assert isinstance(results, SearchResults)
        assert results.num_queries == len(queries)
        assert len(results.results) == len(queries)

    def test_top_k_limits_number_of_results(self, embedder: Embedder) -> None:
        """Search should return at most top_k results per query."""
        corpus = ["a", "b", "c", "d", "e"]
        queries = ["query"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=3)

        assert len(results.results[0]) == 3

    def test_results_are_sorted_by_score_descending(self, embedder: Embedder) -> None:
        """Search results should be sorted by score in descending order."""
        corpus = ["one", "two", "three", "four"]
        queries = ["test"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=4)

        hits = results.results[0]
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_queries_return_empty_results(self, embedder: Embedder) -> None:
        """Searching with no queries should return an empty SearchResults."""
        corpus = ["doc1", "doc2"]

        results = embedder.search(queries=[], corpus=corpus)

        assert results.num_queries == 0

    def test_empty_corpus_returns_no_hits(self, embedder: Embedder) -> None:
        """Searching an empty corpus should return no hits per query."""
        queries = ["q1", "q2"]

        results = embedder.search(queries=queries, corpus=[])

        assert results.num_queries == len(queries)
        for query_results in results.results:
            assert len(query_results) == 0

    def test_invalid_top_k_raises_validation_error(self, embedder: Embedder) -> None:
        """top_k less than 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            embedder.search(queries=["q"], corpus=["doc"], top_k=0)

    def test_invalid_score_function_raises_validation_error(self, embedder: Embedder) -> None:
        """Using an unsupported score_function should raise ValidationError."""
        with pytest.raises(ValidationError):
            embedder.search(queries=["q"], corpus=["doc"], score_function="manhattan")

    def test_dot_product_score_function(self, embedder: Embedder) -> None:
        """Search should accept score_function='dot' and still use cosine for ranking."""
        corpus = ["a", "b", "c"]
        queries = ["query"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=2, score_function="dot")

        assert results.num_queries == 1
        assert len(results.results[0]) == 2

    def test_search_with_precomputed_corpus_embeddings_matches_corpus_search(self, embedder: Embedder) -> None:
        """Search with corpus and with corpus_embeddings should produce the same ranking."""
        corpus = ["zero", "one", "two"]
        queries = ["test"]

        corpus_embeddings = embedder.encode(corpus)

        results_with_corpus = embedder.search(queries=queries, corpus=corpus, top_k=2)
        results_with_embeddings = embedder.search(queries=queries, corpus_embeddings=corpus_embeddings, top_k=2)

        # The rankings should match exactly.
        hits_corpus = results_with_corpus.results[0]
        hits_embeddings = results_with_embeddings.results[0]

        for hit_c, hit_e in zip(hits_corpus, hits_embeddings):
            assert hit_c.corpus_id == hit_e.corpus_id
            assert hit_c.score == pytest.approx(hit_e.score)

    def test_search_with_both_corpus_and_embeddings_raises_validation_error(self, embedder: Embedder) -> None:
        """Supplying both corpus and corpus_embeddings should raise ValidationError."""
        corpus = ["doc"]
        corpus_embeddings = embedder.encode(corpus)

        with pytest.raises(ValidationError):
            embedder.search(queries=["q"], corpus=corpus, corpus_embeddings=corpus_embeddings)

    def test_search_with_neither_corpus_nor_embeddings_raises_validation_error(self, embedder: Embedder) -> None:
        """Omitting both corpus and corpus_embeddings should raise ValidationError."""
        with pytest.raises(ValidationError):
            embedder.search(queries=["q"])

