# tests/test_embedder.py
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from embeddify.config import EmbedderConfig, RuntimeConfig
from embeddify.embedder import Embedder
from embeddify.exceptions import EncodingError, ModelLoadError, ValidationError
from embeddify.models import Embedding, EmbeddingResult, SearchResults, SimilarityScore


class TestEmbedderInitialisation:
    def test_embedder_loads_model_with_valid_config(self, embedder: Embedder, mock_model_path: Path) -> None:
        """Embedder should load a SentenceTransformer instance on initialisation."""
        # The dummy SentenceTransformer stores the path and device attributes.
        assert embedder.config.model_path == str(mock_model_path)
        assert hasattr(embedder, "_model")

        model = embedder._model  # type: ignore[attr-defined]
        # The dummy model exposes ``model_name_or_path`` and ``device`` attributes.
        assert getattr(model, "model_name_or_path") == str(mock_model_path)
        assert getattr(model, "device") == embedder.config.device

    def test_model_name_property_uses_underlying_model(self, embedder: Embedder, mock_model_path: Path) -> None:
        """model_name should reflect the underlying model's identifier when available."""
        assert embedder.model_name == str(mock_model_path)

    def test_device_property_reflects_configured_device(self, embedder: Embedder) -> None:
        """device property should return the device configured for the model."""
        assert embedder.device == embedder.config.device

    def test_from_config_file_constructs_embedder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_config_file should load both model and runtime configuration."""
        # Build a minimal YAML configuration file.
        config_path = tmp_path / "embedder.yaml"
        config_path.write_text(
            """model:
  model_path: "{model_path}"
  device: "cpu"
runtime:
  batch_size: 8
""".format(
                model_path=tmp_path / "model-dir",
            ),
            encoding="utf-8",
        )

        # Ensure the referenced model path exists so EmbedderConfig is valid.
        model_dir = tmp_path / "model-dir"
        model_dir.mkdir()

        embedder = Embedder.from_config_file(str(config_path))

        assert isinstance(embedder, Embedder)
        assert embedder.config.model_path == str(model_dir)
        assert embedder.runtime_config.batch_size == 8

    def test_invalid_model_load_raises_model_load_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedder_config: EmbedderConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        """Errors from SentenceTransformer initialisation must surface as ModelLoadError."""

        def _failing_constructor(*_: Any, **__: Any) -> Any:
            raise RuntimeError("boom")

        import sentence_transformers

        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            _failing_constructor,
            raising=True,
        )

        with pytest.raises(ModelLoadError) as exc_info:
            Embedder(config=embedder_config, runtime_config=runtime_config)

        message = str(exc_info.value)
        assert "Failed to load SentenceTransformer model" in message
        assert "boom" in message

    def test_cache_starts_empty(self, embedder: Embedder) -> None:
        """Embedder should initialise an empty in-memory cache for future steps."""
        cache = cast(dict[str, Embedding], embedder._cache)  # type: ignore[attr-defined]
        assert cache == {}



class TestEmbedderEncode:
    def test_encode_single_text_returns_embedding_result(self, embedder: Embedder) -> None:
        """Encoding a single string should yield one embedding in the result."""
        result = embedder.encode("hello world")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 1

        embedding = result.embeddings[0]
        assert embedding.text == "hello world"
        assert embedding.model_name == embedder.model_name
        assert embedding.normalized == embedder.config.normalize_embeddings

        assert result.model_name == embedder.model_name
        assert result.dimensions == embedding.dimensions

    def test_encode_list_of_texts_produces_matching_embeddings(self, embedder: Embedder) -> None:
        """Encoding a list of texts should yield one embedding per text."""
        texts = ["one", "two", "three"]

        result = embedder.encode(texts)

        assert len(result.embeddings) == len(texts)
        assert result.model_name == embedder.model_name

        dims = {embedding.dimensions for embedding in result.embeddings}
        assert len(dims) == 1  # All embeddings share the same dimensionality.

        for original, embedding in zip(texts, result.embeddings):
            assert embedding.text == original

    def test_encode_empty_list_returns_empty_result(self, embedder: Embedder) -> None:
        """Encoding an empty list should return an empty EmbeddingResult."""
        result = embedder.encode([])

        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == []
        assert result.dimensions == 0

    def test_encode_with_none_entry_raises_encoding_error(self, embedder: Embedder) -> None:
        """None entries must be rejected with a clear EncodingError."""
        with pytest.raises(EncodingError):
            embedder.encode(["valid", None])  # type: ignore[list-item]

    def test_encode_with_empty_string_raises_encoding_error(self, embedder: Embedder) -> None:
        """Empty or whitespace-only strings are not valid inputs."""
        with pytest.raises(EncodingError):
            embedder.encode("   ")

    def test_encode_respects_convert_to_numpy_runtime_flag(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """When convert_to_numpy is enabled vectors should be numpy arrays."""
        runtime_config = RuntimeConfig(convert_to_numpy=True)
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        result = embedder.encode(["hello numpy"])
        assert len(result.embeddings) == 1

        vector = result.embeddings[0].vector
        assert isinstance(vector, np.ndarray)

class TestEncodeCaching:
    def test_cache_disabled_by_default(self, embedder: Embedder) -> None:
        """Cache should be disabled when enable_cache is False."""
        first = embedder.encode(["hello"])
        second = embedder.encode(["hello"])

        assert first.embeddings[0].vector == second.embeddings[0].vector
        # Without caching the returned Embedding instances should be distinct.
        assert first.embeddings[0] is not second.embeddings[0]
        assert embedder._cache == {}  # type: ignore[attr-defined]

    def test_cache_enabled_reuses_embeddings(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """When caching is enabled repeated texts reuse the same Embedding."""
        runtime_config = RuntimeConfig(enable_cache=True)
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        first = embedder.encode(["cached text"])
        second = embedder.encode(["cached text"])

        assert first.embeddings[0] is second.embeddings[0]
        assert list(embedder._cache.keys()) == ["cached text"]  # type: ignore[attr-defined]

    def test_cache_bypassed_when_convert_to_numpy_enabled(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """Enabling convert_to_numpy should disable caching entirely."""
        runtime_config = RuntimeConfig(enable_cache=True, convert_to_numpy=True)
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        first = embedder.encode(["numpy text"])
        second = embedder.encode(["numpy text"])

        assert isinstance(first.embeddings[0].vector, np.ndarray)
        assert isinstance(second.embeddings[0].vector, np.ndarray)
        # Caching is disabled so each call should yield distinct Embedding objects.
        assert first.embeddings[0] is not second.embeddings[0]
        assert embedder._cache == {}  # type: ignore[attr-defined]

    def test_clear_cache_empties_internal_cache(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """clear_cache should remove all cached entries."""
        runtime_config = RuntimeConfig(enable_cache=True)
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        embedder.encode(["to be cached"])
        assert embedder._cache  # type: ignore[attr-defined]

        embedder.clear_cache()
        assert embedder._cache == {}  # type: ignore[attr-defined]

    def test_cache_handles_mixed_cached_and_uncached_texts_in_order(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """Encoding with partially cached inputs must preserve input order."""
        runtime_config = RuntimeConfig(enable_cache=True)
        embedder = Embedder(config=embedder_config, runtime_config=runtime_config)

        first = embedder.encode(["one", "two"])
        # Second call shares "two" and introduces a new text "three".
        second = embedder.encode(["two", "three"])

        assert [e.text for e in second.embeddings] == ["two", "three"]
        # The embedding for "two" should be exactly the same object as in the first result.
        assert second.embeddings[0] is first.embeddings[1]

    def test_cache_is_isolated_per_embedder_instance(
        self,
        embedder_config: EmbedderConfig,
    ) -> None:
        """Each Embedder instance maintains its own cache."""
        runtime_config = RuntimeConfig(enable_cache=True)

        embedder_one = Embedder(config=embedder_config, runtime_config=runtime_config)
        embedder_two = Embedder(config=embedder_config, runtime_config=runtime_config)

        embedder_one.encode(["shared text"])

        assert "shared text" in embedder_one._cache  # type: ignore[attr-defined]
        assert "shared text" not in embedder_two._cache  # type: ignore[attr-defined]


class TestSimilarity:
    def test_similarity_cosine_for_identical_embeddings_is_one(self, embedder: Embedder) -> None:
        """Cosine similarity between identical embeddings should be 1.0."""
        result = embedder.encode(["same text"])
        embedding = result.embeddings[0]

        score = embedder.similarity(embedding, embedding)

        assert isinstance(score, SimilarityScore)
        assert score.metric == "cosine"
        assert score.score == pytest.approx(1.0)

    def test_similarity_dot_product_uses_raw_vectors(self, embedder: Embedder) -> None:
        """Dot product metric should compute the raw vector dot product."""
        result = embedder.encode(["one", "two"])
        emb_one, emb_two = result.embeddings

        score = embedder.similarity(emb_one, emb_two, metric="dot")

        # The dummy model in tests produces simple deterministic vectors; for the
        # first two texts the expected dot product is easily computed.
        assert score.metric == "dot"
        assert score.score == pytest.approx(20.0)

    def test_similarity_raises_for_dimension_mismatch(self, embedder: Embedder) -> None:
        """Embeddings with different dimensions should raise a ValidationError."""
        emb_small = Embedding(vector=[1.0, 2.0, 3.0], model_name="test", normalized=False)
        emb_large = Embedding(vector=[1.0, 2.0, 3.0, 4.0], model_name="test", normalized=False)

        with pytest.raises(ValidationError, match="dimension mismatch"):
            embedder.similarity(emb_small, emb_large)

    def test_similarity_raises_for_invalid_metric(self, embedder: Embedder) -> None:
        """Unsupported metrics should raise a ValidationError before computation."""
        result = embedder.encode(["one", "two"])
        emb_one, emb_two = result.embeddings

        with pytest.raises(ValidationError, match="Unsupported similarity metric"):
            embedder.similarity(emb_one, emb_two, metric="euclidean")

    def test_similarity_batch_returns_score_per_pair(self, embedder: Embedder) -> None:
        """similarity_batch should return one score for each pair of embeddings."""
        result_one = embedder.encode(["a", "b", "c"])
        result_two = embedder.encode(["d", "e", "f"])

        scores = embedder.similarity_batch(result_one.embeddings, result_two.embeddings)

        assert len(scores) == 3
        assert all(isinstance(score, SimilarityScore) for score in scores)

    def test_similarity_batch_length_mismatch_raises(self, embedder: Embedder) -> None:
        """Input lists for similarity_batch must have the same length."""
        result_one = embedder.encode(["a", "b"])
        result_two = embedder.encode(["c"])

        with pytest.raises(ValidationError, match="must have the same length"):
            embedder.similarity_batch(result_one.embeddings, result_two.embeddings)

    def test_similarity_supports_numpy_vectors(self, embedder: Embedder) -> None:
        """similarity should handle embeddings backed by numpy arrays."""
        emb_one = Embedding(
            vector=np.array([1.0, 0.0, 0.0, 0.0]),
            model_name="test",
            normalized=False,
        )
        emb_two = Embedding(
            vector=np.array([1.0, 0.0, 0.0, 0.0]),
            model_name="test",
            normalized=False,
        )

        score = embedder.similarity(emb_one, emb_two)

        assert score.metric == "cosine"
        assert score.score == pytest.approx(1.0)



class TestEmbedderSearch:
    """Tests for the semantic search functionality built on top of embeddings."""

    def _patch_deterministic_encode(self, embedder: Embedder, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace ``encode`` with a deterministic implementation for tests.

        The patched method derives a small numeric vector from each input text
        using a hash-based random generator so that the same text always yields
        the same embedding across calls.
        """

        def fake_encode(self, texts: str | list[str]) -> EmbeddingResult:
            if isinstance(texts, str):
                raw_texts = [texts]
            else:
                raw_texts = list(texts)

            embeddings: list[Embedding] = []
            for text in raw_texts:
                # Derive a stable seed from the text so encodings are
                # repeatable but independent of the underlying model.
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                vector = rng.normal(size=4)

                embeddings.append(
                    Embedding(
                        vector=vector,
                        model_name="deterministic-test-model",
                        normalized=False,
                        text=text,
                    )
                )

            return EmbeddingResult(
                embeddings=embeddings,
                model_name="deterministic-test-model",
                dimensions=4,
            )

        monkeypatch.setattr(Embedder, "encode", fake_encode)

    def test_basic_search_returns_results_for_each_query(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search should return top hits per query with associated corpus text."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        queries = ["alpha", "beta"]
        corpus = ["alpha", "gamma", "beta"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=2)

        assert isinstance(results, SearchResults)
        # One list of hits per query.
        assert len(results.results) == len(queries)
        assert results.query_texts == queries
        # The best hit for each query should be the matching corpus entry.
        first_hits = [hits[0].text for hits in results.results]
        assert first_hits[0] == "alpha"
        assert first_hits[1] == "beta"

    def test_top_k_limits_number_of_results(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ``top_k`` parameter should control the number of hits per query."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        queries = ["alpha"]
        corpus = ["alpha", "beta", "gamma", "delta"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=2)

        assert len(results.results) == 1
        assert len(results.results[0]) == 2

    def test_results_are_sorted_by_score_descending(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Results for a query must be ordered from most to least similar."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        queries = ["alpha"]
        corpus = ["alpha", "beta", "gamma"]

        results = embedder.search(queries=queries, corpus=corpus, top_k=3)

        scores = [hit.score for hit in results.results[0]]
        assert scores == sorted(scores, reverse=True)

    def test_empty_queries_return_empty_results(self, embedder: Embedder) -> None:
        """An empty query list should yield an empty SearchResults container."""
        results = embedder.search(queries=[], corpus=["alpha", "beta"])

        assert isinstance(results, SearchResults)
        assert results.results == []
        assert results.query_texts == []

    def test_empty_corpus_returns_no_hits(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty corpus should produce no hits for each query."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        queries = ["alpha", "beta"]
        results = embedder.search(queries=queries, corpus=[])

        assert len(results.results) == len(queries)
        assert all(hits == [] for hits in results.results)

    def test_invalid_top_k_raises_validation_error(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid ``top_k`` values must raise :class:`ValidationError`."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        corpus = ["alpha", "beta", "gamma"]

        with pytest.raises(ValidationError):
            embedder.search(queries=["alpha"], corpus=corpus, top_k=0)

        with pytest.raises(ValidationError):
            embedder.search(queries=["alpha"], corpus=corpus, top_k=4)

    def test_invalid_score_function_raises_validation_error(self, embedder: Embedder) -> None:
        """Unsupported score functions must be rejected early."""
        with pytest.raises(ValidationError):
            embedder.search(queries=["alpha"], corpus=["alpha"], score_function="euclidean")

    def test_dot_product_score_function(self, embedder: Embedder, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``dot`` score function should be accepted and behave like cosine."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        queries = ["alpha"]
        corpus = ["alpha", "beta"]

        cosine_results = embedder.search(
            queries=queries,
            corpus=corpus,
            top_k=2,
            score_function="cosine",
        )
        dot_results = embedder.search(
            queries=queries,
            corpus=corpus,
            top_k=2,
            score_function="dot",
        )

        assert len(dot_results.results[0]) == len(cosine_results.results[0])
        # Both metrics should pick the same best-matching corpus entry.
        assert dot_results.results[0][0].corpus_id == cosine_results.results[0][0].corpus_id

    def test_search_with_precomputed_corpus_embeddings_matches_corpus_search(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search using pre-computed corpus embeddings should match corpus search.

        The ranking (``corpus_id`` order and scores) must be identical to a
        search that encodes the corpus on the fly, while ``text`` is omitted
        for pre-computed searches.
        """
        self._patch_deterministic_encode(embedder, monkeypatch)

        queries = ["alpha", "beta"]
        corpus = ["alpha", "gamma", "beta"]

        corpus_embeddings = embedder.encode(corpus)

        corpus_results = embedder.search(
            queries=queries,
            corpus=corpus,
            top_k=2,
            score_function="cosine",
        )
        precomputed_results = embedder.search(
            queries=queries,
            corpus_embeddings=corpus_embeddings,
            top_k=2,
            score_function="cosine",
        )

        assert len(precomputed_results.results) == len(corpus_results.results)

        for hits_pre, hits_corpus in zip(precomputed_results.results, corpus_results.results):
            assert len(hits_pre) == len(hits_corpus)
            for hit_pre, hit_corpus in zip(hits_pre, hits_corpus):
                assert hit_pre.corpus_id == hit_corpus.corpus_id
                assert hit_pre.score == pytest.approx(hit_corpus.score)
                # When using pre-computed embeddings, corpus text is not available.
                assert hit_pre.text is None
                assert hit_corpus.text is not None

    def test_search_with_both_corpus_and_embeddings_raises_validation_error(
        self,
        embedder: Embedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Providing both ``corpus`` and ``corpus_embeddings`` must be rejected."""
        self._patch_deterministic_encode(embedder, monkeypatch)

        corpus = ["alpha", "beta"]
        corpus_embeddings = embedder.encode(corpus)

        with pytest.raises(ValidationError):
            embedder.search(
                queries=["alpha"],
                corpus=corpus,
                corpus_embeddings=corpus_embeddings,
                top_k=1,
            )

    def test_search_with_neither_corpus_nor_embeddings_raises_validation_error(
        self,
        embedder: Embedder,
    ) -> None:
        """Omitting both ``corpus`` and ``corpus_embeddings`` must fail fast."""
        with pytest.raises(ValidationError):
            embedder.search(queries=["alpha"])

