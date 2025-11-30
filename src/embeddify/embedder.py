# src/embeddify/embedder.py
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from embeddify.config import EmbedderConfig, RuntimeConfig, load_config_file
from embeddify.exceptions import EncodingError, ModelLoadError, SearchError, ValidationError
from embeddify.models import Embedding, EmbeddingResult, SearchResult, SearchResults, SimilarityScore
logger = logging.getLogger(__name__)


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

    def encode(self, texts: str | list[str]) -> EmbeddingResult:
        """Encode one or more texts into embeddings with optional caching.

        The method accepts either a single string or a list of strings, validates the
        inputs and delegates to the underlying SentenceTransformer model. The raw
        vectors are wrapped in :class:`Embedding` instances and returned inside an
        :class:`EmbeddingResult`. When caching is enabled via
        :class:`RuntimeConfig`, repeated texts will reuse previously computed
        :class:`Embedding` objects.

        Args:
            texts: A single text string or a list of text strings to encode.

        Returns:
            An :class:`EmbeddingResult` containing one :class:`Embedding` per input
            text. When an empty list is supplied the result will contain no
            embeddings and ``dimensions`` will be set to ``0``.

        Raises:
            EncodingError: If any of the supplied texts are invalid or if the
                underlying model raises an exception during encoding.
        """
        # Normalise to a list so the rest of the implementation can assume a
        # consistent shape.
        if isinstance(texts, str):
            input_texts: list[str] = [texts]
        else:
            input_texts = list(texts)

        if not input_texts:
            return EmbeddingResult(embeddings=[], model_name=self.model_name, dimensions=0)

        # Validate the individual text entries before calling into the model so
        # we can raise domain specific errors.
        for index, text in enumerate(input_texts):
            if text is None:
                raise EncodingError(f"Text at index {index} is None; expected a non-empty string.")
            if not isinstance(text, str):
                raise EncodingError(
                    f"Text at index {index} must be a string, got {type(text).__name__!s} instead."
                )
            if not text.strip():
                raise EncodingError(f"Text at index {index} is empty or whitespace only.")

        # Determine whether caching is active for this call.
        use_cache = self.runtime_config.enable_cache and not self.runtime_config.convert_to_numpy
        if self.runtime_config.enable_cache and self.runtime_config.convert_to_numpy:
            logger.debug(
                "Embedding cache disabled because convert_to_numpy=True; "
                "falling back to direct encoding."
            )

        # Split texts into cached and uncached sets while preserving indices so
        # results can be reassembled in the original order.
        cached_embeddings: dict[int, Embedding] = {}
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        if use_cache:
            for index, text in enumerate(input_texts):
                cached = self._cache.get(text)
                if cached is not None:
                    logger.debug("Cache hit for text at index %d", index)
                    cached_embeddings[index] = cached
                else:
                    logger.debug("Cache miss for text at index %d", index)
                    uncached_texts.append(text)
                    uncached_indices.append(index)
        else:
            uncached_texts = input_texts
            uncached_indices = list(range(len(input_texts)))

        new_embeddings_by_index: dict[int, Embedding] = {}
        if uncached_texts:
            try:
                vectors = self._model.encode(
                    uncached_texts,
                    batch_size=self.runtime_config.batch_size,
                    show_progress_bar=self.runtime_config.show_progress_bar,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=self.runtime_config.convert_to_numpy,
                )
            except Exception as exc:
                raise EncodingError(f"Failed to encode texts with underlying model: {exc}") from exc

            # SentenceTransformer.encode may return either a list of vectors or a
            # numpy array with shape (n_texts, dim). We normalise this to an
            # iterable of per-text vectors.
            if isinstance(vectors, np.ndarray):
                if vectors.ndim == 1:
                    per_text_vectors = [vectors]
                else:
                    per_text_vectors = [vectors[i] for i in range(vectors.shape[0])]
            else:
                per_text_vectors = list(vectors)

            for index, text, vector in zip(uncached_indices, uncached_texts, per_text_vectors):
                embedding = Embedding(
                    vector=vector,
                    model_name=self.model_name,
                    normalized=self.config.normalize_embeddings,
                    text=text,
                )
                new_embeddings_by_index[index] = embedding
                if use_cache:
                    self._cache[text] = embedding

        # Reassemble embeddings in original input order using both cached and
        # newly encoded values.
        embeddings: list[Embedding] = []
        for index in range(len(input_texts)):
            if index in cached_embeddings:
                embeddings.append(cached_embeddings[index])
            else:
                embeddings.append(new_embeddings_by_index[index])

        dimensions = embeddings[0].dimensions if embeddings else 0

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimensions=dimensions,
        )
    def clear_cache(self) -> None:
        """Clear all cached embeddings for this embedder instance.

        The cache is maintained per :class:`Embedder` instance and keyed by the
        original text string. This method removes all cached entries, forcing
        subsequent :meth:`encode` calls to recompute embeddings.
        """
        if self._cache:
            logger.debug("Clearing %d cached embeddings", len(self._cache))
        self._cache.clear()


    def similarity(
        self,
        emb1: Embedding,
        emb2: Embedding,
        metric: str = "cosine",
    ) -> SimilarityScore:
        """Compute similarity between two embeddings.

        Args:
            emb1: First embedding.
            emb2: Second embedding.
            metric: Similarity metric to use, either ``"cosine"`` or ``"dot"``.

        Returns:
            A :class:`SimilarityScore` object describing the similarity.

        Raises:
            ValidationError: If the embeddings have different dimensions or an
                unsupported metric is requested.
        """
        metric_normalised = metric.lower()
        if metric_normalised not in {"cosine", "dot"}:
            raise ValidationError(
                f"Unsupported similarity metric {metric!r}. "
                "Supported metrics are 'cosine' and 'dot'."
            )

        # Validate dimensionality before performing any computation.
        vec1 = emb1.vector
        vec2 = emb2.vector

        dim1 = int(vec1.shape[-1]) if isinstance(vec1, np.ndarray) else len(vec1)
        dim2 = int(vec2.shape[-1]) if isinstance(vec2, np.ndarray) else len(vec2)

        if dim1 != dim2:
            raise ValidationError(f"Embedding dimension mismatch: {dim1} vs {dim2}")

        arr1 = np.asarray(vec1, dtype=float)
        arr2 = np.asarray(vec2, dtype=float)

        if metric_normalised == "dot":
            score_value = float(np.dot(arr1, arr2))
        else:
            # Cosine similarity: dot(a, b) / (||a|| * ||b||)
            norm1 = float(np.linalg.norm(arr1))
            norm2 = float(np.linalg.norm(arr2))
            if norm1 == 0.0 or norm2 == 0.0:
                raise ValidationError(
                    "Cannot compute cosine similarity for zero-length embedding."
                )
            score_value = float(np.dot(arr1, arr2) / (norm1 * norm2))

        return SimilarityScore(score=score_value, metric=metric_normalised)

    def similarity_batch(
        self,
        embs1: list[Embedding],
        embs2: list[Embedding],
        metric: str = "cosine",
    ) -> list[SimilarityScore]:
        """Compute pairwise similarities between two lists of embeddings.

        The two lists must have the same length; the *i*th result corresponds to
        the similarity between ``embs1[i]`` and ``embs2[i]``.

        Args:
            embs1: First list of embeddings.
            embs2: Second list of embeddings.
            metric: Similarity metric to use, either ``"cosine"`` or ``"dot"``.

        Returns:
            A list of :class:`SimilarityScore` instances, one per input pair.

        Raises:
            ValidationError: If the input lists have different lengths, if the
                embeddings in a pair have mismatched dimensions or if an
                unsupported metric is requested.
        """
        if len(embs1) != len(embs2):
            raise ValidationError(
                "similarity_batch inputs must have the same length; "
                f"got {len(embs1)} and {len(embs2)}"
            )

        return [self.similarity(e1, e2, metric=metric) for e1, e2 in zip(embs1, embs2)]



    def search(
        self,
        queries: list[str],
        corpus: list[str] | None = None,
        corpus_embeddings: EmbeddingResult | None = None,
        top_k: int = 5,
        score_function: str = "cosine",
    ) -> SearchResults:
        """Perform semantic search over a corpus of texts.

        The search can operate either on raw corpus texts or on pre-computed
        corpus embeddings. Exactly one of ``corpus`` or ``corpus_embeddings``
        must be provided.

        When a plain text ``corpus`` is supplied, both the queries and the
        corpus are encoded on the fly using :meth:`encode`. When
        ``corpus_embeddings`` are provided, only the queries are encoded and the
        given embeddings are reused, which is more efficient for repeated
        searches over a static corpus.

        Args:
            queries: The list of query texts to search for.
            corpus: Optional list of corpus texts to search over. Mutually
                exclusive with ``corpus_embeddings``.
            corpus_embeddings: Optional pre-computed embeddings for the corpus.
                Mutually exclusive with ``corpus``.
            top_k: Maximum number of results to return per query. Must be at
                least 1 and cannot exceed the size of the effective corpus.
            score_function: Similarity metric to use. Supported values are
                ``"cosine"`` and ``"dot"``.

        Returns:
            A :class:`SearchResults` instance containing, for each query, a list
            of :class:`SearchResult` objects describing the best matching corpus
            entries.

        Raises:
            ValidationError: If the configuration of arguments is invalid, if
                ``top_k`` is out of range, or if an unsupported ``score_function``
                is requested.
            SearchError: If encoding or similarity computation fails.
        """
        # Handle trivial case first: no queries means no results regardless of corpus.
        if not queries:
            return SearchResults(results=[], query_texts=[])

        # Exactly one of corpus or corpus_embeddings must be provided.
        if corpus is None and corpus_embeddings is None:
            raise ValidationError(
                "Either 'corpus' or 'corpus_embeddings' must be provided for search."
            )
        if corpus is not None and corpus_embeddings is not None:
            raise ValidationError(
                "Only one of 'corpus' or 'corpus_embeddings' may be provided, not both."
            )

        # Determine the effective corpus size and handle empty-corpus cases.
        if corpus is not None:
            corpus_size = len(corpus)
            if corpus_size == 0:
                # No documents to search; each query simply has no hits.
                return SearchResults(
                    results=[[] for _ in queries],
                    query_texts=queries,
                )
        else:
            corpus_size = len(corpus_embeddings.embeddings)  # type: ignore[union-attr]
            if corpus_size == 0:
                return SearchResults(
                    results=[[] for _ in queries],
                    query_texts=queries,
                )

        metric_normalised = score_function.lower()
        if metric_normalised not in {"cosine", "dot"}:
            raise ValidationError(
                f"Unsupported similarity metric {score_function!r}. "
                "Supported metrics are 'cosine' and 'dot'."
            )

        if top_k < 1:
            raise ValidationError(f"top_k must be at least 1; got {top_k}.")
        if corpus_size and top_k > corpus_size:
            raise ValidationError(
                f"top_k ({top_k}) cannot be greater than corpus size ({corpus_size})."
            )

        try:
            query_result = self.encode(queries)

            if corpus is not None:
                corpus_result = self.encode(corpus)
                corpus_texts: list[str] | None = corpus
            else:
                corpus_result = corpus_embeddings  # type: ignore[assignment]
                corpus_texts = None

            all_results: list[list[SearchResult]] = []

            for query_embedding in query_result.embeddings:
                # Compute similarity between this query and every corpus entry.
                similarities = [
                    self.similarity(query_embedding, corpus_embedding, metric=metric_normalised)
                    for corpus_embedding in corpus_result.embeddings  # type: ignore[union-attr]
                ]

                indexed_scores = list(enumerate(similarities))
                indexed_scores.sort(key=lambda item: item[1].score, reverse=True)
                top_hits = indexed_scores[:top_k]

                query_hits = [
                    SearchResult(
                        corpus_id=corpus_index,
                        score=score.score,
                        text=corpus_texts[corpus_index] if corpus_texts is not None else None,
                    )
                    for corpus_index, score in top_hits
                ]
                all_results.append(query_hits)

            return SearchResults(results=all_results, query_texts=queries)
        except Exception as exc:  # pragma: no cover - defensive; exercised via tests
            raise SearchError(f"Semantic search failed: {exc}") from exc
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