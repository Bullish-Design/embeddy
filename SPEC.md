# src/embeddify/SPEC.md

# **Embeddify Technical Specification**

## **Document Information**

* **Version:** 0.1.0
* **Last Updated:** 2025-11-29
* **Status:** Draft

## **Executive Summary**

Embeddify is a Pydantic-based wrapper for Sentence Transformers that provides type-safe embedding generation and 
semantic search through validated configuration models. It eliminates boilerplate code while adding comprehensive 
validation, reducing typical embedding workflows from 20+ lines to 3-5 lines of configuration-driven code.

## **Requirements**

### **Functional Requirements**

#### **Core Features**

**REQ-001: Model Configuration and Loading**

* **Description:** Load Sentence Transformer models from local filesystem paths with validated configuration 
  parameters. Support device selection (CPU/CUDA), normalization settings, and trust_remote_code option.
* **Inputs:** 
  - `model_path` (str): Absolute path to pre-downloaded model
  - `device` (str): "cpu", "cuda", or "cuda:N"
  - `normalize_embeddings` (bool): Whether to L2-normalize vectors
  - `trust_remote_code` (bool): Allow custom model architectures
* **Outputs:** Initialized `Embedder` instance with loaded SentenceTransformer model
* **Priority:** Critical
* **Acceptance Criteria:**
  * [ ] EmbedderConfig validates all parameters via Pydantic
  * [ ] Invalid model paths raise ModelLoadError before initialization
  * [ ] Device validation checks CUDA availability when requested
  * [ ] Model loads successfully and is accessible via Embedder instance
  * [ ] Configuration can be loaded from YAML/JSON files
  * [ ] Environment variables override config file values

**REQ-002: Text Encoding**

* **Description:** Generate embedding vectors from text inputs with batch processing support. Handle single texts 
  and lists uniformly, apply normalization if configured, and return validated embedding objects.
* **Inputs:**
  - `texts` (list[str]): One or more text strings to encode
  - RuntimeConfig settings (batch_size, show_progress_bar, convert_to_numpy, enable_cache)
* **Outputs:** `EmbeddingResult` containing list of `Embedding` objects with vectors and metadata
* **Priority:** Critical
* **Acceptance Criteria:**
  * [ ] Single text and text lists both accepted
  * [ ] Empty text lists return empty EmbeddingResult
  * [ ] Batch processing respects batch_size configuration
  * [ ] Progress bar displays for large batches when enabled
  * [ ] Vectors normalized if normalize_embeddings=True
  * [ ] Cache lookup/storage when enable_cache=True
  * [ ] Returns list[float] by default, np.ndarray if convert_to_numpy=True
  * [ ] Empty strings and None values raise EncodingError

**REQ-003: Similarity Computation**

* **Description:** Calculate cosine similarity between embedding pairs with validated inputs and outputs. Support 
  both single pair and batch similarity operations.
* **Inputs:**
  - Single: `emb1`, `emb2` (Embedding objects)
  - Batch: `embs1`, `embs2` (list[Embedding])
* **Outputs:**
  - Single: `SimilarityScore` with score float and metric string
  - Batch: `list[SimilarityScore]`
* **Priority:** High
* **Acceptance Criteria:**
  * [ ] Cosine similarity computed correctly (range -1 to 1)
  * [ ] Dimension mismatch raises ValidationError
  * [ ] Batch similarity computes pairwise scores
  * [ ] SimilarityScore includes metric="cosine" metadata
  * [ ] Handles both list[float] and np.ndarray vectors

**REQ-004: Semantic Search**

* **Description:** Find top-k most similar corpus entries for each query using semantic search. Support both 
  on-the-fly corpus encoding and pre-computed embeddings for efficiency.
* **Inputs:**
  - `queries` (list[str]): Query texts
  - `corpus` (list[str] | None): Corpus texts to search (encoded internally)
  - `corpus_embeddings` (EmbeddingResult | None): Pre-computed corpus embeddings
  - `top_k` (int): Number of results per query
  - `score_function` (str): "cosine" or "dot"
* **Outputs:** `SearchResults` with results per query, each containing top_k `SearchResult` objects
* **Priority:** High
* **Acceptance Criteria:**
  * [ ] Exactly one of corpus or corpus_embeddings must be provided
  * [ ] Returns top_k results per query, sorted by score descending
  * [ ] SearchResult includes corpus_id, score, and text (if available)
  * [ ] Supports "cosine" and "dot" score functions
  * [ ] Query encoding reuses encode() method with caching
  * [ ] Empty queries or corpus returns empty results

**REQ-005: Configuration File Support**

* **Description:** Load configuration from YAML or JSON files with environment variable support. Enable 
  configuration-as-code for reproducible embedder setups.
* **Inputs:**
  - File path (str | None): Config file location, or None to read from EMBEDDIFY_CONFIG_PATH env var
  - Config file contents (YAML/JSON)
* **Outputs:** Fully configured `Embedder` instance
* **Priority:** High
* **Acceptance Criteria:**
  * [ ] YAML and JSON formats both supported
  * [ ] Nested config structure: `model:` and `runtime:` sections
  * [ ] Environment variables override file values (EMBEDDIFY_MODEL_PATH, EMBEDDIFY_DEVICE)
  * [ ] from_config_file() class method returns Embedder instance
  * [ ] Invalid config raises ValidationError with clear message
  * [ ] Missing file raises FileNotFoundError

**REQ-006: Error Handling**

* **Description:** Provide clear, actionable error messages with custom exception hierarchy wrapping 
  sentence-transformers and PyTorch errors.
* **Inputs:** Various failure conditions (validation, model loading, encoding, search)
* **Outputs:** Specific exception types with context
* **Priority:** Critical
* **Acceptance Criteria:**
  * [ ] EmbeddifyError base class for all library errors
  * [ ] ModelLoadError for model initialization failures
  * [ ] EncodingError for text encoding failures
  * [ ] ValidationError for Pydantic validation failures
  * [ ] SearchError for semantic search failures
  * [ ] Error messages include actionable guidance
  * [ ] Original exception preserved in __cause__

**REQ-007: Result Metadata**

* **Description:** Include comprehensive metadata in all result objects for debugging and traceability.
* **Inputs:** N/A
* **Outputs:** Metadata fields in Embedding, EmbeddingResult, SearchResult, SearchResults
* **Priority:** Medium
* **Acceptance Criteria:**
  * [ ] Embedding stores model_name, normalized flag, optional text
  * [ ] EmbeddingResult stores model_name, dimensions, embedding count
  * [ ] SearchResult stores corpus_id, score, optional text
  * [ ] SearchResults stores query_texts for reference
  * [ ] SimilarityScore stores metric used

#### **Extended Features**

**REQ-008: Batch Processing Optimization**

* **Description:** Handle large text lists efficiently through batching with progress tracking.
* **Priority:** Medium
* **Dependencies:** REQ-002
* **Acceptance Criteria:**
  * [ ] Configurable batch_size in RuntimeConfig
  * [ ] Progress bar via tqdm when show_progress_bar=True
  * [ ] Memory-efficient batching prevents OOM

**REQ-009: Embedding Caching**

* **Description:** Cache encoded texts to avoid re-computation of identical inputs.
* **Priority:** Medium
* **Dependencies:** REQ-002
* **Acceptance Criteria:**
  * [ ] In-memory dict cache keyed by text hash
  * [ ] enable_cache flag in RuntimeConfig
  * [ ] Cache per Embedder instance
  * [ ] Cache bypassed when convert_to_numpy=True (dict key limitation)

## **Architecture**

### **System Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Layer                           │
│  ┌────────────────────┐      ┌──────────────────────┐          │
│  │  EmbedderConfig    │      │   RuntimeConfig      │          │
│  │  - model_path      │      │   - batch_size       │          │
│  │  - device          │      │   - show_progress    │          │
│  │  - normalize       │      │   - enable_cache     │          │
│  │  - trust_remote    │      │   - convert_to_numpy │          │
│  └────────────────────┘      └──────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Execution Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Embedder                              │  │
│  │  - _model: SentenceTransformer (PrivateAttr)            │  │
│  │  - _cache: dict[str, Embedding] (PrivateAttr)           │  │
│  │                                                          │  │
│  │  + encode(texts) -> EmbeddingResult                     │  │
│  │  + similarity(emb1, emb2) -> SimilarityScore            │  │
│  │  + similarity_batch(...) -> list[SimilarityScore]       │  │
│  │  + search(queries, corpus, ...) -> SearchResults        │  │
│  │  + from_config_file(path) -> Embedder [classmethod]     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Results Layer                              │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Embedding   │  │ EmbeddingResult │  │  SearchResults   │  │
│  │  - vector    │  │ - embeddings    │  │  - results       │  │
│  │  - text      │  │ - model_name    │  │  - query_texts   │  │
│  │  - model     │  │ - dimensions    │  │                  │  │
│  │  - normalized│  └─────────────────┘  └──────────────────┘  │
│  └──────────────┘                                              │
│  ┌──────────────┐  ┌─────────────────┐                        │
│  │SearchResult  │  │ SimilarityScore │                        │
│  │- corpus_id   │  │ - score         │                        │
│  │- score       │  │ - metric        │                        │
│  │- text        │  └─────────────────┘                        │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

### **Core Components**

#### **Class Hierarchy**

```python
# src/embeddify/config.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from pathlib import Path
import os

class EmbedderConfig(BaseModel):
    """Configuration for SentenceTransformer model initialization."""
    
    model_path: str = Field(
        description="Path to pre-downloaded Sentence Transformer model"
    )
    device: str = Field(
        default="cpu",
        description="Device for computation: 'cpu', 'cuda', or 'cuda:N'"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Normalize embedding vectors to unit length"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code when loading models with custom architectures"
    )
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate model path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {v}")
        return v
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string format."""
        if v not in ["cpu", "cuda"] and not v.startswith("cuda:"):
            raise ValueError(f"Invalid device: {v}. Must be 'cpu', 'cuda', or 'cuda:N'")
        
        # Check CUDA availability if requested
        if v.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError(f"CUDA requested but not available. Falling back to CPU.")
            except ImportError:
                raise ValueError("torch not installed, cannot use CUDA")
        
        return v


class RuntimeConfig(BaseModel):
    """Configuration for execution behavior and performance tuning."""
    
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Number of texts to encode per batch"
    )
    show_progress_bar: bool = Field(
        default=False,
        description="Display progress bar for batch operations"
    )
    enable_cache: bool = Field(
        default=False,
        description="Cache encoded texts to avoid re-computation"
    )
    convert_to_numpy: bool = Field(
        default=False,
        description="Return numpy arrays instead of lists for embeddings"
    )


def load_config_file(path: str | None = None) -> dict:
    """Load configuration from YAML or JSON file.
    
    Args:
        path: Config file path. If None, reads from EMBEDDIFY_CONFIG_PATH env var.
        
    Returns:
        Dictionary of configuration values
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config file format invalid
    """
    import yaml
    import json
    
    if path is None:
        path = os.getenv("EMBEDDIFY_CONFIG_PATH")
        if path is None:
            raise ValueError("No config path provided and EMBEDDIFY_CONFIG_PATH not set")
    
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    content = config_path.read_text()
    
    # Try YAML first, fall back to JSON
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError:
        try:
            config = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file is neither valid YAML nor JSON: {e}")
    
    # Apply environment variable overrides
    if "model" not in config:
        config["model"] = {}
    if "runtime" not in config:
        config["runtime"] = {}
    
    if env_model_path := os.getenv("EMBEDDIFY_MODEL_PATH"):
        config["model"]["path"] = env_model_path
    if env_device := os.getenv("EMBEDDIFY_DEVICE"):
        config["model"]["device"] = env_device
    
    return config
```

```python
# src/embeddify/models.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import numpy as np

class Embedding(BaseModel):
    """Single embedding vector with metadata."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    vector: list[float] | np.ndarray = Field(
        description="Embedding vector representation"
    )
    text: str | None = Field(
        default=None,
        description="Original text that was encoded"
    )
    model_name: str = Field(
        description="Name of model used for encoding"
    )
    normalized: bool = Field(
        description="Whether vector is L2-normalized"
    )
    
    @field_validator('vector')
    @classmethod
    def validate_vector_length(cls, v: list[float] | np.ndarray) -> list[float] | np.ndarray:
        """Ensure vector is non-empty."""
        if isinstance(v, np.ndarray):
            if v.size == 0:
                raise ValueError("Embedding vector cannot be empty")
        elif len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        return v


class EmbeddingResult(BaseModel):
    """Collection of embeddings with metadata."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    embeddings: list[Embedding] = Field(
        description="List of embedding objects"
    )
    model_name: str = Field(
        description="Name of model used for encoding"
    )
    dimensions: int = Field(
        gt=0,
        description="Dimensionality of embedding vectors"
    )
    
    @field_validator('embeddings')
    @classmethod
    def validate_embedding_dimensions(cls, v: list[Embedding]) -> list[Embedding]:
        """Ensure all embeddings have same dimensions."""
        if not v:
            return v
        
        first_dim = len(v[0].vector) if isinstance(v[0].vector, list) else v[0].vector.size
        for emb in v[1:]:
            dim = len(emb.vector) if isinstance(emb.vector, list) else emb.vector.size
            if dim != first_dim:
                raise ValueError(f"Inconsistent embedding dimensions: {first_dim} vs {dim}")
        
        return v


class SearchResult(BaseModel):
    """Single search result entry."""
    
    corpus_id: int = Field(
        ge=0,
        description="Index in original corpus"
    )
    score: float = Field(
        description="Similarity score"
    )
    text: str | None = Field(
        default=None,
        description="Corpus text if available"
    )


class SearchResults(BaseModel):
    """Complete search results for all queries."""
    
    results: list[list[SearchResult]] = Field(
        description="Results per query (outer list) with top_k hits (inner list)"
    )
    query_texts: list[str] = Field(
        description="Original query texts for reference"
    )
    
    @field_validator('results')
    @classmethod
    def validate_results_structure(cls, v: list[list[SearchResult]]) -> list[list[SearchResult]]:
        """Ensure each query has sorted results by score descending."""
        for query_results in v:
            if len(query_results) > 1:
                scores = [r.score for r in query_results]
                if scores != sorted(scores, reverse=True):
                    raise ValueError("Search results must be sorted by score descending")
        return v


class SimilarityScore(BaseModel):
    """Similarity score between two embeddings."""
    
    score: float = Field(
        description="Similarity value (range depends on metric)"
    )
    metric: str = Field(
        description="Metric used for similarity computation"
    )
    
    def __float__(self) -> float:
        """Allow direct float conversion."""
        return self.score
    
    def __lt__(self, other: SimilarityScore | float) -> bool:
        """Enable comparison operations."""
        other_score = other.score if isinstance(other, SimilarityScore) else other
        return self.score < other_score
    
    def __gt__(self, other: SimilarityScore | float) -> bool:
        """Enable comparison operations."""
        other_score = other.score if isinstance(other, SimilarityScore) else other
        return self.score > other_score
```

```python
# src/embeddify/exceptions.py
from __future__ import annotations

class EmbeddifyError(Exception):
    """Base exception for all Embeddify errors."""
    pass


class ModelLoadError(EmbeddifyError):
    """Raised when model fails to load from specified path."""
    pass


class EncodingError(EmbeddifyError):
    """Raised when text encoding fails."""
    pass


class ValidationError(EmbeddifyError):
    """Raised when Pydantic validation fails with additional context."""
    pass


class SearchError(EmbeddifyError):
    """Raised when semantic search operation fails."""
    pass
```

```python
# src/embeddify/embedder.py
from __future__ import annotations
from pydantic import BaseModel, PrivateAttr, field_validator
from typing import Any
from pathlib import Path
import hashlib
import numpy as np

from .config import EmbedderConfig, RuntimeConfig, load_config_file
from .models import Embedding, EmbeddingResult, SearchResult, SearchResults, SimilarityScore
from .exceptions import ModelLoadError, EncodingError, SearchError


class Embedder(BaseModel):
    """Main interface for embedding generation and semantic operations."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    config: EmbedderConfig = Field(
        description="Model configuration"
    )
    runtime_config: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime execution configuration"
    )
    
    _model: Any = PrivateAttr()
    _cache: dict[str, Embedding] = PrivateAttr(default_factory=dict)
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize SentenceTransformer after Pydantic validation."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self.config.model_path,
                device=self.config.device,
                trust_remote_code=self.config.trust_remote_code
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from {self.config.model_path}: {e}"
            ) from e
    
    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for input texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            EmbeddingResult containing validated embeddings
            
        Raises:
            EncodingError: If encoding fails
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name=self._get_model_name(),
                dimensions=self._model.get_sentence_embedding_dimension()
            )
        
        # Validate inputs
        for i, text in enumerate(texts):
            if text is None or (isinstance(text, str) and not text.strip()):
                raise EncodingError(f"Invalid text at index {i}: empty or None")
        
        embeddings = []
        
        for text in texts:
            # Check cache if enabled
            if self.runtime_config.enable_cache and not self.runtime_config.convert_to_numpy:
                cache_key = hashlib.md5(text.encode()).hexdigest()
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    continue
            
            # Encode text
            try:
                vector = self._model.encode(
                    [text],
                    batch_size=self.runtime_config.batch_size,
                    show_progress_bar=self.runtime_config.show_progress_bar,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=self.runtime_config.convert_to_numpy
                )[0]
                
                # Convert to list if needed
                if isinstance(vector, np.ndarray) and not self.runtime_config.convert_to_numpy:
                    vector = vector.tolist()
                
                embedding = Embedding(
                    vector=vector,
                    text=text,
                    model_name=self._get_model_name(),
                    normalized=self.config.normalize_embeddings
                )
                
                # Cache if enabled
                if self.runtime_config.enable_cache and not self.runtime_config.convert_to_numpy:
                    self._cache[cache_key] = embedding
                
                embeddings.append(embedding)
                
            except Exception as e:
                raise EncodingError(f"Failed to encode text '{text[:50]}...': {e}") from e
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self._get_model_name(),
            dimensions=self._model.get_sentence_embedding_dimension()
        )
    
    def similarity(self, emb1: Embedding, emb2: Embedding) -> SimilarityScore:
        """Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            SimilarityScore with cosine similarity value
            
        Raises:
            ValidationError: If embeddings have different dimensions
        """
        from sentence_transformers import util
        
        # Validate dimensions match
        dim1 = len(emb1.vector) if isinstance(emb1.vector, list) else emb1.vector.size
        dim2 = len(emb2.vector) if isinstance(emb2.vector, list) else emb2.vector.size
        
        if dim1 != dim2:
            raise ValidationError(
                f"Embedding dimension mismatch: {dim1} vs {dim2}"
            )
        
        # Convert to tensors
        vec1 = np.array(emb1.vector) if isinstance(emb1.vector, list) else emb1.vector
        vec2 = np.array(emb2.vector) if isinstance(emb2.vector, list) else emb2.vector
        
        score = util.cos_sim(vec1, vec2).item()
        
        return SimilarityScore(score=score, metric="cosine")
    
    def similarity_batch(
        self, 
        embs1: list[Embedding], 
        embs2: list[Embedding]
    ) -> list[SimilarityScore]:
        """Compute pairwise similarities between two lists of embeddings.
        
        Args:
            embs1: First list of embeddings
            embs2: Second list of embeddings
            
        Returns:
            List of SimilarityScore objects
            
        Raises:
            ValidationError: If lists have different lengths
        """
        if len(embs1) != len(embs2):
            raise ValidationError(
                f"Embedding list length mismatch: {len(embs1)} vs {len(embs2)}"
            )
        
        return [self.similarity(e1, e2) for e1, e2 in zip(embs1, embs2)]
    
    def search(
        self,
        queries: list[str],
        corpus: list[str] | None = None,
        corpus_embeddings: EmbeddingResult | None = None,
        top_k: int = 10,
        score_function: str = "cosine"
    ) -> SearchResults:
        """Perform semantic search on corpus.
        
        Args:
            queries: Query texts to search with
            corpus: Corpus texts (encoded internally if provided)
            corpus_embeddings: Pre-computed corpus embeddings (mutually exclusive with corpus)
            top_k: Number of top results to return per query
            score_function: Similarity metric - "cosine" or "dot"
            
        Returns:
            SearchResults containing results for each query
            
        Raises:
            SearchError: If search fails
            ValidationError: If both corpus and corpus_embeddings provided or neither provided
        """
        from sentence_transformers import util
        
        # Validate inputs
        if corpus is None and corpus_embeddings is None:
            raise ValidationError("Must provide either corpus or corpus_embeddings")
        if corpus is not None and corpus_embeddings is not None:
            raise ValidationError("Cannot provide both corpus and corpus_embeddings")
        
        if not queries:
            return SearchResults(results=[], query_texts=[])
        
        try:
            # Encode queries
            query_result = self.encode(queries)
            query_vectors = [
                np.array(e.vector) if isinstance(e.vector, list) else e.vector
                for e in query_result.embeddings
            ]
            
            # Get corpus embeddings
            if corpus is not None:
                corpus_result = self.encode(corpus)
                corpus_texts = corpus
            else:
                corpus_result = corpus_embeddings
                corpus_texts = [e.text for e in corpus_result.embeddings]
            
            corpus_vectors = [
                np.array(e.vector) if isinstance(e.vector, list) else e.vector
                for e in corpus_result.embeddings
            ]
            
            # Perform search
            search_results = util.semantic_search(
                query_vectors,
                corpus_vectors,
                top_k=top_k,
                score_function=score_function
            )
            
            # Convert to SearchResult objects
            results = []
            for query_hits in search_results:
                query_results = [
                    SearchResult(
                        corpus_id=hit['corpus_id'],
                        score=hit['score'],
                        text=corpus_texts[hit['corpus_id']] if corpus_texts else None
                    )
                    for hit in query_hits
                ]
                results.append(query_results)
            
            return SearchResults(
                results=results,
                query_texts=queries
            )
            
        except Exception as e:
            raise SearchError(f"Semantic search failed: {e}") from e
    
    @classmethod
    def from_config_file(cls, path: str | None = None) -> Embedder:
        """Load embedder from YAML/JSON config file.
        
        Args:
            path: Config file path. If None, reads from EMBEDDIFY_CONFIG_PATH env var.
            
        Returns:
            Configured Embedder instance
            
        Raises:
            FileNotFoundError: If config file not found
            ValidationError: If config validation fails
        """
        config_dict = load_config_file(path)
        
        # Extract model and runtime configs
        model_config = EmbedderConfig(**config_dict.get("model", {}))
        runtime_config = RuntimeConfig(**config_dict.get("runtime", {}))
        
        return cls(config=model_config, runtime_config=runtime_config)
    
    def _get_model_name(self) -> str:
        """Get the model name from the loaded model."""
        # SentenceTransformer stores model name in various ways
        if hasattr(self._model, 'model_card_data') and self._model.model_card_data:
            return self._model.model_card_data.get('model_name', 'unknown')
        return Path(self.config.model_path).name
```

#### **Component Responsibilities**

* **EmbedderConfig:** Validates and stores model initialization parameters. Ensures model path exists and device 
  is valid before model loading.
* **RuntimeConfig:** Manages execution behavior separate from model concerns. Controls batching, progress display, 
  caching, and output format.
* **Embedder:** Main orchestrator wrapping SentenceTransformer. Manages model lifecycle, caching, and provides 
  type-safe methods for all operations.
* **Embedding/EmbeddingResult:** Immutable result containers with validation. Store vectors with metadata for 
  traceability.
* **SearchResult/SearchResults:** Structured search outputs with corpus mapping. Maintain type safety through 
  search pipeline.
* **SimilarityScore:** Wrapper for similarity values with comparison operators. Enables natural score comparisons.
* **Exception hierarchy:** Clear error categorization for different failure modes. Preserves original exceptions 
  as __cause__.

#### **Data Flow**

**Encoding Pipeline:**
1. Input validation: Check texts non-empty, non-None
2. Cache lookup: If enabled and convert_to_numpy=False, check cache by MD5 hash
3. Batch processing: Split by batch_size, show progress if configured
4. SentenceTransformer.encode(): Call underlying library
5. Normalization: Apply if normalize_embeddings=True
6. Result wrapping: Create Embedding objects with metadata
7. Cache storage: Store if enabled
8. Return EmbeddingResult

**Search Pipeline:**
1. Input validation: Exactly one of corpus or corpus_embeddings
2. Query encoding: Encode queries via encode() method (with caching)
3. Corpus encoding: Encode corpus OR use pre-computed embeddings
4. util.semantic_search(): Delegate to sentence-transformers
5. Result mapping: Map corpus_ids to SearchResult objects with text
6. Result grouping: Group by query, sort by score descending
7. Return SearchResults

**Configuration Loading Pipeline:**
1. File reading: Load YAML/JSON from path or env var
2. Environment override: Apply EMBEDDIFY_MODEL_PATH, EMBEDDIFY_DEVICE overrides
3. Config parsing: Extract model and runtime sections
4. Pydantic validation: Validate via EmbedderConfig and RuntimeConfig
5. Embedder creation: Instantiate Embedder with validated configs
6. Model initialization: Load SentenceTransformer in model_post_init

### **Design Patterns**

* **Composition over Inheritance:** Embedder composes EmbedderConfig and RuntimeConfig rather than inheriting. 
  Allows independent evolution of configuration and execution logic.
* **Facade Pattern:** Embedder provides simplified interface to sentence-transformers complexity. Hides internal 
  caching, batching, and validation details.
* **Factory Method:** from_config_file() class method constructs Embedder from external configuration. Separates 
  object creation from usage.
* **Immutable Results:** All result objects (Embedding, EmbeddingResult, etc.) are Pydantic models without 
  mutating methods. Prevents accidental modification.
* **Validation Boundary:** All inputs validated at API boundaries via Pydantic. Errors caught before reaching 
  sentence-transformers.

## **Data Structures**

### **Input/Output Schemas**

```python
# src/embeddify/models.py (complete definitions above)

# Example usage showing input/output flow:
embedder = Embedder(config=EmbedderConfig(model_path="/models/model"))

# Input: list[str]
texts = ["hello", "world"]

# Output: EmbeddingResult
result = embedder.encode(texts)
# result.embeddings: list[Embedding]
# result.model_name: str
# result.dimensions: int

# Individual embedding access:
emb = result.embeddings[0]
# emb.vector: list[float] | np.ndarray
# emb.text: str | None
# emb.model_name: str
# emb.normalized: bool
```

### **Internal Data Models**

```python
# Cache structure (internal to Embedder)
_cache: dict[str, Embedding]
# Key: MD5 hash of text
# Value: Cached Embedding object
# Only used when enable_cache=True and convert_to_numpy=False

# Example:
# _cache = {
#     "5d41402abc4b2a76b9719d911017c592": Embedding(...),  # "hello"
#     "7d793037a0760186574b0282f2f435e7": Embedding(...)   # "world"
# }
```

### **Validation Rules**

* **EmbedderConfig.model_path:** Must be existing filesystem path
* **EmbedderConfig.device:** Must be "cpu", "cuda", or "cuda:N" format. CUDA availability checked if requested.
* **RuntimeConfig.batch_size:** Must be positive integer (>0)
* **Embedding.vector:** Must be non-empty list or array
* **EmbeddingResult.embeddings:** All embeddings must have identical dimensions
* **SearchResults.results:** Each query's results must be sorted by score descending
* **SimilarityScore.score:** For cosine similarity, should be in range [-1, 1]

## **API Specification**

### **Public Interface**

```python
# src/embeddify/__init__.py
from __future__ import annotations
from .embedder import Embedder
from .config import EmbedderConfig, RuntimeConfig, load_config_file
from .models import (
    Embedding, 
    EmbeddingResult, 
    SearchResult, 
    SearchResults, 
    SimilarityScore
)
from .exceptions import (
    EmbeddifyError,
    ModelLoadError,
    EncodingError,
    ValidationError,
    SearchError
)

__version__ = "0.1.0"

__all__ = [
    "Embedder",
    "EmbedderConfig",
    "RuntimeConfig",
    "load_config_file",
    "Embedding",
    "EmbeddingResult",
    "SearchResult",
    "SearchResults",
    "SimilarityScore",
    "EmbeddifyError",
    "ModelLoadError",
    "EncodingError",
    "ValidationError",
    "SearchError",
]
```

### **Configuration**

```yaml
# Example config file: embeddify_config.yaml
model:
  path: "/models/all-MiniLM-L6-v2"
  device: "cuda"
  normalize_embeddings: true
  trust_remote_code: false

runtime:
  batch_size: 32
  show_progress_bar: false
  enable_cache: true
  convert_to_numpy: false
```

### **Usage Examples**

```python
# Basic usage
from embeddify import Embedder, EmbedderConfig

config = EmbedderConfig(model_path="/models/all-MiniLM-L6-v2")
embedder = Embedder(config=config)

texts = ["Hello world", "Goodbye world"]
result = embedder.encode(texts)

print(f"Dimensions: {result.dimensions}")  # 384
print(f"First vector: {result.embeddings[0].vector[:5]}")

# Load from config file
embedder = Embedder.from_config_file("config.yaml")

# Semantic search
queries = ["greeting"]
corpus = ["hello there", "goodbye", "morning greetings"]
search_results = embedder.search(queries, corpus, top_k=2)

for hit in search_results.results[0]:
    print(f"Score: {hit.score:.4f} - {hit.text}")

# Pre-computed corpus embeddings
corpus_embeddings = embedder.encode(corpus)
search_results = embedder.search(
    queries, 
    corpus_embeddings=corpus_embeddings,
    top_k=2
)

# Similarity computation
emb1 = embedder.encode(["cat"]).embeddings[0]
emb2 = embedder.encode(["dog"]).embeddings[0]
similarity = embedder.similarity(emb1, emb2)
print(f"Similarity: {similarity.score:.4f}")

# Batch similarity
embeddings_a = embedder.encode(["cat", "dog", "bird"])
embeddings_b = embedder.encode(["kitten", "puppy", "eagle"])
similarities = embedder.similarity_batch(
    embeddings_a.embeddings,
    embeddings_b.embeddings
)
for sim in similarities:
    print(f"Score: {sim.score:.4f}")

# With runtime config
from embeddify import RuntimeConfig

runtime_config = RuntimeConfig(
    batch_size=16,
    show_progress_bar=True,
    enable_cache=True
)
embedder = Embedder(config=config, runtime_config=runtime_config)

# Environment variable usage
import os
os.environ["EMBEDDIFY_MODEL_PATH"] = "/models/custom-model"
os.environ["EMBEDDIFY_DEVICE"] = "cuda:0"
embedder = Embedder.from_config_file("config.yaml")  # Overrides applied
```

## **Error Handling**

### **Exception Hierarchy**

```python
# src/embeddify/exceptions.py (complete definitions above)

EmbeddifyError (base)
├── ModelLoadError
├── EncodingError
├── ValidationError
└── SearchError
```

### **Error Scenarios**

* **Invalid Model Path:** ModelLoadError raised during Embedder initialization if model_path doesn't exist or 
  model fails to load. Original exception preserved in __cause__.
* **CUDA Unavailable:** ValidationError raised during EmbedderConfig validation if CUDA requested but not available.
* **Empty/None Texts:** EncodingError raised during encode() if any text is None or empty string.
* **Dimension Mismatch:** ValidationError raised during similarity() if embeddings have different dimensions.
* **Missing Corpus:** ValidationError raised during search() if neither corpus nor corpus_embeddings provided.
* **Both Corpus Provided:** ValidationError raised during search() if both corpus and corpus_embeddings provided.
* **Config File Missing:** FileNotFoundError raised during from_config_file() if config file not found.
* **Invalid Config Format:** ValueError raised during load_config_file() if file is neither valid YAML nor JSON.

### **Error Messages**

All error messages follow this structure:
1. Clear description of what failed
2. Context about the failure (file path, text snippet, dimension values)
3. Actionable guidance for fixing the issue

Examples:
```
ModelLoadError: Failed to load model from /models/invalid: [Errno 2] No such file or directory
EncodingError: Invalid text at index 2: empty or None
ValidationError: Embedding dimension mismatch: 384 vs 768
SearchError: Semantic search failed: Must provide either corpus or corpus_embeddings
```

## **Dependencies**

### **Core Dependencies**

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic>=2.0",
#     "sentence-transformers>=2.0",
#     "torch>=2.0",
#     "transformers>=4.0",
#     "numpy>=1.20",
#     "pyyaml>=6.0",
# ]
# ///
```

### **Optional Dependencies**

* **python-dotenv>=1.0:** Environment variable management from .env files
* **tqdm>=4.60:** Progress bars for batch processing (auto-installed with sentence-transformers)

### **Development Dependencies**

* Testing: pytest>=7.0, pytest-cov>=4.0
* Linting: ruff>=0.1.0
* Type checking: mypy>=1.0
* Documentation: mkdocs>=1.5, mkdocs-material>=9.0

## **Testing Strategy**

### **Unit Tests**

```python
# tests/test_config.py
from __future__ import annotations
import pytest
from pathlib import Path
from embeddify import EmbedderConfig, RuntimeConfig, ValidationError


class TestEmbedderConfig:
    def test_valid_config(self, tmp_path: Path):
        """Test valid configuration creation."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        config = EmbedderConfig(model_path=str(model_path))
        
        assert config.model_path == str(model_path)
        assert config.device == "cpu"
        assert config.normalize_embeddings is True
        assert config.trust_remote_code is False
    
    def test_invalid_model_path(self):
        """Test validation fails for non-existent model path."""
        with pytest.raises(ValueError, match="Model path does not exist"):
            EmbedderConfig(model_path="/nonexistent/path")
    
    def test_invalid_device(self, tmp_path: Path):
        """Test validation fails for invalid device string."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with pytest.raises(ValueError, match="Invalid device"):
            EmbedderConfig(model_path=str(model_path), device="gpu")
    
    def test_cuda_availability_check(self, tmp_path: Path, monkeypatch):
        """Test CUDA availability validation."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        # Mock torch.cuda.is_available to return False
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        
        with pytest.raises(ValueError, match="CUDA requested but not available"):
            EmbedderConfig(model_path=str(model_path), device="cuda")


class TestRuntimeConfig:
    def test_default_values(self):
        """Test default runtime configuration values."""
        config = RuntimeConfig()
        
        assert config.batch_size == 32
        assert config.show_progress_bar is False
        assert config.enable_cache is False
        assert config.convert_to_numpy is False
    
    def test_invalid_batch_size(self):
        """Test validation fails for invalid batch size."""
        with pytest.raises(ValueError):
            RuntimeConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            RuntimeConfig(batch_size=-1)
```

```python
# tests/test_embedder.py
from __future__ import annotations
import pytest
import numpy as np
from embeddify import Embedder, EmbedderConfig, RuntimeConfig
from embeddify import EncodingError, SearchError, ValidationError


class TestEmbedder:
    @pytest.fixture
    def embedder(self, tmp_path):
        """Create embedder with test model."""
        # This assumes you have a test model available
        model_path = tmp_path / "test_model"
        # Setup test model here
        config = EmbedderConfig(model_path=str(model_path))
        return Embedder(config=config)
    
    def test_encode_single_text(self, embedder):
        """Test encoding a single text."""
        result = embedder.encode(["Hello world"])
        
        assert len(result.embeddings) == 1
        assert result.dimensions > 0
        assert result.embeddings[0].text == "Hello world"
        assert result.embeddings[0].model_name is not None
    
    def test_encode_multiple_texts(self, embedder):
        """Test encoding multiple texts."""
        texts = ["Hello", "World", "Test"]
        result = embedder.encode(texts)
        
        assert len(result.embeddings) == 3
        for i, emb in enumerate(result.embeddings):
            assert emb.text == texts[i]
            assert len(emb.vector) == result.dimensions
    
    def test_encode_empty_list(self, embedder):
        """Test encoding empty list returns empty result."""
        result = embedder.encode([])
        
        assert len(result.embeddings) == 0
        assert result.dimensions > 0
    
    def test_encode_empty_text_raises_error(self, embedder):
        """Test encoding empty text raises EncodingError."""
        with pytest.raises(EncodingError, match="empty or None"):
            embedder.encode([""])
    
    def test_encode_none_text_raises_error(self, embedder):
        """Test encoding None raises EncodingError."""
        with pytest.raises(EncodingError, match="empty or None"):
            embedder.encode([None])
    
    def test_encode_with_caching(self, tmp_path):
        """Test caching functionality."""
        model_path = tmp_path / "test_model"
        config = EmbedderConfig(model_path=str(model_path))
        runtime_config = RuntimeConfig(enable_cache=True)
        embedder = Embedder(config=config, runtime_config=runtime_config)
        
        # First encoding
        result1 = embedder.encode(["Test text"])
        
        # Second encoding should use cache
        result2 = embedder.encode(["Test text"])
        
        assert len(embedder._cache) == 1
        assert result1.embeddings[0].vector == result2.embeddings[0].vector
    
    def test_similarity(self, embedder):
        """Test similarity computation."""
        emb1 = embedder.encode(["cat"]).embeddings[0]
        emb2 = embedder.encode(["kitten"]).embeddings[0]
        
        similarity = embedder.similarity(emb1, emb2)
        
        assert -1 <= similarity.score <= 1
        assert similarity.metric == "cosine"
    
    def test_similarity_dimension_mismatch(self, embedder):
        """Test similarity raises error for dimension mismatch."""
        # Mock embeddings with different dimensions
        from embeddify.models import Embedding
        
        emb1 = Embedding(
            vector=[1.0] * 384,
            model_name="test",
            normalized=True
        )
        emb2 = Embedding(
            vector=[1.0] * 768,
            model_name="test",
            normalized=True
        )
        
        with pytest.raises(ValidationError, match="dimension mismatch"):
            embedder.similarity(emb1, emb2)
    
    def test_search_with_corpus(self, embedder):
        """Test semantic search with corpus texts."""
        queries = ["greeting"]
        corpus = ["hello there", "goodbye", "good morning"]
        
        results = embedder.search(queries, corpus, top_k=2)
        
        assert len(results.results) == 1
        assert len(results.results[0]) == 2
        assert results.query_texts == queries
        assert all(r.text in corpus for r in results.results[0])
    
    def test_search_with_precomputed_embeddings(self, embedder):
        """Test semantic search with pre-computed embeddings."""
        queries = ["greeting"]
        corpus = ["hello there", "goodbye", "good morning"]
        
        corpus_embeddings = embedder.encode(corpus)
        results = embedder.search(
            queries,
            corpus_embeddings=corpus_embeddings,
            top_k=2
        )
        
        assert len(results.results) == 1
        assert len(results.results[0]) == 2
    
    def test_search_missing_corpus_raises_error(self, embedder):
        """Test search raises error when neither corpus nor embeddings provided."""
        with pytest.raises(ValidationError, match="Must provide either corpus"):
            embedder.search(["query"])
    
    def test_search_both_corpus_raises_error(self, embedder):
        """Test search raises error when both corpus and embeddings provided."""
        corpus = ["text"]
        corpus_embeddings = embedder.encode(corpus)
        
        with pytest.raises(ValidationError, match="Cannot provide both"):
            embedder.search(
                ["query"],
                corpus=corpus,
                corpus_embeddings=corpus_embeddings
            )
```

```python
# tests/test_models.py
from __future__ import annotations
import pytest
import numpy as np
from embeddify.models import Embedding, EmbeddingResult, SearchResult, SearchResults


class TestEmbedding:
    def test_valid_embedding_list(self):
        """Test creating embedding with list vector."""
        emb = Embedding(
            vector=[1.0, 2.0, 3.0],
            model_name="test-model",
            normalized=True
        )
        
        assert len(emb.vector) == 3
        assert emb.model_name == "test-model"
        assert emb.normalized is True
        assert emb.text is None
    
    def test_valid_embedding_numpy(self):
        """Test creating embedding with numpy array."""
        emb = Embedding(
            vector=np.array([1.0, 2.0, 3.0]),
            model_name="test-model",
            normalized=True
        )
        
        assert emb.vector.size == 3
    
    def test_empty_vector_raises_error(self):
        """Test empty vector raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Embedding(
                vector=[],
                model_name="test",
                normalized=True
            )


class TestEmbeddingResult:
    def test_valid_result(self):
        """Test creating valid embedding result."""
        embeddings = [
            Embedding(vector=[1.0, 2.0], model_name="test", normalized=True),
            Embedding(vector=[3.0, 4.0], model_name="test", normalized=True)
        ]
        
        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test",
            dimensions=2
        )
        
        assert len(result.embeddings) == 2
        assert result.dimensions == 2
    
    def test_inconsistent_dimensions_raises_error(self):
        """Test inconsistent dimensions raises validation error."""
        embeddings = [
            Embedding(vector=[1.0, 2.0], model_name="test", normalized=True),
            Embedding(vector=[3.0, 4.0, 5.0], model_name="test", normalized=True)
        ]
        
        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            EmbeddingResult(
                embeddings=embeddings,
                model_name="test",
                dimensions=2
            )


class TestSearchResults:
    def test_valid_search_results(self):
        """Test creating valid search results."""
        results = [
            [
                SearchResult(corpus_id=0, score=0.9, text="hello"),
                SearchResult(corpus_id=1, score=0.7, text="hi")
            ]
        ]
        
        search_results = SearchResults(
            results=results,
            query_texts=["greeting"]
        )
        
        assert len(search_results.results) == 1
        assert len(search_results.results[0]) == 2
    
    def test_unsorted_results_raises_error(self):
        """Test unsorted results raise validation error."""
        results = [
            [
                SearchResult(corpus_id=0, score=0.7, text="hello"),
                SearchResult(corpus_id=1, score=0.9, text="hi")  # Higher score after lower
            ]
        ]
        
        with pytest.raises(ValueError, match="sorted by score descending"):
            SearchResults(
                results=results,
                query_texts=["greeting"]
            )
```

### **Integration Tests**

* **Full Encoding Pipeline:** Load model from config file, encode texts, verify embeddings have correct dimensions 
  and metadata
* **Search with Pre-computation:** Encode corpus once, perform multiple searches with different queries, verify 
  results correctness
* **Caching Behavior:** Enable caching, encode same texts multiple times, verify cache hits and performance 
  improvement
* **Multi-Query Search:** Search with multiple queries, verify each query gets independent top_k results
* **Config File Loading:** Create config file with various settings, load via from_config_file(), verify all 
  settings applied correctly
* **Environment Override:** Set env vars, load config file, verify env vars override file values

### **Test Coverage Requirements**

* Minimum 90% line coverage across all modules
* 100% coverage for:
  - All validation methods
  - Error handling paths
  - Configuration loading logic
* Edge case coverage for:
  - Empty inputs
  - None values
  - Dimension mismatches
  - Invalid device strings
  - Missing config files

## **Implementation Guidelines**

### **File Organization**

```
embeddify/
├── src/
│   └── embeddify/
│       ├── __init__.py
│       ├── embedder.py
│       ├── config.py
│       ├── models.py
│       ├── exceptions.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_embedder.py
│   ├── test_models.py
│   └── fixtures/
│       └── models/
│           └── test-model/
├── docs/
│   ├── index.md
│   ├── api-reference.md
│   ├── configuration.md
│   └── examples.md
├── examples/
│   ├── basic_usage.py
│   ├── semantic_search.py
│   ├── faq_system.py
│   └── config_file_example.yaml
├── pyproject.toml
├── README.md
├── LICENSE
└── SPEC.md
```

### **Development Workflow**

1. Create feature branch from main
2. Implement feature with comprehensive tests (TDD approach)
3. Run test suite: `uv run pytest --cov=embeddify`
4. Run type checking: `uv run mypy src/embeddify`
5. Run linting: `uv run ruff check src/ tests/`
6. Format code: `uv run ruff format src/ tests/`
7. Update documentation if public API changed
8. Commit with descriptive message following conventional commits
9. Push and create pull request

## **Security Considerations**

* **Input Validation:** All text inputs validated via Pydantic. Empty strings and None values rejected.
* **Path Validation:** Model paths validated to exist before loading. No path traversal vulnerabilities.
* **Trust Remote Code:** Defaults to False. Users must explicitly enable for custom model architectures.
* **Device Validation:** CUDA availability checked before attempting to use GPU. Prevents crashes.

## **Monitoring and Logging**

* **Logging Levels:** 
  - DEBUG: Cache hits/misses, batch processing details
  - INFO: Model loading, encoding operations
  - WARNING: CUDA unavailable fallback, cache disabled due to convert_to_numpy
  - ERROR: All exception scenarios with stack traces

## **Migration and Versioning**

* **Semantic Versioning:** Major.minor.patch following semver.org
* **Breaking Changes:** Allowed with major version bumps and documented in CHANGELOG
* **Backward Compatibility:** No guarantees for personal library usage
* **Migration Guides:** Document API changes between major versions in docs/

---

## **Implementation Checklist**

* [ ] EmbedderConfig implemented with path and device validation
* [ ] RuntimeConfig implemented with batch and cache settings
* [ ] Embedder class with _model and _cache as PrivateAttr
* [ ] encode() method with caching and batch processing
* [ ] similarity() and similarity_batch() methods
* [ ] search() method with corpus/corpus_embeddings handling
* [ ] from_config_file() class method
* [ ] load_config_file() function with YAML/JSON support
* [ ] Environment variable overrides (EMBEDDIFY_MODEL_PATH, EMBEDDIFY_DEVICE)
* [ ] Embedding, EmbeddingResult, SearchResult, SearchResults models
* [ ] SimilarityScore with comparison operators
* [ ] Exception hierarchy (EmbeddifyError, ModelLoadError, EncodingError, ValidationError, SearchError)
* [ ] Unit tests for all components (90%+ coverage)
* [ ] Integration tests for full workflows
* [ ] Type hints complete for all public APIs
* [ ] Documentation with usage examples
* [ ] README with installation and quick start
* [ ] Example config files and usage scripts
