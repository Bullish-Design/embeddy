# /mnt/user-data/outputs/02_basic_models.md

# Step 2: Basic Result Models

**Goal**: Implement foundational Pydantic models for embeddings and similarity scores that will be used throughout the library.

**Context**: These models provide type-safe containers for results. They include validation logic (e.g., embedding vectors can't be empty, dimensions must match) and metadata tracking.

**Contribution**: Establishes the data structures that encode(), similarity(), and search() methods will return.

---

## Prompt

```
Create the basic result models for Embeddify using Pydantic, following TDD.

Project structure:
- src/embeddify/models.py (Pydantic models)
- tests/test_models.py (test suite)

Requirements:

1. Embedding model (represents a single embedding vector):
   - vector: list[float] | np.ndarray (the embedding vector)
   - model_name: str (model that generated this embedding)
   - normalized: bool (whether vector is L2-normalized)
   - text: str | None = None (optional source text)
   - Validation: vector cannot be empty
   - Property: dimensions (returns len(vector))

2. SimilarityScore model (represents similarity between two embeddings):
   - score: float (similarity value, typically -1 to 1 for cosine)
   - metric: str = "cosine" (similarity metric used)
   - Implement comparison operators (__lt__, __le__, __gt__, __ge__, __eq__) based on score
   - Validation: metric must be one of ["cosine", "dot"]

Implementation approach:
1. Write tests first:
   - Test Embedding creation with list[float] and np.ndarray vectors
   - Test Embedding.dimensions property
   - Test empty vector raises ValidationError
   - Test SimilarityScore creation and comparison operators
   - Test SimilarityScore with invalid metric raises ValidationError

2. Implement models.py:
   - Import necessary types from typing and numpy (for type hints)
   - Use Pydantic BaseModel with model_config for strict validation
   - Add custom validators using @field_validator where needed
   - Implement comparison operators for SimilarityScore

3. Verify:
   - All tests pass
   - Type hints complete using `from __future__ import annotations`
   - Models work with both list[float] and np.ndarray seamlessly

Dependencies:
- pydantic>=2.0
- numpy>=1.20

File requirements:
- Filepath comment at top
- from __future__ import annotations
- Lines under 120 characters
- Organize imports: future, stdlib, third-party (pydantic, numpy), local

Create both files as separate artifacts.
```
