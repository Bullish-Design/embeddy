# /mnt/user-data/outputs/03_result_models.md

# Step 3: Complex Result Models

**Goal**: Implement EmbeddingResult and SearchResults models with cross-field validation.

**Context**: These models wrap collections of Embedding/SearchResult objects and validate consistency (e.g., all embeddings have same dimensions, search results are properly sorted).

**Contribution**: Provides validated containers for batch operations, ensuring data integrity.

---

## Prompt

```
Extend the models.py file with collection result models, following TDD.

Files to modify:
- src/embeddify/models.py (add new models)
- tests/test_models.py (add new tests)

Requirements:

1. EmbeddingResult model (batch encoding results):
   - embeddings: list[Embedding]
   - model_name: str
   - dimensions: int
   - Validation: All embeddings must have dimensions matching the dimensions field
   - Property: count (returns len(embeddings))

2. SearchResult model (single search hit):
   - corpus_id: int (index in corpus)
   - score: float (similarity score)
   - text: str | None = None (optional corpus text)
   - Validation: corpus_id >= 0, score is finite

3. SearchResults model (search output for all queries):
   - results: list[list[SearchResult]] (results per query)
   - query_texts: list[str] | None = None (optional query texts)
   - Validation: Each query's results must be sorted by score descending
   - Property: num_queries (returns len(results))

Implementation approach:
1. Write tests first:
   - Test EmbeddingResult with consistent/inconsistent dimensions
   - Test EmbeddingResult.count property
   - Test SearchResult validation (negative corpus_id, non-finite score)
   - Test SearchResults with sorted/unsorted results
   - Test SearchResults.num_queries property

2. Implement models:
   - Add new classes to models.py
   - Use @model_validator for cross-field validation (e.g., dimension consistency)
   - Keep validation logic clear and error messages actionable

3. Verify:
   - All tests pass (new and existing)
   - Validation errors provide helpful messages
   - Models integrate with existing Embedding/SimilarityScore

File requirements:
- Maintain existing code structure
- Add imports as needed (e.g., math.isfinite for validation)
- Keep validators focused and well-documented

Create updated test file as artifact (models.py changes can be described or shown).
```
