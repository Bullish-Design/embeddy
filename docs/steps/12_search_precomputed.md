# /mnt/user-data/outputs/12_search_precomputed.md

# Step 12: Search with Pre-computed Embeddings

**Goal**: Extend search() to support pre-computed corpus embeddings for efficiency.

**Context**: Allows users to encode corpus once and reuse for multiple searches, significantly improving performance.

**Contribution**: Completes search functionality with performance optimization path.

---

## Prompt

```
Enhance search() method to support pre-computed corpus embeddings, following TDD.

Files to modify:
- src/embeddify/embedder.py (enhance search method)
- tests/test_embedder.py (add pre-computed embedding tests)

Requirements:

1. Updated search signature:
   - search(queries: list[str], corpus: list[str] | None = None, 
           corpus_embeddings: EmbeddingResult | None = None,
           top_k: int = 5, score_function: str = "cosine") -> SearchResults
   
2. Behavior:
   - Exactly one of corpus or corpus_embeddings must be provided
   - If corpus provided: encode it (existing behavior)
   - If corpus_embeddings provided: use directly
   - SearchResult.text is None when using corpus_embeddings (no corpus text available)

3. Validation:
   - Raise ValidationError if both corpus and corpus_embeddings provided
   - Raise ValidationError if neither corpus nor corpus_embeddings provided

Implementation approach:
1. Write tests first:
   - Test search with corpus_embeddings parameter
   - Test search with both corpus and corpus_embeddings raises ValidationError
   - Test search with neither raises ValidationError
   - Test SearchResult.text is None when using corpus_embeddings
   - Test pre-computed search produces same results as on-the-fly encoding
   - Test performance improvement with pre-computed embeddings (optional)

2. Implement:
   - Modify search() signature
   - Add mutual exclusivity validation
   - Branch logic: use corpus_embeddings if provided, else encode corpus
   - Handle SearchResult.text appropriately

3. Verify:
   - All tests pass (new and existing)
   - Both code paths work correctly
   - Validation catches invalid combinations

File requirements:
- Update docstring to document both parameters
- Keep logic clear

Create updated test file as artifact.
```
