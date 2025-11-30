# /mnt/user-data/outputs/10_similarity.md

# Step 10: Similarity Computation

**Goal**: Implement similarity() and similarity_batch() methods for computing cosine similarity between embeddings.

**Context**: Used by search() and available as a standalone utility for comparing embeddings.

**Contribution**: Provides similarity computation infrastructure for search and general use.

---

## Prompt

```
Add similarity methods to Embedder class, following TDD.

Files to modify:
- src/embeddify/embedder.py (add similarity methods)
- tests/test_embedder.py (add similarity tests)

Requirements:

1. similarity method:
   - Signature: similarity(emb1: Embedding, emb2: Embedding, metric: str = "cosine") -> SimilarityScore
   - Compute similarity between two embeddings
   - Support metrics: "cosine", "dot"
   - Validate dimensions match
   - Return SimilarityScore object

2. similarity_batch method:
   - Signature: similarity_batch(embs1: list[Embedding], embs2: list[Embedding], 
                                  metric: str = "cosine") -> list[SimilarityScore]
   - Compute pairwise similarities (embs1[i] vs embs2[i])
   - Lists must have same length
   - Return list of SimilarityScore objects

3. Similarity computation:
   - Cosine: use sklearn.metrics.pairwise.cosine_similarity or manual computation
   - Dot: simple dot product
   - Handle both list[float] and np.ndarray vectors

4. Validation:
   - Raise ValidationError if dimensions don't match
   - Raise ValidationError if metric not in ["cosine", "dot"]
   - Raise ValidationError if batch lists have different lengths

Implementation approach:
1. Write tests first:
   - Test similarity with matching dimensions
   - Test similarity with mismatched dimensions raises ValidationError
   - Test cosine similarity produces expected values
   - Test dot product similarity
   - Test similarity_batch with valid inputs
   - Test similarity_batch with length mismatch raises ValidationError
   - Test both list[float] and np.ndarray inputs work

2. Implement:
   - Add similarity() method with metric selection
   - Add similarity_batch() method
   - Add dimension validation helper
   - Use numpy for efficient computation

3. Verify:
   - All tests pass
   - Similarity values are correct
   - Both metrics work

Dependencies:
- numpy for vectorized computation
- Optional: sklearn for cosine_similarity

File requirements:
- Add comprehensive docstrings
- Keep computation logic clear

Create updated test file as artifact.
```
