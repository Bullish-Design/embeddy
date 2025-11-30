# /mnt/user-data/outputs/11_search_basic.md

# Step 11: Semantic Search with Corpus Encoding

**Goal**: Implement search() method that encodes corpus on-the-fly.

**Context**: First version of search that accepts corpus texts and encodes them internally. Pre-computed embeddings will be added in Step 12.

**Contribution**: Enables semantic search functionality with automatic corpus encoding.

---

## Prompt

```
Add basic search() method to Embedder class, following TDD.

Files to modify:
- src/embeddify/embedder.py (add search method)
- tests/test_embedder.py (add search tests)

Requirements:

1. search method signature:
   - search(queries: list[str], corpus: list[str], top_k: int = 5, 
           score_function: str = "cosine") -> SearchResults
   - For now, only support corpus parameter (corpus_embeddings in Step 12)
   
2. Behavior:
   - Encode queries using self.encode()
   - Encode corpus using self.encode()
   - For each query embedding:
     - Compute similarity with all corpus embeddings
     - Find top_k highest scores
     - Create SearchResult for each hit
   - Return SearchResults with results per query

3. SearchResult contents:
   - corpus_id: index in corpus list
   - score: similarity score
   - text: corpus[corpus_id]

4. Validation:
   - Raise SearchError for search failures
   - Raise ValidationError if top_k < 1 or top_k > len(corpus)
   - Raise ValidationError if score_function not in ["cosine", "dot"]
   - Handle empty queries or corpus appropriately

Implementation approach:
1. Write tests first:
   - Test basic search with queries and corpus
   - Test top_k controls number of results
   - Test results sorted by score descending
   - Test empty queries returns empty results
   - Test empty corpus returns empty results
   - Test score_function="dot" works
   - Test SearchResults includes query_texts

2. Implement:
   - Add search() method
   - Encode queries and corpus
   - Compute similarity matrix (queries x corpus)
   - Find top_k for each query
   - Build SearchResult objects
   - Return SearchResults

3. Verify:
   - All tests pass
   - Results are correctly sorted
   - All metadata populated

File requirements:
- Add comprehensive docstring
- Keep search logic organized

Create updated test file as artifact.
```
