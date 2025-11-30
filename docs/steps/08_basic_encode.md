# /mnt/user-data/outputs/08_basic_encode.md

# Step 8: Basic Encode Method (No Caching)

**Goal**: Implement the encode() method for generating embeddings without caching.

**Context**: Core functionality for converting text to vectors. Handles single strings and lists, applies normalization, returns validated EmbeddingResult.

**Contribution**: Provides the fundamental text encoding capability.

---

## Prompt

```
Add the encode() method to Embedder class, following TDD.

Files to modify:
- src/embeddify/embedder.py (add encode method)
- tests/test_embedder.py (add encode tests)

Requirements:

1. encode method signature:
   - encode(texts: str | list[str]) -> EmbeddingResult
   - Accept single string or list of strings
   - Normalize to list internally if single string provided

2. Behavior:
   - Use self._model.encode() from SentenceTransformer
   - Pass normalize_embeddings from config
   - Handle batch_size and show_progress_bar from runtime_config
   - Convert output to list[Embedding] objects
   - Return EmbeddingResult with embeddings, model_name, dimensions

3. Error handling:
   - Raise EncodingError for encoding failures
   - Validate inputs (no None, no empty strings)
   - Wrap SentenceTransformer exceptions

4. Notes:
   - Ignore caching for now (Step 9 adds it)
   - convert_to_numpy handling can be basic (store as is)

Implementation approach:
1. Write tests first:
   - Test encoding single text
   - Test encoding list of texts
   - Test empty list returns empty EmbeddingResult
   - Test None in text list raises EncodingError
   - Test empty string raises EncodingError
   - Test result has correct dimensions and model_name
   - Test normalization is applied when config.normalize_embeddings=True

2. Implement:
   - Add encode() method to Embedder
   - Add input validation
   - Call self._model.encode() with appropriate parameters
   - Create Embedding objects from results
   - Build and return EmbeddingResult

3. Verify:
   - All tests pass
   - Works with actual SentenceTransformer model
   - Error messages are clear

Dependencies:
- numpy (for handling model output)

File requirements:
- Maintain code organization
- Add comprehensive docstring
- Keep method focused

Create updated test file as artifact.
```
