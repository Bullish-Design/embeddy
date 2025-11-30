# /mnt/user-data/outputs/09_encode_caching.md

# Step 9: Add Caching to Encode

**Goal**: Implement in-memory caching for encode() to avoid re-computing identical texts.

**Context**: Caching significantly improves performance when encoding repeated texts. Cache is keyed by text hash and respects runtime_config.enable_cache.

**Contribution**: Adds performance optimization while maintaining correctness.

---

## Prompt

```
Add caching functionality to the encode() method, following TDD.

Files to modify:
- src/embeddify/embedder.py (enhance encode with caching)
- tests/test_embedder.py (add caching tests)

Requirements:

1. Caching behavior:
   - Only cache when runtime_config.enable_cache=True
   - Cache disabled when runtime_config.convert_to_numpy=True (dict key limitation)
   - Cache key: hash of text string
   - Store Embedding objects in self._cache
   - Look up cache before encoding, return cached if found
   - Store newly computed embeddings in cache

2. Cache structure:
   - self._cache: dict[str, Embedding]
   - Key: text (the string itself, not hash - simpler)
   - Value: Embedding object

3. Behavior:
   - When encode() called, check each text in cache first
   - Separate texts into cached and uncached
   - Encode only uncached texts
   - Combine cached and newly encoded results
   - Maintain original order

4. Add cache management method:
   - clear_cache() -> None: Empties self._cache

Implementation approach:
1. Write tests first:
   - Test cache disabled by default (runtime_config.enable_cache=False)
   - Test cache enabled returns same object on second call
   - Test cache bypassed when convert_to_numpy=True
   - Test clear_cache() works
   - Test cache with mixed cached/uncached texts maintains order
   - Test cache is per-Embedder instance

2. Implement:
   - Modify encode() to check cache before encoding
   - Add cache storage after encoding
   - Add cache bypass for convert_to_numpy
   - Add clear_cache() method

3. Verify:
   - All tests pass (new and existing)
   - Performance improves with caching
   - Cache correctness maintained

File requirements:
- Add logging for cache hits/misses at DEBUG level
- Keep caching logic separate and clear

Create updated test file as artifact.
```
