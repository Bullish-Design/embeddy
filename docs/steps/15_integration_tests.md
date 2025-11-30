# /mnt/user-data/outputs/15_integration_tests.md

# Step 15: Integration Tests

**Goal**: Create comprehensive integration tests that validate the entire system working together.

**Context**: Unit tests verify individual components; integration tests ensure they work together correctly in real-world scenarios.

**Contribution**: Provides confidence that the complete library functions as expected end-to-end.

---

## Prompt

```
Create integration test suite to validate complete workflows.

Files to create:
- tests/test_integration.py (integration tests)

Requirements:

1. Test scenarios:
   
   a) Full encoding pipeline:
      - Load embedder from config file
      - Encode multiple texts
      - Verify embeddings have correct dimensions
      - Verify metadata is correct
   
   b) Search with pre-computation:
      - Encode corpus once
      - Perform multiple searches with different queries
      - Verify results correctness
      - Verify no re-encoding of corpus
   
   c) Caching behavior:
      - Enable caching via RuntimeConfig
      - Encode same texts multiple times
      - Verify cache hits
      - Verify performance improvement (time-based)
      - Clear cache and verify re-encoding
   
   d) Multi-query search:
      - Search with multiple queries simultaneously
      - Verify each query gets independent top_k results
      - Verify results are sorted correctly
   
   e) Config file workflow:
      - Create temporary config file
      - Load via from_config_file()
      - Verify all settings applied
      - Test with environment variable overrides
   
   f) Error handling workflow:
      - Test invalid config raises ModelLoadError
      - Test invalid input raises EncodingError
      - Test search validation catches errors

2. Test organization:
   - Use fixtures for temporary config files
   - Use fixtures for test corpus data
   - Group related tests in classes
   - Add docstrings explaining each workflow

Implementation approach:
1. Write integration tests:
   - Focus on realistic usage patterns
   - Test multi-step workflows
   - Verify system behavior, not implementation details
   
2. Use actual SentenceTransformer model:
   - Download a small test model (e.g., all-MiniLM-L6-v2)
   - Store in tests/fixtures/models/
   - Use in fixtures

3. Verify:
   - All integration tests pass
   - Tests are independent (can run in any order)
   - Tests are reproducible

File requirements:
- Comprehensive docstrings
- Clear test names
- Appropriate fixtures

Create test_integration.py as artifact.
```
