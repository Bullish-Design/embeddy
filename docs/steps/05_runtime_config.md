# /mnt/user-data/outputs/05_runtime_config.md

# Step 5: RuntimeConfig

**Goal**: Implement configuration for runtime execution parameters separate from model configuration.

**Context**: RuntimeConfig controls batch processing, caching, and output format - concerns independent of which model is loaded.

**Contribution**: Separates runtime behavior from model configuration, enabling flexible execution control.

---

## Prompt

```
Add RuntimeConfig to config.py and extend test_config.py, following TDD.

Files to modify:
- src/embeddify/config.py (add RuntimeConfig)
- tests/test_config.py (add RuntimeConfig tests)

Requirements:

1. RuntimeConfig model:
   - batch_size: int = 32 (encoding batch size)
   - show_progress_bar: bool = False
   - enable_cache: bool = False
   - convert_to_numpy: bool = False (return np.ndarray instead of list[float])
   
2. Validation:
   - batch_size must be >= 1
   - If enable_cache=True and convert_to_numpy=True, log warning (cache incompatible with numpy)

3. Add class method:
   - from_env() -> RuntimeConfig: Load from environment variables
     - EMBEDDIFY_BATCH_SIZE (optional)
     - EMBEDDIFY_SHOW_PROGRESS_BAR (optional, parse as bool)
     - EMBEDDIFY_ENABLE_CACHE (optional, parse as bool)
     - EMBEDDIFY_CONVERT_TO_NUMPY (optional, parse as bool)

Implementation approach:
1. Write tests first:
   - Test valid RuntimeConfig creation
   - Test batch_size < 1 raises ValidationError
   - Test from_env() with various environment configurations
   - Test boolean parsing from env vars ("true", "false", "1", "0")

2. Implement:
   - Add RuntimeConfig class to config.py
   - Add @field_validator for batch_size
   - Implement from_env() with proper type conversion
   - Add logging for cache/numpy incompatibility warning

3. Verify:
   - All tests pass (new and existing)
   - Environment variable parsing handles edge cases
   - Warning logged when appropriate

Dependencies:
- Add logging import for cache/numpy warning

File requirements:
- Maintain config.py structure
- Keep validators clear and focused

Create updated test file as artifact.
```
