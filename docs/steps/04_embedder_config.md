# /mnt/user-data/outputs/04_embedder_config.md

# Step 4: EmbedderConfig

**Goal**: Implement the configuration model for SentenceTransformer initialization.

**Context**: EmbedderConfig validates all model loading parameters before attempting to load, enabling fail-fast behavior and clear error messages.

**Contribution**: Provides type-safe configuration for the Embedder class with path validation and device checking.

---

## Prompt

```
Create the EmbedderConfig class for validating model loading parameters, following TDD.

Project structure:
- src/embeddify/config.py (configuration models)
- tests/test_config.py (test suite)

Requirements:

1. EmbedderConfig model:
   - model_path: str (path to pre-downloaded model)
   - device: str = "cpu" (cpu, cuda, or cuda:N)
   - normalize_embeddings: bool = True
   - trust_remote_code: bool = False
   
2. Validation:
   - model_path must be non-empty string
   - device must match pattern: "cpu" | "cuda" | "cuda:\d+"
   - If device is "cuda" or "cuda:N", check torch.cuda.is_available()
   - Provide clear error messages for validation failures

3. Add class method:
   - from_env() -> EmbedderConfig: Load from environment variables
     - EMBEDDIFY_MODEL_PATH (required if using from_env)
     - EMBEDDIFY_DEVICE (optional, defaults to "cpu")
     - EMBEDDIFY_NORMALIZE_EMBEDDINGS (optional, defaults to "true")
     - EMBEDDIFY_TRUST_REMOTE_CODE (optional, defaults to "false")

Implementation approach:
1. Write tests first:
   - Test valid configurations
   - Test invalid device strings (e.g., "gpu", "cuda:x")
   - Test CUDA device when CUDA unavailable (should raise ValidationError)
   - Test from_env() with various environment setups
   - Test empty model_path raises ValidationError

2. Implement config.py:
   - Use Pydantic BaseModel
   - Add @field_validator for device validation
   - Implement CUDA availability check
   - Implement from_env() classmethod

3. Verify:
   - All tests pass
   - Import torch only for validation (not at module level if possible)
   - Error messages are actionable

Dependencies:
- pydantic>=2.0
- torch>=2.0 (for cuda availability check)

File requirements:
- Filepath comment at top
- from __future__ import annotations
- Lines under 120 characters

Create both files as separate artifacts.
```
