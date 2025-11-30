# /mnt/user-data/outputs/06_config_file_loading.md

# Step 6: Configuration File Loading

**Goal**: Implement YAML/JSON config file loading with environment variable override support.

**Context**: Enables configuration-as-code for reproducible embedder setups. Environment variables override file values for deployment flexibility.

**Contribution**: Completes the configuration system with file-based config and env var integration.

---

## Prompt

```
Add config file loading utilities to config.py and extend test_config.py, following TDD.

Files to modify:
- src/embeddify/config.py (add load_config_file function)
- tests/test_config.py (add config file tests)
- tests/fixtures/configs/ (create test config files)

Requirements:

1. load_config_file function:
   - Signature: load_config_file(path: str | None = None) -> tuple[EmbedderConfig, RuntimeConfig]
   - If path is None, read from EMBEDDIFY_CONFIG_PATH env var
   - Support both YAML and JSON formats (detect by extension)
   - Parse nested structure:
     ```yaml
     model:
       path: "/models/all-MiniLM-L6-v2"
       device: "cpu"
       normalize_embeddings: true
     runtime:
       batch_size: 32
       enable_cache: true
     ```
   - Environment variables override file values
   - Raise FileNotFoundError if file doesn't exist
   - Raise ValidationError if format invalid

2. Environment variable override logic:
   - EMBEDDIFY_MODEL_PATH overrides model.path
   - EMBEDDIFY_DEVICE overrides model.device
   - EMBEDDIFY_BATCH_SIZE overrides runtime.batch_size
   - etc.

Implementation approach:
1. Write tests first:
   - Create fixture config files (valid.yaml, valid.json, invalid.yaml)
   - Test loading YAML config
   - Test loading JSON config
   - Test env var overrides
   - Test EMBEDDIFY_CONFIG_PATH fallback
   - Test FileNotFoundError for missing file
   - Test ValidationError for malformed config

2. Implement:
   - Add load_config_file function
   - Use PyYAML for YAML, json for JSON
   - Apply env var overrides after loading file
   - Keep error messages clear

3. Verify:
   - All tests pass
   - Both YAML and JSON work identically
   - Override logic is correct

Dependencies:
- pyyaml>=6.0 (for YAML support)
- json (stdlib)
- pathlib for path handling

File requirements:
- Add imports as needed
- Keep function focused and testable
- Use type hints

Create test file and sample config fixtures as artifacts.
```
