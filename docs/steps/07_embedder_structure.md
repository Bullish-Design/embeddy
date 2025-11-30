# /mnt/user-data/outputs/07_embedder_structure.md

# Step 7: Embedder Class Structure and Model Loading

**Goal**: Create the Embedder class with SentenceTransformer model loading and initialization.

**Context**: This is the main interface users interact with. Uses Pydantic BaseModel with PrivateAttr for the model instance and cache.

**Contribution**: Establishes the core class that will provide encode(), search(), and similarity() methods.

---

## Prompt

```
Create the Embedder class with model loading, following TDD.

Project structure:
- src/embeddify/embedder.py (Embedder class)
- tests/test_embedder.py (test suite)
- tests/conftest.py (pytest fixtures)

Requirements:

1. Embedder class (Pydantic BaseModel):
   - config: EmbedderConfig
   - runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
   - _model: SentenceTransformer = PrivateAttr() (loaded model)
   - _cache: dict[str, Embedding] = PrivateAttr(default_factory=dict)
   
2. Initialization:
   - __init__ (or model_post_init): Load SentenceTransformer from config.model_path
   - Wrap SentenceTransformer exceptions in ModelLoadError
   - Set device on loaded model
   - Store model in _model

3. Class method:
   - from_config_file(path: str | None = None) -> Embedder
   - Use load_config_file() to get configs
   - Return initialized Embedder

4. Properties:
   - model_name: str (return model's name/path)
   - device: str (return current device)

Implementation approach:
1. Write fixtures in conftest.py:
   - mock_model_path fixture (path to test model or mock)
   - embedder_config fixture
   - embedder fixture (initialized Embedder)

2. Write tests first:
   - Test Embedder initialization with valid config
   - Test ModelLoadError for invalid model path
   - Test device is set correctly
   - Test from_config_file() creates Embedder
   - Test model_name and device properties

3. Implement embedder.py:
   - Import necessary components
   - Create Embedder class with Pydantic BaseModel
   - Implement model loading in model_post_init
   - Implement from_config_file classmethod
   - Add properties

4. Verify:
   - All tests pass
   - Model loads successfully
   - Error handling works

Dependencies:
- sentence-transformers>=2.0
- Import from config, models, exceptions

File requirements:
- Filepath comment at top
- from __future__ import annotations
- Organize imports properly

Create all three files (embedder.py, test_embedder.py, conftest.py) as artifacts.
```
