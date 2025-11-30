# /mnt/user-data/outputs/01_exceptions.md

# Step 1: Exception Hierarchy

**Goal**: Establish the custom exception hierarchy for Embeddify to provide clear, actionable error messages throughout the library.

**Context**: This is the foundation step. All subsequent components will use these exceptions for error handling. The exception hierarchy must support proper exception chaining (preserving `__cause__`) and provide domain-specific error types.

**Contribution**: Creates the error handling infrastructure that enables fail-fast behavior and clear debugging for users.

---

## Prompt

```
Create the exception hierarchy for the Embeddify library following test-driven development.

Project structure:
- src/embeddify/exceptions.py (exception classes)
- tests/test_exceptions.py (test suite)

Requirements:
1. Base exception: EmbeddifyError (inherits from Exception)
2. Specific exceptions (all inherit from EmbeddifyError):
   - ModelLoadError: Model initialization failures
   - EncodingError: Text encoding failures
   - ValidationError: Pydantic validation failures (note: different from pydantic.ValidationError)
   - SearchError: Semantic search failures

3. All exceptions must:
   - Accept an optional message parameter
   - Preserve the original exception via __cause__ when wrapping
   - Include docstrings explaining when to use them

Implementation approach:
1. Write tests first in test_exceptions.py:
   - Test each exception can be raised with a message
   - Test exception inheritance chain (all inherit from EmbeddifyError)
   - Test exception chaining (verify __cause__ is preserved)
   - Test repr/str methods produce useful output

2. Implement exceptions.py to pass tests:
   - Start with EmbeddifyError base class
   - Add each specific exception type
   - Keep implementation minimal but complete

3. Verify:
   - All tests pass
   - Type hints are complete
   - Import structure works: `from embeddify.exceptions import EmbeddifyError, ModelLoadError, ...`

File requirements:
- Add filepath comment at top: # src/embeddify/exceptions.py
- Use `from __future__ import annotations`
- Keep lines under 120 characters
- No external dependencies (pure Python)

Create both files as separate artifacts.
```
