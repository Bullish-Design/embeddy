# /mnt/user-data/outputs/14_project_config.md

# Step 14: Project Configuration Files

**Goal**: Create pyproject.toml and supporting configuration files for the package.

**Context**: UV-based project configuration with dependencies, build system, and tool configs.

**Contribution**: Enables package installation, dependency management, and development tooling.

---

## Prompt

```
Create project configuration files for UV-based Python package.

Files to create:
- pyproject.toml (project metadata and dependencies)
- .python-version (Python version specification)
- README.md (from provided README template)

Requirements:

1. pyproject.toml:
   - [project] section:
     - name = "embeddify"
     - version = "0.1.0"
     - description = "Pydantic-based wrapper for Sentence Transformers"
     - requires-python = ">=3.10"
     - dependencies: pydantic>=2.0, sentence-transformers>=2.0, torch>=2.0, 
                     transformers>=4.0, numpy>=1.20, pyyaml>=6.0
     - optional-dependencies: dev (pytest, pytest-cov, mypy, ruff, pre-commit)
   
   - [build-system]:
     - requires = ["hatchling"]
     - build-backend = "hatchling.build"
   
   - [tool.uv] section:
     - dev-dependencies listed
   
   - [tool.pytest.ini_options]:
     - testpaths = ["tests"]
     - python_files = "test_*.py"
     - python_classes = "Test*"
     - python_functions = "test_*"
   
   - [tool.ruff] section:
     - line-length = 120
     - target-version = "py310"
     - Select key linting rules
   
   - [tool.mypy] section:
     - strict = true
     - warn_return_any = true
     - warn_unused_configs = true

2. .python-version:
   - Single line: 3.10

3. README.md:
   - Use provided README content from uploaded file

Implementation approach:
1. Create pyproject.toml with all sections
2. Create .python-version
3. Copy README.md content

Verification:
- uv sync should work
- uv run pytest should discover tests
- uv run mypy src/embeddify should run
- uv run ruff check should work

File requirements:
- TOML formatting correct
- Dependencies versions specified

Create pyproject.toml as artifact (README already provided).
```
