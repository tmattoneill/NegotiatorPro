# Testing Guide for NegotiatorPro

Comprehensive testing documentation for the NegotiatorPro RAG system.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [CI/CD Pipeline](#cicd-pipeline)
- [Coverage Reports](#coverage-reports)
- [Troubleshooting](#troubleshooting)

## Overview

NegotiatorPro uses **pytest** as its testing framework with comprehensive coverage across:
- Docker infrastructure
- Unit tests for all modules
- Integration tests for end-to-end workflows
- Security scanning
- Code quality checks

## Test Structure

```
NegotiatorPro/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures and configuration
│   ├── test_docker.py           # Docker infrastructure tests
│   ├── test_admin_config.py     # Admin system tests
│   ├── test_document_manager.py # Document management tests
│   ├── test_model_config.py     # Model configuration tests
│   ├── test_modules.py          # Text preprocessor, prompt manager, embedding tests
│   └── test_integration.py      # End-to-end integration tests
├── pytest.ini                   # Pytest configuration
├── .coveragerc                  # Coverage configuration
├── requirements-test.txt        # Test dependencies
└── .github/workflows/test.yml   # CI/CD pipeline
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Set up environment (optional for most tests)
export OPENAI_API_KEY=your_api_key_here
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only Docker tests
pytest -m docker

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring Docker
pytest -m "not requires_docker"
```

### Run Specific Test Files

```bash
# Run Docker tests only
pytest tests/test_docker.py

# Run admin config tests only
pytest tests/test_admin_config.py

# Run a specific test function
pytest tests/test_admin_config.py::TestAdminConfig::test_password_verification
```

### Run Tests with Different Options

```bash
# Run tests in parallel (faster)
pytest -n auto

# Stop on first failure
pytest -x

# Run failed tests from last run
pytest --lf

# Show print statements
pytest -s

# Generate HTML coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Categories

### 1. Docker Infrastructure Tests (`test_docker.py`)

Tests Docker deployment configuration:
- Dockerfile syntax and best practices
- Multi-stage build verification
- Non-root user configuration
- Health checks
- docker-compose.yml validation
- Port mappings and volumes
- Environment variables
- .dockerignore configuration
- Deployment documentation

**Run Docker tests:**
```bash
pytest tests/test_docker.py -v
```

**Note:** Some Docker tests require Docker daemon to be running.

### 2. Admin System Tests (`test_admin_config.py`)

Tests admin authentication and management:
- Password verification and hashing
- Session creation and validation
- Session expiration
- Password changes
- Usage tracking and statistics
- Configuration persistence

**Run admin tests:**
```bash
pytest tests/test_admin_config.py -v
```

### 3. Document Manager Tests (`test_document_manager.py`)

Tests document handling:
- File upload validation
- Multiple format support (PDF, TXT, DOCX, DOC)
- File size calculations
- Invalid file type rejection
- Document listing with metadata
- Sources directory management

**Run document tests:**
```bash
pytest tests/test_document_manager.py -v
```

### 4. Model Configuration Tests (`test_model_config.py`)

Tests model-specific configuration:
- gpt-4o-mini configuration
- o3-mini configuration (no temperature parameter)
- Unknown model fallback
- Parameter filtering
- Model dictionary validation

**Run model config tests:**
```bash
pytest tests/test_model_config.py -v
```

### 5. Module Tests (`test_modules.py`)

Tests supporting modules:

**TextPreprocessor:**
- Text cleaning and optimization
- Token counting
- Cost calculation
- Whitespace normalization

**PromptManager:**
- Prompt template management
- System/user prompt updates
- Placeholder replacement
- Prompt persistence

**EmbeddingConfig:**
- Embedding model configuration
- Compatibility validation
- Status reporting
- Configuration persistence

**Run module tests:**
```bash
pytest tests/test_modules.py -v
```

### 6. Integration Tests (`test_integration.py`)

Tests end-to-end workflows:
- Complete RAG pipeline
- Document upload and processing
- Model switching (default ↔ premium)
- Admin workflows
- Text preprocessing integration
- Error handling

**Run integration tests:**
```bash
pytest tests/test_integration.py -v
```

## Writing Tests

### Test File Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_with_temp_directory(temp_dir):
    """Use temp_dir fixture for temporary files"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("test")
    assert test_file.exists()

def test_with_mock_env(mock_env_vars):
    """Use mock_env_vars for environment variables"""
    assert os.getenv("OPENAI_API_KEY") == "sk-test-key-1234567890"

def test_with_sample_files(sample_pdf_path, sample_txt_path):
    """Use sample file fixtures"""
    assert Path(sample_pdf_path).exists()
    assert Path(sample_txt_path).exists()
```

### Adding Test Markers

```python
import pytest

@pytest.mark.unit
def test_unit_example():
    """Mark as unit test"""
    pass

@pytest.mark.integration
def test_integration_example():
    """Mark as integration test"""
    pass

@pytest.mark.slow
def test_slow_example():
    """Mark as slow test"""
    pass

@pytest.mark.requires_docker
def test_docker_example():
    """Mark as requiring Docker"""
    pass
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

@patch('main.OpenAIEmbeddings')
@patch('main.ChatOpenAI')
def test_with_mocked_openai(mock_chat, mock_embeddings):
    """Mock OpenAI dependencies"""
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_chat.return_value.invoke.return_value = mock_response

    # Your test code here
    pass
```

### Testing Exceptions

```python
def test_invalid_input():
    """Test exception handling"""
    with pytest.raises(ValueError):
        function_that_should_raise_error()
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/test.yml`) runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual trigger via GitHub Actions UI

### Pipeline Jobs

1. **Test Job** (Python 3.9, 3.10, 3.11)
   - Install dependencies
   - Run linting (flake8)
   - Check code formatting (black)
   - Run unit tests with coverage
   - Upload coverage to Codecov

2. **Docker Job**
   - Build Docker image
   - Validate docker-compose config
   - Run Docker smoke test

3. **Security Job**
   - Run Bandit security scanner
   - Run Safety dependency checker
   - Upload security reports

4. **Test Summary Job**
   - Aggregate results from all jobs

### Required Secrets

Add these secrets in GitHub repository settings:
- `OPENAI_API_KEY` (optional, tests use mock by default)

### Viewing CI/CD Results

1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select a workflow run
4. View job logs and artifacts

## Coverage Reports

### Generating Coverage Reports

```bash
# Terminal report with missing lines
pytest --cov=. --cov-report=term-missing

# HTML report (interactive)
pytest --cov=. --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=. --cov-report=xml
```

### Coverage Goals

- **Minimum target:** 80% coverage
- **Current configuration:** Reports missing lines
- **Excluded from coverage:**
  - Test files
  - Virtual environments
  - `__repr__` and `__str__` methods
  - Abstract methods
  - Debug logging

### Viewing Coverage in HTML

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Troubleshooting

### Tests Failing Due to Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Tests Failing Due to Docker Not Available

```bash
# Skip Docker tests
pytest -m "not requires_docker"
```

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/NegotiatorPro

# Run tests with PYTHONPATH set
PYTHONPATH=. pytest tests/
```

### Permission Errors with Config Files

```bash
# Clean up config files before running tests
rm -f admin_config.json admin_sessions.json usage_stats.json embedding_config.json prompt_config.json
pytest
```

### Slow Tests Taking Too Long

```bash
# Skip slow tests
pytest -m "not slow"

# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

### Coverage Not Showing Correctly

```bash
# Clean coverage data
rm -f .coverage coverage.xml
rm -rf htmlcov/

# Regenerate coverage
pytest --cov=. --cov-report=html
```

### Mock Not Working as Expected

```python
# Ensure mock is patched correctly
# Patch where the object is USED, not where it's defined

# Wrong:
@patch('langchain_openai.OpenAIEmbeddings')

# Correct:
@patch('main.OpenAIEmbeddings')  # Patch in main.py where it's imported
```

## Best Practices

1. **Keep tests isolated:** Use fixtures and temp directories
2. **Mock external dependencies:** Don't make real API calls in tests
3. **Use descriptive names:** Test names should describe what they test
4. **One assertion per test:** Makes failures easier to diagnose
5. **Test edge cases:** Empty inputs, invalid data, errors
6. **Keep tests fast:** Use mocks, avoid I/O when possible
7. **Update tests with code:** Tests should evolve with the codebase
8. **Document complex tests:** Add comments for non-obvious test logic

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

## Support

For issues or questions about testing:
1. Check this documentation
2. Review existing tests for examples
3. Check CI/CD logs for detailed error messages
4. Open an issue on the project repository
