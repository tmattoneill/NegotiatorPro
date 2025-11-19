"""
Shared test fixtures and configuration for NegotiatorPro tests
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")
    return {
        "OPENAI_API_KEY": "sk-test-key-1234567890"
    }


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a mock PDF file for testing"""
    pdf_path = Path(temp_dir) / "test_document.pdf"
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Content) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000293 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
385
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return str(pdf_path)


@pytest.fixture
def sample_txt_path(temp_dir):
    """Create a sample text file for testing"""
    txt_path = Path(temp_dir) / "test_document.txt"
    txt_path.write_text("This is a test negotiation document.\n" * 10)
    return str(txt_path)


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings"""
    mock = Mock()
    mock.embed_query.return_value = [0.1] * 1536  # Standard embedding dimension
    mock.embed_documents.return_value = [[0.1] * 1536]
    return mock


@pytest.fixture
def mock_openai_chat():
    """Mock OpenAI chat completion"""
    mock = Mock()
    mock_response = Mock()
    mock_response.content = "This is a test negotiation response."
    mock.invoke.return_value = mock_response
    return mock


@pytest.fixture
def clean_config_files():
    """Clean up config files before and after tests"""
    config_files = [
        "admin_config.json",
        "admin_sessions.json",
        "usage_stats.json",
        "embedding_config.json",
        "prompt_config.json"
    ]

    # Backup existing files
    backups = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            backup_path = f"{config_file}.backup"
            shutil.copy(config_file, backup_path)
            backups[config_file] = backup_path

    yield

    # Restore backups
    for config_file, backup_path in backups.items():
        if os.path.exists(backup_path):
            shutil.move(backup_path, config_file)

    # Clean up any test-generated files
    for config_file in config_files:
        if os.path.exists(config_file) and config_file not in backups:
            os.remove(config_file)


@pytest.fixture
def mock_vectorstore():
    """Mock FAISS vectorstore"""
    mock = Mock()
    mock.as_retriever.return_value = Mock()
    mock.save_local = Mock()
    return mock


@pytest.fixture
def mock_fastapi_app():
    """Mock FastAPI application"""
    mock_app = Mock()
    mock_app.get = Mock()
    mock_app.post = Mock()
    return mock_app


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # This ensures each test starts with fresh instances
    yield
    # Cleanup code can go here if needed
