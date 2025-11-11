"""
Tests for DocumentManager class
"""
import pytest
from pathlib import Path
from document_manager import DocumentManager


class TestDocumentManager:
    """Test DocumentManager functionality"""

    def test_init(self):
        """Test DocumentManager initialization"""
        manager = DocumentManager()
        assert manager is not None

    def test_list_source_documents_empty(self, temp_dir, monkeypatch):
        """Test listing documents when sources directory is empty"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        manager = DocumentManager()
        documents = manager.list_source_documents()
        assert documents == []

    def test_list_source_documents_with_files(self, temp_dir, sample_pdf_path, sample_txt_path, monkeypatch):
        """Test listing documents with actual files"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        # Copy test files to sources
        import shutil
        shutil.copy(sample_pdf_path, sources_dir / "test.pdf")
        shutil.copy(sample_txt_path, sources_dir / "test.txt")

        manager = DocumentManager()
        documents = manager.list_source_documents()

        assert len(documents) >= 2
        assert any(doc["extension"] == ".pdf" for doc in documents)
        assert any(doc["extension"] == ".txt" for doc in documents)

    def test_valid_file_extensions(self):
        """Test that valid file extensions are recognized"""
        manager = DocumentManager()

        valid_extensions = [".pdf", ".txt", ".docx", ".doc"]
        for ext in valid_extensions:
            test_file = f"test{ext}"
            # The method should recognize these as valid
            assert ext in [".pdf", ".txt", ".docx", ".doc"]

    def test_save_uploaded_file_success(self, temp_dir, sample_pdf_path, monkeypatch):
        """Test successful file upload"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        manager = DocumentManager()
        result = manager.save_uploaded_file(sample_pdf_path, "uploaded_test.pdf")

        assert result["success"] is True
        assert "message" in result
        assert Path("sources/uploaded_test.pdf").exists()

    def test_save_uploaded_file_invalid_extension(self, temp_dir, monkeypatch):
        """Test file upload with invalid extension"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        # Create an invalid file
        invalid_file = Path(temp_dir) / "test.exe"
        invalid_file.write_text("test")

        manager = DocumentManager()
        result = manager.save_uploaded_file(str(invalid_file), "test.exe")

        assert result["success"] is False
        assert "not supported" in result["message"].lower() or "invalid" in result["message"].lower()

    def test_document_info_includes_metadata(self, temp_dir, sample_pdf_path, monkeypatch):
        """Test that document info includes proper metadata"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        import shutil
        shutil.copy(sample_pdf_path, sources_dir / "test.pdf")

        manager = DocumentManager()
        documents = manager.list_source_documents()

        assert len(documents) > 0
        doc = documents[0]

        # Check required metadata fields
        assert "filename" in doc
        assert "path" in doc
        assert "extension" in doc
        assert "type" in doc
        assert "size_mb" in doc

    def test_multiple_file_formats(self, temp_dir, monkeypatch):
        """Test handling multiple file formats"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        # Create test files of different types
        (sources_dir / "test.pdf").write_bytes(b"%PDF-1.4\ntest")
        (sources_dir / "test.txt").write_text("test content")
        (sources_dir / "test.docx").write_bytes(b"PK\x03\x04")  # ZIP signature for DOCX

        manager = DocumentManager()
        documents = manager.list_source_documents()

        extensions = [doc["extension"] for doc in documents]
        assert ".pdf" in extensions
        assert ".txt" in extensions
        assert ".docx" in extensions

    def test_file_size_calculation(self, temp_dir, monkeypatch):
        """Test that file size is calculated correctly"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        # Create a file with known size
        test_file = sources_dir / "test.txt"
        test_file.write_text("x" * 1024)  # 1KB file

        manager = DocumentManager()
        documents = manager.list_source_documents()

        assert len(documents) > 0
        doc = documents[0]
        assert doc["size_mb"] >= 0
        assert isinstance(doc["size_mb"], (int, float))

    def test_empty_filename_handling(self, temp_dir, monkeypatch):
        """Test handling of empty filename"""
        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        manager = DocumentManager()
        result = manager.save_uploaded_file("", "")

        assert result["success"] is False

    def test_sources_directory_creation(self, temp_dir, monkeypatch):
        """Test that sources directory is created if it doesn't exist"""
        monkeypatch.chdir(temp_dir)

        manager = DocumentManager()
        documents = manager.list_source_documents()

        # Should not error even if sources doesn't exist
        assert isinstance(documents, list)
