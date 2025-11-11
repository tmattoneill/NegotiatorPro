"""
Integration tests for NegotiatorPro
Tests end-to-end workflows and component interactions
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGPipeline:
    """Test end-to-end RAG pipeline"""

    @patch('main.OpenAIEmbeddings')
    @patch('main.ChatOpenAI')
    @patch('main.FAISS')
    def test_rag_system_initialization(self, mock_faiss, mock_chat, mock_embeddings, mock_env_vars):
        """Test that RAG system initializes correctly"""
        from main import EnhancedNegotiationRAG

        # Mock the vectorstore loading
        mock_vectorstore = Mock()
        mock_faiss.load_local.return_value = mock_vectorstore

        rag = EnhancedNegotiationRAG()

        assert rag is not None
        assert hasattr(rag, 'vectorstore')
        assert hasattr(rag, 'admin_config')
        assert hasattr(rag, 'document_manager')

    @patch('main.OpenAIEmbeddings')
    @patch('main.ChatOpenAI')
    def test_get_advice_with_default_model(self, mock_chat, mock_embeddings, mock_env_vars):
        """Test getting advice with default model"""
        from main import EnhancedNegotiationRAG

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Test negotiation advice"
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # Mock vectorstore
        with patch.object(EnhancedNegotiationRAG, 'load_vectorstore', return_value=True):
            with patch.object(EnhancedNegotiationRAG, 'get_relevant_context', return_value="Test context"):
                rag = EnhancedNegotiationRAG()
                rag.setup_llms()

                advice = rag.get_advice("How do I negotiate?", use_premium_model=False)

                assert isinstance(advice, str)
                assert len(advice) > 0

    @patch('main.OpenAIEmbeddings')
    @patch('main.ChatOpenAI')
    def test_get_advice_with_premium_model(self, mock_chat, mock_embeddings, mock_env_vars):
        """Test getting advice with premium model"""
        from main import EnhancedNegotiationRAG

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Premium negotiation advice"
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # Mock vectorstore
        with patch.object(EnhancedNegotiationRAG, 'load_vectorstore', return_value=True):
            with patch.object(EnhancedNegotiationRAG, 'get_relevant_context', return_value="Test context"):
                rag = EnhancedNegotiationRAG()
                rag.setup_llms()

                advice = rag.get_advice("How do I negotiate?", use_premium_model=True)

                assert isinstance(advice, str)
                assert len(advice) > 0

    @patch('main.PyPDFLoader')
    @patch('main.TextLoader')
    def test_load_documents_mixed_formats(self, mock_text_loader, mock_pdf_loader, temp_dir, monkeypatch):
        """Test loading documents of mixed formats"""
        from main import EnhancedNegotiationRAG

        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        # Create test files
        (sources_dir / "test.pdf").write_bytes(b"%PDF-1.4\ntest")
        (sources_dir / "test.txt").write_text("test content")

        # Mock loaders
        mock_pdf_doc = Mock()
        mock_pdf_doc.page_content = "PDF content"
        mock_pdf_doc.metadata = {}
        mock_pdf_loader.return_value.load.return_value = [mock_pdf_doc]

        mock_text_doc = Mock()
        mock_text_doc.page_content = "Text content"
        mock_text_doc.metadata = {}
        mock_text_loader.return_value.load.return_value = [mock_text_doc]

        rag = EnhancedNegotiationRAG()
        documents = rag.load_documents()

        assert len(documents) > 0

    @patch('main.OpenAIEmbeddings')
    def test_create_vectorstore(self, mock_embeddings, mock_env_vars):
        """Test vectorstore creation"""
        from main import EnhancedNegotiationRAG

        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        rag = EnhancedNegotiationRAG()

        # Create mock documents
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {}

        with patch('main.FAISS.from_documents') as mock_faiss:
            mock_vectorstore = Mock()
            mock_faiss.return_value = mock_vectorstore

            vectorstore = rag.create_vectorstore([mock_doc])

            assert vectorstore is not None
            mock_faiss.assert_called_once()


class TestDocumentUploadWorkflow:
    """Test document upload and processing workflow"""

    @patch('main.OpenAIEmbeddings')
    def test_document_upload_and_regenerate(self, mock_embeddings, temp_dir, sample_pdf_path, monkeypatch):
        """Test uploading document and regenerating vectorstore"""
        from main import EnhancedNegotiationRAG

        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        rag = EnhancedNegotiationRAG()

        # Upload a document
        result = rag.document_manager.save_uploaded_file(sample_pdf_path, "new_doc.pdf")
        assert result["success"] is True

        # Verify document is in sources
        documents = rag.document_manager.list_source_documents()
        assert any(doc["filename"] == "new_doc.pdf" for doc in documents)


class TestModelSwitching:
    """Test switching between models"""

    @patch('main.OpenAIEmbeddings')
    @patch('main.ChatOpenAI')
    def test_model_switching(self, mock_chat, mock_embeddings, mock_env_vars):
        """Test switching between default and premium models"""
        from main import EnhancedNegotiationRAG

        # Mock LLM instances
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm

        with patch.object(EnhancedNegotiationRAG, 'load_vectorstore', return_value=True):
            with patch.object(EnhancedNegotiationRAG, 'get_relevant_context', return_value="Test context"):
                rag = EnhancedNegotiationRAG()
                rag.setup_llms()

                # Test default model
                advice1 = rag.get_advice("Test question", use_premium_model=False)
                assert isinstance(advice1, str)

                # Test premium model
                advice2 = rag.get_advice("Test question", use_premium_model=True)
                assert isinstance(advice2, str)


class TestAdminWorkflow:
    """Test admin panel workflow"""

    def test_admin_authentication_flow(self, temp_dir, monkeypatch):
        """Test complete admin authentication flow"""
        monkeypatch.chdir(temp_dir)

        from admin_config import AdminConfig

        admin = AdminConfig()

        # Test login
        assert admin.verify_password("admin123")

        # Test session creation
        session_id = "test-session-123"
        admin.create_session(session_id)
        assert admin.is_valid_session(session_id)

        # Test password change
        admin.change_password("newpassword")
        assert not admin.verify_password("admin123")
        assert admin.verify_password("newpassword")

    def test_admin_usage_tracking(self, temp_dir, monkeypatch):
        """Test admin usage tracking workflow"""
        monkeypatch.chdir(temp_dir)

        from admin_config import AdminConfig

        admin = AdminConfig()

        # Log some usage
        admin.log_usage("gpt-4o-mini", 1000)
        admin.log_usage("o3-mini", 2000)

        # Get summary
        summary = admin.get_usage_summary(days=30)

        assert summary["total_requests"] >= 2
        assert summary["total_tokens"] >= 3000
        assert "gpt-4o-mini" in summary["models"]
        assert "o3-mini" in summary["models"]


class TestTextPreprocessingWorkflow:
    """Test text preprocessing integration"""

    @patch('main.ChatOpenAI')
    @patch('main.OpenAIEmbeddings')
    def test_preprocessing_before_query(self, mock_embeddings, mock_chat, mock_env_vars):
        """Test that preprocessing is applied before query"""
        from main import EnhancedNegotiationRAG

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm

        with patch.object(EnhancedNegotiationRAG, 'load_vectorstore', return_value=True):
            with patch.object(EnhancedNegotiationRAG, 'get_relevant_context', return_value="Test context"):
                rag = EnhancedNegotiationRAG()
                rag.setup_llms()

                # Use a long text with extra whitespace
                long_text = "How do I negotiate?  " * 50

                advice = rag.get_advice(long_text, use_preprocessing=True)

                assert isinstance(advice, str)
                assert len(advice) > 0


class TestErrorHandling:
    """Test error handling across the system"""

    @patch('main.ChatOpenAI')
    @patch('main.OpenAIEmbeddings')
    def test_missing_api_key_handling(self, mock_embeddings, mock_chat):
        """Test handling of missing API key"""
        from main import EnhancedNegotiationRAG

        # Simulate missing API key
        with patch.dict('os.environ', {}, clear=True):
            # Should not crash during initialization
            rag = EnhancedNegotiationRAG()
            assert rag is not None

    @patch('main.OpenAIEmbeddings')
    def test_empty_sources_directory(self, mock_embeddings, temp_dir, monkeypatch):
        """Test handling of empty sources directory"""
        from main import EnhancedNegotiationRAG

        monkeypatch.chdir(temp_dir)
        sources_dir = Path("sources")
        sources_dir.mkdir(exist_ok=True)

        rag = EnhancedNegotiationRAG()
        documents = rag.load_documents()

        # Should return empty list, not crash
        assert isinstance(documents, list)

    @patch('main.ChatOpenAI')
    @patch('main.OpenAIEmbeddings')
    def test_llm_error_handling(self, mock_embeddings, mock_chat, mock_env_vars):
        """Test handling of LLM errors"""
        from main import EnhancedNegotiationRAG

        # Mock LLM to raise an error
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_chat.return_value = mock_llm

        with patch.object(EnhancedNegotiationRAG, 'load_vectorstore', return_value=True):
            rag = EnhancedNegotiationRAG()
            rag.setup_llms()

            advice = rag.get_advice("Test question")

            # Should return error message, not crash
            assert isinstance(advice, str)
            assert "error" in advice.lower()
