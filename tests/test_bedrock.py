"""
Tests for BedrockChat API
"""

import os
import pytest
import json
from datetime import datetime
from src.app.flaskapp import app
from unittest.mock import patch, MagicMock, ANY
from src.core.bedrock import call_bedrock, SUPPORTED_MODELS
from src.core.rag import RAGSystem, Document
from src.core.vector_store import VectorStore
from src.utils.user_manager import user_manager
from dotenv import load_dotenv
import io

# Load environment variables at module level
load_dotenv("config/.env")

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_bedrock():
    """Mock Bedrock responses"""
    with patch('src.core.bedrock.boto3.client') as mock_client:
        mock_bedrock = MagicMock()
        mock_client.return_value = mock_bedrock
        
        # Mock typical response structure
        mock_bedrock.invoke_model.return_value = {
            'body': MagicMock()
        }
        yield mock_bedrock

@pytest.fixture
def user_manager_fixture():
    """Create a user manager with mocked S3"""
    with patch('boto3.client') as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        mock_s3.put_object.return_value = {}
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=MagicMock(return_value=b'{"counter": 5}'))
        }
        mock_s3.list_objects_v2.return_value = {'Contents': []}
        yield user_manager

@pytest.fixture
def vector_store():
    """Create a test vector store with its own namespace"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set in .env file")
    store = VectorStore()
    namespace = store.clear()  # Create new namespace
    yield store
    # Cleanup happens automatically as each test gets new namespace

@pytest.fixture
def rag_system():
    """Create a RAG system"""
    return RAGSystem()

# ===== API Authentication Tests =====
def test_chat_endpoint_no_auth(client):
    """Test chat endpoint without API key"""
    response = client.post('/chat/test_user', json={'message': 'Hello'})
    assert response.status_code == 401

def test_chat_endpoint_invalid_auth(client):
    """Test chat endpoint with invalid API key"""
    response = client.post(
        '/chat/test_user',
        json={'message': 'Hello'},
        headers={'X-API-KEY': 'invalid_key'}
    )
    assert response.status_code == 401

# ===== API Endpoint Tests =====
@patch('src.core.bedrock.bedrock')
@patch('src.app.flaskapp.user_manager.validate_api_key')
def test_chat_endpoint_success(mock_validate, mock_bedrock, client):
    """Test successful chat request"""
    # Mock API key validation
    mock_validate.return_value = True
    
    # Mock bedrock response
    mock_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"completion": "Hello there!"}'
            )
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response
    mock_bedrock.apply_guardrail.return_value = {"action": "ALLOW"}

    response = client.post(
        '/chat/test_user',
        json={
            'message': 'Hello',
            'model': 'claude-2'
        },
        headers={'X-API-KEY': 'valid_test_key'}
    )

    assert response.status_code == 200
    data = response.get_json()
    assert 'response' in data

@patch('src.core.bedrock.bedrock')
@patch('src.app.flaskapp.user_manager.validate_api_key')
def test_chat_endpoint_with_rag(mock_validate, mock_bedrock, client):
    """Test chat endpoint with RAG functionality"""
    # Mock API key validation
    mock_validate.return_value = True
    
    # Mock RAG system and Bedrock response
    with patch('src.app.flaskapp.rag_system') as mock_rag:
        mock_rag.retrieve_context.return_value = {
            "success": True,
            "context_chunks": [],
            "chunks_used": 0
        }

        mock_response = {
            'body': MagicMock(
                read=MagicMock(
                    return_value='{"completion": "RAG-enhanced response"}'
                )
            )
        }
        mock_bedrock.invoke_model.return_value = mock_response
        mock_bedrock.apply_guardrail.return_value = {"action": "ALLOW"}

        response = client.post(
            '/chat/test_user',
            json={
                'message': 'What is neuroscience?',
                'model': 'claude-2',
                'use_rag': True,
                'context_source': 'Neuroscience Guide'
            },
            headers={'X-API-KEY': 'valid_test_key'}
        )

        assert response.status_code == 200

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert 'status' in response.json
    assert response.json['status'] == 'healthy'

@patch('src.app.flaskapp.user_manager.create_api_key')
def test_api_key_creation(mock_create_key, client):
    """Test API key creation endpoint"""
    mock_create_key.return_value = "bdrk_241201_abcd123456"
    
    response = client.post(
        '/api-keys',
        json={'user_id': 'test_user'}
    )
    
    assert response.status_code == 201
    assert 'api_key' in response.json
    assert response.json['api_key'] == "bdrk_241201_abcd123456"

# ===== Bedrock Model Tests =====
@patch('src.core.bedrock.bedrock')
def test_multiple_models(mock_bedrock):
    """Test different model calls"""
    # Mock Claude response
    claude_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"completion": "Response"}'
            )
        )
    }
    
    # Mock Nova response
    nova_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"output": {"message": {"content": [{"text": "Response"}]}}}'
            )
        )
    }

    # Test Claude
    mock_bedrock.invoke_model.return_value = claude_response
    response = call_bedrock([{"role": "user", "content": "Hello"}], "claude-2")
    assert response == "Response"
    mock_bedrock.invoke_model.assert_called_with(
        modelId=SUPPORTED_MODELS["claude-2"],
        contentType="application/json",
        accept="application/json",
        body=ANY
    )

    # Test Nova
    mock_bedrock.invoke_model.return_value = nova_response
    response = call_bedrock([{"role": "user", "content": "Hello"}], "nova")
    assert response == "Response"
    mock_bedrock.invoke_model.assert_called_with(
        modelId=SUPPORTED_MODELS["nova"],
        contentType="application/json",
        accept="application/json",
        body=ANY
    )

    # Test invalid model
    with pytest.raises(Exception) as exc:
        call_bedrock([{"role": "user", "content": "Hello"}], "invalid_model")
    assert "Unsupported model: invalid_model" in str(exc.value)

def test_bedrock_supported_models():
    """Test that all expected models are supported"""
    assert "claude-2" in SUPPORTED_MODELS
    assert "nova" in SUPPORTED_MODELS
    assert "mistral" in SUPPORTED_MODELS
    assert len(SUPPORTED_MODELS) >= 3

@patch('src.core.bedrock.bedrock')
def test_bedrock_error_scenarios(mock_bedrock):
    """Test Bedrock error handling scenarios"""
    # Test API error handling
    mock_bedrock.invoke_model.side_effect = Exception("API Error")
    
    with pytest.raises(Exception) as exc_info:
        call_bedrock([{"role": "user", "content": "test"}], "claude-2")
    
    assert "error calling bedrock" in str(exc_info.value).lower()

# ===== User Manager Tests =====
def test_user_manager_user_id_generation(user_manager_fixture):
    """Test user ID generation"""
    # Test basic user ID generation
    user_id = user_manager_fixture.generate_user_id()
    
    assert isinstance(user_id, str)
    assert user_id.startswith("user-")
    
    # Extract the number part
    user_number = user_id.split("-")[1]
    assert user_number.isdigit()
    assert int(user_number) > 0

def test_user_manager_uuid_detection(user_manager_fixture):
    """Test UUID format detection"""
    # Test sequential format (not UUID)
    assert not user_manager_fixture.is_uuid_user_id("user-123")
    assert not user_manager_fixture.is_uuid_user_id("user-456")
    
    # Test UUID format
    uuid_user_id = "user-abc12345-def6-7890-1234-567890abcdef"
    assert user_manager_fixture.is_uuid_user_id(uuid_user_id)
    
    # Test invalid formats
    assert not user_manager_fixture.is_uuid_user_id("invalid")
    assert not user_manager_fixture.is_uuid_user_id("user-")
    assert not user_manager_fixture.is_uuid_user_id("user-abc")

def test_user_manager_normalize_user_id(user_manager_fixture):
    """Test user ID normalization"""
    # Test sequential format (already normalized)
    normalized = user_manager_fixture.normalize_user_id("user-123")
    assert normalized == "user-123"
    
    # Test UUID format (keep for compatibility)
    uuid_id = "user-abc12345-def6-7890-1234-567890abcdef"
    normalized = user_manager_fixture.normalize_user_id(uuid_id)
    assert normalized == uuid_id
    
    # Test empty (generates new)
    normalized = user_manager_fixture.normalize_user_id("")
    assert normalized.startswith("user-")

def test_user_manager_display_names(user_manager_fixture):
    """Test display name generation"""
    # Test sequential format
    display = user_manager_fixture.get_display_name("user-123")
    assert display == "user-123"
    
    # Test UUID format
    uuid_id = "user-abc12345-def6-7890-1234-567890abcdef"
    display = user_manager_fixture.get_display_name(uuid_id)
    assert display.startswith("User abc12345")
    
    # Test empty
    display = user_manager_fixture.get_display_name("")
    assert display == "Unknown User"

def test_user_manager_api_key_operations(user_manager_fixture):
    """Test API key creation and validation"""
    user_id = "user-123"
    
    # Test API key generation
    api_key = user_manager_fixture.create_api_key(user_id)
    
    assert isinstance(api_key, str)
    assert api_key.startswith("bdrk_")
    assert len(api_key) > 15  # Should have format: bdrk_YYMMDD_10chars
    
    # Test API key format
    parts = api_key.split("_")
    assert len(parts) == 3
    assert parts[0] == "bdrk"
    assert len(parts[1]) == 6  # Date format YYMMDD
    assert len(parts[2]) == 10  # UUID part

def test_user_manager_s3_disabled():
    """Test user manager behavior when S3 is disabled"""
    with patch('boto3.client', side_effect=Exception("No S3")):
        from src.utils.user_manager import UserManager
        manager = UserManager()
        
        assert not manager.s3_enabled
        
        # Should still generate user IDs using fallback
        user_id = manager.generate_user_id()
        assert user_id.startswith("user-")
        
        # API key operations should fail gracefully
        with pytest.raises(ValueError, match="S3 not enabled"):
            manager.create_api_key("test_user")

# ===== RAG System Tests =====
def test_rag_text_processing(rag_system):
    """Test RAG text processing functions"""
    # Test text cleaning
    text = "  Hello,  World!  \n  How are you?  "
    cleaned = rag_system.clean_text(text)
    assert cleaned == "Hello, World! How are you?"
    
    # Test chunk creation
    text = "Test sentence one. Test sentence two. Test sentence three."
    chunks = rag_system.create_chunks(text)
    
    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)
    assert all(chunk.endswith('.') for chunk in chunks)

def test_rag_document_handling():
    """Test RAG document creation and management"""
    doc = Document(
        content="This is a test document about neuroscience.",
        metadata={"source": "test.pdf", "page": 1}
    )
    doc.embedding = [0.1, 0.2, 0.3]
    
    assert doc.content == "This is a test document about neuroscience."
    assert doc.metadata["source"] == "test.pdf"
    assert doc.metadata["page"] == 1
    assert doc.embedding == [0.1, 0.2, 0.3]

def test_rag_embedding_generation():
    """Test RAG embedding generation with mocked Pinecone"""
    rag = RAGSystem()
    
    # Mock the Pinecone inference response
    mock_embed_result = MagicMock()
    mock_embed_result.data = [MagicMock()]
    mock_embed_result.data[0].values = [0.1, 0.2, 0.3]
    
    with patch.object(rag.vector_store.pc.inference, 'embed', return_value=mock_embed_result):
        input_text = "Test embedding generation"
        embedding = rag.get_embedding(input_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3
        assert all(isinstance(x, float) for x in embedding)
        assert embedding == [0.1, 0.2, 0.3]

def test_rag_error_handling():
    """Test RAG system error handling"""
    rag = RAGSystem()
    
    # Test empty content processing
    documents = rag.process_document(
        content=b"",
        filename="empty.txt",
        file_type="txt",
        source_key="test",
        doc_type="test"
    )
    assert len(documents) == 0
    
    # Test invalid file type
    documents = rag.process_document(
        content=b"content",
        filename="test.xyz",
        file_type="xyz",
        source_key="test",
        doc_type="test"
    )
    assert len(documents) == 0
    
    # Test embedding error handling
    with patch.object(rag.vector_store.pc.inference, 'embed', side_effect=Exception("Mock error")):
        embedding = rag.get_embedding("test text")
        assert embedding is None

# ===== Vector Store Tests =====
def test_vector_store_namespace_operations(vector_store):
    """Test vector store namespace operations"""
    # Test clearing namespace (creates new one)
    namespace1 = vector_store.clear()
    assert isinstance(namespace1, str)
    assert namespace1.startswith("ns-")
    
    # Test creating another namespace
    import time
    time.sleep(1)  # Ensure different timestamp
    namespace2 = vector_store.clear()
    assert isinstance(namespace2, str)
    assert namespace2.startswith("ns-")

def test_vector_store_document_operations(vector_store):
    """Test vector store document operations"""
    # Create test documents
    docs = [
        Document(
            content="The brain processes information and controls behavior.",
            metadata={"source": "test1.txt", "page": 1}
        ),
        Document(
            content="The heart pumps blood throughout the body.",
            metadata={"source": "test2.txt", "page": 1}
        )
    ]
    
    # Add documents to store
    success = vector_store.add_documents(docs)
    assert success
    
    # Test search
    results = vector_store.search("brain function", top_k=2)
    assert isinstance(results, list)
    
    # Verify result format if we have results
    if results:
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert 0 <= score <= 1

def test_vector_store_error_handling(vector_store):
    """Test vector store error handling"""
    # Test search with invalid namespace
    results = vector_store.search("test", namespace="nonexistent")
    assert isinstance(results, list)
    
    # Test adding empty documents
    success = vector_store.add_documents([])
    assert success is True

# ===== Configuration Tests =====
def test_configuration_loading():
    """Test configuration loading"""
    from src.core.config import (
        CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_MAX_TOKENS,
        VECTOR_DIMENSION, VECTOR_METRIC, ALLOWED_USER_FILE_TYPES
    )
    
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert DEFAULT_TOP_K > 0
    assert DEFAULT_MAX_TOKENS > 0
    assert VECTOR_DIMENSION > 0
    assert VECTOR_METRIC in ["cosine", "euclidean", "dotproduct"]
    assert isinstance(ALLOWED_USER_FILE_TYPES, set)
    assert len(ALLOWED_USER_FILE_TYPES) > 0

# ===== Integration Tests =====
def test_basic_workflow_integration():
    """Test basic workflow components integration"""
    # Test RAG system initialization
    rag_system = RAGSystem()
    assert rag_system is not None
    
    # Test text processing
    test_text = "This is a test document."
    cleaned = rag_system.clean_text(test_text)
    chunks = rag_system.create_chunks(cleaned)
    assert len(chunks) > 0
    
    # Test document creation
    doc = Document(content="Test content", metadata={"source": "test"})
    assert doc.content == "Test content"

# ===== Advanced RAG System Tests =====
def test_rag_file_upload_validation():
    """Test file upload validation and processing"""
    rag = RAGSystem()
    
    # Test invalid file type
    result = rag.upload_user_file(
        file_content=b"test content",
        filename="test.docx",
        user_id="test_user",
        file_type="docx"
    )
    assert result["success"] is False
    assert "Unsupported file type" in result["error"]
    
    # Test empty file content
    result = rag.upload_user_file(
        file_content=b"",
        filename="empty.txt",
        user_id="test_user", 
        file_type="txt"
    )
    assert result["success"] is False
    assert "No content extracted" in result["error"]

@patch('src.core.rag.get_s3_client')
def test_rag_user_file_upload_success(mock_s3_client):
    """Test successful user file upload with mocking"""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_s3_client.return_value = mock_s3
    mock_s3.put_object.return_value = {}
    
    rag = RAGSystem()
    
    # Mock document indexing
    with patch.object(rag, 'index_documents', return_value=True):
        result = rag.upload_user_file(
            file_content=b"This is test content for a text file.",
            filename="test.txt",
            user_id="test_user",
            file_type="txt"
        )
    
    assert result["success"] is True
    assert result["filename"] == "test.txt"
    assert "documents_indexed" in result
    assert "s3_key" in result

@patch('src.core.rag.get_s3_client')
def test_rag_list_user_files(mock_s3_client):
    """Test listing user files"""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_s3_client.return_value = mock_s3
    
    # Mock S3 response
    mock_s3.list_objects_v2.return_value = {
        'Contents': [
            {
                'Key': 'user_uploads/test_user/20241201_120000_test.pdf',
                'Size': 1024,
                'LastModified': datetime(2024, 12, 1, 12, 0, 0)
            }
        ]
    }
    mock_s3.head_object.return_value = {
        'ContentType': 'application/pdf'
    }
    
    rag = RAGSystem()
    files = rag.list_user_files("test_user")
    
    assert isinstance(files, list)
    assert len(files) == 1
    assert files[0]["filename"] == "20241201_120000_test.pdf"
    assert files[0]["size"] == 1024
    assert files[0]["file_type"] == "pdf"

@patch('src.core.rag.get_s3_client')
def test_rag_delete_user_file(mock_s3_client):
    """Test deleting user files"""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_s3_client.return_value = mock_s3
    mock_s3.delete_object.return_value = {}
    
    rag = RAGSystem()
    success = rag.delete_user_file("test_user", "user_uploads/test_user/test.pdf")
    
    assert success is True
    mock_s3.delete_object.assert_called_once()

def test_rag_advanced_text_processing():
    """Test advanced text processing scenarios"""
    rag = RAGSystem()
    
    # Test text with special characters
    text = "Hello, World! This has émojis 🌟 and spëcial chars & symbols."
    cleaned = rag.clean_text(text)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    
    # Test very long text chunking
    long_text = ". ".join([f"This is sentence number {i}" for i in range(50)])
    chunks = rag.create_chunks(long_text)
    assert len(chunks) > 1
    assert all(chunk.endswith('.') for chunk in chunks)
    
    # Test text with excessive whitespace
    messy_text = "   Too    much   \n\n\n   whitespace   everywhere   "
    cleaned = rag.clean_text(messy_text)
    assert cleaned == "Too much whitespace everywhere"

def test_rag_context_retrieval_methods():
    """Test different context retrieval methods"""
    rag = RAGSystem()
    
    # Test neuroscience context when no namespace exists
    context = rag.get_neuroscience_context("test query")
    assert isinstance(context, str)
    
    # Test context with stats
    context, results = rag.get_neuroscience_context_with_stats("test query")
    assert isinstance(context, str)
    assert isinstance(results, list)
    
    # Test user context when no documents exist
    user_context = rag.get_user_context("test query", "test_user")
    assert isinstance(user_context, str)

def test_rag_retrieve_context_comprehensive():
    """Test comprehensive context retrieval"""
    rag = RAGSystem()
    
    # Test different context sources
    result = rag.retrieve_context("test query", context_source="Neuroscience Guide")
    assert result["success"] is True
    assert "context_chunks" in result
    
    result = rag.retrieve_context("test query", context_source="Your Documents", user_id="test_user")
    assert result["success"] is True
    
    result = rag.retrieve_context("test query", context_source="Both", user_id="test_user")
    assert result["success"] is True

@patch('src.core.rag.get_s3_client')
def test_rag_neuroscience_indexing(mock_s3_client):
    """Test neuroscience document indexing"""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_s3_client.return_value = mock_s3
    mock_s3.head_object.return_value = {}
    mock_s3.get_object.return_value = {
        'Body': MagicMock(read=MagicMock(return_value=b"Mock PDF content for testing"))
    }
    
    rag = RAGSystem()
    
    # Mock vector store operations
    with patch.object(rag.vector_store, 'clear', return_value="test-namespace"), \
         patch.object(rag, 'process_document') as mock_process, \
         patch.object(rag, 'index_documents', return_value=True):
        
        mock_process.return_value = [
            Document(content="Test content", metadata={"source": "test"})
        ]
        
        success = rag.index_neuroscience_documents()
        assert success is True

# ===== Initialize RAG Tests =====
@patch('src.core.initialize_rag.validate_required_env_vars')
@patch('src.core.initialize_rag.RAGSystem')
def test_initialize_rag_success(mock_rag_system, mock_validate):
    """Test successful RAG initialization"""
    from src.core.initialize_rag import main
    
    # Mock successful validation
    mock_validate.return_value = []
    
    # Mock successful RAG system
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_neuroscience_documents.return_value = True
    mock_rag_system.return_value = mock_rag_instance
    
    # Should complete without raising exceptions
    main()
    
    mock_validate.assert_called_once()
    mock_rag_system.assert_called_once()
    mock_rag_instance.index_neuroscience_documents.assert_called_once()

@patch('src.core.initialize_rag.validate_required_env_vars')
def test_initialize_rag_missing_env_vars(mock_validate):
    """Test RAG initialization with missing environment variables"""
    from src.core.initialize_rag import main
    
    # Mock missing environment variables
    mock_validate.return_value = ['PINECONE_API_KEY', 'RAG_BUCKET']
    
    # Should handle missing env vars gracefully
    main()
    
    mock_validate.assert_called_once()

@patch('src.core.initialize_rag.validate_required_env_vars')
@patch('src.core.initialize_rag.RAGSystem')
def test_initialize_rag_indexing_failure(mock_rag_system, mock_validate):
    """Test RAG initialization with indexing failure"""
    from src.core.initialize_rag import main
    
    # Mock successful validation
    mock_validate.return_value = []
    
    # Mock failed indexing
    mock_rag_instance = MagicMock()
    mock_rag_instance.index_neuroscience_documents.return_value = False
    mock_rag_system.return_value = mock_rag_instance
    
    # Should handle indexing failure gracefully
    main()
    
    mock_rag_instance.index_neuroscience_documents.assert_called_once()

@patch('src.core.initialize_rag.validate_required_env_vars')
@patch('src.core.initialize_rag.RAGSystem')
def test_initialize_rag_system_exception(mock_rag_system, mock_validate):
    """Test RAG initialization with system exception"""
    from src.core.initialize_rag import main
    
    # Mock successful validation
    mock_validate.return_value = []
    
    # Mock system exception
    mock_rag_system.side_effect = Exception("System initialization failed")
    
    # Should handle exceptions gracefully
    main()
    
    mock_rag_system.assert_called_once()

# ===== Advanced Vector Store Tests =====
def test_vector_store_initialization_errors():
    """Test vector store initialization error handling"""
    with patch('src.core.vector_store.PINECONE_API_KEY', None):
        with pytest.raises(ValueError, match="PINECONE_API_KEY not configured"):
            VectorStore()

def test_vector_store_initialization_with_existing_index():
    """Test vector store initialization with existing index"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set")
    
    # Test basic initialization works
    vector_store = VectorStore()
    assert vector_store.index_name is not None
    assert hasattr(vector_store, 'index')
    assert hasattr(vector_store, 'pc')

def test_vector_store_embedding_batch_processing():
    """Test vector store embedding batch processing"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set")
    
    vector_store = VectorStore()
    
    # Create many documents to test batching
    documents = [
        Document(content=f"Test document {i}", metadata={"id": i})
        for i in range(15)  # More than typical batch size
    ]
    
    # Mock the embedding API to return fake embeddings
    mock_embed_result = MagicMock()
    mock_embed_result.data = [
        MagicMock(values=[0.1, 0.2, 0.3]) for _ in range(len(documents))
    ]
    
    with patch.object(vector_store.pc.inference, 'embed', return_value=mock_embed_result), \
         patch.object(vector_store.index, 'upsert') as mock_upsert:
        
        success = vector_store.add_documents(documents, namespace="test-batch")
        assert success is True
        
        # Verify upsert was called (could be multiple batches)
        assert mock_upsert.called

def test_vector_store_search_error_handling():
    """Test vector store search error handling"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set")
    
    vector_store = VectorStore()
    
    # Mock embedding generation failure
    with patch.object(vector_store.pc.inference, 'embed', side_effect=Exception("API Error")):
        results = vector_store.search("test query")
        assert results == []

def test_vector_store_rate_limiting():
    """Test vector store rate limiting handling"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set")
    
    vector_store = VectorStore()
    
    # Mock rate limiting error followed by success
    mock_embed_result = MagicMock()
    mock_embed_result.data = [MagicMock(values=[0.1, 0.2, 0.3])]
    
    call_count = 0
    def mock_embed_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("429 Too Many Requests")
        return mock_embed_result
    
    with patch.object(vector_store.pc.inference, 'embed', side_effect=mock_embed_side_effect), \
         patch('time.sleep') as mock_sleep:  # Speed up test
        
        results = vector_store.search("test query")
        
        # Should retry and eventually succeed
        assert mock_sleep.called  # Verify sleep was called for rate limiting

def test_vector_store_clear_operations():
    """Test vector store clear operations"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set")
    
    vector_store = VectorStore()
    
    # Test clearing with specific namespace
    with patch.object(vector_store.index, 'delete') as mock_delete:
        namespace = vector_store.clear("specific-namespace")
        mock_delete.assert_called_with(delete_all=True, namespace="specific-namespace")
        assert namespace.startswith("ns-")
    
    # Test clearing with exception handling
    with patch.object(vector_store.index, 'delete', side_effect=Exception("Delete failed")):
        namespace = vector_store.clear()
        assert namespace.startswith("ns-")  # Should still return a namespace

def test_vector_store_search_response_formats():
    """Test vector store handling of different response formats"""
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set")
    
    vector_store = VectorStore()
    
    # Mock embedding generation
    mock_embed_result = MagicMock()
    mock_embed_result.data = [MagicMock(values=[0.1, 0.2, 0.3])]
    
    # Mock query response with dict format
    mock_response = {
        'matches': [
            {
                'id': 'test-1',
                'score': 0.95,
                'metadata': {
                    'text': 'Test content',
                    'source': 'test.pdf'
                }
            }
        ]
    }
    
    with patch.object(vector_store.pc.inference, 'embed', return_value=mock_embed_result), \
         patch.object(vector_store.index, 'query', return_value=mock_response):
        
        results = vector_store.search("test query")
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert doc.content == "Test content"
        assert score == 0.95

print("Added comprehensive tests for RAG, Initialize RAG, and Vector Store!")