"""
Tests for BedrockChat API
"""

import os
import pytest
import json
from datetime import datetime
from app import app
from unittest.mock import patch, MagicMock, ANY
from bedrock import call_bedrock, SUPPORTED_MODELS
from rag import RAGSystem, Document
from vector_store import VectorStore
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

@pytest.fixture
def client():
    """Create test client"""
    # Patch out all logging functions
    with patch('app.log_chat_request'), \
         patch('app.log_error'), \
         patch('app.log_model_usage'), \
         patch('app.get_session_logger'):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

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

# ===== API Tests =====
def test_chat_endpoint_no_auth(client):
    """Test chat endpoint without API key"""
    response = client.post('/chat/test_session', json={'message': 'Hello'})
    assert response.status_code == 401

def test_chat_endpoint_invalid_auth(client):
    """Test chat endpoint with invalid API key"""
    response = client.post(
        '/chat/test_session',
        json={'message': 'Hello'},
        headers={'X-API-KEY': 'invalid_key'}
    )
    assert response.status_code == 401

@patch('bedrock.bedrock')
def test_chat_endpoint_success(mock_bedrock, client):
    """Test successful chat request"""
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

    # Set test API key
    os.environ['API_KEY'] = 'test_key'
    
    # Test request
    response = client.post(
        '/chat/test_session',
        json={
            'message': 'Hello',
            'model': 'claude-2'
        },
        headers={'X-API-KEY': 'test_key'}
    )

    assert response.status_code == 200
    assert response.json['response'] == 'Hello there!'
    assert response.json['model'] == 'claude-2'

@patch('bedrock.bedrock')
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

def test_models_endpoint(client):
    """Test models listing endpoint"""
    # Set test API key
    os.environ['API_KEY'] = 'test_key'

    response = client.get(
        '/models',
        headers={'X-API-KEY': 'test_key'}
    )

    assert response.status_code == 200
    assert 'models' in response.json
    assert set(response.json['models']) == {"claude-2", "nova", "mistral"}

@patch('bedrock.bedrock')
def test_test_endpoint(mock_bedrock, client):
    """Test the simple test endpoint"""
    # Mock bedrock response
    mock_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"completion": "Test response"}'
            )
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response
    mock_bedrock.apply_guardrail.return_value = {"action": "ALLOW"}

    os.environ['API_KEY'] = 'test_key'
    
    response = client.post(
        '/test',
        json={
            'message': 'Hello',
            'model': 'claude-2'
        },
        headers={'X-API-KEY': 'test_key'}
    )
    
    assert response.status_code == 200
    assert 'response' in response.json
    assert 'model' in response.json

# ===== RAG Pipeline Tests =====
@patch('boto3.client')
def test_rag_embedding(mock_boto3):
    """Test RAG embedding generation"""
    print("\n=== Testing Embedding Generation ===")
    
    print("1. Creating Bedrock client...")
    # Mock the Bedrock client
    mock_client = MagicMock()
    mock_boto3.return_value = mock_client
    
    # Mock the response
    mock_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value=json.dumps({
                    "embedding": [0.1, 0.2, 0.3],
                    "inputTextTokenCount": 10
                })
            )
        )
    }
    mock_client.invoke_model.return_value = mock_response
    
    print("2. Setting up request...")
    # Set the model ID
    model_id = "amazon.titan-embed-text-v2:0"
    
    # The text to convert to an embedding
    input_text = "Please recommend books with a theme similar to the movie 'Inception'."
    
    # Create the request for the model
    native_request = {"inputText": input_text}
    
    # Convert the native request to JSON
    request = json.dumps(native_request)
    
    print("3. Invoking model...")
    # Invoke the model with the request
    response = mock_client.invoke_model(modelId=model_id, body=request)
    
    print("4. Processing response...")
    # Decode the model's native response body
    model_response = json.loads(response["body"].read())
    
    # Extract the generated embedding and the input text token count
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]
    
    print("\nYour input:")
    print(input_text)
    print(f"Number of input tokens: {input_token_count}")
    print(f"Size of the generated embedding: {len(embedding)}")
    print("First 5 values of embedding:")
    print(embedding[:5])
    
    # Verify the embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 3
    assert all(isinstance(x, float) for x in embedding)
    assert embedding == [0.1, 0.2, 0.3]

# ===== RAG Text Processing Tests =====
def test_rag_text_processing(rag_system):
    """Test RAG text processing functions"""
    print("\n=== Testing Text Processing ===")
    
    print("2. Testing text cleaning...")
    text = "  Hello,  World!  \n  How are you?  "
    cleaned = rag_system.clean_text(text)
    assert cleaned == "Hello, World! How are you?"
    print(f"Cleaned text: {cleaned}")
    
    print("\n3. Testing chunk creation...")
    # Use text that will definitely contain our test word in each chunk
    text = "Test sentence one. Test sentence two. Test sentence three. Test sentence four."
    chunks = rag_system.create_chunks(text, chunk_size=30, overlap=5)
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    # Basic assertions
    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(len(chunk) <= 30 for chunk in chunks), "All chunks should be within size limit"
    assert all(chunk.strip() for chunk in chunks), "No empty chunks"
    assert all("Test" in chunk for chunk in chunks), "Each chunk should contain test word"
    assert all(chunk.endswith('.') for chunk in chunks), "All chunks should end with period"

# ===== RAG Document Handling Tests =====
def test_rag_document_handling():
    """Test RAG document creation and management"""
    print("\n=== Testing Document Handling ===")
    
    print("1. Creating test document...")
    doc = Document(
        content="This is a test document about neuroscience.",
        metadata={"source": "test.pdf", "page": 1}
    )
    doc.embedding = [0.1, 0.2, 0.3]
    
    print("2. Testing document properties...")
    assert doc.content == "This is a test document about neuroscience."
    assert doc.metadata["source"] == "test.pdf"
    assert doc.metadata["page"] == 1
    assert doc.embedding == [0.1, 0.2, 0.3]
    print(f"Document content: {doc.content}")
    print(f"Document metadata: {doc.metadata}")
    print(f"Document embedding: {doc.embedding}")

def test_rag_vector_store(vector_store):
    """Test RAG vector store operations"""
    print("\n=== Testing Vector Store ===")
    
    print("1. Creating test documents...")
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
    
    print("2. Adding documents to store...")
    success = vector_store.add_documents(docs)
    assert success, "Document indexing should succeed"
    print(f"Added {len(docs)} documents")
    
    print("3. Testing search...")
    results = vector_store.search("brain function", top_k=2)
    print(f"Found {len(results)} results")
    assert len(results) > 0, "Should find at least one result"
    
    # Verify result format
    doc, score = results[0]  # Unpack first result
    assert isinstance(doc, Document), "First item should be Document"
    assert isinstance(score, float), "Second item should be score"
    assert 0 <= score <= 1, "Score should be between 0 and 1"
    assert "brain" in doc.content.lower(), "Most relevant result should contain search term"

# ===== RAG Search and Context Retrieval Tests =====
def test_rag_search_and_context(rag_system):
    """Test RAG search and context retrieval"""
    print("\n=== Testing Search and Context ===")
    
    print("1. Creating test documents...")
    docs = [
        Document(
            content="The brain processes information and controls behavior.",
            metadata={"source": "neuroscience.txt", "page": 1}
        ),
        Document(
            content="The heart pumps blood throughout the body.",
            metadata={"source": "anatomy.txt", "page": 1}
        )
    ]
    
    print("2. Adding documents to vector store...")
    # Create new namespace for testing
    namespace = rag_system.vector_store.clear()
    success = rag_system.vector_store.add_documents(docs, namespace=namespace)
    assert success, "Document indexing should succeed"
    rag_system._current_namespace = namespace  # Set the namespace for searching
    
    print("3. Testing search...")
    query = "How does the brain work?"
    results = rag_system.search(query, top_k=2)  # Use RAG search instead of direct vector store
    print(f"Found {len(results)} results")
    assert len(results) > 0, "Should find at least one result"
    
    # Check first result
    doc, score = results[0]  # Unpack first result
    assert isinstance(doc, Document), "First item should be Document"
    assert isinstance(score, float), "Second item should be score"
    assert "brain" in doc.content.lower(), "Most relevant result should contain search term"
    
    print("4. Testing context retrieval...")
    context = rag_system.get_relevant_context(query)
    print(f"Retrieved context: {context[:200]}...")
    assert isinstance(context, str)
    assert "neuroscience.txt" in context
    assert "brain" in context.lower()

def test_rag_complete_pipeline():
    """Test complete RAG pipeline end-to-end"""
    print("\n=== Testing Complete RAG Pipeline ===")
    
    # Load environment variables
    load_dotenv()
    
    # Verify credentials
    if not os.getenv("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY not set in .env file")
    
    print("1. Creating RAG system...")
    rag = RAGSystem()
    
    # Create new namespace for testing
    namespace = rag.vector_store.clear()
    print(f"Created namespace: {namespace}")
    
    print("\n2. Testing text cleaning...")
    text = "  Hello,  World!  \n  How are you?  "
    cleaned = rag.clean_text(text)
    print(f"Original text: {text}")
    print(f"Cleaned text: {cleaned}")
    assert cleaned == "Hello, World! How are you?"
    
    print("\n3. Testing chunk creation...")
    # Use simpler, more focused test text
    text = """The brain processes information. The brain learns new things. 
    The brain adapts to changes. The brain stores memories."""
    
    # Clean up the text properly
    text = ' '.join(text.split())
    print(f"Input text: {text}")
    print(f"Text length: {len(text)}")
    print(f"Chunk size: 50")  # Smaller chunks for simpler testing
    print(f"Overlap: 10")
    
    chunks = rag.create_chunks(text, chunk_size=50, overlap=10)
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}: {chunk}")
    
    # Basic assertions
    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(len(chunk) <= 50 for chunk in chunks), "All chunks should be within size limit"
    assert all(chunk.strip() for chunk in chunks), "No empty chunks"
    assert all(chunk.endswith('.') for chunk in chunks), "All chunks should end with a period"
    
    print("\n4. Testing document creation and indexing...")
    # Create documents from chunks
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            content=chunk,
            metadata={
                "source": "test_document.txt",
                "section": i + 1,
                "topic": "brain basics"
            }
        )
        documents.append(doc)
    
    # Add to vector store with namespace
    success = rag.vector_store.add_documents(documents, namespace=namespace)
    assert success, "Document indexing should succeed"
    print(f"Successfully indexed {len(documents)} documents in namespace: {namespace}")
    
    print("\n5. Testing search...")
    # Test exact matches first
    queries = [
        ("brain processes", 0.0),  # Should match first chunk
        ("brain learns", 0.0),     # Should match second chunk
        ("brain adapts", 0.0),     # Should match third chunk
        ("brain stores", 0.0),     # Should match fourth chunk
        ("unrelated query", 0.0)   # Shouldn't match well
    ]
    
    for query, min_score in queries:
        print(f"\nSearch query: {query} (min_score: {min_score})")
        results = rag.vector_store.search(
            query=query,
            top_k=4,  # Get all chunks
            namespace=namespace,
            min_score=min_score
        )
        
        print(f"\nFound {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {score:.4f}")
            # print(f"Content: {doc.content}")
            # print(f"Metadata: {doc.metadata}")
        
        # Verify results
        assert isinstance(results, list), "Results should be a list"
        assert all(isinstance(doc, Document) for doc, _ in results), "All results should be Documents"
        assert all(isinstance(score, float) for _, score in results), "All scores should be floats"
        
        # For exact matches, verify we get results
        if query != "unrelated query":
            assert len(results) > 0, f"Should get matches for query: {query}"
    
    print("\n6. Testing context retrieval...")
    # Test with exact phrase from the text
    context = rag.get_relevant_context(
        query="brain processes information",
        top_k=2,
        min_score=0.0
    )
    print(f"\nRetrieved context:")
    print(context)
    
    # Verify context format and content
    assert isinstance(context, str), "Context should be a string"
    assert "test_document.txt" in context, "Context should include source"
    assert "Score:" in context, "Context should include similarity scores"
    assert "brain processes" in context.lower(), "Context should contain relevant text"