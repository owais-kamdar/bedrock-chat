# Test Suite

Automated tests for BedrockChat functionality.

## Files

- **`test_bedrock.py`** - Comprehensive test suite covering all major components

## Test Coverage

- **API Endpoints** - Flask routes and authentication
- **RAG Pipeline** - Document processing, embedding, search
- **Vector Store** - Pinecone operations and initialization
- **Bedrock Integration** - Model calls and response parsing
- **User Management** - ID generation and API key handling

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_bedrock.py::test_chat_endpoint
```

## Requirements

Tests require valid environment variables in `config/.env` for live integration testing. 