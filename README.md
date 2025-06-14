# BedrockChat

A further implemenation of Bedrock chat interface from v1 (week1).
A streamlit based chat interface for Amazon Bedrock foundation models with a Streamlit frontend and Flask API backend. Currently supports Claude, Nova, and Mistral models with session management and usage logging. Supports RAG based knowledge for Neuroscience materials.
## Current Features

- Multiple Bedrock Models:
  - Claude-2
  - Nova
  - Mistral
- Basic Guardrail Support
  - Prompt and response filters
  - Profanity and topic blocking
  - PII Blocks/Masks
- API Key Authentication
- RAG (Retrieval Augmented Generation)
  - PDF document indexing
  - Semantic search with Pinecone
  - Context-aware responses
  - Detailed Usage Logging
- Test Suite
  - Current coverage: 62%

## Setup

1. Clone the repository:
```bash
git clone https://github.com/owais-kamdar/bedrock-chat.git
cd bedrock
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```
# Required
API_KEY=your-secret-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1

# Guardrails and RAG Configuration
BEDROCK_GUARDRAIL_ID=your-guardrail-id
RAG_BUCKET=your-s3-bucket-name
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
```

## Usage

1. Initialize the RAG system (first time only):
```bash
python initialize_rag.py
```

2. Start the Flask API:
```bash
python app.py
```

3. In a separate terminal, start the Streamlit interface:
```bash
python -m streamlit run streamlit.py
```

Access the UI at `http://localhost:8501`

## Architecture

- `app.py` - Flask API with authentication and model endpoints
- `bedrock.py` - Bedrock client and model handling
- `streamlit.py` - Web interface and chat management
- `logger.py` - Session and interaction logging
- `tests.py` - Overall test suite for pipeline
- `initialize_rag.py` - Script for setting up and indexing RAG documents
- `vector_store.py` - Vector database interface for document storage and retrieval
- `rag.py` - RAG system for document processing and context retrieval

## API Endpoints

- `POST /chat/<session_id>` - Send a message in a chat session with optional RAG context
- `GET /models` - List available models (claude-2, nova, mistral)
- `POST /test` - Test endpoint for direct model interaction

All endpoints require an `X-API-KEY` header matching the one in your `.env` file.

## Logging

Logs are stored in `session_logs/` with detailed metrics for each interaction:
- Message counts and timestamps
- Input/output character and token counts
- Model selection
- Response times
- Guardrail results

## Future Plans

### 1. RAG Integration -- WEEK 4
- S3-based document storage
- Document embedding and retrieval
- Context injection into prompts

### 2. Enhanced Guardrails -- WEEK 4
- Custom safety filters
- Topic-based filtering
- Response quality checks

### 3. Additional Models
- Llama
- Deepseek

### 4. Deployment
- AWS deployment options:
  - App Runner
  - EC2
- Monitoring and alerts

### 5. Security
- Rate limiting

### 6. Analytics
- Dashboards
- Reports

## Development

Run tests:
```bash
pytest tests.py -v --cov=. --cov-report=term-missing
```

## Weekly Updates

### Week 4: RAG System Integration
The project now includes a fully functional Retrieval Augmented Generation (RAG) system with the following features:


- **Document Processing and Vector Store**
  - Text chunking with overlap for context preservation
  - Batch processing for embeddings
  - Pinecone serverless vector database integration using GRPC
  - Document chunking and embedding generation

- **Search and Retrieval**
  - Semantic similarity search using Llama embeddings
  - Configurable relevance scoring and filtering

- **Integration with Chat Interface**
  - Toggle for RAG-enhanced responses

- **Added Data for Tracking**
  - source
  - page numbers 
  - chunks and scores
  - timestamps

- **Current Documents: Neuroscience**
  - [Brain Facts Book](https://www.brainfacts.org/-/media/Brainfacts2/BrainFacts-Book/Brain_Facts_BookHighRes.pdf)
  - [Neuroscience: Science of the Brain](https://brain.mcmaster.ca/BrainBee/Neuroscience.Science.of.the.Brain.pdf)

## License

MIT License
