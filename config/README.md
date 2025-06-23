# Configuration

Application configuration and environment variables for BedrockChat.

## Files

- **`.env`** - Environment variables and settings
- **`config.py`** - Centralized configuration module with defaults and validation

## Configuration Sections

### API Configuration
- `API_KEY` - API key

### AWS Configuration  
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_DEFAULT_REGION` - AWS region
- `RAG_BUCKET` - S3 bucket for documents, logs, and user data

### S3 Folder Structure
- `FILE_FOLDER` - Base neuroscience documents folder 
- `USER_UPLOADS_FOLDER` - User uploaded documents
- `API_KEYS_FOLDER` - User API key storage
- `LOGS_FOLDER` - Application logs

### Bedrock Configuration
- `BEDROCK_GUARDRAIL_ID` - Guardrail ID for content filtering
- `GUARDRAIL_VERSION` - Guardrail version

### Pinecone Configuration
- `PINECONE_API_KEY` - Pinecone API key for vector database
- `PINECONE_INDEX_NAME` - Vector database index name
- `PINECONE_INDEX_HOST` - Pinecone index host URL

## System Defaults (configured in config.py)

### RAG Configuration
- `CHUNK_SIZE = 1000` - Text chunk size for document processing
- `CHUNK_OVERLAP = 200` - Overlap between text chunks
- `ALLOWED_USER_FILE_TYPES = {'pdf', 'txt'}` - Supported file types for upload
- `DEFAULT_TOP_K = 5` - Default number of search results to retrieve

### Model Configuration
- `DEFAULT_MAX_TOKENS = 1000` - Default token limit for Claude and Nova models
- `MISTRAL_MAX_TOKENS = 500` - Token limit for Mistral model

### Vector Store Configuration
- `VECTOR_DIMENSION = 1024` - Embedding dimension for llama-text-embed-v2
- `VECTOR_METRIC = "cosine"` - Similarity metric for vector search
- `EMBEDDING_BATCH_SIZE = 96` - Batch size for generating embeddings
- `UPSERT_BATCH_SIZE = 100` - Batch size for uploading vectors to Pinecone

## Usage

All configuration is centralized in `src/core/config.py`