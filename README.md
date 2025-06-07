# BedrockChat
A further implemenation of Bedrock chat interface from v1 (week1).
A streamlit based chat interface for Amazon Bedrock foundation models with a Streamlit frontend and Flask API backend. Currently supports Claude, Nova, and Mistral models with session management and usage logging.

## Current Features

- Multiple Bedrock Models:
  - Claude-2
  - Nova
  - Mistral
- Session Management & Chat History
- Detailed Usage Logging
- Basic Guardrail Support
  - Currently supporting prompt and response filters
  - Profanity and Specific Topic blocking
  - PII Blocks/Masks
- API Key Authentication

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

# Optional
BEDROCK_GUARDRAIL_ID=your-guardrail-id
```

## Usage

1. Start the Flask API:
```bash
python app.py
```

2. In a separate terminal, start the Streamlit interface:
```bash
python -m streamlit run streamlit.py
```

Access the UI at `http://localhost:8501`

## Architecture

- `app.py` - Flask API with authentication and model endpoints
- `bedrock.py` - Bedrock client and model handling
- `streamlit.py` - Web interface and chat management
- `logger.py` - Session and interaction logging

## API Endpoints

- `POST /chat/<session_id>` - Send a message in a chat session
- `GET /models` - List available models
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

### 1. RAG Integration
- S3-based document storage
- Document embedding and retrieval
- Context injection into prompts

### 2. Enhanced Guardrails
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

### 5. Security Enhancements
- Rate limiting

## Development

Run tests:
```bash
pytest tests.py
```

## License

MIT License - See LICENSE file for details
