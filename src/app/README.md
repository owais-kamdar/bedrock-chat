# Flask API Application

REST API for BedrockChat.

## Files

- **`flaskapp.py`** - Main Flask application with API endpoints

## Endpoints

- `POST /chat/{user_id}` - Send message to AI models
- `POST /upload/{user_id}` - Upload user documents for RAG
- `GET /files/{user_id}` - List user's uploaded files
- `DELETE /files/{user_id}?filename=` - Delete user's file
- `POST /api-keys` - Generate API key for user
- `GET /health` - Health check

## Authentication

Requires `X-API-KEY` header. Use `/api-keys` endpoint to generate keys. 