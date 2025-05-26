# Claude Chat Application

A simple chat application using Claude 3 Haiku via AWS Bedrock. This project provides a web interface for chatting with Claude using AWS Bedrock's API.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/owais-kamdar/claude-chat.git
cd claude-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
```

## Running the Application

1. Start the Flask backend:
```bash
python app.py
```

2. Start the Streamlit interface:
```bash
streamlit run streamlit.py
```

3. Open your browser to `http://localhost:8501`

## API Endpoints

- `POST /chat/<session_id>`: Chat with Claude (main endpoint)
- `POST /test`: Test endpoint for direct Claude interaction

## Testing

Run the test suite:
```bash
pytest tests.py
```
