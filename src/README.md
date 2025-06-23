# Source Code 

Core source code for BedrockChat application.

## Structure

- **`core/`** - Core logic (RAG, Bedrock, vector store, configuration)
- **`app/`** - Flask API application 
- **`ui/`** - Streamlit app, dashboard
- **`utils/`** - Logging, user management

## Entry Points

Use the run scripts in the project root:
- `python run_api.py` - Start Flask API
- `python run_streamlit.py` - Start Streamlit UI
- `python run_dash.py` - Start analytics dashboard
- `python run_initialize.py` - Initialize RAG system 