"""
Streamlit interface for BedrockChat
"""

import streamlit as st
import requests
from bedrock import SUPPORTED_MODELS
from dotenv import load_dotenv
import os
from rag import RAGSystem
import numpy as np

# Load environment variables
load_dotenv()

# API settings
API_URL = "http://127.0.0.1:5001"

# Configure page
st.set_page_config(
    page_title="BedrockChat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-user-1"
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY", "")
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
    # Check if documents are already indexed
    stats = st.session_state.rag.vector_store.index.describe_index_stats()
    if stats.total_vector_count == 0:
        with st.spinner("Indexing neuroscience documents..."):
            try:
                st.session_state.rag.index_documents()
                st.success("Successfully indexed documents!")
            except Exception as e:
                st.error(f"Error indexing documents: {str(e)}")
                st.warning("Proceeding with empty index. Some features may be limited.")
    else:
        print(f"Using existing index with {stats.total_vector_count} vectors")
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 3
if "system_status" not in st.session_state:
    st.session_state.system_status = {
        "api_connected": False,
        "vector_store_ready": False,
        "bedrock_connected": False
    }

# Check system status
try:
    # Test API connection
    response = requests.get(f"{API_URL}/models", headers={"X-API-KEY": st.session_state.api_key})
    st.session_state.system_status["api_connected"] = response.status_code == 200
    
    # Test vector store
    st.session_state.system_status["vector_store_ready"] = len(st.session_state.rag.vector_store.documents) > 0
    
    # Test Bedrock connection
    test_embedding = st.session_state.rag.get_embedding("test")
    st.session_state.system_status["bedrock_connected"] = test_embedding is not None
except Exception:
    pass

# Sidebar configuration
with st.sidebar:
    st.title("Settings")
    
    # API Key input
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your API key for authentication"
    )
    st.session_state.api_key = api_key
    
    st.markdown("---")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        list(SUPPORTED_MODELS.keys()),
        help="Choose which model to use for responses"
    )

    st.markdown("---")
    
    # RAG Settings
    st.markdown("### RAG Settings")
    use_rag = st.toggle(
        "Enable Neuroscience Guide",
        help="When enabled, adds relevant neuroscience context to your queries",
        value=False
    )

    # Number of chunks slider
    num_chunks = 3
    if use_rag:
        num_chunks = st.slider(
            "Number of Context Chunks",
            min_value=1,
            max_value=10,
            value=3,
            help="Adjust how many relevant text chunks to include in the context"
        )

    st.markdown("---")
    
    # Environment info
    st.markdown("### Environment")
    guardrail_status = "Enabled" if os.getenv("BEDROCK_GUARDRAIL_ID") else "Disabled"
    st.info(f"Guardrail Status: {guardrail_status}")
    
    st.markdown("---")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("BedrockChat")
st.caption(f"Currently using: {model}")

# Create a container for chat history
chat_container = st.container()

# Display existing chat history
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    if not st.session_state.api_key:
        st.error("Please enter an API key in the sidebar")
    else:
        # Immediately show user message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show assistant response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare request
                    headers = {"X-API-KEY": st.session_state.api_key}
                    data = {
                        "message": prompt,
                        "model": model,
                        "use_rag": use_rag,
                        "num_chunks": num_chunks,
                        "messages": st.session_state.messages[:-1]  # Exclude current message
                    }

                    # Send request to API
                    response = requests.post(
                        f"{API_URL}/chat/{st.session_state.session_id}",
                        json=data,
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    # Show response
                    assistant_reply = response.json()["response"]
                    st.write(assistant_reply)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_reply
                    })

                except requests.exceptions.RequestException as e:
                    if hasattr(e.response, 'json'):
                        error_msg = e.response.json().get('error', str(e))
                    else:
                        error_msg = str(e)
                    st.error(f"Error: {error_msg}")
                    # Remove user message on error
                    st.session_state.messages.pop()
