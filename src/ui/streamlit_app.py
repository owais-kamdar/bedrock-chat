"""
Streamlit interface for BedrockChat
"""

import streamlit as st
import requests
from src.core.bedrock import SUPPORTED_MODELS
from dotenv import load_dotenv
import os
from src.core.rag import RAGSystem
from src.utils.user_manager import user_manager
from src.core.config import ALLOWED_USER_FILE_TYPES, DEFAULT_TOP_K
import numpy as np
import uuid
import io

# Load environment variables
load_dotenv("config/.env")

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
    st.session_state.session_id = str(uuid.uuid4())
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = user_manager.generate_user_id()
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
    # Check if documents are already indexed
    stats = st.session_state.rag.vector_store.index.describe_index_stats()
    if stats.total_vector_count == 0:
        with st.spinner("Indexing neuroscience documents..."):
            try:
                st.session_state.rag.index_neuroscience_documents()
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
if "user_files" not in st.session_state:
    st.session_state.user_files = []

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
    
    # API Key section
    st.markdown("### API Key")
    if not st.session_state.api_key:
        if st.button("Generate New API Key"):
            try:
                response = requests.post(
                    f"{API_URL}/api-keys",
                    json={"user_id": st.session_state.user_id}
                )
                if response.status_code == 201:
                    st.session_state.api_key = response.json()["api_key"]
                    st.success("API key generated successfully!")
                else:
                    st.error("Failed to generate API key")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your API key for authentication"
    )
    st.session_state.api_key = api_key
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### 📁 Upload Documents")
    st.markdown("Upload your own **PDF or TXT files** to add them to your personal context window.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=list(ALLOWED_USER_FILE_TYPES),
        help="Upload PDF or TXT files to add to your personal knowledge base"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }

        st.write("**File Details:**")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        # Upload button
        if st.button("Upload & Index File"):
            with st.spinner("Uploading and indexing file..."):
                try:
                    # Read file content
                    file_content = uploaded_file.read()
                    
                    # determine file type
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    # validate file type
                    if file_type not in ALLOWED_USER_FILE_TYPES:
                        supported_types = ', '.join(ALLOWED_USER_FILE_TYPES).upper()
                        st.error(f"Only {supported_types} files are supported!")
                        st.stop()
                    
                    # upload to RAG system
                    result = st.session_state.rag.upload_user_file(
                        file_content=file_content,
                        filename=uploaded_file.name,
                        user_id=st.session_state.user_id,
                        file_type=file_type
                    )
                    
                    if result["success"]:
                        st.success(f"{result['message']}")
                        # Refresh user files list
                        st.session_state.user_files = st.session_state.rag.list_user_files(st.session_state.user_id)
                        st.rerun()
                    else:
                        st.error(f"{result['error']}")
                        
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
    
    # User Files Management
    if st.session_state.user_files:
        st.markdown("---")
        st.markdown("### Your Documents")
        
        for i, file_info in enumerate(st.session_state.user_files):
            file_type_icon = "📄" if file_info.get('file_type') == 'pdf' else "📝"
            with st.expander(f"{file_type_icon} {file_info['filename']}"):
                st.write(f"**Type:** {file_info.get('file_type', 'unknown').upper()}")
                st.write(f"**Size:** {file_info['size'] / 1024:.1f} KB")
                st.write(f"**Uploaded:** {file_info['upload_date'][:10]}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    if st.session_state.rag.delete_user_file(st.session_state.user_id, file_info['s3_key']):
                        st.success("File deleted successfully!")
                        st.session_state.user_files = st.session_state.rag.list_user_files(st.session_state.user_id)
                        st.rerun()
                    else:
                        st.error("Failed to delete file")
    
    # Load user files on startup
    if not st.session_state.user_files:
        st.session_state.user_files = st.session_state.rag.list_user_files(st.session_state.user_id)
    
    st.markdown("---")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        list(SUPPORTED_MODELS.keys()),
        help="Choose which model to use for responses"
    )

    st.markdown("---")
    
    # RAG Settings
    st.markdown("### Context Settings")
    
    # Context source selection
    context_source = st.radio(
        "Context Source",
        ["Neuroscience Guide", "Your Documents", "Both", "None"],
        help="Choose which documents to use for context"
    )
    
    use_rag = context_source != "None"
    
    # Number of chunks slider
    if use_rag:
        num_chunks = st.slider(
            "Number of Context Chunks",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Adjust how many relevant text chunks to include in the context"
        )
    else:
        num_chunks = DEFAULT_TOP_K

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
        st.error("Please generate or enter an API key in the sidebar")
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
                        "messages": st.session_state.messages[:-1],  # Exclude current message
                        "context_source": context_source,
                        "user_id": st.session_state.user_id
                    }

                    # Send request to API
                    response = requests.post(
                        f"{API_URL}/chat/{st.session_state.user_id}",
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
