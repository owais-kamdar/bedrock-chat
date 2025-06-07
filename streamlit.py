"""
Streamlit interface for BedrockChat
"""

import streamlit as st
import requests
from bedrock import SUPPORTED_MODELS
from dotenv import load_dotenv
import os

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
