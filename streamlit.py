"""
Streamlit interface for the Claude chat application.
"""

import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Basic Chatbot with Claude", layout="centered")
st.title("Basic Chatbot with Claude")

# Initialize session state for chat history
session_id = "streamlit-user-1"
if "history" not in st.session_state:
    st.session_state.history = []

chat_container = st.container()

# Input form at bottom
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Type your message...", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

# Handle form submission
if submitted and user_input:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input})

    try:
        # Send message to Flask backend
        response = requests.post(f"http://127.0.0.1:5000/chat/{session_id}", json={"message": user_input})
        response.raise_for_status()
        assistant_reply = response.json()["response"]
    except Exception as e:
        assistant_reply = f"(Error: {e})"

    # Add assistant's reply to history
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})

# Display chat history
with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.info(f"**You:** {msg['content']}")
        else:
            st.success(f"**Claude:** {msg['content']}")
