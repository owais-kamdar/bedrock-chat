"""
Flask application that provides an API endpoint for Claude-Haiku3 model
"""

from flask import Flask, request, jsonify
from bedrock import call_bedrock

app = Flask(__name__)

# in-memory for chat session history
memory_store = {}

@app.route("/chat/<session_id>", methods=["POST"])

def chat(session_id):
    """
    Messages for each session.
    Args:
        session_id: Unique identifier for the chat session
    Returns:
        JSON response with Claude's reply or error message
    """
    try:
        user_msg = request.json.get("message")
        if not user_msg:
            return jsonify({"error": "Message is required"}), 400

        # Pull full message history
        messages = memory_store.get(session_id, []).copy()
        messages.append({"role": "user", "content": user_msg})

        # Log user message
        print(f"[USER] ({session_id}): {user_msg}")

        response = call_bedrock(messages)
        assistant_reply = response.strip()

        # model reply
        print(f"[CLAUDE] ({session_id}): {assistant_reply}")

        # Update memory with both user and assistant messages
        memory_store.setdefault(session_id, []).append({"role": "user", "content": user_msg})
        memory_store[session_id].append({"role": "assistant", "content": assistant_reply})

        return jsonify({"response": assistant_reply})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test_bedrock():
    """
    Test endpoint direct interaction.
    Returns:
        JSON response with Claude's reply or error message
    """
    try:
        user_msg = request.json.get("message", "Hello, how are you?")
        messages = [{"role": "user", "content": user_msg}]
        response = call_bedrock(messages)
        return jsonify({"response": response.strip()})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
