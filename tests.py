"""
Tests both the chat and test endpoints with different scenarios.
"""

from app import app

# test chat route
def test_chat_route():
    """Test successful chat message handling."""
    client = app.test_client()
    res = client.post("/chat/test", json={"message": "Hello"})
    assert res.status_code == 200
    assert "response" in res.get_json()

# test chat route with missing message
def test_chat_route_missing_message():
    """Test chat endpoint with missing message."""
    client = app.test_client()
    res = client.post("/chat/test", json={})
    assert res.status_code == 400
    assert "error" in res.get_json()

# test test route
def test_test_route():
    """Test the direct Bedrock interaction endpoint."""
    client = app.test_client()
    res = client.post("/test", json={"message": "Hello"})
    assert res.status_code == 200
    assert "response" in res.get_json()
