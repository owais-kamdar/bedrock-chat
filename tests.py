"""
Tests for BedrockChat API
"""

import os
import pytest
import json
from datetime import datetime
from app import app
from unittest.mock import patch, MagicMock, ANY
from bedrock import call_bedrock, SUPPORTED_MODELS

@pytest.fixture
def client():
    """Create test client"""
    # Patch out all logging functions
    with patch('app.log_chat_request'), \
         patch('app.log_error'), \
         patch('app.log_model_usage'), \
         patch('app.get_session_logger'):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

def test_chat_endpoint_no_auth(client):
    """Test chat endpoint without API key"""
    response = client.post('/chat/test_session', json={'message': 'Hello'})
    assert response.status_code == 401

def test_chat_endpoint_invalid_auth(client):
    """Test chat endpoint with invalid API key"""
    response = client.post(
        '/chat/test_session',
        json={'message': 'Hello'},
        headers={'X-API-KEY': 'invalid_key'}
    )
    assert response.status_code == 401

@patch('bedrock.bedrock')
def test_chat_endpoint_success(mock_bedrock, client):
    """Test successful chat request"""
    # Mock bedrock response
    mock_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"completion": "Hello there!"}'
            )
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response
    mock_bedrock.apply_guardrail.return_value = {"action": "ALLOW"}

    # Set test API key
    os.environ['API_KEY'] = 'test_key'
    
    # Test request
    response = client.post(
        '/chat/test_session',
        json={
            'message': 'Hello',
            'model': 'claude-2'
        },
        headers={'X-API-KEY': 'test_key'}
    )

    assert response.status_code == 200
    assert response.json['response'] == 'Hello there!'
    assert response.json['model'] == 'claude-2'

@patch('bedrock.bedrock')
def test_multiple_models(mock_bedrock):
    """Test different model calls"""
    # Mock Claude response
    claude_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"completion": "Response"}'
            )
        )
    }
    
    # Mock Nova response
    nova_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"output": {"message": {"content": [{"text": "Response"}]}}}'
            )
        )
    }

    # Test Claude
    mock_bedrock.invoke_model.return_value = claude_response
    response = call_bedrock([{"role": "user", "content": "Hello"}], "claude-2")
    assert response == "Response"
    mock_bedrock.invoke_model.assert_called_with(
        modelId=SUPPORTED_MODELS["claude-2"],
        contentType="application/json",
        accept="application/json",
        body=ANY
    )

    # Test Nova
    mock_bedrock.invoke_model.return_value = nova_response
    response = call_bedrock([{"role": "user", "content": "Hello"}], "nova")
    assert response == "Response"
    mock_bedrock.invoke_model.assert_called_with(
        modelId=SUPPORTED_MODELS["nova"],
        contentType="application/json",
        accept="application/json",
        body=ANY
    )

    # Test invalid model
    with pytest.raises(Exception) as exc:
        call_bedrock([{"role": "user", "content": "Hello"}], "invalid_model")
    assert "Unsupported model: invalid_model" in str(exc.value)

def test_models_endpoint(client):
    """Test models listing endpoint"""
    # Set test API key
    os.environ['API_KEY'] = 'test_key'

    response = client.get(
        '/models',
        headers={'X-API-KEY': 'test_key'}
    )

    assert response.status_code == 200
    assert 'models' in response.json
    assert set(response.json['models']) == {"claude-2", "nova", "mistral"}

@patch('bedrock.bedrock')
def test_test_endpoint(mock_bedrock, client):
    """Test the simple test endpoint"""
    # Mock bedrock response
    mock_response = {
        'body': MagicMock(
            read=MagicMock(
                return_value='{"completion": "Test response"}'
            )
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response
    mock_bedrock.apply_guardrail.return_value = {"action": "ALLOW"}

    os.environ['API_KEY'] = 'test_key'
    
    response = client.post(
        '/test',
        json={
            'message': 'Hello',
            'model': 'claude-2'
        },
        headers={'X-API-KEY': 'test_key'}
    )
    
    assert response.status_code == 200
    assert 'response' in response.json
    assert 'model' in response.json
