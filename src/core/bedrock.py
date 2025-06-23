"""
AWS Bedrock client for models.
"""

import boto3
import json
from typing import List, Dict, Any, Optional
import os
from src.core.config import get_bedrock_client, DEFAULT_MAX_TOKENS, MISTRAL_MAX_TOKENS

# Initialize client using centralized config
bedrock = get_bedrock_client()

# available models
SUPPORTED_MODELS = {
    "claude-2": "anthropic.claude-v2:1",
    "nova": "amazon.nova-micro-v1:0",
    "mistral": "mistral.mistral-7b-instruct-v0:2"
}

# format messages for each model based on model documentation
def format_messages_for_model(messages: List[Dict[str, str]], model_id: str) -> Dict[str, Any]:
    """
    Format messages according to each model's required structure.
    Args:
        messages: List of message dicts, each with 'role' and 'content'
        model_id: The Bedrock model ID to format for
    Returns:
        Dict containing the properly formatted request body
    """
    # claude 2 prompt format
    if "claude-v2" in model_id:
        prompt = "\n\n"
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n\n"
            else:
                prompt += f"Assistant: {msg['content']}\n\n"
        prompt += "Assistant:"  # Add final assistant prompt
        
        # return the formatted request body
        return {
            "prompt": prompt,
            "max_tokens_to_sample": DEFAULT_MAX_TOKENS,
            "temperature": 0.7,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"
        }
        
    # nova uses content arrays with text field
    elif "nova" in model_id:
        return {
            "inferenceConfig": {
                "max_new_tokens": DEFAULT_MAX_TOKENS
            },
            "messages": [
                {
                    "role": msg["role"],
                    "content": [
                        {
                            "text": msg["content"]
                        }
                    ]
                }
                for msg in messages
            ]
        }
    

    
    # Mistral
    elif "mistral" in model_id:
        combined_prompt = "<s>"
        for msg in messages:
            if msg["role"] == "user":
                combined_prompt += f"[INST] {msg['content']} [/INST]"
            else:
                combined_prompt += msg["content"]
        
        return {
            "prompt": combined_prompt,
            "max_tokens": MISTRAL_MAX_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
        
    raise ValueError(f"Unsupported model: {model_id}")

# parse model response based on model documentation
def parse_model_response(response: Dict[str, Any], model_id: str) -> str:
    """
    Extract the generated text from model responses.
    
    Args:
        response: Raw response from Bedrock API
        model_id: The model ID that generated the response
    
    Returns:
        str: The extracted text response
    """
    if "claude-v2" in model_id:
        return response.get("completion", "").strip()
        
    elif "nova" in model_id:
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "").strip()
        return ""
        
    elif "mistral" in model_id:
        outputs = response.get("outputs", [])
        if outputs and len(outputs) > 0:
            return outputs[0].get("text", "").strip()
        return ""
    

    
    raise ValueError(f"Unsupported model: {model_id}")

# call bedrock api with with default claude-2 model
def call_bedrock(messages: List[Dict[str, str]], model: str = "claude-2") -> str:
    """
    Call AWS Bedrock API with proper formatting for each model.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier

    Returns:
        str: Model's response text
    """
    try:
        # Get model ID
        model_id = SUPPORTED_MODELS.get(model)
        if not model_id:
            raise ValueError(f"Unsupported model: {model}")

        # format request
        body = format_messages_for_model(messages, model_id)
            
        # call bedrock api
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        # parse and return response
        result = json.loads(response["body"].read())
        return parse_model_response(result, model_id)

    except Exception as e:
        raise Exception(f"Error calling Bedrock: {str(e)}")
