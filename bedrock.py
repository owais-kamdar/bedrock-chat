"""
AWS Bedrock client for models.
"""

import boto3
import json
from typing import List, Dict, Any, Optional
import os

# Initialize client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model IDs for supported models
SUPPORTED_MODELS = {
    # These models are not available for on-demand use
    # "deepseek": "deepseek.r1-v1:0",
    # "meta-llama": "meta.llama3-2-3b-instruct-v1:0",
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
    # Claude 2 uses prompt format
    if "claude-v2" in model_id:
        prompt = "\n\n"
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n\n"
            else:
                prompt += f"Assistant: {msg['content']}\n\n"
        prompt += "Assistant:"  # Add final assistant prompt
        
        return {
            "prompt": prompt,
            "max_tokens_to_sample": 1000,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"
        }
        
    # Nova uses content arrays with text field
    elif "nova" in model_id:
        return {
            "inferenceConfig": {
                "max_new_tokens": 1000
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
    
    # Note: Llama and Deepseek are not available for on-demand use
    # elif "llama" in model_id:
    #     return {
    #         "inputs": [messages],
    #         "parameters": {
    #             "max_new_tokens": 512,
    #             "top_p": 0.9,
    #             "temperature": 0.6
    #         }
    #     }
    
    # elif "deepseek" in model_id:
    #     return {
    #         "inferenceConfig": {
    #             "max_tokens": 512
    #         },
    #         "messages": messages
    #     }
    
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
            "max_tokens": 500,
            "temperature": 0.5,
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
    
    # Note: Llama and Deepseek are not available for on-demand use
    # elif "llama" in model_id or "deepseek" in model_id:
    #     return response.get("generation", "").strip()
    
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
    
    Raises:
        ValueError: If model is not supported
        Exception: For API errors
    """
    try:
        # Get model ID
        model_id = SUPPORTED_MODELS.get(model)
        if not model_id:
            raise ValueError(f"Unsupported model: {model}")

        # Format request
        body = format_messages_for_model(messages, model_id)
            
        # Call Bedrock API
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        # Parse and return response
        result = json.loads(response["body"].read())
        return parse_model_response(result, model_id)

    except Exception as e:
        raise Exception(f"Error calling Bedrock: {str(e)}")
