"""
AWS Bedrock for calling Claude-Haiku3 model.
"""

import boto3
import json

# initialize bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def call_bedrock(messages):
    """
    Call Claude model via AWS Bedrock API.
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    Returns:
        str: Model's response text
    """
    # Build request
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7
    })

    # start model
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )

    # parse response for text
    result = json.loads(response["body"].read())
    return "".join(block["text"] for block in result["content"] if block["type"] == "text")
