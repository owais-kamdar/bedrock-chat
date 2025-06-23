"""
Configuration module for BedrockChat
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from config/.env
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_env_path = os.path.join(_project_root, "config", ".env")
load_dotenv(_env_path)

# AWS Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
RAG_BUCKET = os.getenv("RAG_BUCKET")

# S3 Folder Structure
FILE_FOLDER = os.getenv("FILE_FOLDER", "documents")
USER_UPLOADS_FOLDER = os.getenv("USER_UPLOADS_FOLDER", "user_uploads")
API_KEYS_FOLDER = os.getenv("API_KEYS_FOLDER", "api_keys")
LOGS_FOLDER = os.getenv("LOGS_FOLDER", "logs")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bedrock-rag")

# Bedrock Configuration
BEDROCK_REGION = AWS_REGION
GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID")
GUARDRAIL_VERSION = os.getenv("GUARDRAIL_VERSION", "DRAFT")

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ALLOWED_USER_FILE_TYPES = {'pdf', 'txt'}
DEFAULT_TOP_K = 5

# Model Configuration
DEFAULT_MAX_TOKENS = 1000
MISTRAL_MAX_TOKENS = 500

# Vector Store Configuration
VECTOR_DIMENSION = 1024  # Dimension size for llama-text-embed-v2
VECTOR_METRIC = "cosine"  # Similarity metric
EMBEDDING_BATCH_SIZE = 96  # Optimal batch size for llama-text-embed-v2
UPSERT_BATCH_SIZE = 100  # Batch size for uploading vectors to Pinecone

# Document Processing
NEUROSCIENCE_PDF_FILES = [
    f"{FILE_FOLDER}/Brain_Facts_BookHighRes.pdf",
    f"{FILE_FOLDER}/Neuroscience.Science.of.the.Brain.pdf",
    f"{FILE_FOLDER}/psych.pdf"
]

# API Configuration
API_KEY = os.getenv("API_KEY")



def validate_required_env_vars() -> list[str]:
    """
    Validate that all required environment variables are set
    Returns list of missing variables
    """
    required_vars = [
        ("PINECONE_API_KEY", PINECONE_API_KEY),
        ("RAG_BUCKET", RAG_BUCKET),
    ]
    
    missing_vars = []
    for var_name, var_value in required_vars:
        if not var_value:
            missing_vars.append(var_name)
    
    return missing_vars

def get_s3_client():
    """Get configured S3 client"""
    import boto3
    return boto3.client("s3", region_name=AWS_REGION)

def get_bedrock_client():
    """Get configured Bedrock client"""
    import boto3
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION) 