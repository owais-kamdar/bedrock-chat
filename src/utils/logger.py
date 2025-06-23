"""
Dual logging system for BedrockChat
Local file: for debugging | S3: for dashboard analytics
"""

import logging
import os
import boto3
import json
from datetime import datetime
from typing import Dict, Optional
from src.core.config import get_s3_client, RAG_BUCKET, LOGS_FOLDER

# Ensure logs directory exists
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure local logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "logs.txt")),
        logging.StreamHandler()
    ]
)

# S3 logging setup
s3_client = None
s3_enabled = False
try:
    if RAG_BUCKET:
        s3_client = get_s3_client()
        s3_enabled = True
except Exception as e:
    logging.warning(f"S3 logging disabled: {e}")

def _log_to_s3(user_id: str, event_type: str, data: Dict):
    """Simple S3 logging for dashboard"""
    if not s3_enabled:
        return
    
    try:
        log_entry = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Write to daily log file
        date_str = datetime.now().strftime('%Y%m%d')
        s3_key = f'{LOGS_FOLDER}/{date_str}/{user_id}.log'
        
        # Append to existing file
        try:
            # Get existing content
            response = s3_client.get_object(Bucket=RAG_BUCKET, Key=s3_key)
            existing_content = response['Body'].read().decode('utf-8')
            new_content = existing_content + '\n' + json.dumps(log_entry)
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist, create new
            new_content = json.dumps(log_entry)
        
        s3_client.put_object(
            Bucket=RAG_BUCKET,
            Key=s3_key,
            Body=new_content
        )
    except Exception as e:
        logging.error(f"S3 logging failed: {e}")

# log chat request/response
def log_chat_request(user_id: str, message: str, response: str, model: str, duration_ms: float, context_source: str = "none", rag_stats: Optional[Dict] = None, guardrails: Optional[Dict] = None):
    """Log a chat request/response"""
    # Local logging
    logging.info(f"[CHAT] User: {user_id}, Model: {model}, Duration: {duration_ms}ms")
    
    if rag_stats and rag_stats.get('enabled'):
        chunks = rag_stats.get('chunks_retrieved', 0)
        logging.info(f"[RAG] Retrieved {chunks} chunks from {context_source}")
    
    # S3 logging for dashboard
    s3_data = {
        "user_id": user_id,
        "model": model,
        "duration_ms": duration_ms,
        "context_source": context_source,
        "input_chars": len(message) if message else 0,
        "output_chars": len(response) if response else 0,
        "input_tokens": len(message.split()) if message else 0,
        "output_tokens": len(response.split()) if response else 0,
        "user_message": message[:200] + "..." if len(message) > 200 else message,
        "assistant_reply": response[:200] + "..." if len(response) > 200 else response
    }
    
    if rag_stats:
        s3_data.update(rag_stats)
    
    # add guardrail information if available
    if guardrails:
        s3_data["guardrails"] = guardrails
    
    _log_to_s3(user_id, "INTERACTION", s3_data)

def log_error(error_msg: str, user_id: str = "system"):
    """Log an error"""
    logging.error(f"[ERROR] {user_id}: {error_msg}")
    
    # S3 logging for dashboard
    _log_to_s3(user_id, "ERROR", {"user_id": user_id, "message": error_msg})

def log_rag_operation(operation: str, message: str):
    """Log RAG operations"""
    logging.info(f"[RAG-{operation}] {message}")

def log_document_upload(user_id: str, filename: str, success: bool, error: str = None):
    """Log document upload events"""
    status = "SUCCESS" if success else "FAILED"
    logging.info(f"[UPLOAD-{status}] User: {user_id}, File: {filename}")
    if error:
        logging.error(f"[UPLOAD-ERROR] User: {user_id}, File: {filename}, Error: {error}")
    
    # S3 logging for dashboard
    s3_data = {"user_id": user_id, "filename": filename, "success": success}
    if error:
        s3_data["error"] = error
    _log_to_s3(user_id, "DOCUMENT_UPLOAD", s3_data)

def log_initialization_step(step: str, status: str, data: Optional[Dict] = None):
    """Log initialization steps"""
    logging.info(f"[INIT-{step}] {status}")
    if data and data.get('error'):
        logging.error(f"[INIT-ERROR] {data['error']}")

def log_system_event(event: str, message: str):
    """Log system events"""
    logging.info(f"[SYSTEM-{event}] {message}")

def log_user_session(user_id: str, event: str):
    """Log user session events for dashboard"""
    _log_to_s3(user_id, f"USER_{event.upper()}", {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat()
    }) 