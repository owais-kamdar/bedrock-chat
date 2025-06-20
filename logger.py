"""
This module is used to log chat sessions and interactions.
Logs are stored in S3 for better persistence and accessibility.
"""

import boto3
import json
import os
from datetime import datetime
from typing import Dict
from io import StringIO

class S3Logger:
    # S3 folder for logs
    LOGS_FOLDER = 'logs'
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.message_count = 0
        
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        self.bucket = os.getenv('RAG_BUCKET')
        if not self.bucket:
            raise ValueError("RAG_BUCKET environment variable not set")
        
        # Create log buffer
        self.log_buffer = StringIO()
        
        # Log session start
        self.log_event("SESSION_START", {
            "session_id": session_id,
            "start_time": self.start_time.isoformat()
        })
    
    def log_event(self, event_type: str, data: Dict):
        """Log an event in JSON format"""
        event = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Add to buffer
        self.log_buffer.write(json.dumps(event) + "\n")
        
        # Write to S3
        self._write_to_s3()
    
    def _write_to_s3(self):
        """Write current buffer to S3"""
        try:
            # Generate S3 key based on session and date
            date_str = self.start_time.strftime('%Y%m%d')
            key = f'{self.LOGS_FOLDER}/{date_str}/{self.session_id}.log'
            
            # Upload buffer content
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=self.log_buffer.getvalue()
            )
        except Exception as e:
            print(f"Error writing to S3: {str(e)}")
    
    def end_session(self):
        """Log session end with summary metrics"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            "session_id": self.session_id,
            "total_messages": self.message_count,
            "duration_seconds": duration,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        self.log_event("SESSION_END", summary)
        
        # Final write and close buffer
        self._write_to_s3()
        self.log_buffer.close()

# Global store for session loggers
session_loggers: Dict[str, S3Logger] = {}

def get_session_logger(session_id: str) -> S3Logger:
    """Get or create a session logger"""
    if session_id not in session_loggers:
        session_loggers[session_id] = S3Logger(session_id)
    return session_loggers[session_id]

def log_chat_request(session_id: str, model: str, prompt: str, response: str = "", duration_ms: float = 0):
    """Log chat request with metrics"""
    logger = get_session_logger(session_id)
    logger.message_count += 1
    logger.log_event("INTERACTION", {
        "message_number": logger.message_count,
        "model": model,
        "duration_ms": duration_ms,
        "input_text": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "output_text": response[:100] + "..." if len(response) > 100 else response
    })

def log_error(error_msg: str, session_id: str = "system"):
    """Log error messages"""
    logger = get_session_logger(session_id)
    logger.log_event("ERROR", {"message": error_msg})

def end_session(session_id: str):
    """End a session and log summary"""
    if session_id in session_loggers:
        session_loggers[session_id].end_session()
        del session_loggers[session_id]

def log_model_usage(model: str, tokens_used: int, session_id: str = "system"):
    """Log model usage statistics"""
    logger = get_session_logger(session_id)
    logger.log_event("MODEL_USAGE", {
        "model": model,
        "tokens_used": tokens_used
    }) 