"""
This module is used to log chat sessions and interactions.
Will be used to track model usage and performance and create dashboards in the future.
"""

import logging
from datetime import datetime
import os
import json
from typing import Dict

# Create logs directory if it doesn't exist
LOGS_DIR = "session_logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR, mode=0o755)

class SessionLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.message_count = 0
        
        # Create session-specific log file
        self.log_file = os.path.join(LOGS_DIR, f"{session_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configure session logger
        self.logger = logging.getLogger(session_id)
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Add file handler for this session
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Log session start
        self.log_event("SESSION_START", {
            "session_id": session_id,
            "start_time": self.start_time.isoformat()
        })
    
    def log_event(self, event_type: str, data: Dict):
        """Log an event in JSON format"""
        self.logger.info(json.dumps({
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }))
    
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

# Global store for session loggers
session_loggers: Dict[str, SessionLogger] = {}

def get_session_logger(session_id: str) -> SessionLogger:
    """Get or create a session logger"""
    if session_id not in session_loggers:
        session_loggers[session_id] = SessionLogger(session_id)
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