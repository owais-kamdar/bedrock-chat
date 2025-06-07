"""
Flask application that provides an API endpoint for Claude-2 Bedrock foundation models.
Some models (Deepseek, Meta-Llama) are not available for on-demand use.
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from bedrock import call_bedrock, SUPPORTED_MODELS
from logger import log_chat_request, log_error, log_model_usage, get_session_logger
from dotenv import load_dotenv
import boto3
import json
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask and Swagger
app = Flask(__name__)
api = Api(app, version='1.0', 
    title='BedrockChat API',
    description='API for interacting with AWS Bedrock foundation models (Claude-2, Nova, Mistral)',
    doc='/docs',
    authorizations={
        'apikey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-KEY'
        }
    },
    security='apikey'
)

# Define namespaces
ns = api.namespace('', description='AI Chat Operations')

# Define models for request/response documentation
chat_request = api.model('ChatRequest', {
    'message': fields.String(required=True, description='The message to send to the model'),
    'model': fields.String(required=False, description='Model to use (claude-2, nova, mistral)', enum=list(SUPPORTED_MODELS.keys()), default='claude-2'),
})

chat_response = api.model('ChatResponse', {
    'response': fields.String(description='Model response'),
    'model': fields.String(description='Model used')
})

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# chat session history
memory_store = {}

# check if api is valid. going to add further authentication in the future.
def check_api_key():
    """Check if the API key in the request header matches the expected key."""
    api_key = request.headers.get('X-API-KEY')
    expected_key = os.getenv('API_KEY')
    if not api_key or api_key != expected_key:
        api.abort(401, "Invalid or missing API key")

def check_guardrail(text: str, source: str = "INPUT") -> dict:
    """
    Check if text passes guardrail using ApplyGuardrail API from bedrock.
    
    Args:
        text: Text to check
        source: Either "INPUT" or "OUTPUT"
    
    Returns:
        dict with status and reason
    """
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID")
    if not guardrail_id:
        return {"passed": True, "reason": "No guardrail configured"}
        
    try:
        # Call ApplyGuardrail API
        response = bedrock.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion="2",  # Default to version 2 with some customizations
            source=source,
            content=[{
                "text": {
                    "text": text
                }
            }]
        )
        
        # Check if guardrail passed
        passed = response.get("action") != "GUARDRAIL_INTERVENED"
        reason = response.get("actionReason", "Unknown reason") if not passed else "PASS"
            
        return {
            "passed": passed,
            "reason": reason
        }
        
    except Exception as e:
        return {
            "passed": False,
            "reason": f"Guardrail error: {str(e)}"
        }

# list available models
@ns.route('/models')
class ModelList(Resource):
    @api.doc('list_models', 
             description='List available models (claude-2, nova, mistral)',
             responses={200: 'Success', 401: 'Authentication error'})
    def get(self):
        """List available models"""
        check_api_key()
        return {"models": list(SUPPORTED_MODELS.keys())}

# chat endpoint for different sessions
@ns.route('/chat/<string:session_id>')
@api.doc(params={'session_id': 'Unique identifier for the chat session'})
class Chat(Resource):
    @api.expect(chat_request)
    @api.marshal_with(chat_response, code=200, description='Success')
    @api.doc(responses={
        400: 'Validation error',
        401: 'Authentication error',
        500: 'Server error'
    })
    def post(self, session_id):
        """
        Send a message in a chat session
        """
        check_api_key()
        try:
            # Get request data
            data = request.json
            user_msg = data.get("message")
            model = data.get("model", "claude-2")

            # Validate request
            if not user_msg:
                api.abort(400, "Message is required")
            if model not in SUPPORTED_MODELS:
                api.abort(400, f"Unsupported model. Choose from: {list(SUPPORTED_MODELS.keys())}")

            # Guardrail check on input
            input_guardrail = check_guardrail(user_msg, "INPUT")
            if not input_guardrail["passed"]:
                api.abort(400, f"Input content rejected by guardrail: {input_guardrail['reason']}")

            # Pull full message history
            messages = memory_store.get(session_id, []).copy()

            # Add user message
            messages.append({"role": "user", "content": user_msg})

            # Call model and measure duration
            start_time = datetime.now()
            response = call_bedrock(messages, model)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            assistant_reply = response.strip()
            
            # Check response with guardrail
            output_guardrail = check_guardrail(assistant_reply, "OUTPUT")
            if not output_guardrail["passed"]:
                api.abort(400, f"Model response rejected by guardrail: {output_guardrail['reason']}")
            
            # Get session logger
            logger = get_session_logger(session_id)
            
            # Log the interaction with metrics
            logger.log_event("INTERACTION", {
                "message_number": len(messages),
                "model": model,
                "input_chars": len(user_msg),
                "output_chars": len(assistant_reply),
                "input_tokens": len(user_msg.split()),
                "output_tokens": len(assistant_reply.split()),
                "duration_ms": duration_ms,
                "input_text": user_msg,
                "output_text": assistant_reply,
                "total_tokens": len(user_msg.split()) + len(assistant_reply.split()),
                "guardrails": {
                    "input": input_guardrail,
                    "output": output_guardrail
                }
            })

            # Store in memory
            if session_id not in memory_store:
                memory_store[session_id] = []
            memory_store[session_id].extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_reply}
            ])

            return {
                "response": assistant_reply,
                "model": model
            }

        except Exception as e:
            error_msg = str(e)
            log_error(error_msg, session_id)
            return {
                "error": "Server error",
                "details": str(e)
            }, 500

# test endpoint for direct model interaction
@ns.route('/test')
class Test(Resource):
    @api.expect(chat_request)
    @api.doc(responses={
        400: 'Validation error',
        401: 'Authentication error',
        500: 'Server error'
    })
    def post(self):
        """
        Simple test endpoint - just tests model response and guardrail
        """
        check_api_key()
        try:
            # Get and validate request data
            data = request.json
            if not data:
                return {"error": "No JSON data provided"}, 400
                
            user_msg = data.get("message", "Hello!")
            model = data.get("model", "claude-2")

            # Quick guardrail check
            guardrail_result = check_guardrail(user_msg, "INPUT")
            if not guardrail_result["passed"]:
                return {
                    "error": "Guardrail rejected input",
                    "details": guardrail_result["reason"]
                }, 400

            # Simple one message test
            messages = [{"role": "user", "content": user_msg}]
            response = call_bedrock(messages, model)
            
            return {
                "response": response.strip(),
                "model": model
            }
            
        except Exception as e:
            log_error(str(e))
            return {
                "error": "Server error",
                "details": str(e)
            }, 500

if __name__ == "__main__":
    import sys
    sys.stdout = open('logs.txt', 'a')
    sys.stderr = sys.stdout
    app.run(debug=True, host="0.0.0.0", port=5001)
