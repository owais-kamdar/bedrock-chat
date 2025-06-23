"""
Flask application that provides an API endpoint for Bedrock foundation models.
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, Namespace
from src.core.bedrock import call_bedrock, SUPPORTED_MODELS
from src.core.config import get_bedrock_client, GUARDRAIL_ID, GUARDRAIL_VERSION, API_KEY, DEFAULT_TOP_K
from src.core.rag import RAGSystem
from src.utils.logger import log_chat_request, log_error, log_document_upload
from src.utils.user_manager import user_manager
import json
import os
from datetime import datetime

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
    'use_rag': fields.Boolean(required=False, description='Whether to use RAG for neuroscience context', default=False),
    'num_chunks': fields.Integer(required=False, description='Number of context chunks to retrieve', default=DEFAULT_TOP_K),
    'context_source': fields.String(required=False, description='Context source', enum=['Neuroscience Guide', 'Your Documents', 'Both', 'None'], default='Neuroscience Guide'),
    'user_id': fields.String(required=False, description='User ID for user-specific RAG')
})

chat_response = api.model('ChatResponse', {
    'response': fields.String(description='Model response'),
    'model': fields.String(description='Model used'),
    'user_id': fields.String(description='User ID'),
    'context_source': fields.String(description='RAG context source used'),
    'rag_stats': fields.Raw(description='RAG retrieval statistics'),
    'duration_ms': fields.Float(description='Response time in milliseconds'),
    'guardrails_enabled': fields.Boolean(description='Whether guardrails are enabled'),
    'message_count': fields.Integer(description='Number of messages in conversation')
})

api_key_request = api.model('APIKeyRequest', {
    'user_id': fields.String(required=True, description='User ID to create API key for')
})

api_key_response = api.model('APIKeyResponse', {
    'api_key': fields.String(description='Generated API key')
})

# Initialize clients
bedrock = get_bedrock_client()
rag_system = RAGSystem()

# chat user history
memory_store = {}

# check api key function
def check_api_key():
    """Check if the API key in the request header is valid"""
    api_key = request.headers.get('X-API-KEY')
    
    if not api_key:
        api.abort(401, "Invalid or missing API key")
    
    if not (user_manager.validate_api_key(api_key) or (API_KEY and api_key == API_KEY)):
        api.abort(401, "Invalid or missing API key")

# check guardrail function to check if the text is safe to send to the model
def check_guardrail(text, direction):
    """Check text against guardrails if enabled"""
    if not is_guardrail_enabled():
        return {"passed": True}
    
    try:
        guardrail_response = bedrock.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source=direction,
            content=[{
                'text': {
                    'text': text
                }
            }]
        )
        
        action = guardrail_response['action']
        if action == 'GUARDRAIL_INTERVENED':
            outputs = guardrail_response.get('outputs', [])
            reason = outputs[0].get('text', 'Content blocked by guardrail') if outputs else 'Content blocked by guardrail'
            return {"passed": False, "reason": reason}
        else:
            return {"passed": True}
    except Exception as e:
        # If guardrail check fails, log error but allow request to proceed
        log_error(f"Guardrail check failed: {str(e)}")
        return {"passed": True}

# check if guardrails are enabled
def is_guardrail_enabled():
    """Check if guardrails are enabled"""
    return bool(GUARDRAIL_ID)

# get user id from api key
def get_user_id_from_api_key(api_key):
    """Get user ID from API key"""
    # Try to get from user manager first
    user_id = user_manager.get_user_id_from_key(api_key)
    if user_id:
        return user_id
    
    # Fallback to a default user ID if not found
    return "user-default"

# API key management endpoints
@ns.route('/api-keys')
class APIKeys(Resource):
    @api.expect(api_key_request)
    @api.marshal_with(api_key_response, code=201, description='API key created')
    @api.doc(responses={
        201: 'API key created',
        400: 'Validation error'
    })
    def post(self):
        """Create a new API key"""
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            api.abort(400, "user_id is required")
        
        # Normalize user ID to ensure consistency
        normalized_user_id = user_manager.normalize_user_id(user_id)
        api_key = user_manager.create_api_key(normalized_user_id)
        return {'api_key': api_key}, 201

@ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@ns.route('/models')
class ModelList(Resource):
    @api.doc('list_models', 
             description='List available models (claude-2, nova, mistral)',
             responses={200: 'Success', 401: 'Authentication error'})
    def get(self):
        """List available models"""
        check_api_key()
        return {"models": list(SUPPORTED_MODELS.keys())}

# Chat endpoint using user_id 
@ns.route('/chat/<string:user_id>')
@api.doc(params={'user_id': 'User identifier (use "new" to generate a new user)'})
class Chat(Resource):
    @api.expect(chat_request)
    @api.marshal_with(chat_response, code=200, description='Success')
    @api.doc(responses={
        400: 'Validation error',
        401: 'Authentication error',
        500: 'Server error'
    })
    def post(self, user_id):
        """Send a message to the AI model"""
        check_api_key()
        
        try:
            # get api key and validate
            api_key = request.headers.get('X-API-KEY')
            
            # normalize user id (generate new if "new", keep existing, or convert UUID)
            if user_id == "new":
                actual_user_id = user_manager.generate_user_id()
            else:
                actual_user_id = user_manager.normalize_user_id(user_id)
            
            # get and validate request data
            try:
                data = request.get_json(force=True)
            except Exception as e:
                return {"error": f"Invalid JSON data: {str(e)}"}, 400
                
            if not data:
                return {"error": "No JSON data provided"}, 400
                
            user_msg = data.get("message", "")
            if not user_msg.strip():
                return {"error": "Message cannot be empty"}, 400
                
            model = data.get("model", "claude-2")
            if model not in SUPPORTED_MODELS:
                return {"error": f"Unsupported model. Choose from: {', '.join(SUPPORTED_MODELS)}"}, 400
            
            use_rag = data.get("use_rag", True)
            num_chunks = data.get("num_chunks", DEFAULT_TOP_K)

            # input guardrail check
            input_guardrail = check_guardrail(user_msg, "INPUT")
            if not input_guardrail["passed"]:
                api.abort(400, f"Input rejected by guardrail: {input_guardrail['reason']}")

            # get conversation history
            messages = memory_store.get(actual_user_id, [])
            
            # rag context retrieval
            rag_stats = {"enabled": use_rag}
            context_source = "None"
            context_chunks = []
            
            if use_rag:
                context_source = data.get("context_source", "Neuroscience Guide")
                
                # Perform RAG retrieval
                rag_result = rag_system.retrieve_context(
                    query=user_msg,
                    num_chunks=num_chunks,
                    context_source=context_source,
                    user_id=actual_user_id
                )
                
                # update rag stats
                if rag_result["success"]:
                    context_chunks = rag_result["context_chunks"]
                    rag_stats.update({
                        "context_source": context_source,
                        "chunks_retrieved": len(context_chunks),
                        "chunks_used": rag_result["chunks_used"]
                    })
                else:
                    rag_stats.update({
                        "context_source": context_source,
                        "chunks_retrieved": 0,
                        "chunks_used": 0,
                        "error": rag_result.get("error")
                    })
            
            # Prepare the message for the model
            original_user_msg = user_msg  # Preserve original message for logging
            model_user_msg = user_msg     # Message to send to the model
            
            if context_chunks:
                context_text = "\n\n".join([chunk.content for chunk in context_chunks])
                # embed context directly in user message (Claude v2 compatible)
                model_user_msg = f"""Based on this neuroscience context:
                {context_text}

Answer this question: {user_msg}"""
            
            # Add user message to conversation
            messages_for_model = [
                *messages,
                {"role": "user", "content": model_user_msg}
            ]
            
            # Call Bedrock
            start_time = datetime.now()
            try:
                response = call_bedrock(messages_for_model, model)
            except Exception as e:
                log_error(f"Bedrock API call failed: {str(e)}")
                return {
                    "error": "Model API call failed",
                    "details": str(e)
                }, 500
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # output guardrail check
            output_guardrail = check_guardrail(response, "OUTPUT")
            if not output_guardrail["passed"]:
                api.abort(400, f"Output rejected by guardrail: {output_guardrail['reason']}")
            
            # update conversation history break away original user message for dashboard
            messages.append({"role": "user", "content": original_user_msg})
            messages.append({"role": "assistant", "content": response})
            
            # keep only last 10 messages
            if len(messages) > 10:
                messages = messages[-10:]
            
            memory_store[actual_user_id] = messages
            
            # collect guardrail information for logging
            guardrail_info = {
                "input": input_guardrail,
                "output": output_guardrail
            }
            
            # log the interaction 
            log_chat_request(
                user_id=actual_user_id,
                message=original_user_msg,
                response=response,
                model=model,
                duration_ms=duration_ms,
                context_source=context_source,
                rag_stats=rag_stats,
                guardrails=guardrail_info
            )
            
            return {
                "response": response,
                "model": model,
                "user_id": actual_user_id,
                "context_source": context_source,
                "rag_stats": rag_stats,
                "duration_ms": round(duration_ms, 2),
                "guardrails_enabled": is_guardrail_enabled(),
                "message_count": len(messages)
            }
            
        except Exception as e:
            log_error(str(e))
            return {
                "error": "Server error",
                "details": str(e)
            }, 500

# document upload endpoint
@ns.route('/upload/<string:user_id>')
@api.doc('upload_document',
         description='Upload documents for user-specific RAG',
         params={'user_id': 'User ID to upload documents for'},
         responses={200: 'Success', 400: 'Validation error', 401: 'Authentication error', 500: 'Server error'})
class Upload(Resource):
    def post(self, user_id):
        """upload documents for user-specific RAG"""
        check_api_key()
        
        try:
            # Normalize user ID
            actual_user_id = user_manager.normalize_user_id(user_id)
            
            # Check if files were uploaded
            if 'files' not in request.files:
                return {"error": "No files uploaded"}, 400
            
            files = request.files.getlist('files')
            if not files or all(f.filename == '' for f in files):
                return {"error": "No files selected"}, 400
            
            results = []
            
            # process each file
            for file in files:
                if file and file.filename:
                    # read file content
                    file_content = file.read()
                    file_type = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
                    
                    # validate file type before processing
                    from src.core.config import ALLOWED_USER_FILE_TYPES
                    if file_type not in ALLOWED_USER_FILE_TYPES:
                        result = {
                            "success": False,
                            "filename": file.filename,
                            "error": f"Unsupported file type '{file_type}'. Allowed types: {', '.join(ALLOWED_USER_FILE_TYPES)}"
                        }
                    else:
                        # process the document
                        result = rag_system.upload_user_file(
                            file_content=file_content,
                            filename=file.filename,
                            user_id=actual_user_id,
                            file_type=file_type
                        )
                    
                    results.append(result)
                    
                    # log document upload
                    log_document_upload(
                        user_id=actual_user_id,
                        filename=file.filename,
                        success=result.get('success', False),
                        error=result.get('error')
                    )
            
            return {
                "message": f"Processed {len(results)} files",
                "user_id": actual_user_id,
                "results": results
            }
            
        except Exception as e:
            log_error(str(e))
            return {
                "error": "Server error",
                "details": str(e)
            }, 500

# user files endpoint
@ns.route('/files/<string:user_id>')
@api.doc('list_files',
         description='List documents uploaded for a specific user',
         params={'user_id': 'User ID to list documents for'},
         responses={200: 'Success', 401: 'Authentication error', 500: 'Server error'})
class UserFiles(Resource):
    def get(self, user_id):
        """list documents uploaded for a specific user"""
        check_api_key()
        try:
            # Normalize user ID
            actual_user_id = user_manager.normalize_user_id(user_id)
            
            # Get user files
            files = rag_system.list_user_files(actual_user_id)
            
            return {
                "user_id": actual_user_id,
                "files": files,
                "count": len(files)
            }
            
            # return error if no files found
        except Exception as e:
            log_error(str(e))
            return {
                "error": "Server error",
                "details": str(e)
            }, 500
    
    # delete file endpoint
    @api.doc('delete_file',
             description='Delete a document from a user',
             params={'user_id': 'User ID', 'filename': 'Filename to delete'})
    def delete(self, user_id):
        """delete a document from a user"""
        check_api_key()
        try:
            # Normalize user ID
            actual_user_id = user_manager.normalize_user_id(user_id)
            
            # get filename from query parameters
            filename = request.args.get('filename')
            if not filename:
                return {"error": "Filename parameter required"}, 400
            
            # find the s3 key for the file
            user_files = rag_system.list_user_files(actual_user_id)
            s3_key = None
            
            for file_info in user_files:
                if file_info["filename"].endswith(filename):
                    s3_key = file_info["s3_key"]
                    break
            
            if not s3_key:
                return {"error": f"File {filename} not found for user {actual_user_id}"}, 404
            
            # delete from rag system
            success = rag_system.delete_user_file(actual_user_id, s3_key)
            result = {
                "success": success,
                "message": f"Successfully deleted {filename}" if success else "Failed to delete file"
            }
            
            if result["success"]:
                return {
                    "message": result["message"],
                    "user_id": actual_user_id,
                    "filename": filename
                }
            else:
                return {"error": result["error"]}, 500
                
        except Exception as e:
            log_error(str(e))
            return {
                "error": "Server error",
                "details": str(e)
            }, 500

# run the app
if __name__ == '__main__':
    app.run(debug=True)
