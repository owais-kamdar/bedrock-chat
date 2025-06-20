"""
API key management for BedrockChat
"""

import boto3
import os
import uuid
from datetime import datetime
import json

class APIManager:
    # Folder name for API keys in S3
    API_KEYS_FOLDER = 'api_keys'
    
    def __init__(self):
        """Initialize with S3 client"""
        self.s3 = boto3.client('s3')
        self.bucket = os.getenv('RAG_BUCKET')  # Use existing RAG bucket
        if not self.bucket:
            raise ValueError("RAG_BUCKET environment variable not set")
    
    def create_api_key(self, user_id: str) -> str:
        """Create a new API key for a user"""
        # Create a more distinct API key format: bdrk_<timestamp>_<uuid>
        timestamp = datetime.utcnow().strftime('%y%m%d')
        unique_id = str(uuid.uuid4()).replace('-', '')[:10]  # Take first 16 chars
        api_key = f"bdrk_{timestamp}_{unique_id}"
        
        # Store under user_id with API key in content
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f'{self.API_KEYS_FOLDER}/{user_id}',
            Body=json.dumps({
                'api_key': api_key,
                'created_at': datetime.utcnow().isoformat(),
                'is_active': True,
                'usage_count': 0
            })
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Check if an API key is valid"""
        try:
            # List all user objects
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'{self.API_KEYS_FOLDER}/'
            )
            
            # Check each user's API key
            for obj in response.get('Contents', []):
                data = json.loads(
                    self.s3.get_object(
                        Bucket=self.bucket,
                        Key=obj['Key']
                    )['Body'].read()
                )
                if data.get('api_key') == api_key and data.get('is_active', True):
                    return True
            
            return False
        except:
            return False
    
    def get_user_api_key(self, user_id: str) -> str:
        """Get existing API key for a user"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=f'{self.API_KEYS_FOLDER}/{user_id}'
            )
            data = json.loads(response['Body'].read())
            if data.get('is_active', True):
                return data.get('api_key')
        except:
            pass
        return None 
    



