"""
Unified User Manager for BedrockChat
Handles user ID generation and API key management
"""

import json
import boto3
import os
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Use centralized configuration
from src.core.config import RAG_BUCKET, API_KEYS_FOLDER

# Configuration constants
API_KEY_PREFIX = "bdrk"
API_KEY_UNIQUE_ID_LENGTH = 10
FALLBACK_USER_ID_MODULO = 1000

# user manager class
class UserManager:
    
    def __init__(self):
        """Initialize user manager with S3 connection"""
        try:
            self.s3_client = boto3.client('s3')
            self.bucket_name = RAG_BUCKET
            self.user_counter_key = 'user_counter.json'
            self.s3_enabled = bool(self.bucket_name)
            
            if not self.bucket_name:
                raise ValueError("RAG_BUCKET environment variable not set")
        except Exception as e:
            self.s3_client = None
            self.bucket_name = None
            self.s3_enabled = False
    
    # get next user number
    def _get_next_user_number(self) -> int:
        """Get the next sequential user number"""
        if not self.s3_enabled:
            return int(datetime.utcnow().timestamp() * 1000) % FALLBACK_USER_ID_MODULO
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.user_counter_key
            )
            counter_data = json.loads(response['Body'].read().decode('utf-8'))
            current_count = counter_data.get('counter', 0)
        except self.s3_client.exceptions.NoSuchKey:
            current_count = 0
        except Exception:
            return int(datetime.utcnow().timestamp() * 1000) % FALLBACK_USER_ID_MODULO
        
        # new user numbers for sequential IDs
        new_count = current_count + 1
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.user_counter_key,
                Body=json.dumps({
                    'counter': new_count,
                    'last_updated': datetime.utcnow().isoformat()
                })
            )
        except Exception:
            pass
        
        return new_count
    
    def generate_user_id(self) -> str:
        """Generate a sequential user ID: user-1, user-2, etc."""
        # generate new sequential user ID
        user_number = self._get_next_user_number()
        return f"user-{user_number}"
    
    def is_uuid_user_id(self, user_id: str) -> bool:
        """Check if user ID is in old UUID format"""
        if not user_id or not user_id.startswith('user-'):
            return False
        
        # extract the part after 'user-'
        uuid_part = user_id[5:]  # Remove 'user-' prefix
        parts = uuid_part.split('-')
        
        # UUID format: 8-4-4-4-12 characters
        if len(parts) != 5:
            return False
        
        try:
            expected_lengths = [8, 4, 4, 4, 12]
            for i, part in enumerate(parts):
                if len(part) != expected_lengths[i]:
                    return False
                int(part, 16)  # Check if hexadecimal
            return True
        except ValueError:
            return False
    
    def normalize_user_id(self, user_id: str) -> str:
        """Normalize user ID for consistency"""
        if not user_id:
            return self.generate_user_id()
        
        # if already in sequential format, use it
        if user_id.startswith('user-') and not self.is_uuid_user_id(user_id):
            return user_id
        
        # keep UUID format for backwards compatibility
        if self.is_uuid_user_id(user_id):
            return user_id
        
        # Generate new sequential ID for any other format
        return self.generate_user_id()
    
    # get display name for user ID
    def get_display_name(self, user_id: str) -> str:
        """Get display name for user ID"""
        if not user_id:
            return "Unknown User"
        
        if user_id.startswith('user-') and not self.is_uuid_user_id(user_id):
            return user_id  # user-1, user-2, etc.
        elif self.is_uuid_user_id(user_id):
            return f"User {user_id[5:13]}..."  # User abc12345...
        else:
            return f"User {user_id[:8]}..."
    
    # find user ID by API key
    def _find_user_by_api_key(self, api_key: str) -> Optional[str]:
        """find user ID by API key"""
        if not self.s3_enabled:
            return None
        
        try:
            # list all user API key objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f'{API_KEYS_FOLDER}/'
            )
            
            # check each user's API key
            for obj in response.get('Contents', []):
                try:
                    data = json.loads(
                        self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )['Body'].read()
                    )
                    if data.get('api_key') == api_key and data.get('is_active', True):
                        # extract user_id from the S3 key path
                        user_id = obj['Key'].split('/')[-1]
                        return user_id
                except Exception:
                    continue  # Skip corrupted entries
            
            return None
        except Exception:
            return None
    
    # create API key for user
    def create_api_key(self, user_id: str) -> str:
        """Create a new API key for a user"""
        if not self.s3_enabled:
            raise ValueError("S3 not enabled - cannot create API keys")
        
        # create a distinct API key format: bdrk_<timestamp>_<uuid>
        timestamp = datetime.utcnow().strftime('%y%m%d')
        unique_id = str(uuid.uuid4()).replace('-', '')[:API_KEY_UNIQUE_ID_LENGTH]
        api_key = f"{API_KEY_PREFIX}_{timestamp}_{unique_id}"
        
        # Store under user_id with API key in content
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=f'{API_KEYS_FOLDER}/{user_id}',
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
        return self._find_user_by_api_key(api_key) is not None
    
    def get_user_api_key(self, user_id: str) -> Optional[str]:
        """Get existing API key for a user"""
        if not self.s3_enabled:
            return None
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f'{API_KEYS_FOLDER}/{user_id}'
            )
            data = json.loads(response['Body'].read())
            if data.get('is_active', True):
                return data.get('api_key')
        except Exception:
            pass
        return None
    
    # get user ID from API key
    def get_user_id_from_key(self, api_key: str) -> Optional[str]:
        """Get user ID associated with an API key"""
        return self._find_user_by_api_key(api_key)

# Global user manager instance
user_manager = UserManager() 