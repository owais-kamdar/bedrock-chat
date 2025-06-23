"""
Streamlit Dashboard for BedrockChat Analytics
Shows session logs, performance metrics, RAG file information, and user-specific analytics
"""

import streamlit as st
import boto3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from typing import List, Dict, Tuple
from collections import defaultdict

# Load environment variables from config/.env
load_dotenv("config/.env")

# Import configuration
from src.core.config import RAG_BUCKET, FILE_FOLDER, USER_UPLOADS_FOLDER, LOGS_FOLDER, get_s3_client

# Configure page
st.set_page_config(
    page_title="BedrockChat Analytics Dashboard",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Dashboard Analytics Class
class DashboardAnalytics:
    def __init__(self):
        """Initialize S3 client and bucket info"""
        try:
            self.s3 = get_s3_client()
            self.bucket = RAG_BUCKET
            if not self.bucket:
                st.error("RAG_BUCKET environment variable not set")
                st.stop()
        except Exception as e:
            st.error(f"Failed to initialize S3 client: {e}")
            st.stop()
        
        self.logs_folder = LOGS_FOLDER
        self.rag_folder = FILE_FOLDER
        self.user_uploads_folder = USER_UPLOADS_FOLDER
    
    def get_rag_files(self) -> List[Dict]:
        """Get list of files in the RAG documents folder"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'{self.rag_folder}/'
            )
            
            files = []
            for obj in response.get('Contents', []):
                if not obj['Key'].endswith('/'):  # Skip folders
                    files.append({
                        'name': obj['Key'].split('/')[-1],
                        'path': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
            
            return files
        except Exception as e:
            st.error(f"Error fetching RAG files: {str(e)}")
            return []
    
    def get_active_users(self) -> List[str]:
        """Get list of users who have uploaded documents"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'{self.user_uploads_folder}/',
                Delimiter='/'
            )
            
            users = []
            for prefix in response.get('CommonPrefixes', []):
                user_folder = prefix['Prefix'].split('/')[-2]
                if user_folder and user_folder != self.user_uploads_folder:
                    users.append(user_folder)
            
            return sorted(users)
        except Exception as e:
            st.error(f"Error fetching users: {str(e)}")
            return []
    
    def get_user_documents(self, user_id: str = None) -> List[Dict]:
        """Get list of documents uploaded by users"""
        try:
            if user_id:
                prefix = f'{self.user_uploads_folder}/{user_id}/'
            else:
                prefix = f'{self.user_uploads_folder}/'
            
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            documents = []
            for obj in response.get('Contents', []):
                if not obj['Key'].endswith('/'):  # Skip folders
                    key_parts = obj['Key'].split('/')
                    if len(key_parts) >= 3:  # user_uploads/user_id/filename
                        doc_user_id = key_parts[1]
                        filename = key_parts[2]
                        
                        documents.append({
                            'user_id': doc_user_id,
                            'filename': filename,
                            'full_path': obj['Key'],
                            'size': obj['Size'],
                            'upload_date': obj['LastModified'],
                            'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown'
                        })
            
            return documents
        except Exception as e:
            st.error(f"Error fetching user documents: {str(e)}")
            return []
    
    def get_users_from_logs(self, df: pd.DataFrame) -> List[str]:
        """Extract unique user IDs from log data"""
        if df.empty:
            return []
        
        user_ids = set()
        
        # Extract from user_id field in data
        if 'user_id' in df.columns:
            user_ids.update(df['user_id'].dropna().unique())
        
        # Extract from log filenames (format: user_id.log)
        if 'source_file' in df.columns:
            for source_file in df['source_file'].dropna().unique():
                filename = source_file.split('/')[-1]  # Get just the filename
                if filename.endswith('.log'):
                    user_id = filename.replace('.log', '')
                    if user_id and user_id != 'system':
                        user_ids.add(user_id)
        
        return sorted(list(user_ids))
    
    def get_available_dates(self) -> List[str]:
        """Get list of available log dates"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'{self.logs_folder}/',
                Delimiter='/'
            )
            
            dates = []
            for prefix in response.get('CommonPrefixes', []):
                date_folder = prefix['Prefix'].split('/')[-2]
                if len(date_folder) == 8:  # Format: YYYYMMDD
                    dates.append(date_folder)
            
            return sorted(dates, reverse=True)
        except Exception as e:
            st.error(f"Error fetching dates: {str(e)}")
            return []
    
    def load_logs(self, date_filter: List[str] = None) -> pd.DataFrame:
        """Load and parse logs from S3"""
        all_logs = []
        errors = []
        
        try:
            # Get all log files
            if date_filter:
                prefixes = [f'{self.logs_folder}/{date}/' for date in date_filter]
            else:
                prefixes = [f'{self.logs_folder}/']
            
            for prefix in prefixes:
                try:
                    response = self.s3.list_objects_v2(
                        Bucket=self.bucket,
                        Prefix=prefix
                    )
                except Exception as e:
                    errors.append(f"Error listing objects in {prefix}: {str(e)}")
                    continue
                
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.log'):
                        try:
                            # Read log file
                            log_content = self.s3.get_object(
                                Bucket=self.bucket,
                                Key=obj['Key']
                            )['Body'].read().decode('utf-8')
                            
                            # Extract user ID from filename
                            filename = obj['Key'].split('/')[-1]
                            file_user_id = filename.replace('.log', '')
                            
                            # Parse each line as JSON
                            for line_num, line in enumerate(log_content.strip().split('\n'), 1):
                                if line.strip():
                                    try:
                                        log_entry = json.loads(line)
                                        
                                        # Create flattened entry
                                        flattened = {
                                            'timestamp': log_entry.get('timestamp'),
                                            'event': log_entry.get('event'),
                                            'source_file': obj['Key'],
                                            'file_user_id': file_user_id  # User ID from filename
                                        }
                                        
                                        # Add data fields directly (not nested)
                                        data = log_entry.get('data', {})
                                        for key, value in data.items():
                                            flattened[key] = value
                                        
                                        all_logs.append(flattened)
                                        
                                    except json.JSONDecodeError as e:
                                        errors.append(f"JSON parse error in {obj['Key']} line {line_num}: {str(e)}")
                                        continue
                        except Exception as e:
                            errors.append(f"Error reading file {obj['Key']}: {str(e)}")
                            continue
            
            # Convert to DataFrame
            if all_logs:
                df = pd.DataFrame(all_logs)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add session_id column (use file_user_id as session_id for compatibility)
                df['session_id'] = df['file_user_id']
                
                if errors:
                    with st.expander("⚠️ Loading Warnings", expanded=False):
                        st.warning("Some errors occurred while loading logs:")
                        for error in errors[:10]:  # Show first 10 errors
                            st.write(f"- {error}")
                
                return df
            else:
                st.warning("No log entries found in the selected date range")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error loading logs: {str(e)}")
            return pd.DataFrame()
    
    def get_session_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate overall statistics"""
        if df.empty:
            return {}
        
        # Get interaction events only for stats
        interaction_df = df[df['event'] == 'INTERACTION']
        
        stats = {
            'total_events': len(df),
            'unique_sessions': df['session_id'].nunique() if 'session_id' in df.columns else 0,
            'total_interactions': len(interaction_df),
            'total_errors': len(df[df['event'] == 'ERROR']),
            'unique_users': 0,  # Calculate below
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        # Count all users (both document uploaders and chat users)
        all_users = set()
        
        try:
            # Get users from S3 document upload folders
            s3_response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'{self.user_uploads_folder}/',
                Delimiter='/'
            )
            for prefix in s3_response.get('CommonPrefixes', []):
                user_folder = prefix['Prefix'].split('/')[-2]
                if user_folder and user_folder != self.user_uploads_folder:
                    all_users.add(user_folder)
            
            # Also get users from chat logs (users without documents)
            if 'user_id' in df.columns:
                all_users.update(df['user_id'].dropna().unique())
            if 'file_user_id' in df.columns:
                all_users.update(df['file_user_id'].dropna().unique())
            
            # Remove any system or invalid user IDs
            all_users.discard('system')
            all_users.discard('')
            all_users = {user for user in all_users if user and not user.startswith('temp_')}
            
            stats['unique_users'] = len(all_users)
            stats['user_count_source'] = 'S3_folders_and_logs'
            
        except Exception as e:
            # Fallback to log-only counting if S3 access fails
            user_ids = set()
            if 'user_id' in df.columns:
                user_ids.update(df['user_id'].dropna().unique())
            if 'file_user_id' in df.columns:
                user_ids.update(df['file_user_id'].dropna().unique())
            
            # Clean up user IDs
            user_ids.discard('system')
            user_ids.discard('')
            user_ids = {user for user in user_ids if user and not user.startswith('temp_')}
            
            stats['unique_users'] = len(user_ids)
            stats['user_count_source'] = 'logs_only'
        
        # Add interaction-specific stats
        if not interaction_df.empty:
            stats.update({
                'avg_duration_ms': interaction_df['duration_ms'].mean() if 'duration_ms' in interaction_df.columns else 0,
                'total_input_chars': interaction_df['input_chars'].sum() if 'input_chars' in interaction_df.columns else 0,
                'total_output_chars': interaction_df['output_chars'].sum() if 'output_chars' in interaction_df.columns else 0,
                'total_input_tokens': interaction_df['input_tokens'].sum() if 'input_tokens' in interaction_df.columns else 0,
                'total_output_tokens': interaction_df['output_tokens'].sum() if 'output_tokens' in interaction_df.columns else 0,
            })
            
            # Count guardrail failures
            guardrail_failures = 0
            if 'guardrails' in interaction_df.columns:
                for _, row in interaction_df.iterrows():
                    guardrails = row.get('guardrails', {})
                    # Handle cases where guardrails might be NaN (float) instead of dict
                    if guardrails and isinstance(guardrails, dict):
                        input_guard = guardrails.get('input', {})
                        output_guard = guardrails.get('output', {})
                        if not input_guard.get('passed', True) or not output_guard.get('passed', True):
                            guardrail_failures += 1
            stats['total_guardrail_failures'] = guardrail_failures
            
            # Model usage stats
            if 'model' in interaction_df.columns:
                stats['model_usage'] = interaction_df['model'].value_counts().to_dict()
        
        return stats
    
    def get_user_stats(self, df: pd.DataFrame, user_id: str = None) -> Dict:
        """Calculate user-specific statistics"""
        if df.empty:
            return {}
        
        # Filter for specific user if provided
        if user_id:
            user_df = df[df['user_id'] == user_id] if 'user_id' in df.columns else df[df['file_user_id'] == user_id]
        else:
            user_df = df
        
        if user_df.empty:
            return {}
        
        # Calculate user-specific stats
        interaction_df = user_df[user_df['event'] == 'INTERACTION']
        
        stats = {
            'user_id': user_id,
            'total_sessions': user_df['session_id'].nunique() if 'session_id' in user_df.columns else 0,
            'total_interactions': len(interaction_df),
            'total_errors': len(user_df[user_df['event'] == 'ERROR']),
            'first_activity': user_df['timestamp'].min(),
            'last_activity': user_df['timestamp'].max(),
        }
        
        # Add interaction-specific stats
        if not interaction_df.empty:
            # Count guardrail failures for user
            user_guardrail_failures = 0
            if 'guardrails' in interaction_df.columns:
                for _, row in interaction_df.iterrows():
                    guardrails = row.get('guardrails', {})
                    # Handle cases where guardrails might be NaN (float) instead of dict
                    if guardrails and isinstance(guardrails, dict):
                        input_guard = guardrails.get('input', {})
                        output_guard = guardrails.get('output', {})
                        if not input_guard.get('passed', True) or not output_guard.get('passed', True):
                            user_guardrail_failures += 1
            
            stats.update({
                'avg_duration_ms': interaction_df['duration_ms'].mean() if 'duration_ms' in interaction_df.columns else 0,
                'total_input_chars': interaction_df['input_chars'].sum() if 'input_chars' in interaction_df.columns else 0,
                'total_output_chars': interaction_df['output_chars'].sum() if 'output_chars' in interaction_df.columns else 0,
                'total_input_tokens': interaction_df['input_tokens'].sum() if 'input_tokens' in interaction_df.columns else 0,
                'total_output_tokens': interaction_df['output_tokens'].sum() if 'output_tokens' in interaction_df.columns else 0,
                'rag_usage_count': len(interaction_df[interaction_df.get('enabled', False) == True]) if 'enabled' in interaction_df.columns else 0,
                'guardrail_failures': user_guardrail_failures,
                'models_used': interaction_df['model'].value_counts().to_dict() if 'model' in interaction_df.columns else {}
            })
        
        return stats
    
    def get_session_details(self, df: pd.DataFrame, session_id: str) -> Dict:
        """Get detailed information for a specific session"""
        if df.empty:
            return {}
        
        # Filter for the specific session
        session_data = df[df['session_id'].astype(str) == str(session_id)].sort_values('timestamp')
        
        if session_data.empty:
            return {}
        
        # Initialize details dictionary
        details = {
            'session_id': session_id,
            'start_time': session_data['timestamp'].min(),
            'end_time': session_data['timestamp'].max(),
            'duration_minutes': (session_data['timestamp'].max() - session_data['timestamp'].min()).total_seconds() / 60,
            'total_events': len(session_data),
            'interactions': [],
            'models_used': set(),
            'total_input_chars': 0,
            'total_output_chars': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'errors': [],
            'rag_enabled_count': 0,
            'guardrail_failures': 0,
            'avg_duration_ms': 0
        }
        
        total_duration = 0
        interaction_count = 0
        
        for _, row in session_data.iterrows():
            event_type = row['event']
            
            if event_type == 'INTERACTION':
                interaction = {
                    'timestamp': row['timestamp'],
                    'model': row.get('model', 'Unknown'),
                    'duration_ms': row.get('duration_ms', 0),
                    'input_text': row.get('user_message', ''),
                    'output_text': row.get('assistant_reply', ''),
                    'input_chars': row.get('input_chars', 0),
                    'output_chars': row.get('output_chars', 0),
                    'input_tokens': row.get('input_tokens', 0),
                    'output_tokens': row.get('output_tokens', 0),
                    'context_source': row.get('context_source', 'none'),
                    'rag_enabled': row.get('enabled', False),
                    'chunks_retrieved': row.get('chunks_retrieved', 0),
                    'guardrails': row.get('guardrails', {})
                }
                
                details['interactions'].append(interaction)
                if interaction['model'] != 'Unknown':
                    details['models_used'].add(interaction['model'])
                
                # Update totals
                details['total_input_chars'] += interaction['input_chars']
                details['total_output_chars'] += interaction['output_chars']
                details['total_input_tokens'] += interaction['input_tokens']
                details['total_output_tokens'] += interaction['output_tokens']
                
                # Track duration for average calculation
                if interaction['duration_ms'] > 0:
                    total_duration += interaction['duration_ms']
                    interaction_count += 1
                
                # Track RAG usage
                if interaction['rag_enabled']:
                    details['rag_enabled_count'] += 1
                
                # Track guardrail failures
                guardrails = interaction.get('guardrails', {})
                # Handle cases where guardrails might be NaN (float) instead of dict
                if guardrails and isinstance(guardrails, dict):
                    input_guard = guardrails.get('input', {})
                    output_guard = guardrails.get('output', {})
                    
                    if not input_guard.get('passed', True) or not output_guard.get('passed', True):
                        details['guardrail_failures'] += 1
            
            elif event_type == 'ERROR':
                details['errors'].append({
                    'timestamp': row['timestamp'],
                    'message': row.get('message', '')
                })
        
        # Calculate average duration
        details['avg_duration_ms'] = total_duration / interaction_count if interaction_count > 0 else 0
        
        # Convert set to list for JSON serialization
        details['models_used'] = list(details['models_used'])
        
        return details

def format_bytes(size: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

# Initialize dashboard
def get_analytics():
    return DashboardAnalytics()

analytics = get_analytics()

# Load available dates first
available_dates = analytics.get_available_dates()

# Sidebar - RAG Files and Filters
with st.sidebar:
    st.title("Filters & Info")
    
    # Date Filter FIRST - so it can influence user selection
    st.header("📅 Date Range")
    
    if available_dates:
        # Convert to readable format
        date_options = {}
        for date_str in available_dates:
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                readable_date = date_obj.strftime('%Y-%m-%d')
                date_options[readable_date] = date_str
            except:
                continue
        
        # Select dates for the filter
        selected_dates = st.multiselect(
            "Select dates (leave empty for overview of all dates):",
            options=list(date_options.keys()),
            default=[]  # Default to no dates selected for overview
        )
        
        # Convert back to internal format
        selected_date_codes = [date_options[date] for date in selected_dates]
    else:
        st.warning("No log data found")
        selected_date_codes = []
    
    st.markdown("---")
    
    # NOW determine users based on selected dates
    if selected_date_codes:
        # Use selected dates to get relevant users
        df_for_users = analytics.load_logs(selected_date_codes)
        users_from_logs = analytics.get_users_from_logs(df_for_users)
        # Also filter document users by activity on selected dates (if they have log activity)
        users_with_documents = analytics.get_active_users()
        # Only show document users who also have log activity on selected dates
        filtered_doc_users = [user for user in users_with_documents if user in users_from_logs]
        all_active_users = sorted(list(set(filtered_doc_users + users_from_logs)))
    elif available_dates:
        # Default to all available dates for user detection when no specific dates selected
        all_date_codes = available_dates
        df_for_users = analytics.load_logs(all_date_codes)
        users_from_logs = analytics.get_users_from_logs(df_for_users)
        users_with_documents = analytics.get_active_users()
        all_active_users = sorted(list(set(users_with_documents + users_from_logs)))
    else:
        users_from_logs = []
        all_active_users = []
    
    # User Filter Section
    st.header("👤 User Analytics")
    
    # Get users from both document uploads and chat logs
    user_documents = analytics.get_user_documents()
    
    # User selection - now properly filtered by selected dates
    if selected_date_codes and not all_active_users:
        st.info("No users found with activity on selected dates")
        selected_user = st.selectbox(
            "Select User:",
            options=[''],
            format_func=lambda x: "No users available"
        )
    else:
        selected_user = st.selectbox(
            "Select User:",
            options=[''] + all_active_users,
            format_func=lambda x: f"👤 {x}" if x else "All Users"
        )
    
    # User document summary
    if selected_user:
        user_docs = [doc for doc in user_documents if doc['user_id'] == selected_user]
        st.write(f"**{selected_user} Documents:**")
        
        if user_docs:
            st.write(f"Total files: {len(user_docs)}")
            total_size = sum(doc['size'] for doc in user_docs)
            st.write(f"Total size: {format_bytes(total_size)}")
            
            # Show document list
            st.write("**Files:**")
            for doc in user_docs:
                file_type = doc['file_type'].upper()
                size_str = format_bytes(doc['size'])
                upload_date = doc['upload_date'].strftime('%Y-%m-%d')
                st.write(f"• {doc['filename']} ({file_type}, {size_str}, {upload_date})")
        else:
            st.write("Documents: **None**")
    else:
        st.write(f"**All Users Overview:**")
        # Count all users (document uploaders + chat users)
        all_user_set = set(users_with_documents + users_from_logs)
        # Clean up the user set
        all_user_set.discard('system')
        all_user_set.discard('')
        all_user_set = {user for user in all_user_set if user and not user.startswith('temp_')}
        
        st.write(f"Total users: {len(all_user_set)}")
        st.write(f"Total documents: {len(user_documents)}")
        if user_documents:
            total_size = sum(doc['size'] for doc in user_documents)
            st.write(f"Total size: {format_bytes(total_size)}")
    
    
    st.markdown("---")
    
    # RAG Files Section
    st.header("Base RAG Documents")
    rag_files = analytics.get_rag_files()
    
    if rag_files:
        st.write(f"Total documents: {len(rag_files)}")
        for file in rag_files:
            with st.expander(file['name']):
                st.write(f"Size: {format_bytes(file['size'])}")
                st.write(f"Last modified: {file['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                # st.write(f"Path: {file['path']}")
    else:
        st.info("No base RAG documents found")
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Main Content
st.title("BedrockChat Analytics")

# Load and process data
# Determine which dates to load
if not selected_date_codes:
    # Use all available dates for overview
    dates_to_load = available_dates
    header_text = "📊 System Overview (All Dates)"
else:
    # Use selected dates for detailed view
    dates_to_load = selected_date_codes
    # Create date range string for header
    if len(selected_dates) == 1:
        date_str = selected_dates[0]
    elif len(selected_dates) <= 3:
        date_str = ", ".join(selected_dates)
    else:
        date_str = f"{selected_dates[0]} to {selected_dates[-1]} ({len(selected_dates)} days)"
    header_text = f"📊 Overall Performance ({date_str})"

if dates_to_load:
    with st.spinner("Loading log data..."):
        df = analytics.load_logs(dates_to_load)
        stats = analytics.get_session_stats(df)
        
        # Get user-specific stats if user is selected
        if selected_user:
            user_stats = analytics.get_user_stats(df, selected_user)
        else:
            user_stats = {}
    
    if not df.empty:
        # Overall Metrics
        if selected_user:
            st.header(f"📊 Analytics for {selected_user}")
            
            # User-specific metrics (user is already filtered by date selection)
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                st.metric("User Sessions", user_stats.get('total_sessions', 0))
            with col2:
                st.metric("Interactions", user_stats.get('total_interactions', 0))
            with col3:
                st.metric("Avg Response Time", f"{user_stats.get('avg_duration_ms', 0):.0f}ms")
            with col4:
                total_tokens = user_stats.get('total_input_tokens', 0) + user_stats.get('total_output_tokens', 0)
                st.metric("Total Tokens", f"{total_tokens:,}")
            with col5:
                total_chars = user_stats.get('total_input_chars', 0) + user_stats.get('total_output_chars', 0)
                st.metric("Total Characters", f"{total_chars:,}")
            with col6:
                st.metric("RAG Usage", user_stats.get('rag_usage_count', 0))
            with col7:
                st.metric("Guardrail Failures", user_stats.get('guardrail_failures', 0))
            
            # User activity timeline
            if user_stats.get('first_activity') and user_stats.get('last_activity'):
                st.write(f"**Activity Period:** {user_stats['first_activity'].strftime('%Y-%m-%d %H:%M')} to {user_stats['last_activity'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.header(header_text)
            
            # Top-level metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Sessions", stats.get('unique_sessions', 0))
            with col2:
                st.metric("Total Interactions", stats.get('total_interactions', 0))
            with col3:
                st.metric("Avg Response Time", f"{stats.get('avg_duration_ms', 0):.0f}ms")
            with col4:
                total_tokens = stats.get('total_input_tokens', 0) + stats.get('total_output_tokens', 0)
                st.metric("Total Tokens Used", f"{total_tokens:,}")
            with col5:
                st.metric("Guardrail Failures", stats.get('total_guardrail_failures', 0))
            
        st.markdown("---")
        
        # Filter data based on user selection (users are already filtered by date selection in sidebar)
        if selected_user:
            # Filter for the selected user
            user_df = df[df['user_id'] == selected_user] if 'user_id' in df.columns else df[df['file_user_id'] == selected_user]
            filtered_df = user_df
        else:
            # Show all data
            filtered_df = df.copy()

        # Tabs for different views (only show detailed tabs when dates are selected)
        if selected_date_codes:
            tab1, tab2, tab3 = st.tabs(["📊 User Details", "⚡ Performance", "📋 Raw Logs"])
        elif not selected_user:
            # Show overview insights only when no user is selected and no dates are selected
            st.markdown("---")
            st.subheader("Quick Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                if stats.get('unique_sessions', 0) > 0:
                    avg_interactions_per_session = stats.get('total_interactions', 0) / stats.get('unique_sessions', 1)
                    st.write(f"**Average interactions per session:** {avg_interactions_per_session:.1f}")
                
                if stats.get('unique_users', 0) > 0:
                    avg_sessions_per_user = stats.get('unique_sessions', 0) / stats.get('unique_users', 1)
                    st.write(f"**Average sessions per user:** {avg_sessions_per_user:.1f}")
            
            with insights_col2:
                if user_documents:
                    total_doc_size = sum(doc['size'] for doc in user_documents)
                    st.write(f"**Total document storage:** {format_bytes(total_doc_size)}")
                    
                    if stats.get('unique_users', 0) > 0:
                        avg_docs_per_user = len(user_documents) / stats.get('unique_users', 1)
                        st.write(f"**Average documents per user:** {avg_docs_per_user:.1f}")
            
            st.info("Select specific dates above to view detailed analytics and user information.")
        else:
            # When a user is selected but no dates are selected, show a message
            st.markdown("---")
            st.info("Select specific dates above to view detailed analytics and user information for this user.")
            
            
        
        if selected_date_codes:
            with tab1:
                if selected_user:
                    # Get detailed session view (user is already filtered by date selection)
                    session_details = analytics.get_session_details(df, selected_user)
                    
                    if session_details:
                        # Session overview
                        st.subheader("Session Overview")
                        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
                        
                        with overview_col1:
                            st.metric("Duration", f"{session_details['duration_minutes']:.1f} min")
                        with overview_col2:
                            st.metric("Interactions", len(session_details['interactions']))
                        with overview_col3:
                            st.metric("Average Response Time", f"{session_details['avg_duration_ms']:.0f}ms")
                        with overview_col4:
                            st.metric("Errors", len(session_details['errors']))
                        
                        # Additional metrics
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        with metrics_col1:
                            st.metric("Input Tokens", session_details['total_input_tokens'])
                        with metrics_col2:
                            st.metric("Output Tokens", session_details['total_output_tokens'])
                        with metrics_col3:
                            st.metric("RAG Enabled", f"{session_details['rag_enabled_count']} times")
                        with metrics_col4:
                            st.metric("Guardrails Failed", session_details['guardrail_failures'])
                        
                        
                        
                        # Conversation history
                        st.subheader("Conversation History")
                        for i, interaction in enumerate(session_details['interactions'], 1):
                            with st.expander(f"Message {i} - {interaction['timestamp'].strftime('%H:%M:%S')}"):
                                # User message
                                st.write("**User Message:**")
                                st.write(interaction['input_text'])
                                st.write(f"*Chars: {interaction['input_chars']}, Tokens: {interaction['input_tokens']}*")
                                
                                # AI response
                                st.write("**AI Response:**")
                                st.write(interaction['output_text'])
                                st.write(f"*Chars: {interaction['output_chars']}, Tokens: {interaction['output_tokens']}*")
                                
                                # Message details
                                details_col1, details_col2 = st.columns(2)
                                with details_col1:
                                    st.write("**Message Details:**")
                                    st.write(f"- Model: {interaction['model']}")
                                    st.write(f"- Duration: {interaction['duration_ms']:.1f}ms")
                                    total_tokens = interaction['input_tokens'] + interaction['output_tokens']
                                    st.write(f"- Total Tokens: {total_tokens}")
                                
                                with details_col2:
                                    st.write("**Features:**")
                                    rag_status = "Enabled" if interaction['rag_enabled'] else "Disabled"
                                    st.write(f"- RAG: {rag_status}")
                                    
                                    # Context information
                                    context_source = interaction['context_source']
                                    if context_source != 'none':
                                        st.write(f"- Context Source: {context_source}")
                                        chunks = interaction.get('chunks_retrieved', 0)
                                        if chunks > 0:
                                            st.write(f"  - Chunks Retrieved: {chunks}")
                                    
                                    # Guardrail information
                                    guardrails = interaction.get('guardrails', {})
                                    # Handle cases where guardrails might be NaN (float) instead of dict
                                    if guardrails and isinstance(guardrails, dict):
                                        input_guard = guardrails.get('input', {})
                                        output_guard = guardrails.get('output', {})
                                        
                                        input_status = "✅ Passed" if input_guard.get('passed', True) else f"❌ Failed: {input_guard.get('reason', 'Unknown')}"
                                        output_status = "✅ Passed" if output_guard.get('passed', True) else f"❌ Failed: {output_guard.get('reason', 'Unknown')}"
                                        
                                        st.write(f"- Input Guardrail: {input_status}")
                                        st.write(f"- Output Guardrail: {output_status}")
                                    else:
                                        st.write(f"- Guardrails: Not available")
                        # Session info
                        st.write(f"**Start Time:** {session_details['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**End Time:** {session_details['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        # st.write(f"**Models Used:** {', '.join(session_details['models_used'])}")

                        # Errors
                        if session_details['errors']:
                            st.subheader("Errors")
                            for error in session_details['errors']:
                                st.error(f"{error['timestamp'].strftime('%H:%M:%S')}: {error['message']}")
                else:
                    st.info("Select a user to view details")
            
            with tab2:
                
                if not selected_user:
                    st.info("Select a user to view performance details")
                else:
                    # Get interaction data (user is already filtered by date selection)
                    interaction_df = filtered_df[filtered_df['event'] == 'INTERACTION']
                    
                    if not interaction_df.empty and 'duration_ms' in interaction_df.columns:
                        # Create line plot of response times
                        # Add message number if it doesn't exist
                        if 'message_number' not in interaction_df.columns:
                            # Create message numbers based on timestamp order
                            interaction_df = interaction_df.sort_values('timestamp').reset_index(drop=True)
                            interaction_df['message_number'] = range(1, len(interaction_df) + 1)
                        
                        fig = px.line(
                            interaction_df.sort_values('message_number'),
                            x='message_number',
                            y='duration_ms',
                            title="Response Time by Message Number",
                            labels={
                                'message_number': 'Message Number',
                                'duration_ms': 'Response Time (ms)'
                            },
                            markers=True  # Add markers to show individual points
                        )
                        
                        # Customize the layout
                        fig.update_layout(
                            xaxis_title="Message Number",
                            yaxis_title="Response Time (ms)",
                            hovermode='x unified',
                            showlegend=False
                        )
                        
                        # Add hover information
                        fig.update_traces(
                            hovertemplate="Message: %{x}<br>Response Time: %{y:.0f}ms<extra></extra>"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics (with standard deviation)
                        st.write("**Response Time Statistics:**")
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        with stats_col1:
                            st.metric("Average", f"{interaction_df['duration_ms'].mean():.0f}ms")
                        with stats_col2:
                            st.metric("Min", f"{interaction_df['duration_ms'].min():.0f}ms")
                        with stats_col3:
                            st.metric("Max", f"{interaction_df['duration_ms'].max():.0f}ms")
                        with stats_col4:
                            st.metric("Std Dev", f"{interaction_df['duration_ms'].std():.0f}ms")
                    
                    # Token usage visualization
                    if 'input_tokens' in interaction_df.columns and 'output_tokens' in interaction_df.columns:
                        st.markdown("---")
                        # Create DataFrame for plotting
                        # Ensure message_number exists
                        if 'message_number' not in interaction_df.columns:
                            interaction_df = interaction_df.sort_values('timestamp').reset_index(drop=True)
                            interaction_df['message_number'] = range(1, len(interaction_df) + 1)
                        
                        plot_df = interaction_df[['timestamp', 'message_number', 'input_tokens', 'output_tokens']].copy()
                        plot_df.columns = ['timestamp', 'message_number', 'Input Tokens', 'Output Tokens']
                        
                        # Create double line plot
                        fig = go.Figure()
                        
                        # Add input line
                        fig.add_trace(go.Scatter(
                            x=plot_df['message_number'],
                            y=plot_df['Input Tokens'],
                            name='Input Tokens',
                            line=dict(color='blue')
                        ))
                        
                        # Add output line
                        fig.add_trace(go.Scatter(
                            x=plot_df['message_number'],
                            y=plot_df['Output Tokens'],
                            name='Output Tokens',
                            line=dict(color='green')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title="Input vs Output Tokens by Message Number",
                            xaxis_title="Message Number",
                            yaxis_title="Number of Tokens",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics
                        st.write("**Token Usage Statistics:**")
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("Total Input Tokens", f"{plot_df['Input Tokens'].sum():,}")
                        with stats_col2:
                            st.metric("Total Output Tokens", f"{plot_df['Output Tokens'].sum():,}")
                        with stats_col3:
                            st.metric("Total Tokens", f"{(plot_df['Input Tokens'].sum() + plot_df['Output Tokens'].sum()):,}")
            
            with tab3:
                if selected_user:
                    # Show recent events for the selected user
                    if not filtered_df.empty:
                        display_df = filtered_df.sort_values('timestamp', ascending=False)
                    else:
                        display_df = pd.DataFrame()
                    
                    # Select columns to display based on logger.py structure
                    display_columns = [
                        'timestamp', 
                        'event', 
                        'session_id',
                        'message_number',
                        'model',
                        'duration_ms',
                        'input_chars',
                        'output_chars',
                        'input_tokens',
                        'output_tokens',
                        'user_message',
                        'assistant_reply'
                    ]
                    
                    # Filter to existing columns
                    available_columns = [col for col in display_columns if col in display_df.columns]
                    
                    if available_columns and len(display_df) > 0:
                        st.dataframe(
                            display_df[available_columns],
                            use_container_width=True,
                            height=400
                        )
                    
                        # Download option
                        csv = display_df[available_columns].to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=f"user_{selected_user}_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No log data available for this user")
                else:
                    st.info("Select a user to view raw logs")
    
    else:
        st.warning("No data found for the selected date range")

else:
    st.info("No log data available. Please check your S3 bucket configuration.")

# Footer
st.markdown("---")