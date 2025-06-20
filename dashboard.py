"""
Streamlit Dashboard for BedrockChat Analytics
Shows session logs, performance metrics, and RAG file information
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

# Load environment variables
load_dotenv()

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
        # Initialize S3 client and bucket info
        self.s3 = boto3.client('s3')
        self.bucket = os.getenv('RAG_BUCKET')
        if not self.bucket:
            st.error("RAG_BUCKET environment variable not set")
            st.stop()
        
        self.logs_folder = 'logs'
        self.rag_folder = os.getenv('FILE_FOLDER', 'documents')
    
    # Get list of files in the RAG documents folder
    # to display in the sidebar
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
    
    # Get list of available log dates
    # for filter in the sidebar
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
    
    # Load and parse logs from S3 in the log folder
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
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=prefix
                )
                
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.log'):
                        try:
                            # Read log file
                            log_content = self.s3.get_object(
                                Bucket=self.bucket,
                                Key=obj['Key']
                            )['Body'].read().decode('utf-8')
                            
                            # Extract session ID from filename
                            file_session_id = obj['Key'].split('/')[-1].replace('.log', '')
                            
                            # Parse each line as JSON
                            for line_num, line in enumerate(log_content.strip().split('\n'), 1):
                                if line:
                                    try:
                                        log_entry = json.loads(line)
                                        
                                        # Flatten the nested data structure
                                        flattened = {
                                            'timestamp': log_entry.get('timestamp'),
                                            'event': log_entry.get('event'),
                                            'source_file': obj['Key'],
                                            'data.session_id': file_session_id  # Use session ID from filename
                                        }
                                        
                                        # Add flattened data fields
                                        data = log_entry.get('data', {})
                                        for key, value in data.items():
                                            if isinstance(value, dict):
                                                for sub_key, sub_value in value.items():
                                                    flattened[f'data.{key}.{sub_key}'] = sub_value
                                            else:
                                                flattened[f'data.{key}'] = value
                                        
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
                
                # Debug information about the loaded data
                # st.write("**Log Loading Details:**")
                # st.write(f"- Files processed: {len(set(df['source_file']))}")
                # st.write(f"- Total log entries: {len(df)}")
                # st.write(f"- Event types found: {sorted(df['event'].unique())}")
                # st.write(f"- Unique sessions: {df['data.session_id'].nunique()}")
                # st.write(f"- Available columns: {sorted(df.columns)}")
                
                # Show sample of session IDs and their file mappings
                # session_files = df.groupby('data.session_id')['source_file'].unique()
                # st.write("\n**Session to File Mappings:**")
                # for session_id, files in session_files.items():
                #     st.write(f"- Session {session_id}: {files[0]}")
                
                if errors:
                    st.warning("Some errors occurred while loading logs:")
                    for error in errors[:5]:  # Show first 5 errors
                        st.write(f"- {error}")
                
                return df
            else:
                st.warning("No log entries found in the selected date range")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error loading logs: {str(e)}")
            return pd.DataFrame()
    
    # Calculate overall statistics
    def get_session_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate overall statistics"""
        if df.empty:
            return {}
        
        # Initialize stats dictionary
        stats = {
            'total_events': len(df),
            'unique_sessions': df['data.session_id'].nunique() if 'data.session_id' in df.columns else 0,
            'total_interactions': len(df[df['event'] == 'INTERACTION']),
            'total_errors': len(df[df['event'] == 'ERROR']),
            'avg_duration_ms': df[df['event'] == 'INTERACTION']['data.duration_ms'].mean() if 'data.duration_ms' in df.columns else 0,
            'total_tokens': df[df['event'] == 'INTERACTION']['data.total_tokens'].sum() if 'data.total_tokens' in df.columns else 0,
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        
        # Model usage stats from INTERACTION events
        interaction_df = df[df['event'] == 'INTERACTION']
        if not interaction_df.empty and 'data.model' in interaction_df.columns:
            stats['model_usage'] = interaction_df['data.model'].value_counts().to_dict()
        
        return stats
    
    # Get detailed information for a specific session
    def get_session_details(self, df: pd.DataFrame, session_id: str) -> Dict:
        """Get detailed information for a specific session"""
        if df.empty:
            return {}
        
        # Filter for the specific session
        session_data = df[df['data.session_id'].astype(str) == str(session_id)].sort_values('timestamp')
        
        if session_data.empty:
            return {}
        
        # Initialize details dictionary based on the session data from logs
        details = {
            'session_id': session_id,
            'start_time': session_data['timestamp'].min(),
            'end_time': session_data['timestamp'].max(),
            'duration_minutes': (session_data['timestamp'].max() - session_data['timestamp'].min()).total_seconds() / 60,
            'total_events': len(session_data),
            'interactions': [],
            'models_used': set(),
            'total_tokens': 0,
            'total_input_chars': 0,
            'total_output_chars': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'errors': [],
            'rag_enabled_count': 0,
            'guardrails_failed_count': 0,
            'avg_duration_ms': 0  # Initialize average duration
        }
        
        total_duration = 0
        interaction_count = 0
        
        for _, row in session_data.iterrows():
            event_type = row['event']
            
            if event_type == 'INTERACTION':
                interaction = {
                    'timestamp': row['timestamp'],
                    'message_number': row.get('data.message_number', 0),
                    'model': row.get('data.model', 'Unknown'),
                    'duration_ms': row.get('data.duration_ms', 0),
                    'input_text': row.get('data.input_text', ''),
                    'output_text': row.get('data.output_text', ''),
                    'input_chars': row.get('data.input_chars', len(str(row.get('data.input_text', '')))),
                    'output_chars': row.get('data.output_chars', len(str(row.get('data.output_text', '')))),
                    'input_tokens': row.get('data.input_tokens', 0),
                    'output_tokens': row.get('data.output_tokens', 0),
                    'total_tokens': row.get('data.total_tokens', 0),
                    'guardrails': {
                        'input': row.get('data.guardrails.input', {}),
                        'output': row.get('data.guardrails.output', {})
                    },
                    'rag': {
                        'enabled': row.get('data.rag.enabled', False)
                    },
                    'num_chunks_requested': row.get('data.num_chunks_requested')
                }
                
                details['interactions'].append(interaction)
                if interaction['model'] != 'Unknown':
                    details['models_used'].add(interaction['model'])
                
                # Update totals
                details['total_input_chars'] += interaction['input_chars']
                details['total_output_chars'] += interaction['output_chars']
                details['total_input_tokens'] += interaction['input_tokens']
                details['total_output_tokens'] += interaction['output_tokens']
                details['total_tokens'] += interaction['total_tokens']
                
                # Track duration for average calculation
                if interaction['duration_ms'] > 0:
                    total_duration += interaction['duration_ms']
                    interaction_count += 1
                
                # Track RAG and guardrails usage
                if interaction['rag'].get('enabled', False):
                    details['rag_enabled_count'] += 1
                
                guardrails = interaction['guardrails']
                if guardrails:
                    input_passed = guardrails.get('input', {}).get('passed', True)
                    output_passed = guardrails.get('output', {}).get('passed', True)
                    if not input_passed or not output_passed:
                        details['guardrails_failed_count'] += 1
            
            elif event_type == 'ERROR':
                details['errors'].append({
                    'timestamp': row['timestamp'],
                    'message': row.get('data.message', '')
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

# Sidebar - RAG Files and Filters
with st.sidebar:
    st.title("Filters & Info")
    
    # RAG Files Section
    st.header("RAG Documents")
    rag_files = analytics.get_rag_files()
    
    if rag_files:
        st.write(f"Total documents: {len(rag_files)}")
        for file in rag_files:
            with st.expander(file['name']):
                st.write(f"Size: {format_bytes(file['size'])}")
                st.write(f"Last modified: {file['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                # st.write(f"Path: {file['path']}")
    else:
        st.info("No RAG documents found")
    
    st.markdown("---")
    
    # Date Filter
    st.header("Date Range")
    available_dates = analytics.get_available_dates()
    
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
            "Select dates:",
            options=list(date_options.keys()),
            default=list(date_options.keys())[:7] if len(date_options) > 0 else []
        )
        
        # Convert back to internal format
        selected_date_codes = [date_options[date] for date in selected_dates]
    else:
        st.warning("No log data found")
        selected_date_codes = []
    
    # Refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Main Content
st.title("🚀 BedrockChat Analytics")

# Load and process data
if selected_date_codes:
    with st.spinner("Loading log data..."):
        df = analytics.load_logs(selected_date_codes)
        # Debug info for initial data load
        # st.write("**Debug Information:**")
        # st.write(f"Total records loaded: {len(df)}")
        # if not df.empty:
        #     st.write("Available columns:", list(df.columns))
        #     st.write("Sample data:")
        #     st.write(df.head(1).to_dict('records'))
        
        stats = analytics.get_session_stats(df)
    
    if not df.empty:
        # Overall Metrics
        st.header("Overall Performance")
        
        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", stats.get('unique_sessions', 0))
        with col2:
            st.metric("Total Interactions", stats.get('total_interactions', 0))
        with col3:
            st.metric("Avg Response Time", f"{stats.get('avg_duration_ms', 0):.0f}ms")
        with col4:
            st.metric("Total Tokens Used", f"{stats.get('total_tokens', 0):,}")
        
        st.markdown("---")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            # Session filter
            if 'data.session_id' in df.columns:
                unique_sessions = df['data.session_id'].dropna().unique()
                selected_session = st.selectbox(
                    "Select Session:",
                    options=[''] + list(unique_sessions),
                    format_func=lambda x: f"Session {x[:8]}..." if x else "All Sessions"
                )
        
        # with col2:
        #     # Model filter
        #     if 'data.model' in df.columns:
        #         unique_models = df[df['event'] == 'INTERACTION']['data.model'].dropna().unique()
        #         selected_model = st.selectbox(
        #             "Filter by Model:",
        #             options=[''] + list(unique_models),
        #             format_func=lambda x: x if x else "All Models"
        #         )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_session:
            # st.write(f"Filtering for session: {selected_session}")
            # st.write(f"Records before session filter: {len(filtered_df)}")
            
            # Get the file(s) for this session
            session_files = filtered_df[filtered_df['data.session_id'].astype(str) == str(selected_session)]['source_file'].unique()
            
            # Filter by file instead of session_id
            filtered_df = filtered_df[filtered_df['source_file'].isin(session_files)]
            
            # st.write(f"Records after session filter: {len(filtered_df)}")
            if len(filtered_df) > 0:
                # st.write("Events in session:", sorted(filtered_df['event'].unique()))
                # st.write("Log file:", session_files[0])
                # st.write("Sample data from session:")
                # st.write(filtered_df.head(1).to_dict('records'))
                pass
            else:
                st.info("No records found for selected session")
                # st.write("Available session IDs:", sorted(df['data.session_id'].unique()))
        
        # if selected_model:
        #     st.write(f"Filtering for model: {selected_model}")
        #     model_mask = (filtered_df['data.model'] == selected_model) | (filtered_df['event'] != 'INTERACTION')
        #     filtered_df = filtered_df[model_mask]
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Session Details", "Performance", "Raw Logs"])
        
        with tab1:
            if selected_session:
                # Detailed session view
                session_details = analytics.get_session_details(df, selected_session)
                
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
                        st.metric("Guardrails Failed", session_details['guardrails_failed_count'])
                    
                    
                    
                    # Conversation history
                    st.subheader("Conversation History")
                    for interaction in session_details['interactions']:
                        with st.expander(f"Message {interaction['message_number']} - {interaction['timestamp'].strftime('%H:%M:%S')}"):
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
                                st.write(f"- Total Tokens: {interaction['total_tokens']}")
                            
                            with details_col2:
                                st.write("**Features:**")
                                rag_status = "Enabled" if interaction['rag'].get('enabled', False) else "Disabled"
                                st.write(f"- RAG: {rag_status}")
                                # if interaction['rag'].get('enabled', False) and interaction['num_chunks_requested']:
                                #     st.write(f"  - Chunks: {interaction['num_chunks_requested']}")
                                
                                # Guardrails status
                                guardrails = interaction['guardrails']
                                if guardrails:
                                    input_status = "✅" if guardrails.get('input', {}).get('passed', True) else "❌"
                                    output_status = "✅" if guardrails.get('output', {}).get('passed', True) else "❌"
                                    st.write(f"- Guardrails: Input: {input_status} | Output: {output_status}")

                                    
                                    # Show reasons if failed
                                    if not guardrails.get('input', {}).get('passed', True):
                                        st.write(f"  - Input failed: {guardrails['input'].get('reason', 'Unknown')}")
                                    if not guardrails.get('output', {}).get('passed', True):
                                        st.write(f"  - Output failed: {guardrails['output'].get('reason', 'Unknown')}")
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
                    st.info("No details found for this session")
            else:
                st.info("Select a session to view details")
        
        with tab2:
            
            if not selected_session:
                st.info("Select a session to view details")
            else:
                # Response time visualization
                interaction_df = filtered_df[filtered_df['event'] == 'INTERACTION']
                if not interaction_df.empty and 'data.duration_ms' in interaction_df.columns:
                    # Create line plot of response times
                    fig = px.line(
                        interaction_df.sort_values('data.message_number'),
                        x='data.message_number',
                        y='data.duration_ms',
                        title="Response Time by Message Number",
                        labels={
                            'data.message_number': 'Message Number',
                            'data.duration_ms': 'Response Time (ms)'
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
                    
                    # Add summary statistics
                    st.write("**Response Time Statistics:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Average", f"{interaction_df['data.duration_ms'].mean():.0f}ms")
                    with stats_col2:
                        st.metric("Min", f"{interaction_df['data.duration_ms'].min():.0f}ms")
                    with stats_col3:
                        st.metric("Max", f"{interaction_df['data.duration_ms'].max():.0f}ms")
                
                # Character usage visualization
                if 'data.input_chars' in interaction_df.columns and 'data.output_chars' in interaction_df.columns:
                    st.markdown("---")
                    # Create DataFrame for plotting
                    plot_df = interaction_df[['timestamp', 'data.message_number', 'data.input_chars', 'data.output_chars']].copy()
                    plot_df.columns = ['timestamp', 'message_number', 'Input Characters', 'Output Characters']
                    
                    # Create double line plot
                    fig = go.Figure()
                    
                    # Add input line
                    fig.add_trace(go.Scatter(
                        x=plot_df['message_number'],
                        y=plot_df['Input Characters'],
                        name='Input Characters',
                        line=dict(color='blue')
                    ))
                    
                    # Add output line
                    fig.add_trace(go.Scatter(
                        x=plot_df['message_number'],
                        y=plot_df['Output Characters'],
                        name='Output Characters',
                        line=dict(color='green')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Input vs Output Characters by Message",
                        xaxis_title="Message Number",
                        yaxis_title="Number of Characters",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add summary statistics
                    st.write("**Character Usage Statistics:**")
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.metric("Total Input Characters", f"{plot_df['Input Characters'].sum():,}")
                    with stats_col2:
                        st.metric("Total Output Characters", f"{plot_df['Output Characters'].sum():,}")
        
        with tab3:
            st.subheader("Raw Event Logs")
            
            # Show recent events
            display_df = filtered_df.sort_values('timestamp', ascending=False)
            
            # Select columns to display based on logger.py structure
            display_columns = [
                'timestamp', 
                'event', 
                'data.session_id',
                'data.message_number',
                'data.model',
                'data.duration_ms',
                'data.tokens_used',
                'data.input_text',
                'data.output_text',
                'data.message'
            ]
            
            # Filter to existing columns
            available_columns = [col for col in display_columns if col in display_df.columns]
            
            if available_columns:
                st.dataframe(
                    display_df[available_columns],
                    use_container_width=True,
                    height=400
                )
            
                # Download option
                if st.button("Download as CSV"):
                    csv = display_df[available_columns].to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"bedrock_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data to display")
    
    else:
        st.warning("No data found for the selected date range")

else:
    st.info("Please select at least one date to view analytics")

# Footer
st.markdown("---")