"""
RAG system implementation

This module provides two RAG functionalities:
1. Base Neuroscience RAG: pre-indexed neuroscience documents for general use
2. User Document RAG: user-uploaded documents for personalized context
"""

import boto3
from PyPDF2 import PdfReader
from io import BytesIO
import json
import os
from typing import List, Dict, Tuple, Optional
import re
from src.core.vector_store import VectorStore, Document
from src.utils.logger import log_rag_operation, log_error, log_document_upload
import uuid
from datetime import datetime
import time

# use variables from config
from src.core.config import (
    get_s3_client, 
    RAG_BUCKET,
    USER_UPLOADS_FOLDER,
    NEUROSCIENCE_PDF_FILES, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    ALLOWED_USER_FILE_TYPES,
    DEFAULT_TOP_K
)

# rag system class
class RAGSystem:
    def __init__(self):
        """Initialize RAG system with vector store"""
        self.vector_store = VectorStore()
        self.s3 = get_s3_client()
        self._neuroscience_namespace = self._get_neuroscience_namespace()
        
    # get neuroscience namespace for
    def _get_neuroscience_namespace(self) -> Optional[str]:
        """Find existing neuroscience namespace with the most documents (non-user namespace)"""
        # Use the specific namespace that has the properly indexed documents
        return "ns-20250623-021813"
    
    # clean text for chunking
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Clean special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    # create chunks for text based on chunk size and overlap
    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks using CHUNK_SIZE and CHUNK_OVERLAP"""
        if not text:
            return []

        # split into sentences
        sentences = [s.strip().rstrip('.') + '.' for s in text.split('. ') if s.strip()]
        
        chunks = []
        current_chunk = ""
        previous_chunk = ""

        for sentence in sentences:
            # check if sentence fits in current chunk
            if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
                current_chunk += (sentence + " ") if current_chunk else sentence
            else:
                # save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    previous_chunk = current_chunk.strip()
                
                # start new chunk with proper overlap
                if previous_chunk and len(previous_chunk) >= CHUNK_OVERLAP:
                    # take the last CHUNK_OVERLAP characters from previous chunk
                    overlap_text = previous_chunk[-CHUNK_OVERLAP:].strip()
                    # find a good break point using spaces
                    space_idx = overlap_text.find(' ')
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx:].strip()
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

        # add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # ensure chunks end with period and clean up
        chunks = [chunk if chunk.endswith('.') else chunk + '.' for chunk in chunks]
        chunks = [chunk.replace('..', '.') for chunk in chunks]
        
        return chunks
    
    # extract text from pdf
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        
        text_parts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    # extract text from txt
    def extract_text_from_txt(self, txt_content: bytes) -> str:
        """Extract text from TXT bytes"""
        try:
            return txt_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return txt_content.decode('latin-1')
            except:
                return txt_content.decode('utf-8', errors='ignore')
    
    # process either of the documents into chunks and create Document objects
    def process_document(self, content: bytes, filename: str, file_type: str, source_key: str, doc_type: str) -> List[Document]:
        """Process any document into chunks and create Document objects"""
        # Extract text based on file type
        if file_type.lower() == 'pdf':
            raw_text = self.extract_text_from_pdf(content)
        elif file_type.lower() == 'txt':
            raw_text = self.extract_text_from_txt(content)
        else:
            print(f"Unsupported file type: {file_type}")
            return []
        
        if not raw_text.strip():
            return []
        
        # clean and chunk text
        cleaned_text = self.clean_text(raw_text)
        chunks = self.create_chunks(cleaned_text)
        
        # create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source_key,
                "chunk": i + 1,
                "total_chunks": len(chunks),
                "document_type": doc_type,
                "file_type": file_type.lower()
            }
            
            # add page info for PDFs (estimate based on chunk position)
            if file_type.lower() == 'pdf':
                estimated_page = (i // 3) + 1  # Rough estimate
                metadata["page"] = estimated_page
            
            documents.append(Document(content=chunk, metadata=metadata))
        
        return documents
    
    # get embedding for text with rate limiting
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text with rate limiting"""
        max_retries = 3
        base_delay = 5
        import time
        
        for attempt in range(max_retries):
            try:
                result = self.vector_store.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[text],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                return result.data[0].values if result.data else None
                
            # add error handling for rate limiting
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(delay)
                    else:
                        print(f"Rate limit exceeded after {max_retries} attempts")
                        return None
                else:
                    print(f"Error getting embedding: {str(e)}")
                    return None
        
        return None 
    
    # index documents in vector store
    def index_documents(self, documents: List[Document], namespace: str) -> bool:
        """Index documents in vector store"""
        if not documents:
            return True
        
        start_time = time.time()
        log_rag_operation("INDEX_START", f"Starting to index {len(documents)} documents in namespace: {namespace}")
        print(f"Indexing {len(documents)} documents in namespace: {namespace}")
        success = self.vector_store.add_documents(documents, namespace=namespace)
        
        duration_ms = (time.time() - start_time) * 1000
        if success:
            log_rag_operation("INDEX_SUCCESS", f"Successfully indexed {len(documents)} documents in namespace {namespace} ({duration_ms:.1f}ms)")
        else:
            log_rag_operation("INDEX_FAILED", f"Failed to index {len(documents)} documents in namespace {namespace} ({duration_ms:.1f}ms)")
        
        return success
    
    # search documents in vector store
    def search_documents(self, query: str, namespace: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[Document, float]]:
        """Search documents in namespace"""
        start_time = time.time()
        log_rag_operation("SEARCH_START", f"Searching for '{query[:50]}...' in namespace: {namespace}")
        
        try:
            results = self.vector_store.search(query, top_k=top_k, namespace=namespace)
            
            duration_ms = (time.time() - start_time) * 1000
            log_rag_operation("SEARCH_SUCCESS", f"Search completed with {len(results)} results")
            
            return results
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_error(f"Error searching documents in namespace {namespace}: {str(e)}")
            log_rag_operation("SEARCH_FAILED", f"Search failed: {str(e)}")
            print(f"Error searching documents: {str(e)}")
            return []

    # index base neuroscience documents
    # part 1 - BASE NEUROSCIENCE RAG FUNCTIONALITY
    def index_neuroscience_documents(self, namespace: Optional[str] = None) -> bool:
        """Index base neuroscience documents"""
        namespace = namespace or self.vector_store.clear()
        self._neuroscience_namespace = namespace
        
        print("indexing neuroscience documents")
        
        all_documents = []
        for pdf_file in NEUROSCIENCE_PDF_FILES:
            try:
                print(f"Processing: {pdf_file}")
                
                # check file exists
                self.s3.head_object(Bucket=RAG_BUCKET, Key=pdf_file)
                
                # get file content
                response = self.s3.get_object(Bucket=RAG_BUCKET, Key=pdf_file)
                content = response["Body"].read()
                
                # process document
                documents = self.process_document(
                    content=content,
                    filename=pdf_file.split('/')[-1],
                    file_type='pdf',
                    source_key=pdf_file,
                    doc_type='neuroscience_base'
                )
                
                all_documents.extend(documents)
                print(f"Processed {len(documents)} chunks from {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                return False
        
        # index all documents
        success = self.index_documents(all_documents, namespace)
        
        if success:
            print(f"Successfully indexed {len(all_documents)} neuroscience chunks")
        else:
            print(f"Failed to index neuroscience documents")
        
        return success
    
    # get context from neuroscience documents
    def get_neuroscience_context(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Get context from neuroscience documents"""
        formatted_context, _ = self.get_neuroscience_context_with_stats(query, top_k)
        return formatted_context

    def get_neuroscience_context_with_stats(self, query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, List[Tuple[Document, float]]]:
        """Get context from neuroscience documents and return both formatted context and raw results"""
        if not self._neuroscience_namespace:
            return "", []
        
        results = self.search_documents(query, self._neuroscience_namespace, top_k)
        
        if not results:
            return "", []
        
        # format context with source and page number
        context_parts = []
        for doc, score in results:
            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"Source: {source} (Page {page})\nContent: {doc.content}")
        
        formatted_context = "\n\n".join(context_parts)
        return formatted_context, results

    # upload user file
    # part 2 - USER DOCUMENT UPLOAD FUNCTIONALITY
    def upload_user_file(self, file_content: bytes, filename: str, user_id: str, file_type: str) -> Dict:
        """
        Upload a user file to S3 and index it for personalized RAG
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            user_id: User identifier
            file_type: Type of file (pdf or txt only)
            
        Returns:
            Dict with upload and indexing results
        """
        print(f"uploading user document: {filename}")
        
        # validate file type for pdf and txt
        if file_type.lower() not in ALLOWED_USER_FILE_TYPES:
            return {
                "success": False,
                "error": f"Unsupported file type '{file_type}'. Only PDF and TXT files are allowed."
            }
        
        try:
            # create user-specific S3 key
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{USER_UPLOADS_FOLDER}/{user_id}/{timestamp}_{filename}"
            
            print(f"Uploading to S3: {s3_key}")
            
            # upload to S3
            self.s3.put_object(
                Bucket=RAG_BUCKET,
                Key=s3_key,
                Body=file_content,
                ContentType="application/pdf" if file_type == "pdf" else "text/plain"
            )
            
            # process document
            documents = self.process_document(
                content=file_content,
                filename=filename,
                file_type=file_type,
                source_key=s3_key,
                doc_type='user_upload'
            )
            
            if not documents:
                return {"success": False, "error": "No content extracted from file"}
            
            # create user namespace remove 'user-' prefix
            clean_user_id = user_id.replace('user-', '') if user_id.startswith('user-') else user_id
            namespace = f"user_{clean_user_id}_{timestamp}"
            print(f"Using user namespace: {namespace}")
            
            # index documents
            success = self.index_documents(documents, namespace)
            
            # if success, print success message and log document upload
            if success:
                print(f"successfully indexed user document: {filename}")
                log_document_upload(user_id, filename, True)
                return {
                    "success": True,
                    "filename": filename,
                    "s3_key": s3_key,
                    "namespace": namespace,
                    "documents_indexed": len(documents),
                    "message": f"Successfully indexed {len(documents)} document chunks from {filename}"
                }
            else:
                log_document_upload(user_id, filename, False, "Failed to index documents in vector store")
                return {
                    "success": False,
                    "error": "Failed to index documents in vector store"
                }
        # if error, print error message and log document upload
        except Exception as e:
            print(f"Upload failed for {filename}: {str(e)}")
            log_document_upload(user_id, filename, False, f"Upload failed: {str(e)}")
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}"
            }
    
    # get context from user documents
    def get_user_context(self, query: str, user_id: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Get relevant context from user's uploaded documents only"""
        try:
            # Find user namespaces
            stats = self.vector_store.index.describe_index_stats()
            clean_user_id = user_id.replace('user-', '') if user_id.startswith('user-') else user_id
            user_namespaces = [ns for ns in stats.namespaces.keys() if ns.startswith(f"user_{clean_user_id}_")]
            
            if not user_namespaces:
                log_rag_operation("USER_SEARCH", f"No uploaded documents found for user: {user_id}")
                return ""
            
            log_rag_operation("USER_SEARCH", f"Searching across {len(user_namespaces)} user namespaces for user: {user_id}")
            
            # Search across all user namespaces
            all_results = []
            for namespace in user_namespaces:
                results = self.search_documents(query, namespace, top_k)
                all_results.extend(results)
            
            # Sort by score and take top results
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_results = all_results[:top_k]
            
            if top_results:
                context_parts = []
                for doc, score in top_results:
                    source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    file_type = doc.metadata.get('file_type', 'unknown').upper()
                    page_info = f" (Page {doc.metadata.get('page')})" if doc.metadata.get('page') else ""
                    
                    context_parts.append(f"Source: {source_name} [{file_type}]{page_info}\nContent: {doc.content}")
                
                return "\n\n".join(context_parts)
            
            return ""
            
        except Exception as e:
            log_error(f"Error getting user context: {str(e)}")
            return ""
    
    # retrieve context user documents
    def retrieve_context(self, query: str, num_chunks: int = DEFAULT_TOP_K, context_source: str = "Neuroscience Guide", user_id: str = None) -> Dict:
        """
        retrieve relevant context for a query
        
        Args:
            query: Search query
            num_chunks: Number of chunks to retrieve
            context_source: Source of context
            user_id: User ID for user-specific documents
            
        Returns:
            Dict with context_chunks and other metadata
        """
        try:
            context_chunks = []
            
            if context_source in ["Neuroscience Guide", "Both"]:
                # Search neuroscience documents
                if self._neuroscience_namespace:
                    neuro_results = self.search_documents(query, self._neuroscience_namespace, num_chunks)
                    context_chunks.extend([doc for doc, score in neuro_results])
            
            if context_source in ["Your Documents", "Both"] and user_id:
                # Search user documents  
                user_results = self._search_user_documents(query, user_id, num_chunks)
                context_chunks.extend([doc for doc, score in user_results])
            
            # Limit total chunks if "Both" was selected
            if context_source == "Both":
                context_chunks = context_chunks[:num_chunks]
            
            return {
                "success": True,
                "context_chunks": context_chunks,
                "chunks_used": len(context_chunks)
            }
            
        except Exception as e:
            log_error(f"Error retrieving context: {str(e)}")
            return {
                "success": False,
                "context_chunks": [],
                "chunks_used": 0,
                "error": str(e)
            }
    
    def _search_user_documents(self, query: str, user_id: str, top_k: int) -> List[Tuple[Document, float]]:
        """Search across all user namespaces for a user"""
        try:
            # Find user namespaces
            stats = self.vector_store.index.describe_index_stats()
            clean_user_id = user_id.replace('user-', '') if user_id.startswith('user-') else user_id
            user_namespaces = [ns for ns in stats.namespaces.keys() if ns.startswith(f"user_{clean_user_id}_")]
            
            if not user_namespaces:
                return []
            
            # Search across all user namespaces
            all_results = []
            for namespace in user_namespaces:
                results = self.search_documents(query, namespace, top_k)
                all_results.extend(results)
            
            # Sort by score and take top results
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            log_error(f"Error searching user documents: {str(e)}")
            return []
    
    # list user files
    def list_user_files(self, user_id: str) -> List[Dict]:
        """List all files uploaded by a user"""
        try:
            # List objects in user's S3 folder
            response = self.s3.list_objects_v2(
                Bucket=RAG_BUCKET,
                Prefix=f"{USER_UPLOADS_FOLDER}/{user_id}/"
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract filename from S3 key
                    s3_key = obj['Key']
                    filename = s3_key.split('/')[-1]
                    
                    # Get file metadata
                    try:
                        head_response = self.s3.head_object(Bucket=RAG_BUCKET, Key=s3_key)
                        file_type = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
                        
                        files.append({
                            "filename": filename,
                            "s3_key": s3_key,
                            "size": obj['Size'],
                            "upload_date": obj['LastModified'].isoformat(),
                            "content_type": head_response.get('ContentType', 'unknown'),
                            "file_type": file_type
                        })
                    except Exception as e:
                        print(f"Error getting metadata for {s3_key}: {str(e)}")
            
            return files
            
        except Exception as e:
            print(f"Error listing user files: {str(e)}")
            return []
    
    # delete user file
    def delete_user_file(self, user_id: str, s3_key: str) -> bool:
        """Delete a user's uploaded file and its indexed content"""
        try:
            print(f"Deleting user file: {s3_key}")
            
            # delete from S3
            self.s3.delete_object(Bucket=RAG_BUCKET, Key=s3_key)
            
            # print success message
            print(f"successfully deleted file: {s3_key}")
            return True
            
        except Exception as e:
            print(f"Error deleting user file: {str(e)}")
            return False
