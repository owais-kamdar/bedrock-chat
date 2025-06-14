"""
RAG (Retrieval Augmented Generation) system implementation
"""

import boto3
from PyPDF2 import PdfReader
from io import BytesIO
import json
import os
from typing import List, Dict, Tuple, Optional
import re
from vector_store import VectorStore, Document

# Initialize AWS clients
s3 = boto3.client("s3")

# Constants
BUCKET_NAME = os.getenv("RAG_BUCKET")
PDF_FILES = [
    "Brain_Facts_BookHighRes.pdf",
    "Neuroscience.Science.of.the.Brain.pdf"
]

# Chunking parameters
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100  # characters overlap between chunks

# RAG System class
class RAGSystem:
    def __init__(self):
        """Initialize RAG system with vector store"""
        self.vector_store = VectorStore()
        self._current_namespace = None
        
        # Try to get existing namespace
        stats = self.vector_store.index.describe_index_stats()
        namespaces = list(stats.namespaces.keys())
        if namespaces:
            self._current_namespace = namespaces[0]
            print(f"Using existing namespace: {self._current_namespace}")
    
    # Clean text by removing extra whitespace and normalizing while preserving important punctuation
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing while preserving important punctuation"""
        if not text:
            return ""
            
        # Replace various unicode whitespace with standard space
        text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Replace remaining newlines with space
        text = text.replace('\n', ' ')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation and quotes
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    # Split text into overlapping chunks while preserving sentence boundaries
    def create_chunks(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks while preserving sentence boundaries.
        """
        if not text:
            return []

        # Split into sentences and ensure each ends with exactly one period
        sentences = []
        for s in text.split('. '):
            s = s.strip()
            if s:
                # Remove any existing periods and add exactly one
                s = s.rstrip('.')
                sentences.append(s + '.')
        
        chunks = []
        current_chunk = ""
        last_sentence = ""  # Keep track of the last sentence for overlap

        for sentence in sentences:
            # If sentence fits in current chunk, add it
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (sentence + " ") if current_chunk else sentence
                last_sentence = sentence
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from last sentence
                words = last_sentence.split()
                if len(words) >= 2:
                    overlap_words = ' '.join(words[-2:])
                    current_chunk = overlap_words + " " + sentence
                else:
                    current_chunk = sentence
                last_sentence = sentence

                # If the new chunk is too big, split it
                if len(current_chunk) > chunk_size:
                    words = current_chunk.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= chunk_size:
                            temp_chunk += (word + " ")
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                # Keep last two words for overlap
                                overlap_words = ' '.join(temp_chunk.split()[-2:])
                                temp_chunk = overlap_words + " " + word + " "
                            else:
                                # Word itself is too long, force split
                                chunks.append(word[:chunk_size] + ".")
                                temp_chunk = word[chunk_size:] + " "
                    current_chunk = temp_chunk.strip()
                    last_sentence = current_chunk

        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Ensure all chunks end with a period
        chunks = [chunk if chunk.endswith('.') else chunk + '.' for chunk in chunks]
        
        # Clean up any double periods
        chunks = [chunk.replace('..', '.') for chunk in chunks]
        
        return chunks

    # Load and split a PDF into documents
    def load_pdf(self, object_key: str) -> List[Document]:
        """Load and split a PDF into documents"""
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        pdf_content = response["Body"].read()
        
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        
        documents = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():  # Only process non-empty pages
                # Clean the text
                cleaned_text = self.clean_text(text)
                
                # Split into chunks
                chunks = self.create_chunks(cleaned_text)
                
                # Create documents for each chunk
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        metadata={
                            "source": object_key,
                            "page": i + 1,
                            "chunk": j + 1,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    # Process all PDFs and create embeddings
    def index_documents(self, namespace: Optional[str] = None) -> bool:
        """Process all PDFs and create embeddings"""
        namespace = namespace or self._current_namespace
        total_docs = 0
        success = True
        
        # Create new namespace if none provided
        if not namespace:
            namespace = self.vector_store.clear()
            self._current_namespace = namespace
        
        for pdf_file in PDF_FILES:
            try:
                print(f"\nProcessing {pdf_file}...")
                
                # Validate file exists in S3
                try:
                    s3.head_object(Bucket=BUCKET_NAME, Key=pdf_file)
                except s3.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        print(f"Error: File {pdf_file} not found in bucket {BUCKET_NAME}")
                        success = False
                        continue
                    else:
                        raise
                
                docs = self.load_pdf(pdf_file)
                if not docs:
                    print(f"Warning: No documents extracted from {pdf_file}")
                    continue
                
                # print(f"Extracted {len(docs)} documents from {pdf_file}")
                
                # Add documents to vector store
                if not self.vector_store.add_documents(docs, namespace=namespace):
                    success = False
                else:
                    total_docs += len(docs)
                
                # print(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                success = False
                continue
        
        print(f"\nIndexing complete. Successfully indexed {total_docs} documents.")
        return success
    
    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (document, score) tuples
        """
        # If no namespace is set, use the first available namespace
        if not self._current_namespace:
            stats = self.vector_store.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            if namespaces:
                self._current_namespace = namespaces[0]
                print(f"Using namespace: {self._current_namespace}")
            else:
                print("Warning: No namespaces found in index")
                return []

        return self.vector_store.search(
            query=query,
            top_k=top_k,
            namespace=self._current_namespace,
            min_score=min_score
        )
    
    # Get relevant context from documents for a query for the LLM to use
    def get_relevant_context(self, query: str, top_k: int = 3, min_score: float = 0.0) -> str:
        """
        Get relevant context from documents for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            str: Formatted context from relevant documents
        """
        results = self.search(query, top_k, min_score)
        if not results:
            return ""
        
        # Combine relevant chunks with their metadata
        context_parts = []
        for doc, score in results:
            section_info = f"Section {doc.metadata.get('section', 'N/A')}"
            if 'page' in doc.metadata:
                section_info = f"Page {doc.metadata['page']}"
                
            context_parts.append(
                f"From {doc.metadata['source']} ({section_info}) [Score: {score:.3f}]:\n{doc.content}\n"
            )
        
        return "\n".join(context_parts)
