"""
Vector store implementation using Pinecone with GRPC client
"""

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from src.core.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, AWS_REGION, DEFAULT_TOP_K,
    VECTOR_DIMENSION, VECTOR_METRIC, EMBEDDING_BATCH_SIZE, UPSERT_BATCH_SIZE
)

# 
class Document:
    """Represents a document with content and metadata"""
    def __init__(self, content: str, metadata: Dict):
        self.content = content  # The actual text content
        self.metadata = metadata  # Associated metadata like source, page number, etc.
from src.utils.logger import log_rag_operation, log_error

# Vector store class
class VectorStore:
    def __init__(self, index_name: str = None):
        """
        Initialize Pinecone vector store with GRPC client for better performance.        
        """
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not configured")
            
        # initialize Pinecone GRPC client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        self.index_name = index_name or PINECONE_INDEX_NAME
        self._default_namespace = None  # track current namespace
        
        # connect to existing index or create if doesn't exist
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                # Create new index with serverless spec
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=VECTOR_DIMENSION,
                    metric=VECTOR_METRIC,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=AWS_REGION
                    )
                )
                
                print(f"Successfully created index: {self.index_name}")
                
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                import time
                while True:
                    try:
                        desc = self.pc.describe_index(self.index_name)
                        if desc.status.get('ready', False):
                            print("Index is ready")
                            break
                        time.sleep(1)
                    except Exception as e:
                        print(f"Waiting for index... ({str(e)})")
                        time.sleep(1)
            else:
                print(f"Using existing index: {self.index_name}")
            
            # connect to index
            self.index = self.pc.Index(self.index_name)
            
            # check if index is ready
            
        except Exception as e:
            print(f"Error during index creation: {str(e)}")
            print("Attempting to connect to existing index...")
    
    def add_documents(self, documents: List[Document], namespace: Optional[str] = None) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents
            namespace: Optional namespace for the vectors
            
        Returns:
            bool: True if all documents were successfully added
        """
        if not documents:
            return True
            
        namespace = namespace or self._default_namespace
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # step 1: generate embeddings in batches
            print(f"\nGenerating embeddings for {len(documents)} chunks in namespace: {namespace}")
            texts = [doc.content for doc in documents]
            
            # process in batches to respect Pinecone's token limits
            embeddings = []
            import time
            
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + EMBEDDING_BATCH_SIZE]
                print(f"Processing batch {i//EMBEDDING_BATCH_SIZE + 1}/{(len(texts)-1)//EMBEDDING_BATCH_SIZE + 1} ({len(batch)} documents)")
                
                # retry logic for rate limiting
                max_retries = 3
                base_delay = 10  # Start with 10 seconds
                
                for attempt in range(max_retries):
                    try:
                        result = self.pc.inference.embed(
                            model="llama-text-embed-v2",
                            inputs=batch,
                            parameters={
                                "input_type": "passage",
                                "truncate": "END"
                            }
                        )
                        embeddings.extend(result.data)
                        break  # success, exit retry loop
                        
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)  # Exponential backoff
                                print(f"Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                                time.sleep(delay)
                            else:
                                print(f"Rate limit exceeded after {max_retries} attempts. Please try again later.")
                                raise e
                        else:
                            # different error, don't retry
                            raise e
            
            print(f"Generated {len(embeddings)} embeddings")
            
            # step 2: create vectors for Pinecone
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = f"{timestamp}-{i}"
                vector = {
                    "id": doc_id,
                    "values": embedding.values,
                    "metadata": {
                        **doc.metadata,
                        "text": doc.content,
                        "chunk_id": i,
                        "timestamp": timestamp
                    }
                }
                vectors.append(vector)
            
            # step 3: upsert to Pinecone in batches
            for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
                batch = vectors[i:i + UPSERT_BATCH_SIZE]
                batch_num = i // UPSERT_BATCH_SIZE + 1
                total_batches = (len(vectors) - 1) // UPSERT_BATCH_SIZE + 1
                
                print(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                
                try:
                    self.index.upsert(vectors=batch, namespace=namespace)
                except Exception as e:
                    print(f"Failed to upsert batch {batch_num}: {str(e)}")
                    return False
            
            print(f"\nsuccessfully indexed all {len(vectors)} vectors in namespace: {namespace}")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            print(f"Error type: {type(e)}")
            return False
    
    # search for similar documents using text query
    def search(self, query: str, top_k: int = DEFAULT_TOP_K, namespace: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using text query
        
        Args:
            query: Text to search for
            top_k: Maximum number of results to return
            namespace: Optional namespace to search in

        Returns:
            List of (document, similarity_score) tuples, sorted by relevance
        """
        try:
            namespace = namespace or self._default_namespace
            
            # Step 1: Check index status
            log_rag_operation("SEARCH_INDEX_CHECK", "Checking index stats")
            stats = self.index.describe_index_stats()
            log_rag_operation("SEARCH_INDEX_STATS", f"Index stats: {stats}")
            if stats.total_vector_count == 0:
                log_rag_operation("SEARCH_WARNING", "Index appears to be empty!")
            
            # Step 2: Generate query embedding
            log_rag_operation("SEARCH_EMBEDDING", f"Generating embedding for query: {query}")
            
            # Retry logic for rate limiting
            max_retries = 3
            base_delay = 5  # Shorter delay for single query
            import time
            
            for attempt in range(max_retries):
                try:
                    result = self.pc.inference.embed(
                        model="llama-text-embed-v2",
                        inputs=[query],
                        parameters={"input_type": "query", "truncate": "END"}
                    )
                    query_embedding = result.data[0].values
                    break  # success, exit retry loop
                    
                    # if error, log error and return empty list
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Rate limit hit during search. Waiting {delay} seconds...")
                            time.sleep(delay)
                        else:
                            log_rag_operation("SEARCH_FAILED", f"Rate limit exceeded after {max_retries} attempts")
                            return []
                    else:
                        log_rag_operation("SEARCH_FAILED", f"Failed to generate query embedding: {str(e)}")
                        return []
            
            log_rag_operation("SEARCH_EMBEDDING_DONE", f"Generated embedding of dimension: {len(query_embedding)}")
            
            # step 3: query Pinecone
            log_rag_operation("SEARCH_QUERY", f"Querying Pinecone with namespace: {namespace}")
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            # print(f"Raw Pinecone response type: {type(response)}")
            # print(f"Raw Pinecone response: {response}")
            
            # convert results to documents
            documents = []
            matches = response.matches if hasattr(response, 'matches') else response.get('matches', [])
            log_rag_operation("SEARCH_MATCHES", f"Found {len(matches)} matches")
            
            for match in matches:
                # extract match data handling both dict and object formats
                if isinstance(match, dict):
                    match_id = match['id']
                    match_score = float(match['score'])
                    match_metadata = match['metadata']
                else:
                    match_id = match.id
                    match_score = float(match.score)
                    match_metadata = match.metadata
                

                    # if no metadata, log warning and continue
                if not match_metadata:
                    log_rag_operation("SEARCH_WARNING", f"No metadata for match with ID {match_id}")
                    continue
                
                # Convert back to document format
                log_rag_operation("SEARCH_PROCESSING", f"Processing match ID: {match_id}, Score: {match_score}")
                
                doc = Document(
                    content=match_metadata.get("text", ""),
                    metadata={k: v for k, v in match_metadata.items() if k not in ["text", "chunk_id", "timestamp"]}
                )
                documents.append((doc, match_score))
                log_rag_operation("SEARCH_RESULT", f"Added result: score={match_score}, content={doc.content[:50]}...")
            
            log_rag_operation("SEARCH_COMPLETE", f"Returning {len(documents)} results")
            return documents
            
        except Exception as e:
            log_error(f"Error in search: {str(e)}")
            return []
    
    # clear vectors from a namespace and create a new one
    def clear(self, namespace: Optional[str] = None) -> str:
        """
        Clear vectors from a namespace and create a new one.
        """
        try:
            if namespace:
                # Delete specific namespace
                self.index.delete(delete_all=True, namespace=namespace)
            
            # Create new namespace with timestamp
            new_namespace = f"ns-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            self._default_namespace = new_namespace
            print(f"Using namespace: {new_namespace}")
            return new_namespace
            
        except Exception as e:
            print(f"Operation failed: {str(e)}")
            # Return a new namespace anyway to ensure we start fresh
            new_namespace = f"ns-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            self._default_namespace = new_namespace
            return new_namespace 