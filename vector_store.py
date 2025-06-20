"""
Vector store implementation using Pinecone with GRPC client
"""

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime

class Document:
    """Represents a document with content and metadata"""
    def __init__(self, content: str, metadata: Dict):
        self.content = content  # The actual text content
        self.metadata = metadata  # Associated metadata like source, page number, etc.

# Vector store class
class VectorStore:
    def __init__(self, index_name: str = "bedrock-rag"):
        """
        Initialize Pinecone vector store with GRPC client for better performance.
        Creates a new index if one doesn't exist, or connects to an existing one.
        
        Args:
            index_name: Name of the Pinecone index
        """
        # Initialize Pinecone GRPC client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        self.index_name = index_name
        self._default_namespace = None  # Track current namespace
        
        # Connect to existing index or create if doesn't exist
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                # Create new index with serverless spec
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # Dimension size for llama-text-embed-v2
                    metric="cosine",  # Use cosine similarity for better text matching
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
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
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Check if index is ready
            
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
        vectors = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Step 1: Generate embeddings
            print(f"\nGenerating embeddings for {len(documents)} chunks in namespace: {namespace}")
            texts = [doc.content for doc in documents]
            # print("Document texts:")
            # for i, text in enumerate(texts):
            #     print(f"{i+1}: {text}")
            
            # Process in batches to respect Llama's token limits
            embeddings = []
            batch_size = 96  # Llama's batch size limit
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                result = self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=batch,
                    parameters={
                        "input_type": "passage",
                        "truncate": "END"
                    }
                )
                embeddings.extend(result.data)
            
            # print(f"Generated {len(embeddings)} embeddings")
            
            # Create vectors for Pinecone
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
                # print(f"\nPrepared vector {i+1}:")
                # print(f"ID: {doc_id}")
                # print(f"Metadata: {vector['metadata']}")
                # print(f"Vector dimension: {len(vector['values'])}")
            
            # Check initial stats
            # print("\nChecking initial index stats...")
            initial_stats = self.index.describe_index_stats()
            initial_count = initial_stats.total_vector_count
            # print(f"Initial vector count: {initial_count}")
            
            # Upsert to Pinecone in batches with retry
            batch_size = 100
            max_retries = 3
            import time
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                # print(f"\nUpserting batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                # print(f"Batch size: {len(batch)} vectors")
                
                for retry in range(max_retries):
                    try:
                        self.index.upsert(vectors=batch, namespace=namespace)
                        # print("Batch upsert sent successfully")
                        
                        # Wait for vectors to be indexed - give more initial time and check less frequently
                        # print("Waiting for vectors to be indexed...")
                        time.sleep(3)  # Initial wait to let indexing start
                        for _ in range(3):  # Only check 3 times with longer intervals
                            stats = self.index.describe_index_stats()
                            current_count = stats.total_vector_count
                            # print(f"Current vector count: {current_count}")
                            
                            if current_count > initial_count:
                                # print("Vectors successfully indexed!")
                                break
                            time.sleep(3)  # Wait longer between checks
                        else:
                            # print("Warning: Vectors not showing up in index stats")
                            continue
                            
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        print(f"Error upserting batch (attempt {retry + 1}/{max_retries}): {str(e)}")
                        if retry == max_retries - 1:  # Last attempt
                            print(f"Failed to upsert batch after {max_retries} attempts")
                            return False
                        time.sleep(1)  # Wait before retry
            
            # Final verification
            # print("\nVerifying final index state...")
            final_stats = self.index.describe_index_stats()
            # print(f"Final index stats: {final_stats}")
            expected_count = initial_count + len(vectors)
            actual_count = final_stats.total_vector_count
            
            if actual_count < expected_count:
                print(f"Warning: Expected {expected_count} vectors but found {actual_count}")
                return False
                
            print(f"\nSuccessfully indexed all {len(vectors)} vectors")
            print(f"Namespace {namespace} now has {actual_count} total vectors")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            print(f"Error type: {type(e)}")
            return False
    
    def search(self, query: str, top_k: int = 3, namespace: Optional[str] = None, min_score: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using text query
        
        Args:
            query: Text to search for
            top_k: Maximum number of results to return
            namespace: Optional namespace to search in
            min_score: Minimum similarity score (0-1) for results
            
        Returns:
            List of (document, similarity_score) tuples, sorted by relevance
        """
        try:
            namespace = namespace or self._default_namespace
            
            # Step 1: Check index status
            print("\nChecking index stats...")
            stats = self.index.describe_index_stats()
            print(f"Index stats: {stats}")
            if stats.total_vector_count == 0:
                print("Warning: Index appears to be empty!")
            
            # Step 2: Generate query embedding
            print(f"\nGenerating embedding for query: {query}")
            query_embedding = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[query],
                parameters={
                    "input_type": "query",  # Optimize for query embedding
                    "truncate": "END"
                }
            ).data[0].values
            print(f"Generated embedding of dimension: {len(query_embedding)}")
            
            # Step 3: Query Pinecone
            print(f"\nQuerying Pinecone with namespace: {namespace}")
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            # print(f"Raw Pinecone response type: {type(response)}")
            # print(f"Raw Pinecone response: {response}")
            
            # Convert results to documents
            documents = []
            matches = response.matches if hasattr(response, 'matches') else response.get('matches', [])
            print(f"Found {len(matches)} matches")
            
            for match in matches:
                # Extract match data handling both dict and object formats
                if isinstance(match, dict):
                    match_id = match['id']
                    match_score = float(match['score'])
                    match_metadata = match['metadata']
                else:
                    match_id = match.id
                    match_score = float(match.score)
                    match_metadata = match.metadata
                
                # Filter by minimum score
                if match_score < min_score:
                    continue
                    
                if not match_metadata:
                    print(f"Warning: No metadata for match with ID {match_id}")
                    continue
                
                # Convert back to document format
                print(f"\nProcessing match:")
                print(f"ID: {match_id}")
                # print(f"Score: {match_score}")
                # print(f"Metadata: {match_metadata}")
                
                doc = Document(
                    content=match_metadata.get("text", ""),
                    metadata={k: v for k, v in match_metadata.items() if k not in ["text", "chunk_id", "timestamp"]}
                )
                documents.append((doc, match_score))
                print(f"Added result: score={match_score}, content={doc.content[:50]}...")
            
            print(f"Returning {len(documents)} results")
            return documents
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            print(f"Error type: {type(e)}")
            return []
    
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