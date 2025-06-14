"""
Initialize RAG system and build document index
"""

from dotenv import load_dotenv
from rag import RAGSystem
import os

def main():
    # Load environment variables
    load_dotenv()
    
    # Verify Pinecone API key is set
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not found in .env file")
        return
        
    print("Initializing RAG system...")
    rag = RAGSystem()
    
    print("\nCreating new namespace for documents...")
    namespace = rag.vector_store.clear()
    print(f"Created namespace: {namespace}")
    
    print("\nIndexing documents...")
    if rag.index_documents(namespace):
        print("\nSuccessfully indexed all documents!")
        print(f"Documents are stored in namespace: {namespace}")
        # print("\nYou can now use the RAG system for queries.")
    else:
        print("\nError: Failed to index some documents.")
        # print("Please check the logs above for details.")

if __name__ == "__main__":
    main() 