"""
Initialize RAG system and build neuroscience document index
"""

from src.core.config import validate_required_env_vars
from src.core.rag import RAGSystem
from src.utils.logger import log_initialization_step, log_system_event

def main():
    """Initialize the RAG system with neuroscience documents"""
    print("Starting RAG system initialization...")
    log_system_event("INITIALIZATION_START", "Starting RAG system initialization")
    
    # check required environment variables
    missing_vars = validate_required_env_vars()
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        print(f"Error: {error_msg}")
        log_initialization_step("ENV_VALIDATION", "FAILED", {"error": error_msg})
        return
    

    print("Environment validation passed")
    log_initialization_step("ENV_VALIDATION", "SUCCESS")
        
    print("\nInitializing RAG system...")
    
    try:
        rag_system = RAGSystem()
        print("RAG system instance created")
        
        print("\nIndexing neuroscience documents...")
        
        # index based neuroscience documents
        if rag_system.index_neuroscience_documents():
            print("\nSuccessfully indexed all neuroscience documents")
            print("The base neuroscience RAG system is now ready")
            print("Users can also upload their own PDF/TXT files for personalized context.")
            log_system_event("INITIALIZATION_COMPLETE", "RAG system initialization completed successfully")
        else:
            print("\nError: Failed to index some neuroscience documents.")
            print("Please check the logs above for details.")
            log_system_event("INITIALIZATION_FAILED", "RAG system initialization failed")
            
    except Exception as e:
        print(f"\nError initializing RAG system: {str(e)}")
        log_initialization_step("RAG_SYSTEM", "FAILED", {"error": str(e)})
        log_system_event("INITIALIZATION_FAILED", f"RAG system initialization failed: {str(e)}")

if __name__ == "__main__":
    main() 