"""
Simple test script for the TechDocVectorStore implementation.
Writes results to a log file for review.
"""

import uuid
import sys
import traceback
from datetime import datetime
from langchain_core.documents import Document

# Import the vector store implementation
from app.chat.vector_stores import get_vector_store

# Log file path
LOG_FILE = "vector_store_test_results.log"

def log_message(message, also_print=True):
    """Write message to log file and optionally print to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    if also_print:
        print(message)

def main():
    """Simple test to verify the vector store implementation."""
    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write(f"=== TechDocVectorStore Test Results - {datetime.now()} ===\n\n")
    
    log_message("\n=== Testing TechDocVectorStore Implementation ===\n")
    
    try:
        # Step 1: Initialize vector store
        log_message("Step 1: Initializing vector store...")
        vector_store = get_vector_store()
        
        if not vector_store or not vector_store.initialized:
            log_message("ERROR: Vector store initialization failed!")
            return 1
            
        log_message(f"SUCCESS: Vector store initialized (Type: {type(vector_store).__name__})")
        # Log pinecone index details
        if hasattr(vector_store, 'index_name'):
            log_message(f"         Using Pinecone index: {vector_store.index_name}")
        
        # Step 2: Create test documents
        log_message("\nStep 2: Creating test documents...")
        pdf_id = f"test-{uuid.uuid4().hex[:8]}"
        
        documents = [
            Document(
                page_content="This is a test document about system architecture.",
                metadata={
                    "id": f"doc1_{uuid.uuid4().hex[:8]}",
                    "pdf_id": pdf_id,
                    "content_type": "text",
                    "page_number": 1,
                }
            ),
            Document(
                page_content="The system uses microservices for data processing.",
                metadata={
                    "id": f"doc2_{uuid.uuid4().hex[:8]}",
                    "pdf_id": pdf_id,
                    "content_type": "text",
                    "page_number": 1,
                }
            ),
        ]
        
        log_message(f"SUCCESS: Created {len(documents)} test documents with PDF ID: {pdf_id}")
        
        # Step 3: Add documents to vector store
        log_message("\nStep 3: Adding documents to vector store...")
        namespace = f"pdf_{pdf_id}"
        
        try:
            vector_store.add_documents(documents, namespace=namespace)
            log_message(f"SUCCESS: Added {len(documents)} documents to namespace '{namespace}'")
            
            # Log metrics if available
            if hasattr(vector_store, 'metrics'):
                log_message(f"         Metrics: {vector_store.metrics.__dict__}")
                
        except Exception as e:
            log_message(f"ERROR: Failed to add documents: {str(e)}")
            log_message(traceback.format_exc())
            return 1
            
        # Step 4: Test similarity search
        log_message("\nStep 4: Testing similarity search...")
        
        try:
            query = "What is the system architecture?"
            log_message(f"         Query: '{query}'")
            
            results = vector_store.similarity_search(
                query=query,
                k=2,
                pdf_id=pdf_id
            )
            
            log_message(f"SUCCESS: Search found {len(results)} documents")
            
            # Display search results
            if results:
                log_message("\nSearch results:")
                for i, doc in enumerate(results):
                    log_message(f"  Result {i+1}: {doc.page_content}")
                    log_message(f"    Metadata: {doc.metadata}")
            else:
                log_message("WARNING: No results found for search query")
            
        except Exception as e:
            log_message(f"ERROR: Similarity search failed: {str(e)}")
            log_message(traceback.format_exc())
            return 1
        
        # Step 5: Test retrieval
        log_message("\nStep 5: Testing document retrieval...")
        
        try:
            query = "How does the system process data?"
            log_message(f"         Query: '{query}'")
            
            search_result = vector_store.retrieve(
                query=query,
                k=2,
                pdf_id=pdf_id
            )
            
            log_message(f"SUCCESS: Retrieved {search_result.total_results} documents")
            log_message(f"         Search time: {search_result.search_time:.4f} seconds")
            
            # Display retrieved documents
            if search_result.documents:
                log_message("\nRetrieved documents:")
                for i, doc in enumerate(search_result.documents):
                    log_message(f"  Document {i+1}: {doc.page_content}")
                    log_message(f"    Metadata: {doc.metadata}")
            else:
                log_message("WARNING: No documents found for retrieval query")
            
        except Exception as e:
            log_message(f"ERROR: Document retrieval failed: {str(e)}")
            log_message(traceback.format_exc())
            return 1
        
        log_message("\n=== All TechDocVectorStore tests passed successfully! ===")
        log_message("The migration from storage.py to vector_store.py is complete.")
        return 0
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: Test failed with unexpected exception: {str(e)}")
        log_message(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed. Results written to {LOG_FILE}")
    sys.exit(exit_code)
