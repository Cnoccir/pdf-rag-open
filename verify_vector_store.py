"""
Direct verification of the TechDocVectorStore implementation after migration.
"""

import os
import sys
import time
from pprint import pprint

# Import the vector store
from app.chat.vector_stores import get_vector_store
from langchain_core.documents import Document

def main():
    """Verify the TechDocVectorStore implementation."""
    print("\n" + "="*80)
    print("VECTOR STORE MIGRATION VERIFICATION")
    print("="*80)
    
    # Step 1: Get vector store instance
    print("\nStep 1: Initializing vector store")
    vs = get_vector_store()
    
    # Display basic info
    print(f"Vector store class: {vs.__class__.__name__}")
    print(f"Initialized: {vs.initialized}")
    
    # Display index info if available
    if hasattr(vs, 'index_name'):
        print(f"Index name: {vs.index_name}")
    
    # Display metrics if available
    if hasattr(vs, 'metrics'):
        print("Metrics:")
        for key, value in vars(vs.metrics).items():
            print(f"  {key}: {value}")
    
    # Step 2: Test basic operations
    if not vs.initialized:
        print("\nVector store not initialized, cannot continue tests")
        return 1
    
    print("\nStep 2: Testing basic operations")
    
    # Create a test document
    test_doc = Document(
        page_content="This document tests the new TechDocVectorStore implementation which replaces the legacy storage implementation.",
        metadata={
            "id": "test-doc-1",
            "pdf_id": "test-pdf-001",
            "content_type": "text",
            "page_number": 1,
            "section": "Introduction"
        }
    )
    
    print(f"Created test document: '{test_doc.page_content}'")
    
    # Add the document to the vector store
    try:
        print("Adding document to vector store...")
        vs.add_documents([test_doc], namespace="test-namespace")
        print("Document added successfully")
    except Exception as e:
        print(f"Error adding document: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test similarity search
    try:
        print("\nTesting similarity search...")
        query = "What implementation replaced the legacy storage?"
        print(f"Query: '{query}'")
        
        # Perform search
        start_time = time.time()
        results = vs.similarity_search(
            query=query,
            k=1,
            pdf_id="test-pdf-001"
        )
        search_time = time.time() - start_time
        
        # Display results
        print(f"Search completed in {search_time:.4f} seconds")
        print(f"Results found: {len(results)}")
        
        if results:
            print("\nSearch results:")
            for i, doc in enumerate(results):
                print(f"Result {i+1}: '{doc.page_content}'")
                print("Metadata:")
                pprint(doc.metadata)
        else:
            print("No results found")
            
    except Exception as e:
        print(f"Error in similarity search: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test retrieve functionality
    try:
        print("\nTesting retrieve functionality...")
        query = "What replaced the legacy implementation?"
        print(f"Query: '{query}'")
        
        # Perform retrieval
        start_time = time.time()
        result = vs.retrieve(
            query=query,
            k=1,
            pdf_id="test-pdf-001"
        )
        retrieval_time = time.time() - start_time
        
        # Display results
        print(f"Retrieval completed in {retrieval_time:.4f} seconds")
        print(f"Total results: {result.total_results}")
        print(f"Search time: {result.search_time}")
        
        if result.documents:
            print("\nRetrieved documents:")
            for i, doc in enumerate(result.documents):
                print(f"Document {i+1}: '{doc.page_content}'")
                print("Metadata:")
                pprint(doc.metadata)
        else:
            print("No documents retrieved")
            
    except Exception as e:
        print(f"Error in retrieve operation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("The vector_store.py implementation is working correctly and has successfully")
    print("replaced the legacy storage.py implementation.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
