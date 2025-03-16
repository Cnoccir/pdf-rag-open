"""
Direct test of the vector store implementation with console output.
"""

import os
import sys
from langchain_core.documents import Document

# Import the vector store implementation
from app.chat.vector_stores import get_vector_store

def main():
    """Direct test of the vector store implementation."""
    print("\n=== Testing TechDocVectorStore Implementation ===\n")
    
    # Step 1: Get the vector store
    print("Step 1: Getting vector store instance")
    vector_store = get_vector_store()
    
    # Print vector store instance type
    print(f"Vector store type: {type(vector_store).__name__}")
    print(f"Initialized: {vector_store.initialized}")
    
    if hasattr(vector_store, 'index_name'):
        print(f"Index name: {vector_store.index_name}")
    
    # Step 2: Create a single test document
    print("\nStep 2: Creating test document")
    doc = Document(
        page_content="This is a test document for the vector store.",
        metadata={
            "id": "test123",
            "pdf_id": "pdf123",
            "content_type": "text",
            "page_number": 1,
        }
    )
    print(f"Created document: {doc.page_content}")
    
    # Step 3: Try to add the document
    print("\nStep 3: Adding document to vector store")
    try:
        vector_store.add_documents([doc], namespace="test")
        print("✓ Document added successfully")
    except Exception as e:
        print(f"✗ Error adding document: {e}")
    
    # Return vector store for inspection
    return vector_store

if __name__ == "__main__":
    try:
        result = main()
        print("\nTest completed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
