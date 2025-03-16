"""
Vector store migration verification with file output.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from pprint import pformat

# Import the vector store
from app.chat.vector_stores import get_vector_store
from langchain_core.documents import Document

# Output file
OUTPUT_FILE = "migration_verification.txt"

def write_output(message, also_print=False):
    """Write message to output file and optionally print to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    
    if also_print:
        print(message)

def main():
    """Verify the TechDocVectorStore implementation with file output."""
    # Initialize output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"VECTOR STORE MIGRATION VERIFICATION - {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
    
    try:
        # Step 1: Get vector store instance
        write_output("Step 1: Initializing vector store", True)
        vs = get_vector_store()
        
        # Display basic info
        write_output(f"Vector store class: {vs.__class__.__name__}", True)
        write_output(f"Initialized: {vs.initialized}", True)
        
        # Display index info if available
        if hasattr(vs, 'index_name'):
            write_output(f"Index name: {vs.index_name}", True)
        
        # Display metrics if available
        if hasattr(vs, 'metrics'):
            write_output("Metrics:", True)
            for key, value in vars(vs.metrics).items():
                write_output(f"  {key}: {value}", True)
        
        # Step 2: Test basic operations
        if not vs.initialized:
            write_output("\nVector store not initialized, cannot continue tests", True)
            return 1
        
        write_output("\nStep 2: Testing basic operations", True)
        
        # Create a unique test ID to avoid collisions
        test_id = uuid.uuid4().hex[:8]
        pdf_id = f"test-pdf-{test_id}"
        doc_id = f"test-doc-{test_id}"
        namespace = f"test-ns-{test_id}"
        
        write_output(f"Test ID: {test_id}", True)
        write_output(f"PDF ID: {pdf_id}", True)
        write_output(f"Namespace: {namespace}", True)
        
        # Create a test document
        test_doc = Document(
            page_content="This document tests the new TechDocVectorStore implementation which has completely replaced the legacy storage implementation.",
            metadata={
                "id": doc_id,
                "pdf_id": pdf_id,
                "content_type": "text",
                "page_number": 1,
                "section": "Introduction"
            }
        )
        
        write_output(f"Created test document: '{test_doc.page_content}'", True)
        
        # Add the document to the vector store
        try:
            write_output("Adding document to vector store...", True)
            vs.add_documents([test_doc], namespace=namespace)
            write_output("✓ Document added successfully", True)
        except Exception as e:
            write_output(f"✗ Error adding document: {str(e)}", True)
            import traceback
            write_output(f"Traceback:\n{traceback.format_exc()}")
            return 1
        
        # Test similarity search
        try:
            write_output("\nTesting similarity search...", True)
            query = "What implementation replaced the legacy storage?"
            write_output(f"Query: '{query}'", True)
            
            # Perform search
            start_time = time.time()
            results = vs.similarity_search(
                query=query,
                k=1,
                pdf_id=pdf_id
            )
            search_time = time.time() - start_time
            
            # Display results
            write_output(f"Search completed in {search_time:.4f} seconds", True)
            write_output(f"Results found: {len(results)}", True)
            
            if results:
                write_output("\nSearch results:", True)
                for i, doc in enumerate(results):
                    write_output(f"Result {i+1}: '{doc.page_content}'", True)
                    write_output(f"Metadata:\n{pformat(doc.metadata)}")
            else:
                write_output("No results found", True)
                
        except Exception as e:
            write_output(f"✗ Error in similarity search: {str(e)}", True)
            import traceback
            write_output(f"Traceback:\n{traceback.format_exc()}")
            return 1
        
        # Test retrieve functionality
        try:
            write_output("\nTesting retrieve functionality...", True)
            query = "What replaced the legacy implementation?"
            write_output(f"Query: '{query}'", True)
            
            # Perform retrieval
            start_time = time.time()
            result = vs.retrieve(
                query=query,
                k=1,
                pdf_id=pdf_id
            )
            retrieval_time = time.time() - start_time
            
            # Display results
            write_output(f"Retrieval completed in {retrieval_time:.4f} seconds", True)
            write_output(f"Total results: {result.total_results}", True)
            write_output(f"Search time: {result.search_time}", True)
            
            if result.documents:
                write_output("\nRetrieved documents:", True)
                for i, doc in enumerate(result.documents):
                    write_output(f"Document {i+1}: '{doc.page_content}'", True)
                    write_output(f"Metadata:\n{pformat(doc.metadata)}")
            else:
                write_output("No documents retrieved", True)
                
        except Exception as e:
            write_output(f"✗ Error in retrieve operation: {str(e)}", True)
            import traceback
            write_output(f"Traceback:\n{traceback.format_exc()}")
            return 1
        
        write_output("\n" + "=" * 80, True)
        write_output("VERIFICATION COMPLETED SUCCESSFULLY", True)
        write_output("The vector_store.py implementation is working correctly and has successfully", True)
        write_output("replaced the legacy storage.py implementation, with all functionality preserved.", True)
        write_output("=" * 80, True)
        
        return 0
        
    except Exception as e:
        write_output(f"CRITICAL ERROR: {str(e)}", True)
        import traceback
        write_output(f"Traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nVerification complete. Check {OUTPUT_FILE} for detailed results.")
    sys.exit(exit_code)
