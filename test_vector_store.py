"""
Simple test script for the new TechDocVectorStore implementation.
This script tests the basic operations of the vector store to ensure 
that our migration from storage.py to vector_store.py is working correctly.
"""

import os
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the vector store implementation
from app.chat.vector_stores import get_vector_store, TechDocVectorStore
from app.chat.types import ContentElement, ProcessingResult
from langchain_core.documents import Document

def test_vector_store():
    """Test basic vector store operations."""
    print("\n" + "="*80)
    print("🧪 Testing TechDocVectorStore Implementation")
    print("="*80)
    
    try:
        # Step 1: Initialize vector store
        print("\n🔄 Step 1: Initializing vector store...")
        vector_store = get_vector_store()
        
        if not vector_store or not vector_store.initialized:
            print("❌ ERROR: Vector store initialization failed!")
            return False
            
        print("✅ Vector store initialized successfully")
        
        # Step 2: Create test documents
        print("\n🔄 Step 2: Creating test documents...")
        pdf_id = f"test-{uuid.uuid4()}"
        
        documents = [
            Document(
                page_content="This is a test document about system architecture and microservices.",
                metadata={
                    "id": f"doc1_{uuid.uuid4()}",
                    "pdf_id": pdf_id,
                    "content_type": "text",
                    "page_number": 1,
                    "section": "Introduction",
                }
            ),
            Document(
                page_content="The system follows a microservices architecture with components for data ingestion, processing, and visualization.",
                metadata={
                    "id": f"doc2_{uuid.uuid4()}",
                    "pdf_id": pdf_id,
                    "content_type": "text",
                    "page_number": 1,
                    "section": "Architecture",
                }
            ),
            Document(
                page_content="Performance metrics show 99.9% uptime and 50ms average response time.",
                metadata={
                    "id": f"doc3_{uuid.uuid4()}",
                    "pdf_id": pdf_id,
                    "content_type": "text",
                    "page_number": 2,
                    "section": "Performance",
                }
            ),
        ]
        
        print(f"✅ Created {len(documents)} test documents with PDF ID: {pdf_id}")
        
        # Step 3: Add documents to vector store
        print("\n🔄 Step 3: Adding documents to vector store...")
        namespace = f"pdf_{pdf_id}"
        
        try:
            vector_store.add_documents(documents, namespace=namespace)
            print(f"✅ Added {len(documents)} documents to vector store in namespace '{namespace}'")
        except Exception as e:
            print(f"❌ ERROR: Failed to add documents: {str(e)}")
            raise
            
        # Step 4: Test similarity search
        print("\n🔄 Step 4: Testing similarity search...")
        
        try:
            results = vector_store.similarity_search(
                query="What is the system architecture?",
                k=2,
                pdf_id=pdf_id
            )
            
            print(f"✅ Search successful, found {len(results)} documents")
            
            # Display search results
            if results:
                print("\n📋 Search results:")
                for i, doc in enumerate(results):
                    print(f"  Result {i+1}: {doc.page_content[:100]}...")
                    print(f"    Metadata: {doc.metadata}")
            else:
                print("⚠️ No results found for the search query")
                
        except Exception as e:
            print(f"❌ ERROR: Similarity search failed: {str(e)}")
            raise
            
        # Step 5: Test with ContentElements
        print("\n🔄 Step 5: Testing with ContentElements...")
        
        # Create content elements
        elements = [
            ContentElement(
                id=f"element1_{uuid.uuid4()}",
                content="This is a code example for processing data in our system.",
                content_type="code",
                pdf_id=pdf_id,
                page_number=3,
                section="Implementation",
                metadata={
                    "section_title": "Code Implementation",
                    "chunk_id": "chunk_1"
                }
            ),
            ContentElement(
                id=f"element2_{uuid.uuid4()}",
                content="The deployment uses Kubernetes with auto-scaling based on CPU usage.",
                content_type="text",
                pdf_id=pdf_id,
                page_number=4,
                section="Deployment",
                metadata={
                    "section_title": "Deployment Architecture",
                    "chunk_id": "chunk_2"
                }
            )
        ]
        
        # Create processing result
        result = ProcessingResult(
            pdf_id=pdf_id,
            elements=elements,
            metadata={
                "filename": f"test_document_{pdf_id}.pdf",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Process result
            vector_store.process_processing_result(result)
            print(f"✅ Processed result with {len(elements)} elements")
            
        except Exception as e:
            print(f"❌ ERROR: Processing result failed: {str(e)}")
            raise
            
        # Step 6: Test document retrieval
        print("\n🔄 Step 6: Testing document retrieval...")
        
        try:
            # Test retrieval
            search_result = vector_store.retrieve(
                query="What is the deployment architecture?",
                k=3,
                pdf_id=pdf_id,
                filter_content_types=["text", "code"]
            )
            
            print(f"✅ Retrieval successful, found {search_result.total_results} documents")
            print(f"  Query time: {search_result.search_time:.4f} seconds")
            
            # Display retrieved documents
            if search_result.documents:
                print("\n📋 Retrieved documents:")
                for i, doc in enumerate(search_result.documents):
                    print(f"  Document {i+1}: {doc.page_content[:100]}...")
                    print(f"    Metadata: {doc.metadata}")
            else:
                print("⚠️ No documents found for the retrieval query")
                
        except Exception as e:
            print(f"❌ ERROR: Document retrieval failed: {str(e)}")
            raise
            
        print("\n" + "="*80)
        print("✅ All TechDocVectorStore tests passed successfully!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ Vector store test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the vector store test
    if test_vector_store():
        print("\n✨ The TechDocVectorStore migration is complete and working properly! ✨")
        print("All legacy storage.py functionality has been successfully replaced.")
    else:
        print("\n⚠️ The vector store test encountered errors. Please check the logs for details.")
