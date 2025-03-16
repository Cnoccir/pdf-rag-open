"""
Test script to validate LangGraph integration with the new vector store implementation.
Tests the document processing and retrieval pipeline with the new vector store.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from uuid import uuid4
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chat.langgraph.state import GraphState, QueryState, DocumentState
from app.chat.langgraph.nodes.document_processor import process_document
from app.chat.langgraph.nodes.retriever import retrieve_content
from app.chat.vector_stores.vector_store import TechDocVectorStore, get_vector_store
from app.chat.types import ContentElement, ContentType, ContentMetadata, ProcessingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestLangGraphVectorStoreIntegration(unittest.TestCase):
    """Test the integration of LangGraph with our new vector store implementation."""
    
    @patch('app.chat.vector_stores.vector_store.TechDocVectorStore')
    @patch('app.chat.langgraph.nodes.document_processor.TechDocVectorStore')
    async def test_document_processor_integration(self, mock_doc_processor_store, mock_vector_store):
        """Test that the document processor correctly integrates with the vector store."""
        # Configure the mock
        mock_instance = mock_vector_store.return_value
        mock_doc_processor_store.return_value = mock_instance
        mock_instance.process_processing_result = MagicMock()
        
        # Create a test PDF ID
        pdf_id = f"test-pdf-{uuid4()}"
        
        # Process a document
        result = await process_document(pdf_id)
        
        # Verify vector store was used
        self.assertTrue(mock_doc_processor_store.called)
        
        # Verify processing result was passed to the vector store
        if hasattr(result, 'elements') and result.elements:
            self.assertTrue(mock_instance.process_processing_result.called)
    
    @patch('app.chat.vector_stores.vector_store.get_vector_store')
    @patch('app.chat.langgraph.nodes.retriever.get_vector_store')
    async def test_retriever_integration(self, mock_retriever_get_store, mock_get_store):
        """Test that the retriever correctly integrates with the vector store."""
        # Configure the mocks
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [
            MagicMock(page_content="Test content", metadata={"pdf_id": "test-pdf"})
        ]
        mock_vector_store.retrieve.return_value = MagicMock(results=[
            MagicMock(page_content="Test content", metadata={"pdf_id": "test-pdf"})
        ])
        
        mock_retriever_get_store.return_value = mock_vector_store
        mock_get_store.return_value = mock_vector_store
        
        # Create query state
        query_state = QueryState(
            query="test query",
            pdf_ids=["test-pdf"],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Create graph state
        state = GraphState(
            conversation=MagicMock(),
            query=query_state,
            retrieval=None,
            generation=None,
            documents=None
        )
        
        # Retrieve content
        updated_state = await retrieve_content(state)
        
        # Verify vector store was used
        self.assertTrue(mock_retriever_get_store.called)
        
        # Verify similarity_search or retrieve was called
        self.assertTrue(
            mock_vector_store.similarity_search.called or 
            mock_vector_store.retrieve.called
        )
        
        # Verify retrieval state was updated
        self.assertIsNotNone(updated_state.retrieval)
    
    @patch('app.chat.vector_stores.vector_store.TechDocVectorStore')
    async def test_end_to_end_document_processing(self, mock_vector_store):
        """Test the end-to-end document processing with the vector store."""
        # Configure the mock
        mock_instance = mock_vector_store.return_value
        mock_instance.process_processing_result = MagicMock()
        
        # Create a test document state
        document_state = DocumentState(
            pdf_id="test-pdf",
            status="processing",
            elements=[
                ContentElement(
                    id=str(uuid4()),
                    content="Test heading",
                    content_type=ContentType.HEADING,
                    metadata=ContentMetadata(page_number=1)
                ),
                ContentElement(
                    id=str(uuid4()),
                    content="Test paragraph",
                    content_type=ContentType.PARAGRAPH,
                    metadata=ContentMetadata(page_number=1)
                )
            ],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Create a processing result
        result = ProcessingResult(
            pdf_id="test-pdf",
            status="completed",
            elements=document_state.elements,
            metadata={},
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Process the result with the vector store
        vector_store = get_vector_store()
        vector_store.process_processing_result(result)
        
        # Verify process_processing_result was called
        mock_instance.process_processing_result.assert_called_once()


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    # Use asyncio to run async tests
    asyncio.run(unittest.main())
