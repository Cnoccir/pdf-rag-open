"""
Test for the new LangGraph-aligned vector store implementation.
Tests the core functionality of the TechDocVectorStore class.
"""

import os
import sys
import logging
import unittest
from unittest.mock import MagicMock, patch
import asyncio
from datetime import datetime
import uuid

# Add parent directory to path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Langchain imports
from langchain_core.documents import Document

# App imports
from app.chat.vector_stores.vector_store import (
    TechDocVectorStore, 
    get_vector_store,
    VectorStoreMetrics,
    CachedEmbeddings
)
from app.chat.types import ContentElement, ContentMetadata, ContentType, ProcessingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestVectorStoreMetrics(unittest.TestCase):
    """Test the vector store metrics functionality."""
    
    def setUp(self):
        self.metrics = VectorStoreMetrics()
    
    def test_record_query_time(self):
        """Test that query times are properly recorded."""
        self.metrics.record_query_time(0.5)
        self.metrics.record_query_time(1.5)
        
        self.assertEqual(self.metrics.total_queries, 2)
        self.assertEqual(self.metrics.query_times, [0.5, 1.5])
        self.assertEqual(self.metrics.avg_query_time, 1.0)
    
    def test_record_batch(self):
        """Test that batch sizes are properly recorded."""
        self.metrics.record_batch(10)
        self.metrics.record_batch(20)
        
        self.assertEqual(self.metrics.total_embeddings, 30)
        self.assertEqual(self.metrics.batch_sizes, [10, 20])
    
    def test_record_error(self):
        """Test that errors are properly recorded."""
        self.metrics.record_error("Test error")
        
        self.assertEqual(self.metrics.error_count, 1)
        self.assertEqual(self.metrics.last_error, "Test error")


class TestTechDocVectorStore(unittest.TestCase):
    """Test the TechDocVectorStore implementation."""
    
    @patch("pinecone.Pinecone")
    @patch("langchain_openai.OpenAIEmbeddings")
    def setUp(self, mock_embeddings, mock_pinecone):
        """Set up the test with mocked dependencies."""
        # Mock the OpenAI embeddings
        self.mock_embedding = mock_embeddings.return_value
        self.mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        self.mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock Pinecone
        self.mock_pc = mock_pinecone.return_value
        mock_indexes = MagicMock()
        mock_indexes.names.return_value = ["test-index"]
        self.mock_pc.list_indexes.return_value = mock_indexes
        
        # Mock index
        self.mock_index = MagicMock()
        self.mock_pc.Index.return_value = self.mock_index
        
        # Create the vector store
        with patch.dict(os.environ, {
            "PINECONE_API_KEY": "test-key",
            "PINECONE_ENVIRONMENT": "test-env",
            "PINECONE_INDEX": "test-index"
        }):
            self.vector_store = TechDocVectorStore()
    
    def test_initialization(self):
        """Test that the vector store initializes correctly."""
        self.assertTrue(self.vector_store.initialized)
        self.assertEqual(self.vector_store.index_name, "test-index")
    
    @patch("time.time")
    def test_similarity_search(self, mock_time):
        """Test similarity search functionality."""
        # Mock time for query timing
        mock_time.side_effect = [100, 101]  # 1 second elapsed
        
        # Mock vector store search results
        test_doc = Document(
            page_content="Test content",
            metadata={"pdf_id": "test-pdf", "page_number": 1}
        )
        self.vector_store.vectorstore.similarity_search.return_value = [test_doc]
        
        # Perform search
        results = self.vector_store.similarity_search(
            query="test query",
            k=5,
            pdf_id="test-pdf"
        )
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Test content")
        self.assertEqual(results[0].metadata["pdf_id"], "test-pdf")
        
        # Check metrics
        self.assertEqual(self.vector_store.metrics.total_queries, 1)
        self.assertEqual(self.vector_store.metrics.total_retrievals, 1)
        self.assertEqual(self.vector_store.metrics.total_filter_ops, 1)
        self.assertEqual(self.vector_store.metrics.query_times[0], 1)
    
    def test_add_documents(self):
        """Test adding documents to the vector store."""
        # Create test documents
        docs = [
            Document(
                page_content="Test content 1",
                metadata={"pdf_id": "test-pdf", "page_number": 1}
            ),
            Document(
                page_content="Test content 2",
                metadata={"pdf_id": "test-pdf", "page_number": 2}
            )
        ]
        
        # Add documents
        self.vector_store.add_documents(docs)
        
        # Check metrics
        self.assertEqual(self.vector_store.metrics.total_embeddings, 2)
        
        # Verify vectorstore method was called
        self.vector_store.vectorstore.add_documents.assert_called_once_with(
            documents=docs,
            namespace="default"
        )
    
    def test_process_document_elements(self):
        """Test processing document elements."""
        # Create test elements
        elements = [
            ContentElement(
                id=str(uuid.uuid4()),
                content="Test heading",
                content_type=ContentType.HEADING,
                metadata=ContentMetadata(
                    page_number=1,
                    section="Section 1"
                )
            ),
            ContentElement(
                id=str(uuid.uuid4()),
                content="Test paragraph",
                content_type=ContentType.PARAGRAPH,
                metadata=ContentMetadata(
                    page_number=1,
                    section="Section 1"
                )
            )
        ]
        
        # Process elements
        self.vector_store.process_document_elements(elements, "test-pdf")
        
        # Check metrics
        self.assertEqual(self.vector_store.metrics.total_embeddings, 2)
        
        # Verify vectorstore method was called with documents
        call_args = self.vector_store.vectorstore.add_documents.call_args[1]
        self.assertEqual(len(call_args["documents"]), 2)
        self.assertEqual(call_args["documents"][0].page_content, "Test heading")
        self.assertEqual(call_args["documents"][1].page_content, "Test paragraph")
    
    def test_process_processing_result(self):
        """Test processing a ProcessingResult."""
        # Create test result
        element1 = ContentElement(
            id=str(uuid.uuid4()),
            content="Test heading",
            content_type=ContentType.HEADING,
            metadata=ContentMetadata(page_number=1)
        )
        element2 = ContentElement(
            id=str(uuid.uuid4()),
            content="Test paragraph",
            content_type=ContentType.PARAGRAPH,
            metadata=ContentMetadata(page_number=1)
        )
        
        result = ProcessingResult(
            pdf_id="test-pdf",
            status="completed",
            elements=[element1, element2],
            metadata={},
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Process result
        self.vector_store.process_processing_result(result)
        
        # Check metrics
        self.assertEqual(self.vector_store.metrics.total_embeddings, 2)
        
        # Verify vectorstore method was called with documents
        call_args = self.vector_store.vectorstore.add_documents.call_args[1]
        self.assertEqual(len(call_args["documents"]), 2)


class TestCachedEmbeddings(unittest.TestCase):
    """Test the CachedEmbeddings implementation."""
    
    def setUp(self):
        """Set up the test."""
        # Mock embedding model
        self.mock_embedding = MagicMock()
        self.mock_embedding.embed_documents.return_value = [
            [0.1, 0.2, 0.3], 
            [0.4, 0.5, 0.6]
        ]
        self.mock_embedding.embed_query.return_value = [0.7, 0.8, 0.9]
        
        # Create cached embeddings
        self.cached_embeddings = CachedEmbeddings(
            self.mock_embedding,
            pdf_id="test-pdf",
            cache_size=10
        )
    
    def test_embed_documents_caching(self):
        """Test that document embeddings are cached."""
        # First call should miss cache
        texts = ["text1", "text2"]
        embeddings1 = self.cached_embeddings.embed_documents(texts)
        
        # Check results
        self.assertEqual(len(embeddings1), 2)
        self.assertEqual(embeddings1[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings1[1], [0.4, 0.5, 0.6])
        
        # Check cache metrics
        self.assertEqual(self.cached_embeddings.hits, 0)
        self.assertEqual(self.cached_embeddings.misses, 2)
        
        # Second call should hit cache
        embeddings2 = self.cached_embeddings.embed_documents(texts)
        
        # Check results again
        self.assertEqual(embeddings2, embeddings1)
        
        # Check updated cache metrics
        self.assertEqual(self.cached_embeddings.hits, 2)
        self.assertEqual(self.cached_embeddings.misses, 2)
        
        # The underlying model should only be called once
        self.mock_embedding.embed_documents.assert_called_once()
    
    def test_embed_query_caching(self):
        """Test that query embeddings are cached."""
        # First call should miss cache
        query = "test query"
        embedding1 = self.cached_embeddings.embed_query(query)
        
        # Check result
        self.assertEqual(embedding1, [0.7, 0.8, 0.9])
        
        # Check cache metrics
        self.assertEqual(self.cached_embeddings.hits, 0)
        self.assertEqual(self.cached_embeddings.misses, 1)
        
        # Second call should hit cache
        embedding2 = self.cached_embeddings.embed_query(query)
        
        # Check result again
        self.assertEqual(embedding2, embedding1)
        
        # Check updated cache metrics
        self.assertEqual(self.cached_embeddings.hits, 1)
        self.assertEqual(self.cached_embeddings.misses, 1)
        
        # The underlying model should only be called once
        self.mock_embedding.embed_query.assert_called_once()


class TestVectorStoreIntegration(unittest.TestCase):
    """Test the integration with get_vector_store function."""
    
    @patch("app.chat.vector_stores.vector_store.TechDocVectorStore")
    def test_get_vector_store(self, mock_vector_store):
        """Test that get_vector_store returns a singleton instance."""
        # Configure the mock
        mock_instance = mock_vector_store.return_value
        
        # First call should create the instance
        store1 = get_vector_store()
        self.assertEqual(store1, mock_instance)
        
        # Second call should return the same instance
        store2 = get_vector_store()
        self.assertEqual(store2, mock_instance)
        
        # Constructor should only be called once
        mock_vector_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
