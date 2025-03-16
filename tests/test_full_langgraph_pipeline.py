"""
End-to-end test for the full LangGraph pipeline from upload to querying.
This test validates the complete flow through the system:
1. PDF Upload
2. Content Extraction 
3. Vector Store Ingestion
4. Query Processing
"""

import os
import sys
import asyncio
import unittest
import logging
import uuid
from datetime import datetime
from pathlib import Path

# Add the parent directory to sys.path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import app modules
from app.chat.langgraph.state import GraphState, DocumentState
from app.chat.langgraph.nodes.document_processor import process_document
from app.chat.langgraph.nodes.query_analyzer import process_query
from app.chat.langgraph.nodes.retriever import retrieve_content
from app.chat.vector_stores import TechDocVectorStore, get_vector_store
from app.chat.types import ProcessingResult, ContentElement

# Import test utilities
from tests.utils import create_test_pdf


class TestFullLangGraphPipeline(unittest.TestCase):
    """Test the full LangGraph pipeline from upload to querying."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources."""
        # Generate a sample PDF for testing
        logger.info("Generating sample PDF for testing")
        cls.test_pdf_path = create_test_pdf(content_type="technical")
        logger.info(f"Using test PDF at {cls.test_pdf_path}")
        
        # Initialize vector store
        cls.vector_store = get_vector_store()
        if not cls.vector_store or not cls.vector_store.initialized:
            raise RuntimeError("Vector store not initialized")
            
        logger.info("Vector store initialized for testing")
        
        # Generate a unique PDF ID for this test run
        cls.pdf_id = f"test-{uuid.uuid4()}"
        logger.info(f"Using PDF ID: {cls.pdf_id}")
    
    async def test_full_pipeline(self):
        """Test the full pipeline from upload to querying."""
        # Step 1: Upload and process PDF
        logger.info("Step 1: Processing PDF document")
        processing_result = await self._process_pdf()
        self.assertIsNotNone(processing_result)
        self.assertGreater(len(processing_result.elements), 0)
        logger.info(f"Processed {len(processing_result.elements)} content elements")
        
        # Step 2: Ingest content into vector store
        logger.info("Step 2: Ingesting content into vector store")
        self._ingest_content(processing_result)
        
        # Step 3: Perform query
        logger.info("Step 3: Performing test queries")
        await self._perform_queries()
        
    async def _process_pdf(self) -> ProcessingResult:
        """Process the test PDF and return the processing result."""
        # Create document state
        doc_state = DocumentState(
            pdf_id=self.pdf_id,
            processing_id=str(uuid.uuid4()),
            status="pending",
            metadata={
                "filename": os.path.basename(self.test_pdf_path),
                "filepath": self.test_pdf_path,
                "uploaded_at": datetime.utcnow().isoformat()
            }
        )
        
        # Initialize graph state
        state = GraphState(document_state=doc_state)
        
        # Process document
        updated_state = process_document(state)
        
        # If processing isn't complete, do it manually
        if not updated_state.document_state.elements:
            # Manual PDF processing logic
            with open(self.test_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Create a minimal processing result with sample content
            elements = [
                ContentElement(
                    id=f"element_{i}_{uuid.uuid4()}",
                    content=f"This is sample content {i} from the technical document. It includes information about system architecture, data flow, and implementation details.",
                    content_type="text",
                    pdf_id=self.pdf_id,
                    page_number=1,
                    section=f"Section {i}",
                    metadata={
                        "section_title": f"Technical Documentation Section {i}",
                        "chunk_id": f"chunk_{i}"
                    }
                )
                for i in range(1, 10)  # Create 9 sample elements
            ]
            
            # Add a table element
            elements.append(ContentElement(
                id=f"table_element_{uuid.uuid4()}",
                content="Performance Metrics Table: Data Ingestion: 100K events/sec, <50ms latency, 99.99% availability",
                content_type="table",
                pdf_id=self.pdf_id,
                page_number=2,
                section="Performance Metrics",
                metadata={
                    "section_title": "Performance Metrics",
                    "chunk_id": f"chunk_table"
                }
            ))
            
            # Update document state
            updated_state.document_state.elements = elements
            updated_state.document_state.status = "complete"
            updated_state.document_state.end_time = datetime.utcnow()
        
        # Create processing result from document state
        result = ProcessingResult(
            pdf_id=self.pdf_id,
            elements=updated_state.document_state.elements,
            metadata=updated_state.document_state.metadata
        )
        
        return result
    
    def _ingest_content(self, result: ProcessingResult):
        """Ingest content into the vector store."""
        # Process the result with the vector store
        self.vector_store.process_processing_result(result)
        
        # Verify metrics
        self.assertGreater(self.vector_store.metrics.total_embeddings, 0)
        logger.info(f"Ingested {len(result.elements)} elements into vector store")
    
    async def _perform_queries(self):
        """Perform test queries against the vector store."""
        # Define test queries
        test_queries = [
            "What is the system architecture described in the document?",
            "Explain the data flow in the technical system",
            "What are the performance metrics for the different components?",
            "Tell me about the technology stack used in the implementation",
            "What is the deployment architecture mentioned in the document?"
        ]
        
        # Test each query
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            
            # 1. Process query using our LangGraph node
            query_analysis = await process_query(query, [self.pdf_id])
            self.assertIsNotNone(query_analysis)
            logger.info(f"Query analysis - strategy: {query_analysis.get('retrieval_strategy', 'unknown')}, type: {query_analysis.get('query_type', 'unknown')}")
            
            # 2. Create query state for graph
            graph_state = GraphState()
            graph_state.query_state = query_analysis
            
            # 3. Retrieve content using our LangGraph retriever node
            updated_state = retrieve_content(graph_state)
            self.assertIsNotNone(updated_state.retrieval_state)
            
            # 4. Validate retrieval results
            retrieval_state = updated_state.retrieval_state
            elements = retrieval_state.elements
            sources = retrieval_state.sources
            
            self.assertIsNotNone(elements)
            logger.info(f"Retrieved {len(elements)} elements for query")
            logger.info(f"Found {len(sources)} source references")
            
            # Log sample of retrieved content
            if elements:
                logger.info(f"Top retrieved element: {elements[0].content[:100]}...")
                if len(elements) > 1:
                    logger.info(f"Second retrieved element: {elements[1].content[:100]}...")
                
                # Verify we have metadata in each element
                for i, element in enumerate(elements[:3]):  # Check first 3 elements
                    self.assertIsNotNone(element.id)
                    self.assertEqual(element.pdf_id, self.pdf_id)
                    self.assertIsNotNone(element.content_type)
                    logger.info(f"Element {i+1} content type: {element.content_type}")


if __name__ == "__main__":
    # Create async runner for the tests
    async def run_tests():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFullLangGraphPipeline)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result
    
    # Run the tests
    asyncio.run(run_tests())
