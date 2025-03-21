# Save as tests/integration_test.py

import os
import sys
import time
import json
import logging
from datetime import datetime
from pprint import pprint

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("integration_test")

# Import app modules
from app.chat.chat_manager import ChatManager
from app.chat.models import ChatArgs
from app.chat.types import ResearchMode
from app.chat.vector_stores import get_vector_store, get_mongo_store, get_qdrant_store
from app.chat.vector_stores.initialization import initialize_and_verify_stores, verify_pdf_exists

def test_database_initialization():
    """Test database initialization."""
    logger.info("=== Testing Database Initialization ===")

    # Run initialization
    init_results = initialize_and_verify_stores(max_retries=2)

    # Print results
    logger.info(f"Initialization Results: {json.dumps(init_results, indent=2)}")

    # Verify success
    if init_results["success"]:
        logger.info("✓ Database initialization successful")
    else:
        logger.error("✗ Database initialization failed!")
        for error in init_results["errors"]:
            logger.error(f"  - {error}")

    return init_results["success"]

def test_pdf_verification(pdf_id):
    """Test PDF verification in vector stores."""
    logger.info(f"=== Testing PDF Verification for {pdf_id} ===")

    # Run verification
    results = verify_pdf_exists(pdf_id)

    # Print results
    logger.info(f"Verification Results: {json.dumps(results, indent=2)}")

    # Check for success
    exists_in_mongo = results.get("exists_in_mongo", False)
    exists_in_qdrant = results.get("exists_in_qdrant", False)

    if exists_in_mongo and exists_in_qdrant:
        logger.info(f"✓ PDF {pdf_id} exists in both MongoDB and Qdrant")
        return True
    elif exists_in_mongo:
        logger.warning(f"⚠ PDF {pdf_id} exists in MongoDB but not in Qdrant")
        return "partial"
    elif exists_in_qdrant:
        logger.warning(f"⚠ PDF {pdf_id} exists in Qdrant but not in MongoDB")
        return "partial"
    else:
        logger.error(f"✗ PDF {pdf_id} not found in either MongoDB or Qdrant")
        return False

def test_semantic_search(pdf_id, query="what is this document about?"):
    """Test semantic search for a PDF."""
    logger.info(f"=== Testing Semantic Search for PDF {pdf_id} ===")
    logger.info(f"Query: {query}")

    # Get vector store
    vector_store = get_vector_store()

    # Run search
    start_time = time.time()
    results = vector_store.semantic_search(
        query=query,
        k=5,
        pdf_id=pdf_id
    )
    elapsed_time = time.time() - start_time

    # Print results
    logger.info(f"Retrieved {len(results)} results in {elapsed_time:.2f} seconds")

    if results:
        # Show summary of first result
        first_result = results[0]
        content_preview = first_result.page_content[:150] + "..." if len(first_result.page_content) > 150 else first_result.page_content
        logger.info(f"First result content: {content_preview}")
        logger.info(f"First result metadata: {json.dumps(first_result.metadata, default=str)}")

        # Show score distribution
        if len(results) > 1:
            scores = [r.metadata.get("score", 0) for r in results]
            logger.info(f"Score range: {min(scores):.4f} - {max(scores):.4f}")

        logger.info("✓ Semantic search returned results")
        return True
    else:
        logger.error("✗ Semantic search returned no results")
        return False

def test_chat_workflow(pdf_id, query="What is the main topic of this document?"):
    """Test the complete chat workflow."""
    logger.info(f"=== Testing Complete Chat Workflow for PDF {pdf_id} ===")
    logger.info(f"Query: {query}")

    # Create chat manager
    chat_args = ChatArgs(
        pdf_id=pdf_id,
        research_mode=ResearchMode.SINGLE,
        stream_enabled=False
    )

    chat_manager = ChatManager(chat_args)

    # Initialize chat manager
    logger.info("Initializing chat manager...")
    chat_manager.initialize()

    # Run query
    logger.info("Processing query...")
    start_time = time.time()
    result = chat_manager.query(query)
    elapsed_time = time.time() - start_time

    # Check result
    if "error" in result:
        logger.error(f"✗ Chat workflow failed: {result['error']}")
        return False

    logger.info(f"Query processed in {elapsed_time:.2f} seconds")
    logger.info(f"Chat response: {result['response'][:150]}...")
    logger.info(f"Citations: {len(result.get('citations', []))}")

    # Check if conversation was saved
    if chat_manager.conversation_id:
        logger.info(f"Conversation saved with ID: {chat_manager.conversation_id}")
        logger.info(f"Message count: {len(chat_manager.conversation_state.messages)}")
        logger.info("✓ Chat workflow completed successfully")
        return True
    else:
        logger.error("✗ Conversation was not saved")
        return False

def main():
    """Run integration tests."""
    logger.info("Starting RAG System Integration Tests")

    # Get PDF ID from environment or use a default
    pdf_id = os.environ.get("TEST_PDF_ID", "")

    if not pdf_id:
        logger.error("No PDF ID provided. Set TEST_PDF_ID environment variable.")
        return False

    # Run tests
    test_results = {}

    # Test 1: Database initialization
    test_results["database_init"] = test_database_initialization()

    # Test 2: PDF verification
    test_results["pdf_verification"] = test_pdf_verification(pdf_id)

    # If the PDF doesn't exist, skip further tests
    if test_results["pdf_verification"] is not True:
        logger.error("PDF verification failed, skipping further tests")
        summarize_results(test_results)
        return False

    # Test 3: Semantic search
    test_results["semantic_search"] = test_semantic_search(pdf_id)

    # Test 4: Chat workflow
    test_results["chat_workflow"] = test_chat_workflow(pdf_id)

    # Summarize results
    success = summarize_results(test_results)

    return success

def summarize_results(test_results):
    """Summarize test results."""
    logger.info("\n=== Test Summary ===")

    all_passed = True
    for test_name, result in test_results.items():
        status = "✓ PASS" if result is True else "⚠ PARTIAL" if result == "partial" else "✗ FAIL"
        if result is not True:
            all_passed = False
        logger.info(f"{status} - {test_name}")

    if all_passed:
        logger.info("\n✅ All tests passed!")
    else:
        logger.info("\n❌ Some tests failed or partially passed.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
