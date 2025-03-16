"""
Test runner script to execute all tests for the LangGraph migration.
This script will run all the tests related to the LangGraph architecture
and the new vector store implementation.
"""

import os
import sys
import unittest
import logging
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_tests():
    """Run all tests for the LangGraph migration."""
    logger.info("Starting LangGraph migration tests")
    logger.info(f"Test run started at: {datetime.now().isoformat()}")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test modules
    test_loader = unittest.TestLoader()
    
    # Add vector store tests
    logger.info("Adding vector store tests")
    try:
        from tests.test_vector_store import (
            TestVectorStoreMetrics,
            TestTechDocVectorStore,
            TestCachedEmbeddings,
            TestVectorStoreIntegration
        )
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestVectorStoreMetrics))
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestTechDocVectorStore))
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestCachedEmbeddings))
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestVectorStoreIntegration))
        logger.info("Vector store tests added successfully")
    except Exception as e:
        logger.error(f"Error adding vector store tests: {str(e)}")
    
    # Add LangGraph integration tests
    logger.info("Adding LangGraph vector store integration tests")
    try:
        # Import async test cases
        from tests.test_langgraph_vector_store import TestLangGraphVectorStoreIntegration
        
        # Create instance of the test case
        test_case = TestLangGraphVectorStoreIntegration()
        
        # Add all test methods from the case
        for method_name in dir(test_case):
            if method_name.startswith('test_'):
                test_method = getattr(test_case, method_name)
                if asyncio.iscoroutinefunction(test_method):
                    # Wrap async test in a sync wrapper
                    def create_sync_test(test_method):
                        def wrapped_test(*args, **kwargs):
                            return asyncio.run(test_method(*args, **kwargs))
                        wrapped_test.__name__ = test_method.__name__
                        return wrapped_test
                    
                    # Add wrapped test to suite
                    setattr(test_case, method_name, create_sync_test(test_method))
        
        test_suite.addTest(test_loader.loadTestsFromTestCase(TestLangGraphVectorStoreIntegration))
        logger.info("LangGraph vector store integration tests added successfully")
    except Exception as e:
        logger.error(f"Error adding LangGraph vector store integration tests: {str(e)}")
    
    # Add conversation state tests
    logger.info("Adding conversation state tests")
    try:
        test_suite.addTest(test_loader.discover('tests', pattern='test_conversation_state.py'))
        logger.info("Conversation state tests added successfully")
    except Exception as e:
        logger.error(f"Error adding conversation state tests: {str(e)}")
    
    # Create test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    logger.info("Running all tests")
    result = test_runner.run(test_suite)
    
    # Report results
    logger.info(f"Tests completed at: {datetime.now().isoformat()}")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    return result


if __name__ == "__main__":
    # Run all tests
    run_all_tests()
