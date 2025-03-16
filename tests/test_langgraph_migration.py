"""
Test script to validate LangGraph migration
Tests the entire PDF RAG pipeline from upload to query with research mode
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from uuid import uuid4
import argparse

# Add the parent directory to sys.path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chat.chat_manager import ChatManager
from app.chat.types import ChatArgs, ResearchMode
from app.chat.memories.memory_manager import MemoryManager
from app.chat.models.conversation import ConversationState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample PDF data - replace this with your test PDF path
SAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "sample.pdf")

async def upload_test_pdf(file_path):
    """
    Upload a test PDF file for processing
    
    Args:
        file_path: Path to the test PDF file
        
    Returns:
        pdf_id: ID of the uploaded PDF
    """
    logger.info(f"Uploading test PDF: {file_path}")
    
    # Generate a unique ID for the PDF
    pdf_id = f"test-pdf-{uuid4()}"
    
    try:
        # In a real scenario, we would use the file upload API
        # For testing, we'll simulate the upload by using the ID directly
        # and ensuring the file exists
        if not os.path.exists(file_path):
            logger.error(f"Test PDF file not found: {file_path}")
            return None
            
        logger.info(f"Successfully uploaded PDF with ID: {pdf_id}")
        return pdf_id
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return None

async def process_test_pdf(pdf_id):
    """
    Process the test PDF for embeddings and extraction
    
    Args:
        pdf_id: ID of the PDF to process
        
    Returns:
        bool: Success status
    """
    logger.info(f"Processing PDF ID: {pdf_id}")
    
    try:
        # Import here to avoid circular imports
        from app.web.api import process_document
        
        # Process the document
        result = await process_document(pdf_id)
        
        if "error" in result:
            logger.error(f"Error processing document: {result['error']}")
            return False
            
        logger.info(f"Successfully processed PDF: {pdf_id}")
        logger.info(f"Processing result: {result}")
        return True
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        return False

async def test_single_mode_query(pdf_id, query_text="What is this document about?"):
    """
    Test querying the PDF in single document mode
    
    Args:
        pdf_id: ID of the PDF to query
        query_text: Query text to use
        
    Returns:
        dict: Query result
    """
    logger.info(f"Testing single mode query on PDF {pdf_id}: '{query_text}'")
    
    try:
        # Create conversation
        memory_manager = MemoryManager()
        conversation = await memory_manager.create_conversation(
            title=f"Test conversation for {pdf_id}",
            pdf_id=pdf_id,
            metadata={
                "pdf_id": pdf_id,
                "research_mode": False,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Initialize chat manager
        chat_args = ChatArgs(
            conversation_id=conversation.id,
            pdf_id=pdf_id,
            research_mode=ResearchMode.SINGLE
        )
        
        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()
        
        # Query the PDF
        result = await chat_manager.query(query_text)
        
        if "error" in result:
            logger.error(f"Error in single mode query: {result['error']}")
            return None
            
        logger.info(f"Single mode query successful")
        logger.info(f"Response: {result['response']}")
        return result
    except Exception as e:
        logger.error(f"Error in single mode query: {str(e)}")
        return None

async def test_research_mode_query(pdf_ids, query_text="Compare the content of these documents"):
    """
    Test querying multiple PDFs in research mode
    
    Args:
        pdf_ids: List of PDF IDs to include in research
        query_text: Query text to use
        
    Returns:
        dict: Query result
    """
    logger.info(f"Testing research mode query on PDFs {pdf_ids}: '{query_text}'")
    
    try:
        # Create conversation with primary PDF
        primary_pdf_id = pdf_ids[0]
        memory_manager = MemoryManager()
        conversation = await memory_manager.create_conversation(
            title=f"Test research for {primary_pdf_id}",
            pdf_id=primary_pdf_id,
            metadata={
                "pdf_id": primary_pdf_id,
                "research_mode": True,
                "research_documents": pdf_ids,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Initialize chat manager with research mode
        chat_args = ChatArgs(
            conversation_id=conversation.id,
            pdf_id=primary_pdf_id,
            research_mode=ResearchMode.RESEARCH
        )
        
        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()
        
        # Query the PDFs
        result = await chat_manager.query(query_text)
        
        if "error" in result:
            logger.error(f"Error in research mode query: {result['error']}")
            return None
            
        logger.info(f"Research mode query successful")
        logger.info(f"Response: {result['response']}")
        return result
    except Exception as e:
        logger.error(f"Error in research mode query: {str(e)}")
        return None

async def test_conversation_persistence(conversation_id):
    """
    Test that conversation state is properly persisted
    
    Args:
        conversation_id: ID of the conversation to test
        
    Returns:
        bool: Success status
    """
    logger.info(f"Testing conversation persistence for ID: {conversation_id}")
    
    try:
        # Load conversation from memory manager
        memory_manager = MemoryManager()
        conversation = await memory_manager.get_conversation(conversation_id)
        
        if not conversation:
            logger.error(f"Failed to retrieve conversation: {conversation_id}")
            return False
            
        # Verify messages were saved
        message_count = len([m for m in conversation.messages if m.type != "system"])
        logger.info(f"Retrieved conversation with {message_count} messages")
        
        # Print a few messages for verification
        for i, msg in enumerate(conversation.messages):
            if i >= 3:  # Only print the first few messages
                break
            logger.info(f"Message {i+1}: {msg.type} - {msg.content[:50]}...")
            
        return True
    except Exception as e:
        logger.error(f"Error testing conversation persistence: {str(e)}")
        return False

async def run_full_pipeline_test():
    """Run the full pipeline test from upload to query"""
    
    # Check if sample PDF exists, create if it doesn't
    if not os.path.exists(SAMPLE_PDF_PATH):
        logger.warning(f"Sample PDF not found at {SAMPLE_PDF_PATH}")
        logger.info("Creating a simple test PDF...")
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(SAMPLE_PDF_PATH)
            c.drawString(100, 750, "Test PDF for LangGraph Migration")
            c.drawString(100, 700, "This is a sample PDF created for testing purposes.")
            c.drawString(100, 650, "It contains information about artificial intelligence and machine learning.")
            c.drawString(100, 600, "Large Language Models (LLMs) are transforming how we interact with text.")
            c.drawString(100, 550, "PDF RAG systems allow querying of document content using natural language.")
            c.save()
            logger.info(f"Created sample PDF at {SAMPLE_PDF_PATH}")
        except ImportError:
            logger.error("ReportLab not installed. Please create a sample PDF manually.")
            return
    
    # Upload a test PDF
    pdf_id_1 = await upload_test_pdf(SAMPLE_PDF_PATH)
    if not pdf_id_1:
        logger.error("Failed to upload first test PDF. Aborting test.")
        return
    
    # Create a second test PDF for research mode
    second_pdf_path = os.path.join(os.path.dirname(__file__), "sample2.pdf")
    if not os.path.exists(second_pdf_path):
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(second_pdf_path)
            c.drawString(100, 750, "Second Test PDF for LangGraph Migration")
            c.drawString(100, 700, "This PDF contains different information than the first test PDF.")
            c.drawString(100, 650, "Neural networks are a class of machine learning models.")
            c.drawString(100, 600, "Transformer architecture revolutionized NLP tasks.")
            c.drawString(100, 550, "Python is the most popular programming language for AI development.")
            c.save()
            logger.info(f"Created second sample PDF at {second_pdf_path}")
        except ImportError:
            logger.error("ReportLab not installed. Test will continue with single PDF only.")
    
    # Upload second PDF if it exists
    pdf_id_2 = None
    if os.path.exists(second_pdf_path):
        pdf_id_2 = await upload_test_pdf(second_pdf_path)
    
    # Process the PDFs
    processing_result_1 = await process_test_pdf(pdf_id_1)
    if not processing_result_1:
        logger.error("Failed to process first PDF. Aborting test.")
        return
    
    if pdf_id_2:
        processing_result_2 = await process_test_pdf(pdf_id_2)
        if not processing_result_2:
            logger.warning("Failed to process second PDF. Continuing with single PDF test only.")
            pdf_id_2 = None
    
    # Test single mode query
    single_result = await test_single_mode_query(pdf_id_1)
    if not single_result:
        logger.error("Single mode query test failed. Aborting test.")
        return
    
    # Save conversation ID for persistence test
    single_conversation_id = single_result.get("conversation_id")
    
    # Test research mode if we have multiple PDFs
    research_result = None
    if pdf_id_2:
        research_result = await test_research_mode_query([pdf_id_1, pdf_id_2])
        if not research_result:
            logger.error("Research mode query test failed.")
        else:
            logger.info("Research mode query test successful.")
    else:
        logger.warning("Skipping research mode test as only one PDF is available.")
    
    # Test conversation persistence
    if single_conversation_id:
        persistence_result = await test_conversation_persistence(single_conversation_id)
        if persistence_result:
            logger.info("Conversation persistence test successful.")
        else:
            logger.error("Conversation persistence test failed.")
    
    logger.info("Full pipeline test completed.")
    
    # Return results summary
    return {
        "single_mode_test": bool(single_result),
        "research_mode_test": bool(research_result),
        "persistence_test": bool(persistence_result) if single_conversation_id else False
    }

def main():
    """Main entry point for the test script"""
    global SAMPLE_PDF_PATH  # Declare global at the beginning of the function
    
    parser = argparse.ArgumentParser(description="Test LangGraph migration")
    parser.add_argument("--pdf-path", help="Path to test PDF", default=SAMPLE_PDF_PATH)
    args = parser.parse_args()
    
    # Update the sample PDF path if provided
    if args.pdf_path:
        SAMPLE_PDF_PATH = args.pdf_path
    
    # Run the full pipeline test
    logger.info("Starting LangGraph migration test...")
    results = asyncio.run(run_full_pipeline_test())
    
    # Print overall results
    if results:
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        logger.info(f"Test Results: {success_count}/{total_count} tests passed")
        
        for test, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            logger.info(f"  {test}: {status}")
        
        if success_count == total_count:
            logger.info("All tests passed! LangGraph migration is complete.")
        else:
            logger.warning("Some tests failed. LangGraph migration may need additional work.")
    else:
        logger.error("Test failed to complete. Check logs for details.")

if __name__ == "__main__":
    main()
