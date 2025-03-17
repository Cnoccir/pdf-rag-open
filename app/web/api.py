"""
Enhanced API layer for PDF RAG application.
Provides improved async support and better error handling.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from app.chat.chat_manager import ChatManager
from app.chat.types import ChatArgs, ResearchMode
from app.chat.memories.memory_manager import MemoryManager
from app.web.async_wrapper import async_handler, run_async

logger = logging.getLogger(__name__)

# Initialize memory manager for API-level operations
memory_manager = MemoryManager()

async def process_query(
    query: str,
    conversation_id: Optional[str] = None,
    pdf_id: Optional[str] = None,
    research_mode: str = "single",
    stream: bool = False
) -> Dict[str, Any]:
    """
    Process a query using the LangGraph-based architecture.

    Args:
        query: User query text
        conversation_id: Optional conversation ID for history
        pdf_id: Optional PDF ID to query against
        research_mode: Mode for research ("single" or "research")
        stream: Whether to stream the response

    Returns:
        Response with answer and metadata
    """
    try:
        logger.info(f"Processing query in conversation {conversation_id}, research_mode={research_mode}")

        # Convert research_mode string to enum
        research_mode_enum = ResearchMode.RESEARCH if research_mode.lower() in ["research", "true"] else ResearchMode.SINGLE

        # Initialize chat manager with appropriate args
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            pdf_id=pdf_id,
            research_mode=research_mode_enum,
            stream_enabled=stream
        )

        chat_manager = ChatManager(chat_args)

        # Initialize with conversation history if available
        await chat_manager.initialize()

        # Process query
        result = await chat_manager.query(query=query)

        # Return response including conversation_id for future reference
        if not result.get("conversation_id") and chat_manager.conversation_id:
            result["conversation_id"] = chat_manager.conversation_id

        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {"error": f"Query processing failed: {str(e)}"}

async def process_document(pdf_id: str) -> Dict[str, Any]:
    """
    Process a document using the LangGraph-based architecture.

    Args:
        pdf_id: ID of the PDF to process

    Returns:
        Processing result
    """
    try:
        logger.info(f"Processing document {pdf_id}")

        # Initialize chat manager with PDF ID
        chat_args = ChatArgs(pdf_id=pdf_id)
        chat_manager = ChatManager(chat_args)

        # Process document
        result = await chat_manager.process_document(pdf_id)
        return result

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {"error": f"Document processing failed: {str(e)}"}

async def get_conversation_history(conversation_id: str) -> Dict[str, Any]:
    """
    Get conversation history using the memory manager.

    Args:
        conversation_id: ID of the conversation

    Returns:
        Conversation history
    """
    try:
        logger.info(f"Getting conversation history for {conversation_id}")

        # Load conversation from memory manager
        conversation_state = await memory_manager.get_conversation(conversation_id)

        if not conversation_state:
            return {"error": f"Conversation {conversation_id} not found"}

        # Initialize chat manager to format the conversation
        chat_args = ChatArgs(conversation_id=conversation_id)
        chat_manager = ChatManager(chat_args)

        # Load the conversation into the chat manager
        chat_manager.conversation_state = conversation_state

        # Get formatted history
        history = chat_manager.get_conversation_history()

        return {
            "conversation_id": conversation_id,
            "messages": history,
            "metadata": conversation_state.metadata
        }

    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        return {"error": f"Failed to retrieve conversation history: {str(e)}"}

async def clear_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Clear a conversation's history.

    Args:
        conversation_id: ID of the conversation to clear

    Returns:
        Status of the operation
    """
    try:
        logger.info(f"Clearing conversation {conversation_id}")

        # Initialize chat manager
        chat_args = ChatArgs(conversation_id=conversation_id)
        chat_manager = ChatManager(chat_args)

        # Initialize first (which loads the conversation)
        await chat_manager.initialize()

        # Clear conversation
        success = chat_manager.clear_conversation()

        if not success:
            return {"error": f"Conversation {conversation_id} not found or could not be cleared"}

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "message": f"Conversation {conversation_id} cleared successfully"
        }

    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        return {"error": f"Failed to clear conversation: {str(e)}"}

# Synchronous wrapper for the async functions
def sync_process_query(*args, **kwargs):
    """Synchronous wrapper for process_query"""
    return run_async(process_query(*args, **kwargs))

def sync_process_document(*args, **kwargs):
    """Synchronous wrapper for process_document"""
    return run_async(process_document(*args, **kwargs))

def sync_get_conversation_history(*args, **kwargs):
    """Synchronous wrapper for get_conversation_history"""
    return run_async(get_conversation_history(*args, **kwargs))

def sync_clear_conversation(*args, **kwargs):
    """Synchronous wrapper for clear_conversation"""
    return run_async(clear_conversation(*args, **kwargs))
