from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio
from flask import jsonify, abort

from app.chat.chat_manager import ChatManager
from app.chat.types import ChatArgs, ResearchMode
from app.chat.memories.memory_manager import MemoryManager

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
        # Convert research_mode string to enum
        research_mode_enum = ResearchMode.RESEARCH if research_mode.lower() == "research" else ResearchMode.SINGLE
        
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
        return {"error": f"Query processing failed: {str(e)}"}, 500

async def process_document(pdf_id: str) -> Dict[str, Any]:
    """
    Process a document using the LangGraph-based architecture.
    
    Args:
        pdf_id: ID of the PDF to process
        
    Returns:
        Processing result
    """
    try:
        # Initialize chat manager with PDF ID
        chat_args = ChatArgs(pdf_id=pdf_id)
        chat_manager = ChatManager(chat_args)
        
        # Process document
        result = await chat_manager.process_document(pdf_id)
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {"error": f"Document processing failed: {str(e)}"}, 500

def get_conversation_history(conversation_id: str) -> Dict[str, Any]:
    """
    Get conversation history using the memory manager.
    
    Args:
        conversation_id: ID of the conversation
        
    Returns:
        Conversation history
    """
    try:
        # Load conversation from memory manager
        conversation_state = run_async(memory_manager.load_conversation(conversation_id))
        
        if not conversation_state:
            return {"error": f"Conversation {conversation_id} not found"}, 404
        
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
        return {"error": f"Failed to retrieve conversation history: {str(e)}"}, 500

def clear_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Clear a conversation's history.
    
    Args:
        conversation_id: ID of the conversation to clear
        
    Returns:
        Status of the operation
    """
    try:
        # Initialize chat manager
        chat_args = ChatArgs(conversation_id=conversation_id)
        chat_manager = ChatManager(chat_args)
        
        # Initialize first (which loads the conversation)
        run_async(chat_manager.initialize())
        
        # Clear conversation
        success = chat_manager.clear_conversation()
        
        if not success:
            return {"error": f"Conversation {conversation_id} not found or could not be cleared"}, 404
            
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "message": f"Conversation {conversation_id} cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        return {"error": f"Failed to clear conversation: {str(e)}"}, 500

# Helper function to run async functions in sync contexts
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()