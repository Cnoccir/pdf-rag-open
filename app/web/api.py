"""
API layer for PDF RAG application.
Provides simplified interface to the LangGraph workflow.
"""

from typing import Dict, List, Any, Optional
import logging

import inspect
from app.chat.types import ResearchMode
from app.chat.models import ChatArgs
from app.chat.chat_manager import ChatManager
from app.chat.memories.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# Shared memory manager for global access
memory_manager = MemoryManager()

def process_query(
    query: str,
    conversation_id: Optional[str] = None,
    pdf_id: Optional[str] = None,
    research_mode: str = "single",
    stream: bool = False,
    pdf_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process a query using the LangGraph workflow.

    Args:
        query: User query
        conversation_id: Optional conversation ID for history
        pdf_id: Optional PDF ID to query against
        research_mode: Mode to use ("single" or "research")
        stream: Whether to stream the response
        pdf_ids: Optional list of PDF IDs for research mode

    Returns:
        Query results
    """
    try:
        logger.info(f"Processing query in conversation {conversation_id}, research_mode={research_mode}")

        # Convert research_mode string to enum
        research_mode_enum = ResearchMode.RESEARCH if research_mode.lower() in ["research", "multi", "true"] else ResearchMode.SINGLE

        # Initialize chat manager
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            pdf_id=pdf_id,
            research_mode=research_mode_enum,
            stream_enabled=stream
        )

        chat_manager = ChatManager(chat_args)

        # Initialize with conversation history
        chat_manager.initialize()

        # Override PDF IDs for research mode
        if research_mode_enum == ResearchMode.RESEARCH and pdf_ids:
            pdf_ids_to_use = pdf_ids
        else:
            pdf_ids_to_use = [pdf_id] if pdf_id else None

        # Process query
        result = chat_manager.query(query, pdf_ids_to_use)

        # Ensure conversation_id is included
        if not result.get("conversation_id") and chat_manager.conversation_id:
            result["conversation_id"] = chat_manager.conversation_id

        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {
            "error": f"Query processing failed: {str(e)}",
            "conversation_id": conversation_id
        }

def process_document(pdf_id: str) -> Dict[str, Any]:
    """
    Process a document using the document processing workflow.

    Args:
        pdf_id: ID of PDF to process

    Returns:
        Processing results
    """
    try:
        logger.info(f"Processing document {pdf_id}")

        # Initialize chat manager with PDF ID
        chat_args = ChatArgs(pdf_id=pdf_id)
        chat_manager = ChatManager(chat_args)

        # Process document
        result = chat_manager.process_document(pdf_id)
        return result

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "pdf_id": pdf_id,
            "error": str(e)
        }

def get_conversation_history(conversation_id: str) -> Dict[str, Any]:
    """
    Get conversation history.

    Args:
        conversation_id: Conversation ID

    Returns:
        Conversation history
    """
    try:
        logger.info(f"Getting conversation history for {conversation_id}")

        # Get conversation from memory manager
        conversation_state = memory_manager.get_conversation(conversation_id)

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

def clear_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Clear a conversation's history.

    Args:
        conversation_id: Conversation ID to clear

    Returns:
        Success status
    """
    try:
        logger.info(f"Clearing conversation {conversation_id}")

        # Initialize chat manager
        chat_args = ChatArgs(conversation_id=conversation_id)
        chat_manager = ChatManager(chat_args)

        # Initialize first (which loads the conversation)
        chat_manager.initialize()

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

def list_conversations(pdf_id: Optional[str] = None) -> Dict[str, Any]:
    """
    List all conversations, optionally filtered by PDF ID.

    Args:
        pdf_id: Optional PDF ID to filter by

    Returns:
        List of conversations
    """
    try:
        logger.info(f"Listing conversations" + (f" for PDF {pdf_id}" if pdf_id else ""))

        # Get conversations from memory manager
        conversations = memory_manager.list_conversations(pdf_id)

        # Format for return
        conversation_list = []
        for conv in conversations:
            conversation_list.append({
                "id": conv.conversation_id,
                "title": conv.title,
                "pdf_id": conv.pdf_id,
                "updated_at": conv.updated_at.isoformat() if hasattr(conv.updated_at, "isoformat") else str(conv.updated_at),
                "message_count": len([msg for msg in conv.messages if msg.type.value != "system"]),
                "metadata": conv.metadata
            })

        return {
            "conversations": conversation_list,
            "count": len(conversation_list)
        }

    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}", exc_info=True)
        return {"error": f"Failed to list conversations: {str(e)}"}
