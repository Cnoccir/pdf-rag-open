from flask import Blueprint, g, request, jsonify
from app.web.hooks import login_required
import logging
import json
import asyncio
from datetime import datetime
from uuid import uuid4

from app.chat.types import ResearchMode
from app.chat.chat_manager import ChatManager
from app.chat.types import ChatArgs
from app.chat.memories.memory_manager import MemoryManager
from app.chat.models.conversation import ConversationState
from app.web.api import (
    process_query, 
    process_document, 
    run_async
)

logger = logging.getLogger(__name__)

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")

async def update_conversation_metadata(conversation_state: ConversationState) -> None:
    """Update conversation metadata with proper error handling."""
    try:
        # Get PDF name from metadata
        pdf_id = conversation_state.pdf_id
        pdf_name = conversation_state.metadata.get("pdf_name", "Document")

        # Get the first user message content for context
        user_messages = [msg for msg in conversation_state.messages if msg.type == "user"]
        first_message = user_messages[0] if user_messages else None

        # Create a descriptive title
        if first_message and first_message.content:
            # Truncate content to reasonable length for title
            content_preview = first_message.content[:50] + "..." if len(first_message.content) > 50 else first_message.content
            conversation_state.title = f"{pdf_name}: {content_preview}"
        else:
            conversation_state.title = f"{pdf_name} - Conversation {conversation_state.id[:8]}"

        # Update last_updated timestamp is handled automatically by the model

        # Save the updated conversation state
        memory_manager = MemoryManager()
        await memory_manager.save_conversation(conversation_state)

        logger.info(f"Updated metadata for conversation: {conversation_state.id}")
    except Exception as e:
        logger.error(f"Error updating conversation metadata for {conversation_state.id}: {str(e)}")
        raise

def ensure_research_mode(chat_args: ChatArgs):
    """
    Ensure research mode is properly initialized.
    
    Args:
        chat_args: Chat arguments including PDF ID and research mode
        
    Returns:
        Updated chat arguments with properly initialized research mode
    """
    logger.info(f"Initializing research mode: {chat_args.research_mode}")
    
    # If research mode is explicitly set, use it
    if chat_args.research_mode is not None:
        return chat_args
        
    # Otherwise default to single document mode
    chat_args.research_mode = ResearchMode.SINGLE
    return chat_args

@bp.route("/<pdf_id>", methods=["GET"])
@login_required
async def list_conversations(pdf_id):
    """List conversations for a PDF."""
    try:
        memory_manager = MemoryManager()
        conversations = await memory_manager.list_conversations(pdf_id=pdf_id)
        
        # Format for API response
        result = []
        for conv in conversations:
            # Skip conversations marked for deletion
            if conv.metadata.get("is_deleted", False):
                continue
                
            # Format the conversation data
            result.append({
                "id": conv.id,
                "title": conv.title,
                "pdf_id": conv.pdf_id,
                "last_updated": conv.updated_at.isoformat(),
                "message_count": len([m for m in conv.messages if m.type != "system"]),
                "metadata": conv.metadata
            })
            
        return jsonify({"conversations": result})
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<pdf_id>", methods=["POST"])
@login_required
async def create_conversation(pdf_id):
    """Create a new conversation for a PDF."""
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Create a new conversation
        conversation = await memory_manager.create_conversation(
            title=f"Conversation about {pdf_id}",
            pdf_id=pdf_id,
            metadata={
                "pdf_id": pdf_id,
                "research_mode": False,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Return the conversation data
        return jsonify({
            "conversation_id": conversation.id,
            "title": conversation.title,
            "pdf_id": conversation.pdf_id
        })
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/messages", methods=["POST"])
@login_required
async def create_message(conversation_id):
    """Create and process a new message."""
    try:
        data = request.json
        if not data or not data.get("message"):
            return jsonify({"error": "Message is required"}), 400
            
        user_message = data.get("message")
        stream_enabled = data.get("stream", False)
        
        # Initialize chat manager with the conversation
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            stream_enabled=stream_enabled
        )
        
        # Ensure research mode is properly set
        chat_args = ensure_research_mode(chat_args)
        
        # Initialize chat manager
        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()
        
        if not chat_manager.conversation_state:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Process the message
        if stream_enabled:
            return await process_streaming_message(chat_manager, user_message)
        else:
            result = await chat_manager.query(user_message)
            
            # Check for errors
            if "error" in result:
                return jsonify({"error": result["error"]}), 500
                
            # Update conversation metadata
            await update_conversation_metadata(chat_manager.conversation_state)
            
            return jsonify({
                "response": result["response"],
                "conversation_id": result["conversation_id"],
                "citations": result.get("citations", [])
            })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/messages", methods=["GET"])
@login_required
async def get_messages(conversation_id):
    """Retrieve all messages for a specific conversation."""
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Get the conversation
        conversation = await memory_manager.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Format messages for API response
        messages = []
        for msg in conversation.messages:
            # Skip system messages
            if msg.type == "system":
                continue
                
            messages.append({
                "id": msg.id,
                "role": msg.type,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "metadata": msg.metadata
            })
            
        return jsonify({"messages": messages})
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/research/<document_id>", methods=["POST", "DELETE"])
@login_required
async def manage_research_document(conversation_id, document_id):
    """Add or remove a document from research mode."""
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Get the conversation
        conversation = await memory_manager.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Get research documents list from metadata
        research_documents = conversation.metadata.get("research_documents", [])
        
        if request.method == "POST":
            # Add document to research mode if not already present
            if document_id not in research_documents:
                research_documents.append(document_id)
                
            # Enable research mode
            conversation.metadata["research_mode"] = True
            conversation.metadata["research_documents"] = research_documents
            
            # Save the updated conversation
            await memory_manager.save_conversation(conversation)
            
            return jsonify({
                "message": f"Added document {document_id} to research mode",
                "research_documents": research_documents
            })
        else:  # DELETE
            # Remove document from research mode
            if document_id in research_documents:
                research_documents.remove(document_id)
                
            # Update metadata
            conversation.metadata["research_documents"] = research_documents
            
            # If no documents left, disable research mode
            if not research_documents:
                conversation.metadata["research_mode"] = False
                
            # Save the updated conversation
            await memory_manager.save_conversation(conversation)
            
            return jsonify({
                "message": f"Removed document {document_id} from research mode",
                "research_documents": research_documents
            })
    except Exception as e:
        logger.error(f"Error managing research document: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/research", methods=["POST"])
@login_required
async def toggle_research_mode(conversation_id):
    """Toggle research mode for a conversation."""
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Get the conversation
        conversation = await memory_manager.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Toggle research mode
        current_mode = conversation.metadata.get("research_mode", False)
        conversation.metadata["research_mode"] = not current_mode
        
        # If enabling research mode, ensure we have a research_documents list
        if not current_mode:
            # Add the primary PDF to research documents if not present
            pdf_id = conversation.pdf_id
            if pdf_id:
                research_documents = conversation.metadata.get("research_documents", [])
                if pdf_id not in research_documents:
                    research_documents.append(pdf_id)
                conversation.metadata["research_documents"] = research_documents
                
        # Save the updated conversation
        await memory_manager.save_conversation(conversation)
        
        return jsonify({
            "research_mode": conversation.metadata.get("research_mode", False),
            "research_documents": conversation.metadata.get("research_documents", [])
        })
    except Exception as e:
        logger.error(f"Error toggling research mode: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/clear", methods=["POST"])
@login_required
async def clear_conversation_history(conversation_id):
    """Clear conversation history."""
    try:
        # Initialize chat manager with the conversation
        chat_args = ChatArgs(
            conversation_id=conversation_id
        )
        
        # Initialize chat manager
        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()
        
        if not chat_manager.conversation_state:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Clear the conversation
        research_mode = chat_manager.conversation_state.metadata.get("research_mode", False)
        pdf_id = chat_manager.conversation_state.pdf_id
            
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Create a new conversation state (keeping the same ID)
        conversation = await memory_manager.create_conversation(
            id=conversation_id,
            title=f"Conversation about {pdf_id}",
            pdf_id=pdf_id,
            metadata={
                "pdf_id": pdf_id,
                "research_mode": research_mode,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        return jsonify({"message": "Conversation cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>", methods=["DELETE"])
@login_required
async def delete_conversation(conversation_id):
    """Delete a conversation."""
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Get the conversation
        conversation = await memory_manager.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
            
        # Mark as deleted in metadata rather than actually deleting
        conversation.metadata["is_deleted"] = True
        
        # Save the updated conversation
        await memory_manager.save_conversation(conversation)
        
        return jsonify({"message": "Conversation deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/process/<pdf_id>", methods=["POST"])
@login_required
async def process_pdf(pdf_id):
    """Process a document using LangGraph architecture."""
    try:
        result = await process_document(pdf_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

async def process_streaming_message(chat_manager, user_message):
    """Process a streaming message with the chat manager."""
    # This is a placeholder for the streaming implementation
    # You would need to implement the actual streaming logic here
    response_generator = chat_manager.stream_query(user_message)
    
    # Return a streaming response
    return run_async(response_generator)
