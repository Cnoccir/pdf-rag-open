"""
Conversation views for PDF RAG application.
Provides API endpoints for conversation management with improved error handling.
"""

from flask import Blueprint, g, request, jsonify, Response, stream_with_context
import logging
import json
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, List

from app.chat.types import ResearchMode
from app.chat.models import ChatArgs
from app.chat.chat_manager import ChatManager
from app.web.async_wrapper import async_handler
from app.web.hooks import login_required
from app.web.db.models import Conversation, Message
from app.web.db import db
from app.web.api import (
    process_query,
    get_conversation_history,
    clear_conversation,
    list_conversations
)

logger = logging.getLogger(__name__)

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")

@bp.route("/<string:pdf_id>", methods=["GET"])
@login_required
def list_pdf_conversations(pdf_id):
    """List conversations for a PDF."""
    try:
        logger.info(f"Listing conversations for PDF {pdf_id}")

        # Query the database directly for better performance
        conversations = Conversation.query.filter_by(
            pdf_id=pdf_id,
            user_id=g.user.id,
            is_deleted=False
        ).order_by(Conversation.last_updated.desc()).all()

        # Format for API response
        result = []
        for conv in conversations:
            # Format the conversation
            result.append({
                "id": conv.id,
                "title": conv.title,
                "pdf_id": pdf_id,
                "last_updated": conv.last_updated.isoformat() if hasattr(conv.last_updated, "isoformat") else str(conv.last_updated),
                "message_count": len([m for m in conv.messages if m.role != "system"]),
                "metadata": conv.json_metadata,
                "messages": []  # Initialize empty messages array for frontend compatibility
            })

        # Return as array in "conversations" field for frontend
        return jsonify({"conversations": result})

    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<string:pdf_id>", methods=["POST"])
@login_required
def create_conversation(pdf_id):
    """Create a new conversation for a PDF."""
    try:
        logger.info(f"Creating new conversation for PDF {pdf_id}")

        from app.chat.memories.memory_manager import MemoryManager
        memory_manager = MemoryManager()

        # Create a new conversation
        conversation_state = memory_manager.create_conversation(
            title=f"Conversation about {pdf_id}",
            pdf_id=pdf_id,
            metadata={
                "user_id": g.user.id,
                "pdf_id": pdf_id,
                "created_at": datetime.now().isoformat(),
                "research_mode": {
                    "active": False,
                    "pdf_ids": [pdf_id],
                    "document_names": {}
                }
            }
        )

        # Create database entry
        db_conversation = Conversation(
            id=conversation_state.conversation_id,
            title=conversation_state.title,
            pdf_id=pdf_id,
            user_id=g.user.id,
            json_metadata=conversation_state.metadata
        )

        # Add system message
        system_message = Message(
            conversation_id=conversation_state.conversation_id,
            role="system",
            content="You are an AI assistant specialized in answering questions about documents."
        )

        # Save to database
        db.session.add(db_conversation)
        db.session.add(system_message)
        db.session.commit()

        # Return the new conversation
        response_data = {
            "id": conversation_state.conversation_id,
            "title": conversation_state.title,
            "pdf_id": pdf_id,
            "messages": [],  # Initialize as empty array for frontend
            "metadata": conversation_state.metadata
        }

        logger.info(f"Created conversation {conversation_state.conversation_id} for PDF {pdf_id}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        logger.error(traceback.format_exc())
        db.session.rollback()
        return jsonify({"error": f"Failed to create conversation: {str(e)}"}), 500

@bp.route("/<string:conversation_id>/messages", methods=["POST"])
@login_required
def create_message(conversation_id):
    """Create and process a new message."""
    try:
        data = request.json
        if not data:
            logger.warning(f"No JSON data in request for conversation {conversation_id}")
            return jsonify({"error": "Message content is required"}), 400

        # Support both "message" and "input" fields for compatibility
        user_message = data.get("message") or data.get("input")
        if not user_message:
            return jsonify({"error": "Message content is required"}), 400

        # Get conversation to validate and retrieve PDF ID
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Validate user owns the conversation
        if conversation.user_id != g.user.id:
            return jsonify({"error": "Unauthorized access to conversation"}), 403

        # Get streaming and research mode preferences
        stream_enabled = data.get("useStreaming", data.get("stream", False))
        use_research = data.get("useResearch", False)
        active_docs = data.get("activeDocs", [])

        # Update conversation metadata for research mode if active_docs provided
        if use_research and active_docs and len(active_docs) > 1:
            if not conversation.json_metadata:
                conversation.json_metadata = {}
            if "research_mode" not in conversation.json_metadata:
                conversation.json_metadata["research_mode"] = {}
            conversation.json_metadata["research_mode"]["active"] = True
            conversation.json_metadata["research_mode"]["pdf_ids"] = active_docs
            db.session.add(conversation)
            db.session.commit()

        # Add the user message to the database
        user_db_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_message
        )
        db.session.add(user_db_message)
        db.session.commit()

        logger.info(f"Processing message for conversation {conversation_id}, research mode: {use_research}")

        # If streaming is enabled, delegate to the streaming response
        if stream_enabled:
            return stream_chat_response(
                conversation_id=conversation_id,
                user_message=user_message,
                research_mode_str="research" if use_research else "single",
                pdf_id=str(conversation.pdf_id) if conversation.pdf_id else None,
                active_docs=active_docs
            )
        else:
            # Process non-streaming query
            research_mode_str = "research" if use_research else "single"
            pdf_ids = active_docs if use_research and active_docs else [str(conversation.pdf_id)] if conversation.pdf_id else None

            result = process_query(
                query=user_message,
                conversation_id=conversation_id,
                pdf_id=str(conversation.pdf_id) if conversation.pdf_id else None,
                research_mode=research_mode_str,
                stream=False,
                pdf_ids=pdf_ids
            )

            if "error" in result:
                logger.error(f"Error in message processing: {result['error']}")
                return jsonify({"error": result["error"]}), 500

            # Add assistant message to database if not already handled by ChatManager
            if not db.session.query(Message).filter_by(
                conversation_id=conversation_id,
                role="assistant",
                content=result["response"]
            ).first():
                assistant_db_message = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result["response"],
                    meta_json=json.dumps({
                        "citations": result.get("citations", []),
                        "research_mode": conversation.json_metadata.get("research_mode") if conversation.json_metadata else None
                    })
                )
                db.session.add(assistant_db_message)
                db.session.commit()

            return jsonify({
                "response": result["response"],
                "conversation_id": result["conversation_id"],
                "citations": result.get("citations", []),
                "research_mode": conversation.json_metadata.get("research_mode") if conversation.json_metadata else None
            })

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<string:conversation_id>/messages", methods=["GET"])
@login_required
def get_messages(conversation_id):
    """Retrieve all messages for a conversation."""
    try:
        # Get conversation history
        result = get_conversation_history(conversation_id)

        if "error" in result:
            return jsonify({"error": result["error"]}), 404

        # Return messages
        return jsonify({"messages": result["messages"]})

    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<string:conversation_id>/research", methods=["POST"])
@login_required
def update_research_mode(conversation_id):
    """Update research mode for a conversation."""
    try:
        data = request.json or {}

        # Check if activating or deactivating
        is_active = data.get("active", False)

        # Get the conversation
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Validate user owns the conversation
        if conversation.user_id != g.user.id:
            return jsonify({"error": "Unauthorized access to conversation"}), 403

        # Initialize metadata if needed
        if not conversation.json_metadata:
            conversation.json_metadata = {}

        # Ensure research mode entry exists
        if "research_mode" not in conversation.json_metadata:
            conversation.json_metadata["research_mode"] = {
                "active": False,
                "pdf_ids": [str(conversation.pdf_id)] if conversation.pdf_id else [],
                "document_names": {}
            }

        if is_active:
            # Activating research mode - need PDF IDs
            pdf_ids = data.get("pdf_ids", [])

            # Validate we have at least 2 PDFs for research mode
            if not pdf_ids or len(pdf_ids) < 2:
                return jsonify({"error": "At least two PDF IDs are required for research mode"}), 400

            # Get document names if provided
            document_names = data.get("document_names", {})

            # Update metadata
            conversation.json_metadata["research_mode"]["active"] = True
            conversation.json_metadata["research_mode"]["pdf_ids"] = pdf_ids

            # Update document names if provided
            if document_names:
                conversation.json_metadata["research_mode"]["document_names"] = document_names
        else:
            # Deactivating research mode
            conversation.json_metadata["research_mode"]["active"] = False

        # Save changes
        db.session.add(conversation)
        db.session.commit()

        logger.info(f"{'Activated' if is_active else 'Deactivated'} research mode for conversation {conversation_id}")

        # Return updated research mode status
        return jsonify({
            "status": "success",
            "research_mode": conversation.json_metadata["research_mode"]
        })

    except Exception as e:
        logger.error(f"Error updating research mode: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@bp.route("/<string:conversation_id>/clear", methods=["POST"])
@login_required
def clear_conversation_history(conversation_id):
    """Clear conversation history."""
    try:
        # Get the conversation
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Validate user owns the conversation
        if conversation.user_id != g.user.id:
            return jsonify({"error": "Unauthorized access to conversation"}), 403

        # Clear conversation history
        result = clear_conversation(conversation_id)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify({"message": "Conversation cleared successfully"})

    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<string:conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation(conversation_id):
    """Delete a conversation."""
    try:
        # Get the conversation
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Validate user owns the conversation
        if conversation.user_id != g.user.id:
            return jsonify({"error": "Unauthorized access to conversation"}), 403

        # Mark as deleted
        conversation.is_deleted = True

        # Update metadata
        if not conversation.json_metadata:
            conversation.json_metadata = {}

        conversation.json_metadata["is_deleted"] = True
        conversation.json_metadata["deleted_at"] = datetime.now().isoformat()

        # Save changes
        db.session.add(conversation)
        db.session.commit()

        return jsonify({"message": "Conversation deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

def stream_chat_response(conversation_id, user_message, research_mode_str, pdf_id=None, active_docs=None):
    """Stream chat response for a message."""
    def generate():
        # Initialize chat manager
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            pdf_id=pdf_id,
            research_mode=ResearchMode.RESEARCH if research_mode_str == "research" else ResearchMode.SINGLE,
            stream_enabled=True
        )

        chat_manager = ChatManager(chat_args)

        try:
            # Initialize with conversation history
            chat_manager.initialize()

            # Set initial message
            yield json.dumps({
                "type": "status",
                "status": "processing",
                "message": "Processing your query..."
            }) + "\n"

            # Stream response
            pdf_ids_to_use = active_docs if research_mode_str == "research" and active_docs else None
            response_chunks = []

            # Use the streamable ChatManager.stream_query method
            for chunk in chat_manager.stream_query(user_message, pdf_ids_to_use):
                if "error" in chunk:
                    yield json.dumps({
                        "type": "error",
                        "error": chunk["error"]
                    }) + "\n"
                    return

                if "status" in chunk:
                    if chunk["status"] == "processing":
                        yield json.dumps({
                            "type": "status",
                            "status": "processing",
                            "message": chunk["message"]
                        }) + "\n"
                    elif chunk["status"] == "complete":
                        # The conversation is already saved by the ChatManager
                        yield json.dumps({
                            "type": "end",
                            "message": chunk["response"],
                            "conversation_id": conversation_id,
                            "citations": chunk.get("citations", [])
                        }) + "\n"
                        return

                if "type" in chunk and chunk["type"] == "stream":
                    # Add to accumulated response for database
                    response_chunks.append(chunk["chunk"])

                    # Yield chunk to client
                    yield json.dumps({
                        "type": "stream",
                        "chunk": chunk["chunk"],
                        "index": chunk.get("index", 0),
                        "is_complete": chunk.get("is_complete", False)
                    }) + "\n"

            # If we reached here without a complete message
            # save the response to the database as a fallback
            try:
                full_response = "".join(response_chunks)
                if full_response:
                    # Check if this response is already in the database
                    existing = db.session.query(Message).filter_by(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_response
                    ).first()

                    if not existing:
                        # Add assistant message
                        assistant_message = Message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=full_response,
                            meta_json=json.dumps({
                                "citations": []
                            })
                        )
                        db.session.add(assistant_message)
                        db.session.commit()

                # Send final message
                yield json.dumps({
                    "type": "end",
                    "message": full_response,
                    "conversation_id": conversation_id,
                    "citations": []
                }) + "\n"

            except Exception as db_error:
                logger.error(f"Error saving streamed response: {str(db_error)}")

        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }) + "\n"

    # Return streaming response
    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson'
    )
