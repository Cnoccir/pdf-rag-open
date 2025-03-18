"""
Conversation views for PDF RAG application with LangGraph integration.
Handles conversation management, message streaming and research mode.
"""

from flask import Blueprint, g, request, jsonify, Response, stream_with_context
import logging
import json
import asyncio
from datetime import datetime
import uuid
import traceback
import sys
from typing import List, Dict, Any, Optional

from app.chat.types import ResearchMode, ChatArgs
from app.chat.chat_manager import ChatManager
from app.chat.memories.memory_manager import MemoryManager
from app.web.async_wrapper import async_handler, run_async
from app.web.hooks import login_required
from app.web.db.models import Conversation, Message
from app.web.db import db

logger = logging.getLogger(__name__)

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")

# Memory manager instance to be used across routes
memory_manager = MemoryManager()

@bp.route("/<string:pdf_id>", methods=["GET"])
@login_required
def list_conversations(pdf_id):
    """List conversations for a PDF."""
    try:
        logger.info(f"Listing conversations for PDF {pdf_id}")

        # Query the database directly for better performance
        conversations = Conversation.query.filter_by(
            pdf_id=pdf_id,
            user_id=g.user.id,
            is_deleted=False
        ).order_by(Conversation.last_updated.desc()).all()

        # Format for API response - critical for frontend compatibility
        result = []
        for conv in conversations:
            # Format the conversation for the frontend
            result.append({
                "id": conv.id,
                "title": conv.title,
                "pdf_id": pdf_id,
                "last_updated": conv.last_updated.isoformat() if hasattr(conv.last_updated, "isoformat") else str(conv.last_updated),
                "message_count": len([m for m in conv.messages if m.role != "system"]),
                "metadata": conv.json_metadata,
                "messages": []  # Initialize empty messages array for frontend compatibility
            })

        # Return as array in "conversations" field for frontend compatibility
        return jsonify({"conversations": result})
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<pdf_id>", methods=["POST"])
@login_required
def create_conversation(pdf_id):
    """Create a new conversation for a PDF."""
    try:
        logger.info(f"Creating new conversation for PDF {pdf_id}")

        # Ensure PDF ID is properly formatted
        pdf_id_str = str(pdf_id)

        # Create a new conversation in the database
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=f"Conversation about {pdf_id_str}",
            pdf_id=pdf_id,
            user_id=g.user.id,
            json_metadata={
                "pdf_id": pdf_id_str,
                "created_at": datetime.utcnow().isoformat(),
                "research_mode": {
                    "active": False,
                    "pdf_ids": [pdf_id_str],
                    "document_names": {}
                }
            }
        )

        # Save to database
        db.session.add(conversation)
        db.session.commit()

        # Add system message
        system_message = Message(
            conversation_id=conversation.id,
            role="system",
            content="You are an AI assistant specialized in answering questions about documents."
        )

        db.session.add(system_message)
        db.session.commit()

        # Return the new conversation in the format frontend expects
        response_data = {
            "id": conversation.id,
            "title": conversation.title,
            "pdf_id": pdf_id_str,
            "messages": [],  # CRITICAL: Initialize as empty array for frontend
            "metadata": conversation.json_metadata
        }

        logger.info(f"Created conversation {conversation.id} for PDF {pdf_id}")
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

        # Get streaming preference
        stream_enabled = data.get("useStreaming", data.get("stream", False))

        # Get research mode settings
        use_research = data.get("useResearch", False)

        # Get active document IDs for research mode
        active_docs = data.get("activeDocs", [])

        # If active_docs is provided, update conversation metadata
        if use_research and active_docs and len(active_docs) > 1:
            if not conversation.json_metadata:
                conversation.json_metadata = {}

            # Ensure research_mode key exists
            if "research_mode" not in conversation.json_metadata:
                conversation.json_metadata["research_mode"] = {}

            # Update research mode data
            conversation.json_metadata["research_mode"]["active"] = True
            conversation.json_metadata["research_mode"]["pdf_ids"] = active_docs

            # Save changes to conversation
            db.session.add(conversation)
            db.session.commit()

        # Add user message to database first
        user_db_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_message
        )
        db.session.add(user_db_message)
        db.session.commit()

        logger.info(f"Processing message for conversation {conversation_id}, research mode: {use_research}")

        # Initialize chat manager with correct parameters
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            pdf_id=str(conversation.pdf_id) if conversation.pdf_id else None,
            stream_enabled=stream_enabled,
            research_mode=ResearchMode.RESEARCH if use_research else ResearchMode.SINGLE
        )

        # Process message based on streaming preference
        if stream_enabled:
            # Return streaming response
            return stream_chat_response(
                conversation_id=conversation_id,
                user_message=user_message,
                research_mode_str="research" if use_research else "single",
                pdf_id=str(conversation.pdf_id) if conversation.pdf_id else None,
                active_docs=active_docs
            )
        else:
            # Process message in current thread with async handler
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Initialize chat manager
                chat_manager = ChatManager(chat_args)
                loop.run_until_complete(chat_manager.initialize())

                # Process the query
                result = loop.run_until_complete(chat_manager.query(user_message))

                # Check for errors
                if "error" in result:
                    logger.error(f"Error in message processing: {result['error']}")
                    return jsonify({"error": result["error"]}), 500

                # Add assistant message to database
                assistant_db_message = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result["response"],
                    meta_json={
                        "citations": result.get("citations", []),
                        "research_mode": conversation.json_metadata.get("research_mode") if conversation.json_metadata else None
                    }
                )
                db.session.add(assistant_db_message)
                db.session.commit()

                # Return response
                return jsonify({
                    "response": result["response"],
                    "conversation_id": result["conversation_id"],
                    "citations": result.get("citations", []),
                    "research_mode": conversation.json_metadata.get("research_mode") if conversation.json_metadata else None
                })
            finally:
                # Clean up tasks and close loop
                try:
                    tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    if tasks:
                        for task in tasks:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                    loop.close()
                except Exception as e:
                    logger.error(f"Error cleaning up async tasks: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/messages", methods=["GET"])
@login_required
def get_messages(conversation_id):
    """Retrieve all messages for a specific conversation."""
    try:
        # Get the conversation
        logger.info(f"Fetching messages for conversation {conversation_id}")
        conversation = Conversation.query.filter_by(id=conversation_id).first()

        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Validate user owns the conversation
        if conversation.user_id != g.user.id:
            return jsonify({"error": "Unauthorized access to conversation"}), 403

        # Format messages for API response
        messages = []
        for msg in conversation.messages:
            # Skip system messages - frontend doesn't need them
            if msg.role == "system":
                continue

            # Create message object
            message_obj = {
                "id": str(uuid.uuid4()),  # Generate ID for frontend tracking
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_on.isoformat() if hasattr(msg.created_on, 'isoformat') else str(msg.created_on),
                "metadata": msg.msg_metadata or {}
            }

            messages.append(message_obj)

        return jsonify({"messages": messages})
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/research", methods=["POST"])
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
            # Keep PDF IDs for reference

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
        logger.error(traceback.format_exc())
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/clear", methods=["POST"])
@login_required
def clear_conversation_history(conversation_id):
    """Clear conversation history."""
    try:
        logger.info(f"Clearing conversation {conversation_id}")

        # Get the conversation
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Validate user owns the conversation
        if conversation.user_id != g.user.id:
            return jsonify({"error": "Unauthorized access to conversation"}), 403

        # Get the system messages to keep
        system_messages = [msg for msg in conversation.messages if msg.role == "system"]

        # Delete all other messages
        for msg in conversation.messages:
            if msg.role != "system":
                db.session.delete(msg)

        # If no system message, add one
        if not system_messages:
            system_message = Message(
                conversation_id=conversation_id,
                role="system",
                content="You are an AI assistant specialized in answering questions about documents."
            )
            db.session.add(system_message)

        # Update conversation
        conversation.last_updated = datetime.utcnow()
        db.session.add(conversation)
        db.session.commit()

        return jsonify({"message": "Conversation cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation(conversation_id):
    """Delete a conversation."""
    try:
        logger.info(f"Deleting conversation {conversation_id}")

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
        conversation.json_metadata["deleted_at"] = datetime.utcnow().isoformat()

        # Save changes
        db.session.add(conversation)
        db.session.commit()

        return jsonify({"message": "Conversation deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
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

        # Create a new loop for this request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Initialize chat manager
            chat_manager = ChatManager(chat_args)
            loop.run_until_complete(chat_manager.initialize())

            # Set initial message to indicate processing
            yield json.dumps({
                "type": "status",
                "status": "processing",
                "message": "Processing your query..."
            }) + "\n"

            # Create async generator for streaming
            async def stream_response():
                try:
                    async for chunk in chat_manager.stream_query(user_message):
                        if "error" in chunk:
                            yield {
                                "type": "error",
                                "error": chunk["error"]
                            }
                            return

                        if "status" in chunk and chunk["status"] == "complete":
                            # Final message with full content
                            yield {
                                "type": "end",
                                "message": chunk.get("response", ""),
                                "conversation_id": conversation_id,
                                "citations": chunk.get("citations", [])
                            }

                            # Save the complete message to database
                            try:
                                assistant_message = Message(
                                    conversation_id=conversation_id,
                                    role="assistant",
                                    content=chunk.get("response", ""),
                                    meta_json={
                                        "citations": chunk.get("citations", [])
                                    }
                                )
                                db.session.add(assistant_message)
                                db.session.commit()
                            except Exception as db_error:
                                logger.error(f"Error saving assistant message: {str(db_error)}")

                            return

                        if "chunk" in chunk:
                            # Streaming chunk
                            yield {
                                "type": "stream",
                                "chunk": chunk["chunk"],
                                "index": chunk.get("index", 0),
                                "is_complete": chunk.get("is_complete", False)
                            }
                except Exception as e:
                    logger.error(f"Error in stream_response: {str(e)}", exc_info=True)
                    yield {
                        "type": "error",
                        "error": str(e)
                    }

            # Process the streaming generator
            stream_gen = stream_response()
            while True:
                try:
                    chunk = loop.run_until_complete(anext_async_generator(stream_gen))
                    yield json.dumps(chunk) + "\n"

                    # If we got the final message, we're done
                    if chunk.get("type") == "end" or chunk.get("type") == "error":
                        break

                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield json.dumps({
                        "type": "error",
                        "error": "Stream timeout"
                    }) + "\n"
                    break

        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }) + "\n"
        finally:
            # Clean up tasks
            try:
                tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.close()
            except Exception as e:
                logger.error(f"Error cleaning up async tasks: {str(e)}")

    # Return streaming response
    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson'
    )

async def anext_async_generator(agen, timeout=10.0):
    """Get next item from async generator with timeout."""
    try:
        return await asyncio.wait_for(agen.__anext__(), timeout=timeout)
    except StopAsyncIteration:
        raise
    except asyncio.TimeoutError:
        logger.warning(f"Timeout waiting for next chunk from stream")
        raise
