"""
Final fixed version of conversation views with proper client compatibility.
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

logger = logging.getLogger(__name__)

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")

# Memory manager instance to be used across routes
memory_manager = MemoryManager()

@bp.route("/<string:pdf_id>", methods=["GET"])
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
            if conv.metadata and conv.metadata.get("is_deleted", False):
                continue

            # Format the conversation data
            result.append({
                "id": conv.id,
                "title": conv.title,
                "pdf_id": conv.pdf_id,
                "last_updated": conv.updated_at.isoformat() if hasattr(conv.updated_at, "isoformat") else str(conv.updated_at),
                "message_count": len([m for m in conv.messages if m.type != "system"]),
                "metadata": conv.metadata
            })

        # Log response for debugging
        logger.info(f"Returning {len(result)} conversations for PDF {pdf_id}")

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

        # Ensure PDF ID is properly formatted as string
        pdf_id_str = str(pdf_id)

        if not pdf_id_str:
            return jsonify({"error": "PDF ID is required"}), 400

        # Create a new conversation asynchronously with minimal structure
        # This reduces the chance of errors
        conversation = run_async(memory_manager.create_conversation(
            title=f"Conversation about {pdf_id_str}",
            pdf_id=pdf_id_str,
            metadata={
                "pdf_id": pdf_id_str,
                "created_at": datetime.utcnow().isoformat()
            }
        ))

        if not conversation:
            logger.error(f"Failed to create conversation for PDF {pdf_id}")
            return jsonify({"error": "Failed to create conversation"}), 500

        # Return the conversation data in the expected format for the client
        response_data = {
            "id": conversation.id,  # This is what the client expects
            "title": conversation.title or f"Conversation about {pdf_id_str}",
            "pdf_id": pdf_id_str
        }

        logger.info(f"Created conversation {conversation.id} for PDF {pdf_id}")
        return jsonify(response_data)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error(f"Error creating conversation: {str(e)}")
        logger.error("Exception traceback:")
        traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in traceback_lines:
            logger.error(line.rstrip())
        return jsonify({"error": str(e)}), 500

@bp.route("/<string:conversation_id>/messages", methods=["POST"])
@login_required
async def create_message(conversation_id):
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

        stream_enabled = data.get("useStreaming", data.get("stream", False))

        # Get research mode settings from the request
        use_research = data.get("useResearch", False)
        active_docs = data.get("activeDocs", [])

        logger.info(f"Processing message for conversation {conversation_id}, research mode: {use_research}")

        # Initialize chat manager with the conversation
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            stream_enabled=stream_enabled,
            research_mode=ResearchMode.RESEARCH if use_research else ResearchMode.SINGLE
        )

        # Initialize chat manager
        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()

        if not chat_manager.conversation_state:
            logger.error(f"Failed to initialize conversation state for ID {conversation_id}")
            return jsonify({"error": "Conversation not found"}), 404

        # Process the message
        if stream_enabled:
            return await process_streaming_message(chat_manager, user_message)
        else:
            result = await chat_manager.query(user_message)

            # Check for errors
            if "error" in result:
                logger.error(f"Error in message processing: {result['error']}")
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
def get_messages(conversation_id):
    """Retrieve all messages for a specific conversation."""
    try:
        # Get the conversation
        logger.info(f"Fetching messages for conversation {conversation_id}")
        conversation = run_async(memory_manager.get_conversation(conversation_id))

        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Format messages for API response
        messages = []
        for msg in conversation.messages:
            # Skip system messages
            if msg.type == "system":
                continue

            messages.append({
                "id": str(uuid.uuid4()),  # Generate ID if not present
                "role": msg.type,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if hasattr(msg.created_at, 'isoformat') else str(msg.created_at),
                "metadata": msg.metadata or {}
            })

        return jsonify({"messages": messages})
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/research/activate", methods=["POST"])
@login_required
def activate_research_mode(conversation_id):
    """Activate research mode for a conversation."""
    try:
        data = request.json or {}
        pdf_ids = data.get("pdf_ids", [])

        if not pdf_ids or len(pdf_ids) < 2:
            return jsonify({"error": "At least two PDF IDs are required for research mode"}), 400

        # Ensure all PDF IDs are strings
        pdf_ids = [str(pdf_id) for pdf_id in pdf_ids]

        logger.info(f"Activating research mode for conversation {conversation_id} with PDFs: {pdf_ids}")

        # Initialize chat manager with the conversation
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            research_mode=ResearchMode.RESEARCH
        )

        # Initialize chat manager
        chat_manager = run_async(async_init_chat_manager(chat_args))

        if not chat_manager or not chat_manager.conversation_state:
            return jsonify({"error": "Conversation not found"}), 404

        # Update metadata for research mode
        if not chat_manager.conversation_state.metadata:
            chat_manager.conversation_state.metadata = {}

        chat_manager.conversation_state.metadata["research_mode"] = {
            "active": True,
            "pdf_ids": pdf_ids,
            "document_names": {}
        }

        # Save updated conversation
        run_async(memory_manager.save_conversation(chat_manager.conversation_state))

        return jsonify({
            "status": "success",
            "research_mode": True,
            "pdf_ids": pdf_ids
        })
    except Exception as e:
        logger.error(f"Error activating research mode: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Make sure we handle both POST and GET for deactivate
@bp.route("/<conversation_id>/research/deactivate", methods=["POST", "GET"])
@login_required
def deactivate_research_mode(conversation_id):
    """Deactivate research mode for a conversation."""
    try:
        logger.info(f"Deactivating research mode for conversation {conversation_id}")

        # Initialize chat manager with the conversation
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            research_mode=ResearchMode.SINGLE
        )

        # Initialize chat manager
        chat_manager = run_async(async_init_chat_manager(chat_args))

        if not chat_manager or not chat_manager.conversation_state:
            return jsonify({"error": "Conversation not found"}), 404

        # Update metadata to disable research mode
        if not chat_manager.conversation_state.metadata:
            chat_manager.conversation_state.metadata = {}

        chat_manager.conversation_state.metadata["research_mode"] = {
            "active": False,
            "pdf_ids": [],
            "document_names": {}
        }

        # Save updated conversation
        run_async(memory_manager.save_conversation(chat_manager.conversation_state))

        return jsonify({
            "status": "success",
            "research_mode": False
        })
    except Exception as e:
        logger.error(f"Error deactivating research mode: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>/clear", methods=["POST"])
@login_required
def clear_conversation_history(conversation_id):
    """Clear conversation history."""
    try:
        logger.info(f"Clearing conversation {conversation_id}")

        # Initialize chat manager with the conversation
        chat_args = ChatArgs(
            conversation_id=conversation_id
        )

        # Initialize chat manager
        chat_manager = run_async(async_init_chat_manager(chat_args))

        if not chat_manager or not chat_manager.conversation_state:
            return jsonify({"error": "Conversation not found"}), 404

        # Get metadata for new conversation
        research_mode = chat_manager.conversation_state.metadata.get("research_mode", {"active": False})
        pdf_id = chat_manager.conversation_state.pdf_id

        # Ensure PDF ID is a string
        pdf_id_str = str(pdf_id) if pdf_id else None

        # Create a new conversation state (keeping the same ID)
        conversation = run_async(memory_manager.create_conversation(
            id=conversation_id,
            title=f"Conversation about {pdf_id_str}",
            pdf_id=pdf_id_str,
            metadata={
                "pdf_id": pdf_id_str,
                "research_mode": research_mode,
                "created_at": datetime.utcnow().isoformat()
            }
        ))

        return jsonify({"message": "Conversation cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route("/<conversation_id>", methods=["DELETE"])
@login_required
def delete_conversation(conversation_id):
    """Delete a conversation."""
    try:
        logger.info(f"Deleting conversation {conversation_id}")

        # Get the conversation
        conversation = run_async(memory_manager.get_conversation(conversation_id))

        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        # Mark as deleted in metadata rather than actually deleting
        if not conversation.metadata:
            conversation.metadata = {}
        conversation.metadata["is_deleted"] = True

        # Save the updated conversation
        run_async(memory_manager.save_conversation(conversation))

        return jsonify({"message": "Conversation deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def stream_chat_response(conversation_id, user_message, research_mode_str, pdf_id=None):
    """Stream chat response for a message."""
    try:
        # Set up streaming response
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
                chat_manager = loop.run_until_complete(async_init_chat_manager(chat_args))

                # Set initial message to indicate processing
                yield json.dumps({
                    "type": "status",
                    "status": "processing",
                    "message": "Processing your query..."
                }) + "\n"

                # Stream the response
                stream_gen = chat_manager.stream_query(user_message)
                while True:
                    try:
                        chunk = loop.run_until_complete(anext_with_timeout(stream_gen))

                        if "error" in chunk:
                            yield json.dumps({
                                "type": "error",
                                "error": chunk["error"]
                            }) + "\n"
                            break

                        if "status" in chunk and chunk["status"] == "complete":
                            # Final message with full content
                            yield json.dumps({
                                "type": "end",
                                "message": chunk.get("response", ""),
                                "conversation_id": conversation_id,
                                "citations": chunk.get("citations", [])
                            }) + "\n"
                            break

                        if "chunk" in chunk:
                            # Streaming chunk
                            yield json.dumps({
                                "type": "stream",
                                "chunk": chunk["chunk"],
                                "index": chunk.get("index", 0),
                                "is_complete": chunk.get("is_complete", False)
                            }) + "\n"
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
                logger.error(traceback.format_exc())
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
    except Exception as e:
        logger.error(f"Error setting up streaming: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Helper functions

async def anext_with_timeout(agen, timeout=10.0):
    """Get next item from async generator with timeout."""
    try:
        return await asyncio.wait_for(agen.__anext__(), timeout=timeout)
    except StopAsyncIteration:
        raise
    except asyncio.TimeoutError:
        logger.warning(f"Timeout waiting for next chunk from stream")
        raise

async def async_init_chat_manager(chat_args: ChatArgs):
    """Initialize chat manager and return it."""
    try:
        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()
        return chat_manager
    except Exception as e:
        logger.error(f"Error initializing chat manager: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return None

async def process_query(
    query: str,
    conversation_id: str,
    pdf_id: Optional[str] = None,
    research_mode: str = "single"
):
    """Process a query and return the result."""
    try:
        # Initialize chat manager
        chat_args = ChatArgs(
            conversation_id=conversation_id,
            pdf_id=pdf_id,
            research_mode=ResearchMode.RESEARCH if research_mode == "research" else ResearchMode.SINGLE,
            stream_enabled=False
        )

        chat_manager = ChatManager(chat_args)
        await chat_manager.initialize()

        # Process the query
        result = await chat_manager.query(query)

        # Update conversation metadata
        if chat_manager.conversation_state:
            await memory_manager.save_conversation(chat_manager.conversation_state)

        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return {"error": str(e)}
