"""
Streaming API views for PDF RAG chat application.
Supports the streaming interface for the frontend.
"""

from flask import Blueprint, g, request, jsonify, Response, stream_with_context
import json
import logging
import asyncio
from datetime import datetime

from app.web.hooks import login_required
from app.web.db.models import Pdf
from app.web.db import db
from app.chat.chat_manager import ChatManager
from app.chat.types import ChatArgs, ResearchMode

logger = logging.getLogger(__name__)

bp = Blueprint("stream", __name__, url_prefix="/api/stream")

@bp.route("/<string:conversation_id>/chat", methods=["POST"])
@login_required
def stream_chat(conversation_id):
    """
    Stream chat responses for a conversation.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        query = request.json.get("input")
        if not query or not query.strip():
            return jsonify({"error": "Message content is required"}), 400
            
        # Get optional PDF ID
        pdf_id = request.json.get("pdf_id")
        
        # Get research mode if specified
        research_mode_str = request.json.get("research_mode", "single")
        research_mode = ResearchMode.RESEARCH if research_mode_str.lower() == "research" else ResearchMode.SINGLE
            
        # Create a streaming response
        def generate():
            # Create event loop in this thread/context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Initialize chat manager with streaming enabled
                chat_args = ChatArgs(
                    conversation_id=conversation_id,
                    pdf_id=pdf_id,
                    research_mode=research_mode,
                    stream_enabled=True
                )
                
                chat_manager = ChatManager(chat_args)
                
                # Initialize with conversation history if available
                loop.run_until_complete(chat_manager.initialize())
                
                # Set initial message to indicate processing
                yield json.dumps({
                    "type": "status",
                    "status": "processing",
                    "message": "Processing your query..."
                }) + "\n"
                
                # Create an async generator to stream responses
                async def stream_response():
                    async for chunk in chat_manager.stream_query(query):
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
                            return
                            
                        if "chunk" in chunk:
                            # Streaming chunk
                            yield {
                                "type": "stream",
                                "chunk": chunk["chunk"],
                                "index": chunk.get("index", 0),
                                "is_complete": chunk.get("is_complete", False)
                            }
                
                # Execute the streaming generator and yield results
                stream_gen = stream_response()
                while True:
                    try:
                        chunk = loop.run_until_complete(stream_gen.__anext__())
                        yield json.dumps(chunk) + "\n"
                        
                        # If we got the final message, we're done
                        if chunk.get("type") == "end" or chunk.get("type") == "error":
                            break
                            
                    except StopAsyncIteration:
                        break
                        
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                yield json.dumps({
                    "type": "error",
                    "error": str(e)
                }) + "\n"
            finally:
                # Clean up
                try:
                    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
                    for task in tasks:
                        task.cancel()
                        
                    if tasks:
                        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                    loop.close()
                except Exception as e:
                    logger.error(f"Error cleaning up: {str(e)}")
        
        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='application/x-ndjson'
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
