"""
Simplified ChatManager for the PDF RAG system.
Uses LangGraph for workflow and improves async/sync handling.
"""

import logging
from typing import Dict, List, Any, Optional, Generator, AsyncGenerator
from datetime import datetime
import uuid
import time

from app.chat.models import ChatArgs
from app.chat.types import ResearchMode
from app.chat.langgraph.state import GraphState, QueryState, MessageType, ConversationState
from app.chat.langgraph.graph import create_query_graph, create_research_graph, create_document_graph
from app.chat.memories.memory_manager import MemoryManager
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

class ChatManager:
    """
    ChatManager implementing LangGraph-based architecture for PDF RAG system.
    Simplified implementation with improved async/sync handling.
    """

    # System prompts
    BASE_SYSTEM_PROMPT = """You are an AI assistant specialized in providing information about technical documentation.
    Your goal is to give accurate, clear answers using only the provided context information. Be detailed and precise.
    If you don't know an answer or if the information isn't in the context, simply acknowledge that you don't have that information
    rather than making something up."""

    RESEARCH_MODE_PROMPT = """You are now in RESEARCH MODE. This means you should:
    1. Analyze relationships between concepts across multiple documents
    2. Identify shared patterns, inconsistencies, or complementary information
    3. Synthesize insights that wouldn't be apparent from a single document
    4. Be explicit about which document you're referencing when providing information"""

    def __init__(self, chat_args: ChatArgs):
        """
        Initialize ChatManager with the provided arguments.

        Args:
            chat_args: Chat configuration
        """
        self.chat_args = chat_args
        self.conversation_id = getattr(chat_args, "conversation_id", None)
        self.pdf_id = getattr(chat_args, "pdf_id", None)
        self.research_mode = getattr(chat_args, "research_mode", ResearchMode.SINGLE)
        self.stream_enabled = getattr(chat_args, "stream_enabled", False)
        self.stream_chunk_size = getattr(chat_args, "stream_chunk_size", 20)

        # Initialize conversation state
        self.conversation_state = None

        # Initialize memory manager
        self.memory_manager = MemoryManager()

        # Create LangGraph instances - don't compile until needed
        self._query_graph = None
        self._research_graph = None
        self._document_graph = None

        # Initialize vector store connection status
        self.vector_store_ready = False

        logger.info(f"ChatManager initialized with PDF ID: {self.pdf_id}, "
                   f"Conversation ID: {self.conversation_id}, "
                   f"Research Mode: {self.research_mode}")

    def _get_query_graph(self):
        """Get or create query graph"""
        if not self._query_graph:
            self._query_graph = create_query_graph()
        return self._query_graph

    def _get_research_graph(self):
        """Get or create research graph"""
        if not self._research_graph:
            self._research_graph = create_research_graph()
        return self._research_graph

    def _get_document_graph(self):
        """Get or create document graph"""
        if not self._document_graph:
            self._document_graph = create_document_graph()
        return self._document_graph

    def initialize(self):
        """
        Initialize the ChatManager by loading conversation history and checking vector store.
        """
        # Check vector store readiness
        vector_store = get_vector_store()
        self.vector_store_ready = vector_store._initialized

        if not self.vector_store_ready:
            logger.warning("Vector store not initialized, attempting to initialize...")
            self.vector_store_ready = vector_store.initialize()

        # Load conversation history
        if self.conversation_id:
            # Load conversation from memory manager
            self.conversation_state = self.memory_manager.get_conversation(self.conversation_id)

            if self.conversation_state:
                logger.info(f"Loaded conversation {self.conversation_id} with "
                          f"{len(self.conversation_state.messages)} messages")

                # Check if research mode is active in conversation metadata
                if self.conversation_state.metadata and "research_mode" in self.conversation_state.metadata:
                    research_metadata = self.conversation_state.metadata["research_mode"]
                    if isinstance(research_metadata, dict) and research_metadata.get("active", False):
                        # Override research mode from args if metadata indicates it's active
                        self.research_mode = ResearchMode.RESEARCH
                        logger.info("Research mode activated from conversation metadata")
            else:
                # Create new conversation state
                metadata = {
                    "pdf_id": self.pdf_id,
                    "research_mode": {
                        "active": self.research_mode == ResearchMode.RESEARCH,
                        "pdf_ids": [self.pdf_id] if self.pdf_id else []
                    },
                    "created_at": datetime.now().isoformat()
                }

                self.conversation_state = self.memory_manager.create_conversation(
                    title=f"Conversation about {self.pdf_id}" if self.pdf_id else "New Conversation",
                    pdf_id=self.pdf_id,
                    metadata=metadata,
                    id=self.conversation_id
                )

                # Add system message
                system_prompt = self.BASE_SYSTEM_PROMPT
                if self.research_mode == ResearchMode.RESEARCH:
                    system_prompt = f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"

                self.conversation_state.add_message(MessageType.SYSTEM, system_prompt)

                logger.info(f"Created new conversation state for {self.conversation_id}")
        else:
            # Generate new conversation ID if not provided
            self.conversation_id = str(uuid.uuid4())

            # Create new conversation state
            metadata = {
                "pdf_id": self.pdf_id,
                "research_mode": {
                    "active": self.research_mode == ResearchMode.RESEARCH,
                    "pdf_ids": [self.pdf_id] if self.pdf_id else []
                },
                "created_at": datetime.now().isoformat()
            }

            self.conversation_state = self.memory_manager.create_conversation(
                title=f"Conversation about {self.pdf_id}" if self.pdf_id else "New Conversation",
                pdf_id=self.pdf_id,
                metadata=metadata,
                id=self.conversation_id
            )

            # Add system message
            system_prompt = self.BASE_SYSTEM_PROMPT
            if self.research_mode == ResearchMode.RESEARCH:
                system_prompt = f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"

            self.conversation_state.add_message(MessageType.SYSTEM, system_prompt)

            logger.info(f"Generated new conversation ID: {self.conversation_id}")

        # Ensure conversation has PDF ID in metadata
        if self.pdf_id and self.conversation_state:
            if not self.conversation_state.metadata:
                self.conversation_state.metadata = {}

            self.conversation_state.metadata["pdf_id"] = self.pdf_id

            # Ensure research_mode key exists in metadata
            if "research_mode" not in self.conversation_state.metadata:
                self.conversation_state.metadata["research_mode"] = {
                    "active": self.research_mode == ResearchMode.RESEARCH,
                    "pdf_ids": [self.pdf_id] if self.pdf_id else []
                }

    def process_document(self, pdf_id: str) -> Dict[str, Any]:
        """
        Process a document using the document processing graph.

        Args:
            pdf_id: Document ID to process

        Returns:
            Processing results
        """
        logger.info(f"Processing document: {pdf_id}")

        # Ensure vector store is ready
        if not self.vector_store_ready:
            vector_store = get_vector_store()
            self.vector_store_ready = vector_store.initialize()

        if not self.vector_store_ready:
            logger.error("Vector store initialization failed")
            return {
                "status": "error",
                "pdf_id": pdf_id,
                "error": "Vector store initialization failed"
            }

        # Create initial state
        state = GraphState(
            document_state={"pdf_id": pdf_id}
        )

        try:
            # Get document graph
            document_graph = self._get_document_graph()

            # Process document
            result = document_graph.invoke(state)

            logger.info(f"Document processing complete for: {pdf_id}")
            return {
                "status": "success",
                "pdf_id": pdf_id,
                "elements": result.document_state.get("element_count", 0) if result.document_state else 0,
                "processing_time": result.document_state.get("processing_time", 0) if result.document_state else 0
            }
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "pdf_id": pdf_id,
                "error": str(e)
            }

    def query(self, query: str, pdf_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a query using LangGraph with enhanced error handling.

        Args:
            query: User query text
            pdf_ids: Optional list of PDF IDs to query

        Returns:
            Query results
        """
        logger.info(f"Processing query: {query}")

        # Initialize if needed
        if not self.conversation_state:
            self.initialize()

        # Use the PDF ID from init if not provided
        if not pdf_ids and self.pdf_id:
            pdf_ids = [self.pdf_id]
            logger.info(f"Using PDF ID from initialization: {pdf_ids}")

        # Get PDF IDs from conversation metadata if in research mode
        if self.research_mode == ResearchMode.RESEARCH and self.conversation_state and self.conversation_state.metadata:
            research_mode = self.conversation_state.metadata.get("research_mode", {})
            if isinstance(research_mode, dict) and research_mode.get("active") and research_mode.get("pdf_ids"):
                pdf_ids = research_mode.get("pdf_ids")
                logger.info(f"Using PDF IDs from research mode metadata: {pdf_ids}")

        # Check if streaming is requested
        if self.stream_enabled:
            # For streaming, collect all chunks
            chunks = []
            final_response = None

            for chunk in self.stream_query(query, pdf_ids):
                if "status" in chunk and chunk["status"] == "complete":
                    final_response = chunk
                elif "chunk" in chunk:
                    chunks.append(chunk["chunk"])

            if final_response:
                return final_response
            else:
                # Build response from chunks
                return {
                    "response": "".join(chunks),
                    "conversation_id": self.conversation_id
                }

        # Create initial state
        state = GraphState(
            query_state=QueryState(
                query=query,
                pdf_ids=pdf_ids or []
            ),
            conversation_state=self.conversation_state
        )

        # Initialize conversation metadata if needed
        if state.conversation_state and not state.conversation_state.metadata:
            state.conversation_state.metadata = {}

        # IMPORTANT: Reset cycle count for new query
        if state.conversation_state and state.conversation_state.metadata:
            state.conversation_state.metadata["cycle_count"] = 0
            state.conversation_state.metadata["processed_response"] = False
            state.conversation_state.metadata["query_start_time"] = datetime.now().isoformat()
            logger.debug("Reset cycle count and processed_response flag for new query")

        # Choose the appropriate graph
        try:
            logger.debug("Creating LangGraph instance")
            graph = self._get_research_graph() if self.research_mode == ResearchMode.RESEARCH else self._get_query_graph()
            logger.debug("LangGraph instance created successfully")
        except Exception as graph_error:
            logger.error(f"Error creating graph: {str(graph_error)}", exc_info=True)
            return {
                "status": "error",
                "query": query,
                "error": f"Error creating graph: {str(graph_error)}",
                "conversation_id": self.conversation_id
            }

        try:
            # Ensure vector store is ready
            if not self.vector_store_ready:
                vector_store = get_vector_store()
                self.vector_store_ready = vector_store.initialize()

                if not self.vector_store_ready:
                    logger.error("Vector store initialization failed")
                    return {
                        "status": "error",
                        "query": query,
                        "error": "Vector store initialization failed",
                        "conversation_id": self.conversation_id
                    }

            # Run LangGraph with timeout protection to prevent hanging
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            with ThreadPoolExecutor() as executor:
                future = executor.submit(graph.invoke, state)
                try:
                    # Add a reasonable timeout (30 seconds)
                    result = future.result(timeout=30)
                    logger.info("LangGraph processing completed successfully")
                except TimeoutError:
                    logger.error("LangGraph processing timed out after 30 seconds")
                    return {
                        "status": "error",
                        "query": query,
                        "error": "Processing timed out. Your query might be too complex.",
                        "conversation_id": self.conversation_id
                    }

            # Update conversation state
            if result.conversation_state:
                self.conversation_state = result.conversation_state

                # Save updated conversation
                self.memory_manager.save_conversation(self.conversation_state)
                logger.info(f"Saved conversation with {len(self.conversation_state.messages)} messages")

            # Check for missing response
            if not result.generation_state or not result.generation_state.response:
                logger.error("No response generated by LangGraph")
                return {
                    "status": "error",
                    "query": query,
                    "error": "Failed to generate a response",
                    "conversation_id": self.conversation_id
                }

            # Prepare response
            response = {
                "query": query,
                "response": result.generation_state.response,
                "citations": result.generation_state.citations if result.generation_state and hasattr(result.generation_state, "citations") else [],
                "pdf_ids": pdf_ids or [],
                "conversation_id": self.conversation_id,
                "research_mode": self.conversation_state.metadata.get("research_mode") if self.conversation_state and self.conversation_state.metadata else None
            }

            logger.info(f"Query processing complete, response length: {len(response['response']) if response.get('response') else 0}")
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "conversation_id": self.conversation_id
            }

    def stream_query(self, query: str, pdf_ids: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a query response.

        Args:
            query: User query text
            pdf_ids: Optional list of PDF IDs to query

        Yields:
            Streaming query results
        """
        # Make sure streaming is enabled
        self.stream_enabled = True

        try:
            # First yield a processing message
            yield {
                "status": "processing",
                "message": "Processing your query..."
            }

            # Use the PDF ID from init if not provided
            if not pdf_ids and self.pdf_id:
                pdf_ids = [self.pdf_id]

            # Get PDF IDs from conversation metadata if in research mode
            if self.research_mode == ResearchMode.RESEARCH and self.conversation_state and self.conversation_state.metadata:
                research_mode = self.conversation_state.metadata.get("research_mode", {})
                if isinstance(research_mode, dict) and research_mode.get("active") and research_mode.get("pdf_ids"):
                    pdf_ids = research_mode.get("pdf_ids")
                    logger.info(f"Using PDF IDs from research mode metadata: {pdf_ids}")

            # Initialize if needed
            if not self.conversation_state:
                self.initialize()

            # Ensure vector store is ready
            if not self.vector_store_ready:
                yield {
                    "status": "processing",
                    "message": "Initializing vector database..."
                }

                vector_store = get_vector_store()
                self.vector_store_ready = vector_store.initialize()

                if self.vector_store_ready:
                    yield {
                        "status": "processing",
                        "message": "Vector database ready, processing your query..."
                    }
                else:
                    yield {
                        "status": "error",
                        "error": "Vector store initialization failed"
                    }
                    return

            # Create initial state
            state = GraphState(
                query_state=QueryState(
                    query=query,
                    pdf_ids=pdf_ids or []
                ),
                conversation_state=self.conversation_state
            )

            # Add the user message to conversation state
            if self.conversation_state:
                self.conversation_state.add_message(MessageType.USER, query)

                # Reset cycle count
                if not self.conversation_state.metadata:
                    self.conversation_state.metadata = {}
                self.conversation_state.metadata["cycle_count"] = 0
                self.conversation_state.metadata["processed_response"] = False

            # Choose the appropriate graph
            graph = self._get_research_graph() if self.research_mode == ResearchMode.RESEARCH else self._get_query_graph()

            # Create string buffer for accumulating response
            accumulated_response = ""

            # Process steps
            if self.research_mode == ResearchMode.RESEARCH:
                steps = [
                    ("query_analyzer", "Analyzing query..."),
                    ("retriever", "Searching across documents..."),
                    ("research_synthesizer", "Synthesizing information..."),
                    ("knowledge_generator", "Generating insights..."),
                    ("response_generator", "Creating response...")
                ]
            else:
                steps = [
                    ("query_analyzer", "Analyzing query..."),
                    ("retriever", "Searching documents..."),
                    ("knowledge_generator", "Processing information..."),
                    ("response_generator", "Creating response...")
                ]

            # Run graph with manual stepping for streaming
            current_state = state
            for node_name, status_message in steps:
                # Yield status update
                yield {
                    "status": "processing",
                    "message": status_message
                }

                # Run the current node
                try:
                    next_node = graph.get_node(node_name)
                    current_state = next_node(current_state)

                    # Check if we have a generated response
                    if (node_name == "response_generator" and
                        current_state.generation_state and
                        current_state.generation_state.response):

                        response = current_state.generation_state.response

                        # Stream in chunks
                        for i in range(0, len(response), self.stream_chunk_size):
                            chunk = response[i:i+self.stream_chunk_size]
                            accumulated_response += chunk

                            yield {
                                "type": "stream",
                                "chunk": chunk,
                                "index": i // self.stream_chunk_size,
                                "is_complete": False
                            }

                            # Add a small delay for smoother streaming
                            time.sleep(0.05)

                except Exception as node_error:
                    logger.error(f"Error in {node_name} node: {str(node_error)}")
                    yield {
                        "status": "error",
                        "error": f"Error in processing: {str(node_error)}"
                    }
                    return

            # Complete conversation processing
            try:
                # Add the assistant message to conversation state
                if (self.conversation_state and
                    current_state.generation_state and
                    current_state.generation_state.response):

                    self.conversation_state.add_message(
                        MessageType.ASSISTANT,
                        current_state.generation_state.response,
                        {"citations": current_state.generation_state.citations if hasattr(current_state.generation_state, "citations") else []}
                    )

                    # Mark response as processed
                    if not self.conversation_state.metadata:
                        self.conversation_state.metadata = {}
                    self.conversation_state.metadata["processed_response"] = True

                    # Save conversation
                    self.memory_manager.save_conversation(self.conversation_state)
            except Exception as save_error:
                logger.error(f"Error saving conversation: {str(save_error)}")

            # Send final complete message
            yield {
                "status": "complete",
                "response": current_state.generation_state.response if current_state.generation_state else accumulated_response,
                "conversation_id": self.conversation_id,
                "citations": current_state.generation_state.citations if current_state.generation_state and hasattr(current_state.generation_state, "citations") else []
            }

        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
            yield {
                "status": "error",
                "error": str(e)
            }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get formatted conversation history.

        Returns:
            List of message dictionaries
        """
        if not self.conversation_state:
            return []

        history = []
        for msg in self.conversation_state.messages:
            # Skip system messages
            if msg.type == MessageType.SYSTEM:
                continue

            # Format message
            history.append({
                "role": "user" if msg.type == MessageType.USER else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if hasattr(msg.created_at, "isoformat") else str(msg.created_at),
                "message_id": msg.id if hasattr(msg, "id") else None,
                "metadata": msg.metadata or {}
            })

        return history

    def clear_conversation(self) -> bool:
        """
        Clear the current conversation history.

        Returns:
            True if successful
        """
        if not self.conversation_id:
            return False

        # Delete from memory manager
        success = self.memory_manager.delete_conversation(self.conversation_id)

        if success:
            # Reset conversation state
            metadata = {
                "pdf_id": self.pdf_id,
                "research_mode": {
                    "active": self.research_mode == ResearchMode.RESEARCH,
                    "pdf_ids": [self.pdf_id] if self.pdf_id else []
                },
                "created_at": datetime.now().isoformat()
            }

            self.conversation_state = ConversationState(
                conversation_id=self.conversation_id,
                title=f"Conversation about {self.pdf_id}" if self.pdf_id else "New Conversation",
                pdf_id=self.pdf_id or "",
                metadata=metadata
            )

            # Add system message
            system_prompt = self.BASE_SYSTEM_PROMPT
            if self.research_mode == ResearchMode.RESEARCH:
                system_prompt = f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"

            self.conversation_state.add_message(MessageType.SYSTEM, system_prompt)

            # Save the new conversation
            self.memory_manager.save_conversation(self.conversation_state)

            logger.info(f"Cleared conversation {self.conversation_id}")

        return success
