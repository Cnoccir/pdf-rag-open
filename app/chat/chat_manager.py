"""
LangGraph-based ChatManager for the PDF RAG system.
This class serves as the main entry point for the chat functionality.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
import uuid
import traceback

from app.chat.types import ChatArgs, ContentType, ResearchMode, ResearchContext
from app.chat.langgraph.state import GraphState, QueryState, MessageType, ConversationState
from app.chat.langgraph.graph import create_query_graph, create_document_graph, create_research_graph
from app.chat.memories.memory_manager import MemoryManager
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)


class ChatManager:
    """
    ChatManager implementing LangGraph-based architecture for PDF RAG system.

    This class orchestrates the document processing and query answering workflow
    using the LangGraph architecture for flexibility and modularity.

    Features:
    - Document processing with hierarchical understanding
    - Query analysis with dynamic retrieval strategy selection
    - Research mode for cross-document analysis
    - Streaming support for interactive responses
    - Conversation history tracking
    """

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
        Initialize the ChatManager with the provided arguments.

        Args:
            chat_args: Chat configuration arguments
        """
        self.chat_args = chat_args
        self.conversation_id = getattr(chat_args, "conversation_id", None)
        self.pdf_id = getattr(chat_args, "pdf_id", None)
        self.research_mode = getattr(chat_args, "research_mode", ResearchMode.SINGLE)
        self.stream_enabled = getattr(chat_args, "stream_enabled", False)
        self.stream_chunk_size = getattr(chat_args, "stream_chunk_size", 20)

        # Initialize memory manager
        self.memory_manager = MemoryManager()

        # Initialize conversation state
        self.conversation_state = None

        # Create LangGraph instances
        self.query_graph = create_query_graph()
        self.document_graph = create_document_graph()
        self.research_graph = create_research_graph()

        # Initialize Neo4j connection status
        self.neo4j_initialized = False

        logger.info(f"ChatManager initialized with PDF ID: {self.pdf_id}, "
                   f"Conversation ID: {self.conversation_id}, "
                   f"Research Mode: {self.research_mode}")

    async def initialize(self) -> None:
        """
        Initialize the ChatManager by loading conversation history and initializing Neo4j.
        """
        # Initialize Neo4j store
        try:
            vector_store = get_vector_store()
            if not vector_store.initialized:
                logger.warning("Neo4j vector store not initialized, initializing...")
                await vector_store.initialize_database()
                self.neo4j_initialized = True
            else:
                self.neo4j_initialized = True

            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            logger.error(traceback.format_exc())
            self.neo4j_initialized = False

        # Load conversation history
        if self.conversation_id:
            # Load conversation from memory manager
            self.conversation_state = await self.memory_manager.get_conversation(self.conversation_id)

            if self.conversation_state:
                logger.info(f"Loaded conversation {self.conversation_id} with "
                           f"{len(self.conversation_state.messages)} messages")

                # Check if research mode is active in conversation metadata
                if self.conversation_state.metadata and self.conversation_state.metadata.get("research_mode"):
                    research_metadata = self.conversation_state.metadata.get("research_mode")
                    if research_metadata.get("active", False) and len(research_metadata.get("pdf_ids", [])) > 1:
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
                    "created_at": datetime.utcnow().isoformat()
                }

                self.conversation_state = await self.memory_manager.create_conversation(
                    title=f"Conversation about {self.pdf_id}" if self.pdf_id else f"New Conversation",
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
                "created_at": datetime.utcnow().isoformat()
            }

            self.conversation_state = await self.memory_manager.create_conversation(
                title=f"Conversation about {self.pdf_id}" if self.pdf_id else f"New Conversation",
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

    async def process_document(self, pdf_id: str) -> Dict[str, Any]:
        """
        Process a document using the document processing graph.

        Args:
            pdf_id: ID of the PDF to process

        Returns:
            Document processing results
        """
        logger.info(f"Processing document: {pdf_id}")

        # Ensure Neo4j is initialized
        if not self.neo4j_initialized:
            try:
                vector_store = get_vector_store()
                await vector_store.initialize_database()
                self.neo4j_initialized = True
                logger.info("Initialized Neo4j for document processing")
            except Exception as e:
                logger.error(f"Error initializing Neo4j: {str(e)}")
                self.neo4j_initialized = False
                return {
                    "status": "error",
                    "pdf_id": pdf_id,
                    "error": f"Neo4j initialization failed: {str(e)}"
                }

        # Create initial state for document processing
        state = GraphState(
            document_state={"pdf_id": pdf_id}
        )

        # Execute the document processing graph
        try:
            result = await self.document_graph.ainvoke(state)
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

    async def query(self, query: str, pdf_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a query using the appropriate LangGraph.

        Args:
            query: User query
            pdf_ids: Optional list of PDF IDs to search

        Returns:
            Query results
        """
        if self.stream_enabled:
            # Collect all chunks for a complete response
            chunks = []
            final_response = None

            async for chunk in self.stream_query(query, pdf_ids):
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
        else:
            # Process as non-streaming query
            return await self.aquery(query, pdf_ids)

    async def aquery(self, query: str, pdf_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a query using the query graph.

        Args:
            query: User query
            pdf_ids: Optional list of PDF IDs to search

        Returns:
            Query results including generated response
        """
        logger.info(f"Processing query: {query}")

        # Use the PDF ID from init if not provided
        if not pdf_ids and self.pdf_id:
            pdf_ids = [self.pdf_id]

        # Get PDF IDs from conversation metadata if in research mode
        if self.research_mode == ResearchMode.RESEARCH and self.conversation_state and self.conversation_state.metadata:
            research_mode = self.conversation_state.metadata.get("research_mode", {})
            if research_mode.get("active") and research_mode.get("pdf_ids"):
                pdf_ids = research_mode.get("pdf_ids")
                logger.info(f"Using PDF IDs from research mode metadata: {pdf_ids}")

        # Initialize if needed
        if not self.conversation_state:
            await self.initialize()

        # Create initial state
        state = GraphState(
            query_state=QueryState(
                query=query,
                pdf_ids=pdf_ids or []
            ),
            conversation_state=self.conversation_state
        )

        # Choose the appropriate graph based on research mode
        graph = self.research_graph if self.research_mode in [ResearchMode.RESEARCH, ResearchMode.MULTI] else self.query_graph

        # Execute the graph
        try:
            # Ensure Neo4j is initialized
            if not self.neo4j_initialized:
                try:
                    vector_store = get_vector_store()
                    if not vector_store.initialized:
                        await vector_store.initialize_database()
                    self.neo4j_initialized = True
                    logger.info("Initialized Neo4j for query processing")
                except Exception as e:
                    logger.error(f"Error initializing Neo4j: {str(e)}")
                    self.neo4j_initialized = False
                    return {
                        "status": "error",
                        "query": query,
                        "error": f"Neo4j initialization failed: {str(e)}",
                        "conversation_id": self.conversation_id
                    }

            # Run LangGraph
            result = await graph.ainvoke(state)

            # Update conversation state from result
            if result.conversation_state:
                self.conversation_state = result.conversation_state

                # Save updated conversation
                await self.memory_manager.save_conversation(self.conversation_state)

                logger.info(f"Saved conversation with {len(self.conversation_state.messages)} messages")

            # Prepare response
            response = {
                "query": query,
                "response": result.generation_state.response if result.generation_state else None,
                "citations": result.generation_state.citations if result.generation_state and hasattr(result.generation_state, "citations") else [],
                "pdf_ids": pdf_ids or [],
                "conversation_id": self.conversation_id,
                "research_mode": self.conversation_state.metadata.get("research_mode") if self.conversation_state and self.conversation_state.metadata else None
            }

            logger.info(f"Query processing complete, response length: {len(response['response']) if response['response'] else 0}")
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "conversation_id": self.conversation_id
            }

    async def stream_query(self, query: str, pdf_ids: Optional[List[str]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a query response.

        Args:
            query: User query
            pdf_ids: Optional list of PDF IDs to search

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
                if research_mode.get("active") and research_mode.get("pdf_ids"):
                    pdf_ids = research_mode.get("pdf_ids")
                    logger.info(f"Using PDF IDs from research mode metadata: {pdf_ids}")

            # Initialize if needed
            if not self.conversation_state:
                await self.initialize()

            # Create initial state
            state = GraphState(
                query_state=QueryState(
                    query=query,
                    pdf_ids=pdf_ids or []
                ),
                conversation_state=self.conversation_state
            )

            # Choose the appropriate graph based on research mode
            graph = self.research_graph if self.research_mode in [ResearchMode.RESEARCH, ResearchMode.MULTI] else self.query_graph

            # Ensure Neo4j is initialized
            if not self.neo4j_initialized:
                try:
                    yield {
                        "status": "processing",
                        "message": "Initializing vector database..."
                    }

                    vector_store = get_vector_store()
                    if not vector_store.initialized:
                        await vector_store.initialize_database()
                    self.neo4j_initialized = True

                    yield {
                        "status": "processing",
                        "message": "Vector database ready, processing your query..."
                    }
                except Exception as e:
                    logger.error(f"Error initializing Neo4j: {str(e)}")
                    self.neo4j_initialized = False
                    yield {
                        "status": "error",
                        "error": f"Neo4j initialization failed: {str(e)}"
                    }
                    return

            # Execute graph with event handlers for streaming
            try:
                # Create a string buffer for response chunks
                accumulated_response = ""

                # Define callback that will be called as the graph runs
                async def on_graph_state_change(event_data: Dict[str, Any]) -> None:
                    nonlocal accumulated_response

                    # Extract relevant data about current state
                    state_type = event_data.get("type", "none")
                    node_name = event_data.get("node_name", "")
                    state = event_data.get("state", {})

                    if state_type == "on_node_start":
                        # Node starting - provide progress update
                        if node_name == "retriever":
                            yield {
                                "status": "processing",
                                "message": "Retrieving relevant information..."
                            }
                        elif node_name == "knowledge_generator":
                            yield {
                                "status": "processing",
                                "message": "Analyzing documents..."
                            }
                        elif node_name == "research_synthesizer" and self.research_mode == ResearchMode.RESEARCH:
                            yield {
                                "status": "processing",
                                "message": "Synthesizing across documents..."
                            }
                        elif node_name == "response_generator":
                            yield {
                                "status": "processing",
                                "message": "Generating response..."
                            }

                    elif state_type == "on_node_end" and node_name == "response_generator":
                        # Response is being generated - stream it in chunks
                        if state and hasattr(state, "generation_state") and state.generation_state:
                            response = state.generation_state.response

                            if response and response != accumulated_response:
                                # Only send the new part
                                new_content = response[len(accumulated_response):]

                                # Break into smaller chunks for smoother streaming
                                chunk_size = self.stream_chunk_size
                                for i in range(0, len(new_content), chunk_size):
                                    chunk = new_content[i:i+chunk_size]
                                    is_last_chunk = (i + chunk_size) >= len(new_content)

                                    yield {
                                        "chunk": chunk,
                                        "index": (len(accumulated_response) + i) // chunk_size,
                                        "is_complete": False
                                    }

                                # Update accumulated response
                                accumulated_response = response

                    elif state_type == "on_chain_end":
                        # End of graph execution - return final response
                        if state and hasattr(state, "generation_state") and state.generation_state:
                            # Save conversation state
                            if hasattr(state, "conversation_state") and state.conversation_state:
                                self.conversation_state = state.conversation_state
                                await self.memory_manager.save_conversation(self.conversation_state)

                            # Get citations from the state
                            citations = []
                            if hasattr(state.generation_state, "citations"):
                                citations = state.generation_state.citations

                            # Send final complete response
                            yield {
                                "status": "complete",
                                "response": state.generation_state.response,
                                "conversation_id": self.conversation_id,
                                "citations": citations
                            }

                # Create callback handler that will send chunks through our generator
                async def callback_handler(event_data: Dict[str, Any]) -> None:
                    async for chunk in on_graph_state_change(event_data):
                        yield chunk

                # Set up callbacks
                callbacks = {"on_state_change": callback_handler}

                # Run graph with callbacks
                result_iterator = await graph.astream(state, callbacks)

                # Process all events from graph
                async for chunk_generator in result_iterator:
                    async for chunk in chunk_generator:
                        yield chunk

                # Ensure we have a final state in case callbacks missed it
                async for event in result_iterator:
                    if event.get("type") == "on_chain_end":
                        state = event.get("state")
                        if state and state.generation_state:
                            # Final sanity check - make sure we send complete response
                            yield {
                                "status": "complete",
                                "response": state.generation_state.response,
                                "conversation_id": self.conversation_id,
                                "citations": state.generation_state.citations if hasattr(state.generation_state, "citations") else []
                            }
                            break

            except Exception as e:
                logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
                yield {
                    "status": "error",
                    "error": str(e)
                }

        except Exception as e:
            logger.error(f"Stream query processing failed: {str(e)}", exc_info=True)
            logger.error(traceback.format_exc())
            yield {
                "status": "error",
                "error": str(e),
                "conversation_id": self.conversation_id
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

            history.append({
                "role": "user" if msg.type == MessageType.USER else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if hasattr(msg.created_at, 'isoformat') else str(msg.created_at),
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
        self.memory_manager.delete_conversation(self.conversation_id)

        # Reset conversation state
        metadata = {
            "pdf_id": self.pdf_id,
            "research_mode": {
                "active": self.research_mode == ResearchMode.RESEARCH,
                "pdf_ids": [self.pdf_id] if self.pdf_id else []
            },
            "created_at": datetime.utcnow().isoformat()
        }

        self.conversation_state = ConversationState(
            conversation_id=self.conversation_id,
            title=f"Conversation about {self.pdf_id}" if self.pdf_id else f"New Conversation",
            pdf_id=self.pdf_id or "",
            metadata=metadata
        )

        # Add system message
        system_prompt = self.BASE_SYSTEM_PROMPT
        if self.research_mode == ResearchMode.RESEARCH:
            system_prompt = f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"

        self.conversation_state.add_message(MessageType.SYSTEM, system_prompt)

        logger.info(f"Cleared conversation {self.conversation_id}")
        return True
