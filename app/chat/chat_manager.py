"""
LangGraph-based ChatManager for the PDF RAG system.
This class serves as the main entry point for the chat functionality.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
import uuid

from app.chat.types import ChatArgs, ContentType, ResearchMode, ResearchContext
from app.chat.langgraph.state import GraphState, QueryState, MessageType, ConversationState
from app.chat.langgraph.graph import create_query_graph, create_document_graph, create_research_graph
from app.chat.memories.memory_manager import MemoryManager
from app.chat.utils.langgraph_helpers import format_conversation_for_llm

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
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        
        # Initialize conversation state
        self.conversation_state = None
        
        # Create LangGraph instances
        self.query_graph = create_query_graph()
        self.document_graph = create_document_graph()
        self.research_graph = create_research_graph()
        
        logger.info(f"ChatManager initialized with PDF ID: {self.pdf_id}, "
                   f"Conversation ID: {self.conversation_id}, "
                   f"Research Mode: {self.research_mode}")
    
    async def initialize(self) -> None:
        """
        Initialize the ChatManager by loading conversation history.
        """
        if self.conversation_id:
            # Load conversation from memory manager
            self.conversation_state = await self.memory_manager.get_conversation(self.conversation_id)
            
            if self.conversation_state:
                logger.info(f"Loaded conversation {self.conversation_id} with "
                           f"{len(self.conversation_state.messages)} messages")
            else:
                # Create new conversation state
                metadata = {
                    "pdf_id": self.pdf_id,
                    "research_mode": self.research_mode == ResearchMode.RESEARCH
                }
                
                self.conversation_state = await self.memory_manager.create_conversation(
                    title=f"Conversation about {self.pdf_id}",
                    pdf_id=self.pdf_id,
                    metadata=metadata
                )
                
                # Add system message
                system_prompt = self.BASE_SYSTEM_PROMPT if self.research_mode == ResearchMode.SINGLE else f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"
                self.conversation_state.add_message("system", system_prompt)
                
                logger.info(f"Created new conversation state for {self.conversation_id}")
        else:
            # Generate new conversation ID if not provided
            self.conversation_id = str(uuid.uuid4())
            
            # Create new conversation state
            metadata = {
                "pdf_id": self.pdf_id,
                "research_mode": self.research_mode == ResearchMode.RESEARCH
            }
            
            self.conversation_state = await self.memory_manager.create_conversation(
                title=f"Conversation about {self.pdf_id}",
                pdf_id=self.pdf_id,
                metadata=metadata
            )
            
            # Add system message
            system_prompt = self.BASE_SYSTEM_PROMPT if self.research_mode == ResearchMode.SINGLE else f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"
            self.conversation_state.add_message("system", system_prompt)
            
            logger.info(f"Generated new conversation ID: {self.conversation_id}")
        
        # Ensure conversation has PDF ID in metadata
        if self.pdf_id and self.conversation_state:
            self.conversation_state.metadata["pdf_id"] = self.pdf_id
    
    async def process_document(self, pdf_id: str) -> Dict[str, Any]:
        """
        Process a document using the document processing graph.
        
        Args:
            pdf_id: ID of the PDF to process
            
        Returns:
            Document processing results
        """
        logger.info(f"Processing document: {pdf_id}")
        
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
                "conversation_id": self.conversation_id
            }
            
            logger.info(f"Query processing complete, response length: {len(response['response']) if response['response'] else 0}")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
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
        # Simplified streaming implementation
        try:
            # Get full response first
            full_response = await self.aquery(query, pdf_ids)
            
            if "error" in full_response:
                yield full_response
                return
                
            # Simulate streaming by yielding chunks
            response_text = full_response.get("response", "")
            if not response_text:
                yield {
                    "status": "error",
                    "error": "No response generated",
                    "conversation_id": self.conversation_id
                }
                return
                
            # Split into chunks (in a real implementation, this would be token-by-token)
            chunk_size = self.chat_args.stream_chunk_size or 20  # Characters per chunk
            chunks = [response_text[i:i+chunk_size] for i in range(0, len(response_text), chunk_size)]
            
            # Yield each chunk
            for i, chunk in enumerate(chunks):
                yield {
                    "chunk": chunk,
                    "index": i,
                    "total_chunks": len(chunks),
                    "is_complete": i == len(chunks) - 1,
                    "conversation_id": self.conversation_id
                }
                
                # Simulate some delay
                await asyncio.sleep(0.05)
                
            # Yield final complete message
            yield {
                "status": "complete",
                "response": response_text,
                "citations": full_response.get("citations", []),
                "conversation_id": self.conversation_id
            }
            
        except Exception as e:
            logger.error(f"Stream query processing failed: {str(e)}", exc_info=True)
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
            if msg.type == "system":
                continue
                
            history.append({
                "role": "user" if msg.type == "user" else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "message_id": msg.id if hasattr(msg, "id") else None
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
            "research_mode": self.research_mode == ResearchMode.RESEARCH
        }
        
        self.conversation_state = ConversationState(
            conversation_id=self.conversation_id,
            title=f"Conversation about {self.pdf_id}",
            pdf_id=self.pdf_id,
            metadata=metadata
        )
        
        # Add system message
        system_prompt = self.BASE_SYSTEM_PROMPT if self.research_mode == ResearchMode.SINGLE else f"{self.BASE_SYSTEM_PROMPT}\n\n{self.RESEARCH_MODE_PROMPT}"
        self.conversation_state.add_message(MessageType.SYSTEM, system_prompt)
        
        logger.info(f"Cleared conversation {self.conversation_id}")
        return True