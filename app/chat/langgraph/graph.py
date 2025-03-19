"""
LangGraph graph definitions for the PDF RAG system.
These graphs define the flow of data through the LangGraph nodes.
"""

import logging
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from app.chat.langgraph.state import GraphState

logger = logging.getLogger(__name__)

def create_document_graph() -> StateGraph:
    """
    Create a document processing graph.
    This graph handles the ingestion and processing of documents.

    Returns:
        StateGraph for document processing
    """
    # Import document processor node
    from app.chat.langgraph.nodes import document_processor

    # Create new graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("document_processor", document_processor)

    # Define edges
    graph.set_entry_point("document_processor")
    graph.add_edge("document_processor", END)

    # Compile the graph
    return graph.compile()

def create_query_graph() -> StateGraph:
    """
    Create a query processing graph.
    This graph handles the analysis of queries, retrieval, and response generation.

    Returns:
        StateGraph for query processing
    """
    # Import nodes
    from app.chat.langgraph.nodes import (
        conversation_memory,
        query_analyzer,
        retriever,
        knowledge_generator,
        response_generator
    )

    # Create new graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("conversation_memory", conversation_memory)
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("retriever", retriever)
    graph.add_node("knowledge_generator", knowledge_generator)
    graph.add_node("response_generator", response_generator)

    # Define edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")

    # Add final conversation memory processing to capture generated response
    graph.add_edge("response_generator", "conversation_memory")

    # Define final edge to END state
    def should_end(state: GraphState) -> str:
        """Determine if we should end the graph."""
        # If we have a response and have processed it, end the graph
        if (state.generation_state and
            state.generation_state.response and
            state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            return END

        # Ensure we have conversation state to avoid attribute errors
        if not state.conversation_state:
            # End if no conversation state exists
            return END

        # Enhanced safety check - verify if we've been through this cycle too many times
        # This helps prevent infinite recursion
        cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
        if cycle_count > 5:  # Limit recursion to 5 cycles
            logger.warning("Ending graph due to excessive cycling")
            return END

        # Increment cycle count to track recursion
        if not state.conversation_state.metadata:
            state.conversation_state.metadata = {}
        state.conversation_state.metadata["cycle_count"] = cycle_count + 1

        return "query_analyzer"

    graph.add_conditional_edges(
        "conversation_memory",
        should_end
    )

    # Compile the graph
    return graph.compile()

def create_research_graph() -> StateGraph:
    """
    Create a research graph for multi-document analysis.
    This graph extends the query graph with additional research capabilities.

    Returns:
        StateGraph for research
    """
    # Import nodes
    from app.chat.langgraph.nodes import (
        conversation_memory,
        query_analyzer,
        retriever,
        knowledge_generator,
        response_generator
    )

    # Import research synthesizer
    from app.chat.langgraph.nodes.research_synthesizer import research_synthesize

    # Create new graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("conversation_memory", conversation_memory)
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("retriever", retriever)
    graph.add_node("research_synthesizer", research_synthesize)
    graph.add_node("knowledge_generator", knowledge_generator)
    graph.add_node("response_generator", response_generator)

    # Define edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "research_synthesizer")
    graph.add_edge("research_synthesizer", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")

    # Add final conversation memory processing to capture generated response
    graph.add_edge("response_generator", "conversation_memory")

    # Define final edge to END state
    def should_end(state: GraphState) -> str:
        """Determine if we should end the graph."""
        # If we already have a response and have been through conversation_memory twice, end
        if (state.generation_state and state.generation_state.response and
            state.conversation_state and state.conversation_state.metadata.get("processed_response", False)):
            return END
        return "query_analyzer"

    graph.add_conditional_edges(
        "conversation_memory",
        should_end
    )

    # Compile the graph
    return graph.compile()
