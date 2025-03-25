"""
LangGraph workflow definition for the PDF RAG system.
Improved graph structure with clear, deterministic routing.
"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import END, StateGraph, START
from langgraph.errors import GraphRecursionError

from app.chat.langgraph.state import GraphState

logger = logging.getLogger(__name__)

def create_query_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze_query
    from app.chat.langgraph.nodes.retriever import retrieve_content
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
    from app.chat.langgraph.nodes.response_generator import generate_response

    graph = StateGraph(GraphState)

    # Add nodes to the graph
    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve_content)
    graph.add_node("knowledge_generator", generate_knowledge)
    graph.add_node("response_generator", generate_response)

    # Set entry point and default edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")
    graph.add_edge("response_generator", "conversation_memory")

    def should_end(state: GraphState) -> str:
        """
        Determine whether to continue processing or end the graph execution.
        Improved to better detect completion and prevent infinite recursion.
        """
        logger.info(f"Evaluating should_end condition in query graph")

        # Check for response completion - if we have a response, we're done
        if state.generation_state and state.generation_state.response:
            logger.info("Response generated, ending graph")
            return END

        # Check metadata flags for completion
        if state.conversation_state and state.conversation_state.metadata:
            # Check if response has been processed
            if state.conversation_state.metadata.get("processed_response", False):
                logger.info("Response processed flag is True, ending graph")
                return END

            # Check for cycle count safety limits
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            logger.info(f"Current cycle count: {cycle_count}")

            # Very conservative limit of 3 cycles to prevent recursion
            if cycle_count >= 3:
                logger.warning(f"Ending graph after {cycle_count} cycles (safety limit)")
                # Set flag to avoid future recursion
                state.conversation_state.metadata["processed_response"] = True
                return END

        # Check for retrieval state with no elements but metadata (already tried fallbacks)
        if (state.retrieval_state and
            hasattr(state.retrieval_state, 'elements') and
            len(state.retrieval_state.elements) == 0 and
            state.retrieval_state.metadata and
            state.retrieval_state.metadata.get("warning") == "No relevant content found for this query"):
            logger.warning("No content found after retrieval, ending graph")
            return END

        # Default path - continue to query analyzer
        logger.info("Continuing to query_analyzer")
        return "query_analyzer"

    # Add the conditional edge with improved safety checks
    graph.add_conditional_edges(
        "conversation_memory",
        should_end,
        {END: END, "query_analyzer": "query_analyzer"}
    )

    # Return compiled graph with recursion_limit configuration
    graph_config = {"recursion_limit": 10}
    if config:
        graph_config.update(config)

    logger.info(f"Compiling query graph with config: {graph_config}")
    return graph.compile(graph_config)

def create_research_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze_query
    from app.chat.langgraph.nodes.retriever import retrieve_content
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
    from app.chat.langgraph.nodes.response_generator import generate_response
    from app.chat.langgraph.nodes.research_synthesizer import synthesize_research

    graph = StateGraph(GraphState)

    # Add nodes to the graph
    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve_content)
    graph.add_node("research_synthesizer", synthesize_research)
    graph.add_node("knowledge_generator", generate_knowledge)
    graph.add_node("response_generator", generate_response)

    # Set entry point and default edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "research_synthesizer")
    graph.add_edge("research_synthesizer", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")
    graph.add_edge("response_generator", "conversation_memory")

    def should_end(state: GraphState) -> str:
        """
        Determine whether to continue processing or end the research graph execution.
        Improved to better detect completion and prevent infinite recursion.
        """
        logger.info(f"Evaluating should_end condition in research graph")

        # Check for response completion - if we have a response, we're done
        if state.generation_state and state.generation_state.response:
            logger.info("Response generated, ending research graph")

            # Make sure we mark it as processed
            if state.conversation_state and state.conversation_state.metadata:
                state.conversation_state.metadata["processed_response"] = True

            return END

        # Check metadata flags for completion
        if state.conversation_state and state.conversation_state.metadata:
            # Check if response has been processed
            if state.conversation_state.metadata.get("processed_response", False):
                logger.info("Response processed flag is True, ending research graph")
                return END

            # Check for cycle count safety limits
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            logger.info(f"Current research cycle count: {cycle_count}")

            # Even more conservative limit for research mode (more complex)
            if cycle_count >= 3:
                logger.warning(f"Ending research graph after {cycle_count} cycles (safety limit)")
                # Set flag to avoid future recursion
                state.conversation_state.metadata["processed_response"] = True
                return END

        # Check for retrieval state with no elements but metadata (already tried fallbacks)
        if (state.retrieval_state and
            hasattr(state.retrieval_state, 'elements') and
            len(state.retrieval_state.elements) == 0 and
            state.retrieval_state.metadata and
            state.retrieval_state.metadata.get("warning") == "No relevant content found for this query"):
            logger.warning("No content found after retrieval, ending research graph")
            return END

        # Default path - continue to query analyzer
        logger.info("Continuing to query_analyzer in research graph")
        return "query_analyzer"

    # Add the conditional edge with improved safety checks
    graph.add_conditional_edges(
        "conversation_memory",
        should_end,
        {END: END, "query_analyzer": "query_analyzer"}
    )

    # Return compiled graph with recursion_limit configuration
    graph_config = {"recursion_limit": 10}
    if config:
        graph_config.update(config)

    logger.info(f"Compiling research graph with config: {graph_config}")
    return graph.compile(graph_config)

def create_document_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    from app.chat.langgraph.nodes.document_processor import process_document

    graph = StateGraph(GraphState)

    graph.add_node("document_processor", process_document)
    graph.set_entry_point("document_processor")
    graph.add_edge("document_processor", END)

    # Return compiled graph with default config
    graph_config = {"recursion_limit": 5}
    if config:
        graph_config.update(config)

    return graph.compile(graph_config)

__all__ = ["create_query_graph", "create_research_graph", "create_document_graph"]
