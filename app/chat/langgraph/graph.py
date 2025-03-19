"""
LangGraph workflow definition for the PDF RAG system.
Improved graph structure with clear, deterministic routing.
"""

import logging
from typing import Dict, Any, List
from langgraph.graph import END, StateGraph

from app.chat.langgraph.state import GraphState

logger = logging.getLogger(__name__)

def create_query_graph() -> StateGraph:
    """
    Create a query processing graph for single-document RAG.

    Returns:
        StateGraph for query processing
    """
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze_query
    from app.chat.langgraph.nodes.retriever import retrieve_content
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
    from app.chat.langgraph.nodes.response_generator import generate_response

    # Create graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve_content)
    graph.add_node("knowledge_generator", generate_knowledge)
    graph.add_node("response_generator", generate_response)

    # Define edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")
    graph.add_edge("response_generator", "conversation_memory")

    # Define conditional edge to end the graph
    def should_end(state: GraphState) -> str:
        # If we have processed the response, end the graph
        if (state.generation_state and
            state.generation_state.response and
            state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            return END

        # Check if we've gone through too many cycles
        if state.conversation_state and state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            if cycle_count > 3:
                logger.warning(f"Ending graph after {cycle_count} cycles")
                return END

        # Continue processing
        return "query_analyzer"

    graph.add_conditional_edges("conversation_memory", should_end)

    # Compile the graph
    return graph.compile()

def create_research_graph() -> StateGraph:
    """
    Create a research graph for multi-document analysis.

    Returns:
        StateGraph for research processing
    """
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze_query
    from app.chat.langgraph.nodes.retriever import retrieve_content
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
    from app.chat.langgraph.nodes.response_generator import generate_response
    from app.chat.langgraph.nodes.research_synthesizer import synthesize_research

    # Create graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve_content)
    graph.add_node("research_synthesizer", synthesize_research)
    graph.add_node("knowledge_generator", generate_knowledge)
    graph.add_node("response_generator", generate_response)

    # Define edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "research_synthesizer")
    graph.add_edge("research_synthesizer", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")
    graph.add_edge("response_generator", "conversation_memory")

    # Define conditional edge to end the graph
    def should_end(state: GraphState) -> str:
        # If we have processed the response, end the graph
        if (state.generation_state and
            state.generation_state.response and
            state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            return END

        # Check if we've gone through too many cycles
        if state.conversation_state and state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            if cycle_count > 3:
                logger.warning(f"Ending graph after {cycle_count} cycles")
                return END

        # Continue processing
        return "query_analyzer"

    graph.add_conditional_edges("conversation_memory", should_end)

    # Compile the graph
    return graph.compile()

def create_document_graph() -> StateGraph:
    """
    Create a document processing graph.

    Returns:
        StateGraph for document processing
    """
    from app.chat.langgraph.nodes.document_processor import process_document

    # Create graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("document_processor", process_document)

    # Define edges
    graph.set_entry_point("document_processor")
    graph.add_edge("document_processor", END)

    # Compile the graph
    return graph.compile()
