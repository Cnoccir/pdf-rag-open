"""
LangGraph graph definitions for the PDF RAG system.
These graphs define the flow of data through the LangGraph nodes.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from app.chat.langgraph.state import (
    GraphState, 
    DocumentState, 
    QueryState, 
    RetrievalState,
    GenerationState
)
from app.chat.langgraph.nodes import (
    document_processor,
    query_analyzer,
    retriever,
    knowledge_generator,
    response_generator,
    conversation_memory
)

logger = logging.getLogger(__name__)

def create_document_graph() -> StateGraph:
    """
    Create a document processing graph.
    This graph handles the ingestion and processing of documents.
    
    Returns:
        StateGraph for document processing
    """
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


    """
    Create a research graph for multi-document analysis.
    This graph extends the query graph with additional research capabilities.
    
    Returns:
        StateGraph for research
    """
    # Create new graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("conversation_memory", conversation_memory)
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("retriever", retriever)
    graph.add_node("knowledge_generator", knowledge_generator)
    graph.add_node("response_generator", response_generator)
    
    # Add research-specific node (could be expanded in the future)
    def research_synthesizer(state):
        """Placeholder for future research synthesis node"""
        # In a real implementation, this would perform cross-document synthesis
        # For now, it just passes through
        return state
    
    graph.add_node("research_synthesizer", research_synthesizer)
    
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

def create_research_graph() -> StateGraph:
    """
    Create a research graph for multi-document analysis.
    Enhanced with Neo4j graph capabilities for analyzing relationships.
    
    Returns:
        StateGraph for research
    """
    # Create new graph
    graph = StateGraph(GraphState)
    
    # Import nodes
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze as query_analyzer
    from app.chat.langgraph.nodes.retriever import retrieve_content as retriever
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge as knowledge_generator
    from app.chat.langgraph.nodes.response_generator import generate_response as response_generator
    from app.chat.langgraph.nodes.research_synthesizer import research_synthesize as research_synthesizer
    
    # Add nodes
    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("retriever", retriever)
    graph.add_node("research_synthesizer", research_synthesizer)  # Neo4j-aware research synthesizer
    graph.add_node("knowledge_generator", knowledge_generator)
    graph.add_node("response_generator", response_generator)
    
    # Define edges
    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "research_synthesizer")  # Flow through research synthesizer
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
# Example usage functions
async def process_document_async(pdf_id: str) -> Dict[str, Any]:
    """
    Process a document using the document graph.
    
    Args:
        pdf_id: ID of the PDF to process
        
    Returns:
        Results of document processing
    """
    # Create document graph
    graph = create_document_graph()
    
    # Create initial state
    state = GraphState(
        document_state=DocumentState(pdf_id=pdf_id)
    )
    
    # Execute the graph
    result = await graph.ainvoke(state)
    
    # Return results
    return {
        "state": result,
        "pdf_id": pdf_id,
        "status": result.document_state.status if result.document_state else "unknown"
    }

async def answer_query_async(query: str, pdf_ids: List[str] = None) -> Dict[str, Any]:
    """
    Answer a query using the query graph.
    
    Args:
        query: User query
        pdf_ids: Optional list of PDF IDs to search within
        
    Returns:
        Query results with generated response
    """
    # Create query graph
    graph = create_query_graph()
    
    # Create initial state
    state = GraphState(
        query_state=QueryState(
            query=query,
            pdf_ids=pdf_ids or []
        )
    )
    
    # Execute the graph
    result = await graph.ainvoke(state)
    
    # Return results
    return {
        "query": query,
        "response": result.generation_state.response if result.generation_state else None,
        "citations": result.generation_state.citations if result.generation_state else [],
        "pdf_ids": pdf_ids or []
    }
