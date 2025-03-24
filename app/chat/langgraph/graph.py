"""
LangGraph workflow definition for the PDF RAG system.
Improved graph structure with clear, deterministic routing and proper input mapping.
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import END, StateGraph

from app.chat.langgraph.state import GraphState, QueryState, RetrievalStrategy

logger = logging.getLogger(__name__)

def create_query_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """
    Create a query processing graph for single-document RAG.

    Args:
        config: Optional configuration dictionary

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

    # Define input mapper function to correctly handle direct invocations
    def map_input_to_state(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map raw input to proper GraphState structure.
        Handles both direct invocations and internal calls.
        """
        logger.info(f"Mapping input: {input_data}")

        # Handle different possible input structures
        query = input_data.get("question", input_data.get("query", ""))
        pdf_id = input_data.get("pdf_id", "")
        pdf_ids = input_data.get("pdf_ids", [])

        # Ensure pdf_id is included in pdf_ids if provided
        if pdf_id and not pdf_ids:
            pdf_ids = [pdf_id]

        # Extract conversation_id if present (for memory retrieval)
        conversation_id = input_data.get("conversation_id")

        # Create query state
        query_state = QueryState(
            query=query,
            pdf_ids=pdf_ids,
            retrieval_strategy=RetrievalStrategy.HYBRID  # Default strategy
        )

        # Create a minimal conversation state if needed
        conversation_state = None
        if conversation_id:
            from app.chat.langgraph.state import ConversationState
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                pdf_id=pdf_id if pdf_id else "",
                metadata={"input_mapped": True}
            )

        # Return structured state for graph
        result = {"query_state": query_state}
        if conversation_state:
            result["conversation_state"] = conversation_state

        logger.info(f"Mapped input to: {result}")
        return result

    # Set input mapper for the graph
    graph.set_input_mapper(map_input_to_state)

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

    # Define conditional edge to end the graph - THIS NEEDS FIXING
    def should_end(state: GraphState) -> str:
        # CRITICAL FIX: More explicit logging
        logger.info(f"Evaluating should_end condition in query graph")

        # If there's no conversation state, continue to query analyzer
        if not state.conversation_state:
            logger.info("No conversation state, continuing to query_analyzer")
            return "query_analyzer"

        # Log the critical values we're checking
        if state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            processed = state.conversation_state.metadata.get("processed_response", False)
            logger.info(f"Current cycle_count={cycle_count}, processed_response={processed}")

        # CRITICAL FIX: Check generation state first - if response exists, end immediately
        if state.generation_state and state.generation_state.response:
            logger.info("Response generated, ending graph")
            return END

        # If response is processed according to metadata, end
        if (state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            logger.info("Response processed flag is True, ending graph")
            return END

        # CRITICAL SAFETY: Force end after cycle limit regardless of other conditions
        if (state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("cycle_count", 0) >= 5):  # REDUCED FROM 10 TO 5
            logger.warning(f"Forcing end after {state.conversation_state.metadata.get('cycle_count')} cycles (safety limit)")
            return END

        # Continue processing
        logger.info("Continuing to query_analyzer")
        return "query_analyzer"

    # Critical - ensure this edge is defined PROPERLY
    graph.add_conditional_edges(
        "conversation_memory",
        should_end,
        {END: END, "query_analyzer": "query_analyzer"}
    )

    # Compile the graph
    return graph.compile()

def create_research_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """
    Create a research graph for multi-document analysis.

    Args:
        config: Optional configuration dictionary

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

    # Use the same input mapper as the query graph for consistency
    def map_input_to_state(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map raw input to proper GraphState structure.
        Handles both direct invocations and internal calls.
        """
        logger.info(f"Mapping input for research graph: {input_data}")

        # Handle different possible input structures
        query = input_data.get("question", input_data.get("query", ""))
        pdf_id = input_data.get("pdf_id", "")
        pdf_ids = input_data.get("pdf_ids", [])

        # Ensure pdf_id is included in pdf_ids if provided
        if pdf_id and not pdf_ids:
            pdf_ids = [pdf_id]

        # Extract conversation_id if present (for memory retrieval)
        conversation_id = input_data.get("conversation_id")

        # Create query state
        query_state = QueryState(
            query=query,
            pdf_ids=pdf_ids,
            retrieval_strategy=RetrievalStrategy.HYBRID  # Default strategy
        )

        # Create a minimal conversation state if needed
        conversation_state = None
        if conversation_id:
            from app.chat.langgraph.state import ConversationState
            conversation_state = ConversationState(
                conversation_id=conversation_id,
                pdf_id=pdf_id if pdf_id else "",
                metadata={"input_mapped": True, "research_mode": True}
            )

        # Return structured state for graph
        result = {"query_state": query_state}
        if conversation_state:
            result["conversation_state"] = conversation_state

        logger.info(f"Mapped research input to: {result}")
        return result

    # Set input mapper for the graph
    graph.set_input_mapper(map_input_to_state)

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
        # Log state for debugging
        if state.conversation_state and state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            processed = state.conversation_state.metadata.get("processed_response", False)
            logger.debug(f"Evaluating research graph end condition: cycle_count={cycle_count}, processed_response={processed}")

        # If we have processed the response, end the graph
        if (state.generation_state and
            state.generation_state.response and
            state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            logger.info("Ending research graph: response processed")
            return END

        # Safety check - if we've gone through too many cycles
        if state.conversation_state and state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            # Increase the safety limit from 3 to 10 for more headroom
            if cycle_count > 10:
                logger.warning(f"Ending research graph after {cycle_count} cycles (safety limit)")
                # Force processed flag before ending
                state.conversation_state.metadata["processed_response"] = True
                return END

        # Continue processing
        return "query_analyzer"

    graph.add_conditional_edges("conversation_memory", should_end)

    # Compile the graph
    return graph.compile()

def create_document_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """
    Create a document processing graph.

    Args:
        config: Optional configuration dictionary

    Returns:
        StateGraph for document processing
    """
    from app.chat.langgraph.nodes.document_processor import process_document

    # Create graph
    graph = StateGraph(GraphState)

    # Define document-specific input mapper
    def map_input_to_state(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map document processing input to proper GraphState structure.
        """
        logger.info(f"Mapping document input: {input_data}")

        # Extract PDF ID
        pdf_id = input_data.get("pdf_id", "")
        if not pdf_id:
            logger.error("No PDF ID provided for document processing")
            return {"document_state": {"error": "No PDF ID provided"}}

        # Create document state
        document_state = {"pdf_id": pdf_id}

        # Add additional configuration if provided
        if "config" in input_data:
            document_state["config"] = input_data["config"]

        logger.info(f"Mapped document input to state with pdf_id: {pdf_id}")
        return {"document_state": document_state}

    # Set input mapper for the document graph
    graph.set_input_mapper(map_input_to_state)

    # Add nodes
    graph.add_node("document_processor", process_document)

    # Define edges
    graph.set_entry_point("document_processor")
    graph.add_edge("document_processor", END)

    # Compile the graph
    return graph.compile()

# Expose graph creation functions for LangGraph Studio
__all__ = ["create_query_graph", "create_research_graph", "create_document_graph"]
