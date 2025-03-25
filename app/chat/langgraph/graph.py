"""
LangGraph workflow definition for the PDF RAG system.
Improved graph structure with clear, deterministic routing.
"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import END, StateGraph
from langgraph.managed.is_last_step import RemainingSteps

from app.chat.langgraph.state import GraphState

logger = logging.getLogger(__name__)

def create_query_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze_query
    from app.chat.langgraph.nodes.retriever import retrieve_content
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
    from app.chat.langgraph.nodes.response_generator import generate_response

    graph = StateGraph(GraphState)

    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve_content)
    graph.add_node("knowledge_generator", generate_knowledge)
    graph.add_node("response_generator", generate_response)

    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")
    graph.add_edge("response_generator", "conversation_memory")

    def should_end(state: GraphState) -> str:
        logger.info(f"Evaluating should_end condition in query graph")

        if state.remaining_steps is not None and state.remaining_steps <= 2:
            logger.warning("Ending graph to avoid hitting recursion limit")
            return END

        if not state.conversation_state:
            logger.info("No conversation state, continuing to query_analyzer")
            return "query_analyzer"

        if state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            processed = state.conversation_state.metadata.get("processed_response", False)
            logger.info(f"Current cycle_count={cycle_count}, processed_response={processed}")

        if state.generation_state and state.generation_state.response:
            logger.info("Response generated, ending graph")
            return END

        if (state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            logger.info("Response processed flag is True, ending graph")
            return END

        if (state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("cycle_count", 0) >= 5):
            logger.warning(f"Forcing end after {state.conversation_state.metadata.get('cycle_count')} cycles (safety limit)")
            return END

        logger.info("Continuing to query_analyzer")
        return "query_analyzer"

    graph.add_conditional_edges(
        "conversation_memory",
        should_end,
        {END: END, "query_analyzer": "query_analyzer"}
    )

    return graph.compile()

def create_research_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
    from app.chat.langgraph.nodes.query_analyzer import analyze_query
    from app.chat.langgraph.nodes.retriever import retrieve_content
    from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
    from app.chat.langgraph.nodes.response_generator import generate_response
    from app.chat.langgraph.nodes.research_synthesizer import synthesize_research

    graph = StateGraph(GraphState)

    graph.add_node("conversation_memory", process_conversation_memory)
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve_content)
    graph.add_node("research_synthesizer", synthesize_research)
    graph.add_node("knowledge_generator", generate_knowledge)
    graph.add_node("response_generator", generate_response)

    graph.set_entry_point("conversation_memory")
    graph.add_edge("conversation_memory", "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "research_synthesizer")
    graph.add_edge("research_synthesizer", "knowledge_generator")
    graph.add_edge("knowledge_generator", "response_generator")
    graph.add_edge("response_generator", "conversation_memory")

    def should_end(state: GraphState) -> str:
        if state.remaining_steps is not None and state.remaining_steps <= 2:
            logger.warning("Ending research graph to avoid hitting recursion limit")
            return END

        if state.conversation_state and state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            processed = state.conversation_state.metadata.get("processed_response", False)
            logger.debug(f"Evaluating research graph end condition: cycle_count={cycle_count}, processed_response={processed}")

        if (state.generation_state and
            state.generation_state.response and
            state.conversation_state and
            state.conversation_state.metadata and
            state.conversation_state.metadata.get("processed_response", False)):
            logger.info("Ending research graph: response processed")
            return END

        if state.conversation_state and state.conversation_state.metadata:
            cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
            if cycle_count > 10:
                logger.warning(f"Ending research graph after {cycle_count} cycles (safety limit)")
                state.conversation_state.metadata["processed_response"] = True
                return END

        return "query_analyzer"

    graph.add_conditional_edges("conversation_memory", should_end)

    return graph.compile()

def create_document_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    from app.chat.langgraph.nodes.document_processor import process_document

    graph = StateGraph(GraphState)

    graph.add_node("document_processor", process_document)
    graph.set_entry_point("document_processor")
    graph.add_edge("document_processor", END)

    return graph.compile()

__all__ = ["create_query_graph", "create_research_graph", "create_document_graph"]
