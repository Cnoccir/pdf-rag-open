"""
LangGraph implementation for the PDF RAG system.
Provides core graph structure and components for document processing and querying.
"""

from app.chat.langgraph.state import (
    GraphState,
    QueryState,
    RetrievalState,
    GenerationState,
    ResearchState,
    ConversationState,
    MessageType,
    RetrievalStrategy,
    ContentType
)

# Export node functions
from app.chat.langgraph.nodes.document_processor import process_document
from app.chat.langgraph.nodes.query_analyzer import analyze_query
from app.chat.langgraph.nodes.retriever import retrieve_content
from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
from app.chat.langgraph.nodes.response_generator import generate_response
from app.chat.langgraph.nodes.research_synthesizer import synthesize_research
from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory

# Export graph creation functions
from app.chat.langgraph.graph import (
    create_document_graph,
    create_query_graph,
    create_research_graph
)

__all__ = [
    # States
    "GraphState",
    "QueryState",
    "RetrievalState",
    "GenerationState",
    "ResearchState",
    "ConversationState",
    "MessageType",
    "RetrievalStrategy",
    "ContentType",

    # Nodes
    "process_document",
    "analyze_query",
    "retrieve_content",
    "generate_knowledge",
    "generate_response",
    "synthesize_research",
    "process_conversation_memory",

    # Graphs
    "create_document_graph",
    "create_query_graph",
    "create_research_graph"
]
