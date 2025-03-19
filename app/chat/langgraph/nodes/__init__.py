"""
Nodes for LangGraph PDF RAG implementation.
This module exposes all node functions used in the LangGraph workflow.
"""

import logging
from typing import Dict, Any, Callable

from app.chat.langgraph.state import GraphState

# Import node functions
from app.chat.langgraph.nodes.document_processor import process_document
from app.chat.langgraph.nodes.query_analyzer import analyze_query
from app.chat.langgraph.nodes.retriever import retrieve_content
from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
from app.chat.langgraph.nodes.response_generator import generate_response
from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory
from app.chat.langgraph.nodes.research_synthesizer import synthesize_research

# Export node functions with consistent naming
document_processor = process_document
query_analyzer = analyze_query
retriever = retrieve_content
knowledge_generator = generate_knowledge
response_generator = generate_response
conversation_memory = process_conversation_memory
research_synthesizer = synthesize_research

# Create a registry of nodes for easy access
node_registry: Dict[str, Callable[[GraphState], GraphState]] = {
    "document_processor": document_processor,
    "query_analyzer": query_analyzer,
    "retriever": retrieve_content,
    "knowledge_generator": generate_knowledge,
    "response_generator": generate_response,
    "conversation_memory": conversation_memory,
    "research_synthesizer": research_synthesizer
}

__all__ = [
    # Node functions
    "document_processor",
    "query_analyzer",
    "retriever",
    "knowledge_generator",
    "response_generator",
    "conversation_memory",
    "research_synthesizer",

    # Registry
    "node_registry"
]
