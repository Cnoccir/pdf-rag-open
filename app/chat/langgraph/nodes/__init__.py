"""
Nodes for LangGraph PDF RAG implementation.
"""

from app.chat.langgraph.nodes.document_processor import process_document
from app.chat.langgraph.nodes.query_analyzer import process_query
from app.chat.langgraph.nodes.retriever import retrieve_content
from app.chat.langgraph.nodes.knowledge_generator import generate_knowledge
from app.chat.langgraph.nodes.response_generator import generate_response
from app.chat.langgraph.nodes.conversation_memory import process_conversation_memory

# Functions representing graph nodes
document_processor = process_document
query_analyzer = process_query
retriever = retrieve_content
knowledge_generator = generate_knowledge
response_generator = generate_response
conversation_memory = process_conversation_memory

__all__ = [
    "document_processor",
    "query_analyzer",
    "retriever",
    "knowledge_generator",
    "response_generator",
    "conversation_memory"
]
