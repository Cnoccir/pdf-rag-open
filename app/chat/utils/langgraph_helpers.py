"""
Helper utilities for LangGraph-based PDF RAG system.
These utilities support the LangGraph architecture with specialized functions.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime

from app.chat.langgraph.state import GraphState, MessageType, ConversationState
from app.chat.types import ContentElement, ContentType

logger = logging.getLogger(__name__)


def extract_technical_terms_simple(text: str) -> List[str]:
    """
    Extract technical terms from text using simple regex patterns.
    This is a lightweight implementation for use in LangGraph nodes.
    
    Args:
        text: Text to extract terms from
        
    Returns:
        List of technical terms
    """
    patterns = [
        r'\b[A-Z][A-Z0-9]+\b',                  # Acronyms like PDF, HTTP, API
        r'\b[A-Za-z]+\d+[A-Za-z0-9]*\b',        # Technical codes like GPT3, T5, B2B
        r'\b[a-z]+[-_][a-z]+\b',                # Hyphenated terms like machine-learning
        r'\b[A-Z][a-z]+[A-Z][a-z]+[a-zA-Z]*\b', # CamelCase terms like LangGraph, GraphState
    ]
    
    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)
    
    # Filter out common non-technical terms
    non_technical = {'And', 'The', 'This', 'That', 'With', 'From', 'Into'}
    return list(term for term in terms if term not in non_technical)


def format_conversation_for_llm(state: GraphState) -> List[Dict[str, str]]:
    """
    Format conversation history for LLM input in a standardized format.
    
    Args:
        state: Graph state containing conversation history
        
    Returns:
        List of message dictionaries formatted for LLM input
    """
    if not state.conversation_state or not state.conversation_state.messages:
        return []
        
    formatted_messages = []
    
    for msg in state.conversation_state.messages:
        if msg.type == MessageType.HUMAN:
            formatted_messages.append({"role": "user", "content": msg.content})
        elif msg.type == MessageType.AI:
            formatted_messages.append({"role": "assistant", "content": msg.content})
        elif msg.type == MessageType.SYSTEM:
            formatted_messages.append({"role": "system", "content": msg.content})
            
    return formatted_messages


def parse_citations_from_text(text: str) -> Dict[str, Any]:
    """
    Parse citation markers from generated text.
    Extracts citation markers like [1], [2], etc. and maps them to sources.
    
    Args:
        text: Generated text with citation markers
        
    Returns:
        Dictionary with citation information
    """
    # Simple regex-based citation extraction
    citations = {}
    citation_pattern = r'\[(\d+)\]'
    
    matches = re.findall(citation_pattern, text)
    for match in matches:
        citation_id = int(match)
        if citation_id not in citations:
            citations[citation_id] = {
                "id": citation_id,
                "count": 1
            }
        else:
            citations[citation_id]["count"] += 1
            
    return {
        "citations": list(citations.values()),
        "has_citations": len(citations) > 0
    }


def elements_to_context(elements: List[ContentElement]) -> str:
    """
    Convert retrieved content elements to a formatted context string.
    Optimized for LLM consumption with clear separation between sources.
    
    Args:
        elements: List of content elements
        
    Returns:
        Formatted context string
    """
    if not elements:
        return ""
        
    context_parts = []
    
    for i, element in enumerate(elements):
        # Add source identifier
        source_info = f"[{i+1}] "
        if element.metadata and "page" in element.metadata:
            source_info += f"Page {element.metadata['page']} - "
            
        if element.metadata and "section" in element.metadata:
            source_info += f"{element.metadata['section']}"
        
        # Format based on content type
        if element.content_type == ContentType.TEXT or element.content_type == ContentType.MARKDOWN:
            context_parts.append(f"{source_info}\n{element.content}\n")
        elif element.content_type == ContentType.TABLE:
            context_parts.append(f"{source_info}\nTable: {element.content}\n")
        elif element.content_type == ContentType.IMAGE:
            if element.text_content:
                context_parts.append(f"{source_info}\nImage Caption: {element.text_content}\n")
        else:
            context_parts.append(f"{source_info}\n{element.content}\n")
    
    return "\n\n".join(context_parts)


def create_empty_graph_state() -> GraphState:
    """
    Create an empty graph state with initialized conversation state.
    Useful for starting new conversations or testing.
    
    Returns:
        Empty graph state with conversation state
    """
    conversation_state = ConversationState(
        conversation_id=f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        messages=[],
        technical_concepts=[],
        metadata={}
    )
    
    return GraphState(
        conversation_state=conversation_state
    )
