"""
Conversation memory node for LangGraph-based PDF RAG system.
This node handles conversation history management and technical concept extraction.
"""

import logging
from typing import Dict, List, Any, Optional
import re

from app.chat.langgraph.state import GraphState, MessageType

logger = logging.getLogger(__name__)


def extract_technical_terms(text: str) -> List[str]:
    """
    Extract technical terms from text.
    This is a simplified implementation that could be enhanced with ML/NLP techniques.
    
    Args:
        text: Text to extract terms from
        
    Returns:
        List of technical terms
    """
    # Simple pattern matching for technical terms (words with digits, acronyms, etc.)
    patterns = [
        r'\b[A-Z][A-Z0-9]+\b',                   # Acronyms like PDF, HTTP, API
        r'\b[A-Za-z]+\d+[A-Za-z0-9]*\b',         # Technical codes like GPT3, T5, B2B
        r'\b[a-z]+[-_][a-z]+\b',                 # Hyphenated terms like machine-learning
    ]
    
    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)
    
    return list(terms)


def process_conversation_memory(state: GraphState) -> GraphState:
    """
    Process conversation memory and extract technical concepts.
    This node maintains conversation context and extracts technical terms.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state
    """
    # Initialize conversation state if not present
    if not state.conversation_state:
        from app.chat.langgraph.state import ConversationState
        state.conversation_state = ConversationState()
        logger.info("Initialized new conversation state")
    
    # If we have a generation_state with response but haven't processed it yet
    if (state.generation_state and state.generation_state.response and
        not state.conversation_state.metadata.get("processed_response", False)):
        
        # Extract any additional technical terms from the AI response
        response_terms = extract_technical_terms(state.generation_state.response)
        
        # Update the conversation state with these technical terms
        if response_terms:
            state.conversation_state.technical_concepts.extend(
                [term for term in response_terms if term not in state.conversation_state.technical_concepts]
            )
            
        # Add the AI response to conversation history
        state.add_message(
            MessageType.AI, 
            state.generation_state.response,
            {"citations": state.generation_state.citations if hasattr(state.generation_state, "citations") else []}
        )
        
        # Mark the response as processed
        state.conversation_state.metadata["processed_response"] = True
        
        logger.info(f"Added AI response to conversation history with {len(response_terms)} technical terms")
        return state
    
    # Add system message if not already present and no processed_response flag
    if not state.conversation_state.metadata.get("processed_response", False):
        if not any(msg.type == MessageType.SYSTEM for msg in state.conversation_state.messages):
            # Default system prompt if not specified
            system_prompt = state.conversation_state.system_prompt or (
                "You are an AI assistant specialized in answering questions about technical documents."
            )
            state.add_message(MessageType.SYSTEM, system_prompt)
            logger.info("Added system message to conversation")
        
        # Process human message if present in query state
        if state.query_state and state.query_state.query:
            # Extract technical terms from user query
            technical_terms = extract_technical_terms(state.query_state.query)
            
            # Update the conversation state with these technical terms
            if technical_terms:
                state.conversation_state.technical_concepts.extend(
                    [term for term in technical_terms if term not in state.conversation_state.technical_concepts]
                )
                logger.info(f"Extracted technical terms: {technical_terms}")
                
                # Also add them to query state for current retrieval
                if state.query_state:
                    state.query_state.concepts.extend(
                        [term for term in technical_terms if term not in state.query_state.concepts]
                    )
                    
            # Add the human message to conversation history if not already there
            # Simple check to avoid duplicates - more robust implementation would check message content
            if state.conversation_state.messages and state.conversation_state.messages[-1].type == MessageType.HUMAN:
                if state.conversation_state.messages[-1].content != state.query_state.query:
                    state.add_message(MessageType.HUMAN, state.query_state.query)
            else:
                state.add_message(MessageType.HUMAN, state.query_state.query)
                
            logger.info(f"Added human message to conversation history: {state.query_state.query[:50]}...")
    
    return state
