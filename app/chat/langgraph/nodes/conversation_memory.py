"""
Conversation memory node for LangGraph-based PDF RAG system.
Handles conversation history management and technical concept extraction.
"""

import logging
import re
from typing import Dict, List, Any
from datetime import datetime

from app.chat.langgraph.state import GraphState, MessageType
from app.chat.utils.extraction import extract_technical_terms

logger = logging.getLogger(__name__)

def process_conversation_memory(state: GraphState) -> dict:
    """
    Process conversation memory and extract technical concepts.
    Maintains conversation context and tracks message history.

    Args:
        state: Current graph state

    Returns:
        Dictionary with updated conversation_state
    """
    # Initialize conversation state if not present
    if not state.conversation_state:
        logger.warning("No conversation state found, creating new one")
        from app.chat.langgraph.state import ConversationState
        state.conversation_state = ConversationState()

    # Initialize metadata to track processing status
    if not state.conversation_state.metadata:
        state.conversation_state.metadata = {}

    # IMPROVED CYCLE TRACKING: Increment first before any other processing
    cycle_count = state.conversation_state.metadata.get("cycle_count", 0)
    state.conversation_state.metadata["cycle_count"] = cycle_count + 1

    # Log the cycle count every time for better observability
    logger.info(f"Conversation memory processing cycle: {cycle_count + 1}")

    # CRITICAL FIX: Force end if we've cycled too many times
    # Increased from 3 to 10 for more headroom
    if cycle_count > 10:
        state.conversation_state.metadata["processed_response"] = True
        logger.warning(f"Forcing end of processing after {cycle_count} cycles")
        return {"conversation_state": state.conversation_state}

    # Check if we have a generation_state with response but haven't processed it yet
    has_generation = (state.generation_state and state.generation_state.response)
    already_processed = state.conversation_state.metadata.get("processed_response", False)

    if has_generation and not already_processed:
        logger.info("Processing AI response and adding to conversation history")

        # Extract any additional technical terms from the AI response
        response_terms = []
        try:
            response_terms = extract_technical_terms(state.generation_state.response)
        except Exception as e:
            logger.warning(f"Error extracting technical terms: {str(e)}")

        # Update the conversation state with these technical terms
        if response_terms:
            for term in response_terms:
                if term not in state.conversation_state.technical_concepts:
                    state.conversation_state.technical_concepts.append(term)

        # Add the AI response to conversation history
        metadata = {}
        if hasattr(state.generation_state, "citations") and state.generation_state.citations:
            metadata["citations"] = state.generation_state.citations

        if hasattr(state.generation_state, "metadata") and state.generation_state.metadata:
            for key, value in state.generation_state.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        state.conversation_state.add_message(
            MessageType.ASSISTANT,
            state.generation_state.response,
            metadata
        )

        # CRITICAL: Mark the response as processed to stop recursion
        state.conversation_state.metadata["processed_response"] = True

        # Reset cycle count after processing
        state.conversation_state.metadata["cycle_count"] = 0

        logger.info(f"Added AI response to conversation history with {len(response_terms)} technical terms")
        logger.debug(f"Set processed_response flag to TRUE")
        return {"conversation_state": state.conversation_state}

    # Process query state if not already done
    if not already_processed and state.query_state and state.query_state.query:
        # Make sure we have a system message
        if not any(msg.type == MessageType.SYSTEM for msg in state.conversation_state.messages):
            # Default system prompt
            system_prompt = "You are an AI assistant specialized in answering questions about technical documents."

            # Enhance with research mode if applicable
            if state.query_state and state.query_state.pdf_ids and len(state.query_state.pdf_ids) > 1:
                system_prompt += "\n\nYou are in RESEARCH MODE, which means you should synthesize information across multiple documents and highlight connections between concepts."

            state.conversation_state.add_message(MessageType.SYSTEM, system_prompt)
            logger.info("Added system message to conversation")

        # Extract technical terms from user query
        technical_terms = []
        try:
            technical_terms = extract_technical_terms(state.query_state.query)
        except Exception as e:
            logger.warning(f"Error extracting technical terms from query: {str(e)}")

        # Update the conversation state with these technical terms
        if technical_terms:
            for term in technical_terms:
                if term not in state.conversation_state.technical_concepts:
                    state.conversation_state.technical_concepts.append(term)

            logger.info(f"Extracted technical terms: {technical_terms}")

            # Also add them to query state for current retrieval
            if state.query_state:
                for term in technical_terms:
                    if term not in state.query_state.concepts:
                        state.query_state.concepts.append(term)

        # Check if this query is already in the messages to avoid duplicates
        user_messages = [msg for msg in state.conversation_state.messages if msg.type == MessageType.USER]
        is_duplicate = any(msg.content == state.query_state.query for msg in user_messages)

        if not is_duplicate:
            state.conversation_state.add_message(MessageType.USER, state.query_state.query)
            logger.info(f"Added human message to conversation history: {state.query_state.query[:50]}...")

            # Ensure PDF ID is properly set
            if not state.conversation_state.pdf_id and state.query_state.pdf_ids:
                state.conversation_state.pdf_id = state.query_state.pdf_ids[0]

    # Store conversation update timestamp
    state.conversation_state.updated_at = datetime.now()

    return {"conversation_state": state.conversation_state}
