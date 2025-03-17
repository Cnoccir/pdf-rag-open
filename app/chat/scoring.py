"""
Conversation scoring utilities for the PDF RAG system.
Compatible with LangGraph architecture but maintains API compatibility
with legacy code.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Simple in-memory storage for scores (replace with database in production)
_scores_data = {}

def score_conversation(
    conversation_id: str,
    score: float,
    llm: str = None,
    retriever: str = None,
    memory: str = None
) -> Dict[str, Any]:
    """
    Score a conversation for feedback and model improvement.
    
    Args:
        conversation_id: Conversation ID
        score: Score value (-1 to 1)
        llm: LLM model used
        retriever: Retriever used
        memory: Memory implementation used
        
    Returns:
        Result dictionary
    """
    logger.info(f"Scoring conversation {conversation_id} with score {score}")
    
    # Create score object
    score_data = {
        "conversation_id": conversation_id,
        "score": score,
        "timestamp": datetime.utcnow().isoformat(),
        "llm": llm or "gpt-4",
        "retriever": retriever or "neo4j",
        "memory": memory or "langgraph"
    }
    
    # Store score
    if conversation_id not in _scores_data:
        _scores_data[conversation_id] = []
    
    _scores_data[conversation_id].append(score_data)
    
    return {"status": "success", "message": "Score recorded"}

def get_scores() -> List[Dict[str, Any]]:
    """
    Get all recorded scores.
    
    Returns:
        List of score records
    """
    logger.info(f"Retrieving scores (count: {sum(len(scores) for scores in _scores_data.values())})")
    
    # Convert to flat list
    scores = []
    for conversation_scores in _scores_data.values():
        scores.extend(conversation_scores)
    
    # Sort by timestamp (newest first)
    scores.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return scores

def get_conversation_scores(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Get scores for a specific conversation.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        List of score records for the conversation
    """
    return _scores_data.get(conversation_id, [])