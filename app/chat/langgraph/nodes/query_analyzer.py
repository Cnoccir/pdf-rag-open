"""
Query analyzer node for LangGraph-based PDF RAG system.
Determines the optimal retrieval strategy for a given query.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType

logger = logging.getLogger(__name__)

def analyze_query(state: GraphState) -> GraphState:
    """
    Analyze a user query to determine the optimal retrieval strategy.
    Simplified implementation with clear deterministic rules.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with query analysis
    """
    try:
        # Validate state
        if not state.query_state:
            logger.error("Query state is required for analysis")
            return state

        # Get query from state
        query = state.query_state.query.lower() if state.query_state.query else ""
        logger.info(f"Analyzing query: {query[:50]}...")

        # Extract keywords (simple implementation, could be enhanced with NLP)
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "with", "for", "to", "from"}
        words = re.findall(r'\b[a-z0-9]+\b', query.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Update keywords in state
        state.query_state.keywords = keywords

        # Detect specific content types
        focused_elements = []

        # Check for table references
        if any(term in query for term in ["table", "column", "row", "cell", "spreadsheet", "tabular"]):
            focused_elements.append(ContentType.TABLE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for image references
        elif any(term in query for term in ["image", "figure", "diagram", "chart", "picture", "photo", "graph"]):
            focused_elements.append(ContentType.IMAGE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for code references
        elif any(term in query for term in ["code", "function", "class", "program", "script", "api"]):
            focused_elements.append(ContentType.CODE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for mathematical content
        elif any(term in query for term in ["equation", "formula", "math", "calculation"]):
            focused_elements.append(ContentType.EQUATION.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check if query is best suited for concept search
        elif any(term in query for term in ["related", "connection", "relationship", "similar"]):
            state.query_state.retrieval_strategy = RetrievalStrategy.CONCEPT

        # Check if query is best suited for semantic search
        elif any(term in query for term in ["about", "like", "meaning", "explain", "understand", "concept"]):
            state.query_state.retrieval_strategy = RetrievalStrategy.SEMANTIC

        # Check if query is best suited for keyword search
        elif any(term in query for term in ["exactly", "specific", "find", "locate", "where"]):
            state.query_state.retrieval_strategy = RetrievalStrategy.KEYWORD

        # Default to hybrid search
        else:
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Update focused elements
        state.query_state.focused_elements = focused_elements

        # Detect query type based on patterns
        if query.startswith("how"):
            state.query_state.query_type = "instructional"
        elif query.startswith("why"):
            state.query_state.query_type = "explanatory"
        elif query.startswith("what") or query.startswith("who") or query.startswith("when") or query.startswith("where"):
            state.query_state.query_type = "factual"
        elif "compare" in query or "difference" in query or "versus" in query or "vs" in query:
            state.query_state.query_type = "comparative"
        elif "list" in query or "enumerate" in query or "what are" in query:
            state.query_type = "listing"
        else:
            state.query_state.query_type = "general"

        # Extract technical concepts
        technical_concepts = extract_technical_terms(query)

        # Update concepts in state
        state.query_state.concepts = technical_concepts

        # Update metadata
        if not state.query_state.metadata:
            state.query_state.metadata = {}

        state.query_state.metadata["analyzed_at"] = datetime.now().isoformat()
        state.query_state.metadata["strategy_reasoning"] = f"Selected {state.query_state.retrieval_strategy} based on query patterns"

        logger.info(f"Query analysis: strategy={state.query_state.retrieval_strategy}, "
                   f"type={state.query_state.query_type}, concepts={technical_concepts[:5]}")

        return state

    except Exception as e:
        logger.error(f"Query analysis error: {str(e)}", exc_info=True)

        # Ensure we have valid query state with defaults
        if state.query_state:
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID
            state.query_state.query_type = "general"

        return state

def extract_technical_terms(text: str) -> List[str]:
    """
    Extract technical terms from text.
    Simple regex-based approach for pattern matching.

    Args:
        text: Input text

    Returns:
        List of technical terms
    """
    if not text:
        return []

    # Define patterns for technical terms
    patterns = [
        r'\b[A-Z][A-Z0-9]+\b',                   # Acronyms like PDF, API, etc.
        r'\b[A-Za-z]+\d+[A-Za-z0-9]*\b',         # Technical codes like GPT3, T5, etc.
        r'\b[a-z]+[-_][a-z]+\b',                 # Hyphenated terms like machine-learning
        r'\b[A-Z][a-z]+[A-Z][a-z]+[a-zA-Z]*\b',  # CamelCase terms like LangGraph
    ]

    # Extract terms
    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)

    # Filter common non-technical terms
    non_technical = {'And', 'The', 'This', 'That', 'With', 'From', 'Into'}
    return list(term for term in terms if term not in non_technical)
