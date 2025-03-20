"""
Query analyzer node for LangGraph-based PDF RAG system.
Enhanced with multi-level chunking and multi-embedding strategy support.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType
from app.chat.types import ChunkLevel, EmbeddingType

logger = logging.getLogger(__name__)

def analyze_query(state: GraphState) -> GraphState:
    """
    Analyze a user query to determine optimal retrieval strategy.
    Enhanced with chunk level and embedding type selection.

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

        # Check for procedure references
        elif any(term in query for term in ["procedure", "step", "how to", "process", "instruction", "task", "follow"]):
            focused_elements.append(ContentType.PROCEDURE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.PROCEDURE
            state.query_state.procedure_focused = True

        # Check for parameter references
        elif any(term in query for term in ["parameter", "setting", "value", "configuration", "option", "variable"]):
            focused_elements.append(ContentType.PARAMETER.value)
            state.query_state.parameter_query = True

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

        # Determine chunk level preference based on query patterns
        chunk_level, chunk_reasoning = determine_chunk_level(query)
        state.query_state.preferred_chunk_level = chunk_level

        # Determine embedding type preference based on query patterns
        embedding_type, embedding_reasoning = determine_embedding_type(query)
        state.query_state.preferred_embedding_type = embedding_type

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
        state.query_state.metadata["chunk_level_reasoning"] = chunk_reasoning
        state.query_state.metadata["embedding_type_reasoning"] = embedding_reasoning

        logger.info(f"Query analysis: strategy={state.query_state.retrieval_strategy}, "
                   f"type={state.query_state.query_type}, "
                   f"chunk_level={state.query_state.preferred_chunk_level}, "
                   f"embedding_type={state.query_state.preferred_embedding_type}, "
                   f"concepts={technical_concepts[:5]}")

        return state

    except Exception as e:
        logger.error(f"Query analysis error: {str(e)}", exc_info=True)

        # Ensure we have valid query state with defaults
        if state.query_state:
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID
            state.query_state.query_type = "general"
            state.query_state.preferred_chunk_level = ChunkLevel.SECTION
            state.query_state.preferred_embedding_type = EmbeddingType.GENERAL

        return state

def determine_chunk_level(query: str) -> Tuple[ChunkLevel, str]:
    """
    Determine the appropriate chunk level based on query content.

    Args:
        query: User query

    Returns:
        Tuple of (preferred chunk level, reasoning)
    """
    # Check for document-level indicators (broad questions about the entire document)
    if any(phrase in query.lower() for phrase in [
        "what is this document about",
        "summary of the document",
        "overview of",
        "main topics",
        "generally describe",
        "what does this document cover",
        "main points of",
        "summary of"
    ]):
        return ChunkLevel.DOCUMENT, "Query requests document-level information"

    # Check for section-level indicators
    if any(phrase in query.lower() for phrase in [
        "section on",
        "chapter about",
        "part where",
        "segment that",
        "topic of",
        "explain the concept of",
        "tell me about the",
        "what does it say about"
    ]):
        return ChunkLevel.SECTION, "Query targets specific sections or topics"

    # Check for procedure-level indicators
    if any(phrase in query.lower() for phrase in [
        "how to",
        "steps to",
        "procedure for",
        "process of",
        "instructions for",
        "steps for",
        "how do i",
        "how do you"
    ]):
        return ChunkLevel.PROCEDURE, "Query asks about specific procedures or how-to information"

    # Check for step-level indicators
    if any(phrase in query.lower() for phrase in [
        "step by step",
        "specific step",
        "parameter for",
        "configuration of",
        "settings for",
        "value of",
        "parameter"
    ]):
        return ChunkLevel.STEP, "Query requests detailed steps or parameter information"

    # Default to section level for most queries
    return ChunkLevel.SECTION, "Default chunk level for general queries"

def determine_embedding_type(query: str) -> Tuple[EmbeddingType, str]:
    """
    Determine the appropriate embedding type based on query content.

    Args:
        query: User query

    Returns:
        Tuple of (preferred embedding type, reasoning)
    """
    # Check for conceptual understanding queries
    if any(phrase in query.lower() for phrase in [
        "what is",
        "concept of",
        "explain",
        "describe",
        "definition of",
        "understand",
        "overview of",
        "introduction to"
    ]):
        return EmbeddingType.CONCEPTUAL, "Query seeks conceptual understanding"

    # Check for task-oriented queries
    if any(phrase in query.lower() for phrase in [
        "how to",
        "steps to",
        "procedure for",
        "process of",
        "instructions for",
        "perform",
        "execute",
        "implement"
    ]):
        return EmbeddingType.TASK, "Query asks how to perform a task"

    # Check for technical parameter queries
    if any(phrase in query.lower() for phrase in [
        "parameter",
        "setting",
        "value",
        "configure",
        "specification",
        "technical details",
        "measurement",
        "dimensions"
    ]):
        return EmbeddingType.TECHNICAL, "Query seeks technical details or parameters"

    # Default to general for most queries
    return EmbeddingType.GENERAL, "Default embedding type for general queries"

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
