"""
Query analyzer node for LangGraph-based PDF RAG system.
Enhanced to support multi-level chunking and multi-embedding strategies.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType
from app.chat.types import ChunkLevel, EmbeddingType
from app.chat.utils.extraction import extract_technical_terms

logger = logging.getLogger(__name__)

def analyze_query(state: GraphState) -> dict:
    """
    Analyze a user query to determine optimal retrieval strategy.
    Enhanced with chunk level and embedding type selection.

    Args:
        state: Current graph state

    Returns:
        Dictionary with updated query_state
    """
    try:
        # Validate state
        if not state.query_state:
            logger.error("Query state is required for analysis")
            return {"query_state": state.query_state}

        # Get query from state
        query = state.query_state.query.lower() if state.query_state.query else ""
        logger.info(f"Analyzing query: {query[:50]}...")

        # Extract keywords with enhanced stop word removal
        stop_words = {
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "by",
            "with", "for", "to", "from", "is", "are", "was", "were", "be", "been",
            "about", "have", "has", "had", "this", "these", "those", "that"
        }
        words = re.findall(r'\b[a-z0-9]+\b', query.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Update keywords in state
        state.query_state.keywords = keywords

        # Detect specific content types
        focused_elements = []

        # Check for table references
        if contains_pattern(query, [
            r'\btable\b', r'\btabular\b', r'\bcolumn\b', r'\brow\b', r'\bcell\b',
            r'\bspreadsheet\b', r'\bgrid\b', r'\bmatrix\b'
        ]):
            focused_elements.append(ContentType.TABLE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for image references
        elif contains_pattern(query, [
            r'\bimage\b', r'\bfigure\b', r'\bdiagram\b', r'\bchart\b',
            r'\bpicture\b', r'\bphoto\b', r'\bgraph\b', r'\bdrawing\b'
        ]):
            focused_elements.append(ContentType.IMAGE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for code references
        elif contains_pattern(query, [
            r'\bcode\b', r'\bfunction\b', r'\bclass\b', r'\bprogram\b',
            r'\bscript\b', r'\bapi\b', r'\bsyntax\b', r'\bmethod\b'
        ]):
            focused_elements.append(ContentType.CODE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for mathematical content
        elif contains_pattern(query, [
            r'\bequation\b', r'\bformula\b', r'\bmath\b', r'\bcalculation\b',
            r'\balgebra\b', r'\bcompute\b', r'\bnumerical\b'
        ]):
            focused_elements.append(ContentType.EQUATION.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check for procedure references
        elif contains_pattern(query, [
            r'\bprocedure\b', r'\bstep\b', r'\bhow to\b', r'\bprocess\b',
            r'\binstruction\b', r'\btask\b', r'\bfollow\b', r'\bsequence\b',
            r'\bworkflow\b', r'\boperation\b'
        ]):
            focused_elements.append(ContentType.PROCEDURE.value)
            state.query_state.retrieval_strategy = RetrievalStrategy.PROCEDURE
            state.query_state.procedure_focused = True

        # Check for parameter references
        elif contains_pattern(query, [
            r'\bparameter\b', r'\bsetting\b', r'\bvalue\b', r'\bconfiguration\b',
            r'\boption\b', r'\bvariable\b', r'\bproperty\b', r'\battribute\b',
            r'\bfield\b', r'\bsettings\b'
        ]):
            focused_elements.append(ContentType.PARAMETER.value)
            state.query_state.parameter_query = True
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

        # Check if query is best suited for concept search
        elif contains_pattern(query, [
            r'\brelated\b', r'\bconnection\b', r'\brelationship\b', r'\bsimilar\b',
            r'\blinked\b', r'\bassociated\b', r'\bcorrelated\b'
        ]):
            state.query_state.retrieval_strategy = RetrievalStrategy.CONCEPT

        # Check if query is best suited for semantic search
        elif contains_pattern(query, [
            r'\babout\b', r'\blike\b', r'\bmeaning\b', r'\bexplain\b',
            r'\bunderstand\b', r'\bconcept\b', r'\bdescribe\b', r'\bsummarize\b'
        ]):
            state.query_state.retrieval_strategy = RetrievalStrategy.SEMANTIC

        # Check if query is best suited for keyword search
        elif contains_pattern(query, [
            r'\bexactly\b', r'\bspecific\b', r'\bfind\b', r'\blocate\b', r'\bwhere\b',
            r'\bmentions\b', r'\bcontains\b', r'\blists\b'
        ]):
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
            state.query_state.query_type = "listing"
        else:
            state.query_state.query_type = "general"

        # Extract technical concepts with enhanced extraction
        technical_concepts = extract_technical_terms(query)

        # Update concepts in state
        state.query_state.concepts = technical_concepts

        # Check for research mode indicators - cross-document analysis
        is_research_mode = len(state.query_state.pdf_ids) > 1 if state.query_state.pdf_ids else False

        # Also check query for explicit research indicators
        if not is_research_mode and contains_pattern(query, [
            r'\bacross documents\b', r'\bcompare documents\b', r'\bmultiple documents\b',
            r'\bdocument comparison\b', r'\bcross-reference\b', r'\bresearch mode\b'
        ]):
            # Mark as research-oriented even if not explicitly in research mode
            state.query_state.metadata = state.query_state.metadata or {}
            state.query_state.metadata["research_oriented"] = True

        # Update metadata
        if not state.query_state.metadata:
            state.query_state.metadata = {}

        state.query_state.metadata["analyzed_at"] = datetime.now().isoformat()
        state.query_state.metadata["strategy_reasoning"] = f"Selected {state.query_state.retrieval_strategy} based on query patterns"
        state.query_state.metadata["chunk_level_reasoning"] = chunk_reasoning
        state.query_state.metadata["embedding_type_reasoning"] = embedding_reasoning
        state.query_state.metadata["is_research_mode"] = is_research_mode

        logger.info(f"Query analysis: strategy={state.query_state.retrieval_strategy}, "
                   f"type={state.query_state.query_type}, "
                   f"chunk_level={state.query_state.preferred_chunk_level}, "
                   f"embedding_type={state.query_state.preferred_embedding_type}, "
                   f"research_mode={is_research_mode}, "
                   f"concepts={technical_concepts[:5] if technical_concepts else []}")

        return {"query_state": state.query_state}

    except Exception as e:
        logger.error(f"Query analysis error: {str(e)}", exc_info=True)

        # Ensure we have valid query state with defaults
        if state.query_state:
            state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID
            state.query_state.query_type = "general"
            state.query_state.preferred_chunk_level = ChunkLevel.SECTION
            state.query_state.preferred_embedding_type = EmbeddingType.GENERAL

        return {"query_state": state.query_state}

def contains_pattern(text: str, patterns: List[str]) -> bool:
    """
    Check if text contains any of the specified regex patterns.

    Args:
        text: Text to check
        patterns: List of regex patterns

    Returns:
        True if any pattern matches, False otherwise
    """
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def determine_chunk_level(query: str) -> Tuple[ChunkLevel, str]:
    """
    Determine the appropriate chunk level based on query content.

    Args:
        query: User query

    Returns:
        Tuple of (preferred chunk level, reasoning)
    """
    # Check for document-level indicators (broad questions about the entire document)
    if contains_pattern(query, [
        r"what is this document about",
        r"summary of the document",
        r"overview of",
        r"main topics",
        r"generally describe",
        r"what does this document cover",
        r"main points of",
        r"summary of",
        r"gist of",
        r"purpose of this document"
    ]):
        return ChunkLevel.DOCUMENT, "Query requests document-level information"

    # Check for section-level indicators
    if contains_pattern(query, [
        r"section on",
        r"chapter about",
        r"part where",
        r"segment that",
        r"topic of",
        r"explain the concept of",
        r"tell me about the",
        r"what does it say about",
        r"details on",
        r"information about"
    ]):
        return ChunkLevel.SECTION, "Query targets specific sections or topics"

    # Check for procedure-level indicators
    if contains_pattern(query, [
        r"how to",
        r"steps to",
        r"procedure for",
        r"process of",
        r"instructions for",
        r"steps for",
        r"how do i",
        r"how do you",
        r"methodology for",
        r"implementation of"
    ]):
        return ChunkLevel.PROCEDURE, "Query asks about specific procedures or how-to information"

    # Check for step-level indicators
    if contains_pattern(query, [
        r"step by step",
        r"specific step",
        r"parameter for",
        r"configuration of",
        r"settings for",
        r"value of",
        r"parameter",
        r"attribute",
        r"detailed instructions",
        r"configuration parameter"
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
    if contains_pattern(query, [
        r"what is",
        r"concept of",
        r"explain",
        r"describe",
        r"definition of",
        r"understand",
        r"overview of",
        r"introduction to",
        r"meaning of",
        r"theory behind"
    ]):
        return EmbeddingType.CONCEPTUAL, "Query seeks conceptual understanding"

    # Check for task-oriented queries
    if contains_pattern(query, [
        r"how to",
        r"steps to",
        r"procedure for",
        r"process of",
        r"instructions for",
        r"perform",
        r"execute",
        r"implement",
        r"accomplish",
        r"achieve"
    ]):
        return EmbeddingType.TASK, "Query asks how to perform a task"

    # Check for technical parameter queries
    if contains_pattern(query, [
        r"parameter",
        r"setting",
        r"value",
        r"configure",
        r"specification",
        r"technical details",
        r"measurement",
        r"dimensions",
        r"property",
        r"attribute"
    ]):
        return EmbeddingType.TECHNICAL, "Query seeks technical details or parameters"

    # Default to general for most queries
    return EmbeddingType.GENERAL, "Default embedding type for general queries"
