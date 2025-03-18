"""
Query analyzer node for LangGraph-based PDF RAG system.
This node determines the optimal retrieval strategy for a given query.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from openai import AsyncOpenAI

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType

logger = logging.getLogger(__name__)

QUERY_ANALYSIS_SYSTEM_PROMPT = """You are an AI specialized in analyzing search queries to determine the optimal retrieval strategy.
Given a user question, you need to:
1. Identify if this is a general question or a specific lookup
2. Determine the best retrieval strategy based on the query content
3. Detect if the query refers to specific content types (text, tables, images)
4. Extract key concepts and keywords

Retrieval Strategies:
- SEMANTIC: For conceptual questions that require deep understanding
- KEYWORD: For specific term lookups
- HYBRID: Combination of semantic and keyword (default)
- CONCEPT: For questions about relationships between ideas
- TABLE: For questions specifically about tabular data
- IMAGE: For questions about visual information
- COMBINED: When multiple strategies would be helpful

Content Types:
- TEXT: General text information
- TABLE: Structured tabular data
- FIGURE: Images, diagrams, charts
- EQUATION: Mathematical formulas
- CODE: Programming code snippets
"""

QUERY_ANALYSIS_PROMPT = """Analyze the following user query: "{query}"

Determine:
- retrieval_strategy: The optimal retrieval strategy from [SEMANTIC, KEYWORD, HYBRID, CONCEPT, TABLE, IMAGE, COMBINED]
- query_type: The general intent (e.g., "lookup", "understanding", "comparison", "procedural")
- focused_elements: List of content element types to focus on (e.g., "text", "table", "image")
"""

async def process_query(query: str, pdf_ids: List[str] = None) -> Dict[str, Any]:
    """
    Process a user query to determine the optimal retrieval strategy.

    Args:
        query: The user's query text
        pdf_ids: Optional list of PDF IDs to search within

    Returns:
        Dictionary with query analysis results
    """
    try:
        # Initialize query analysis response
        result = {
            "query": query,
            "pdf_ids": pdf_ids or [],
            "retrieval_strategy": RetrievalStrategy.HYBRID,  # default strategy
            "query_type": "general",
            "focused_elements": ["text"],
            "timestamp": datetime.utcnow().isoformat()
        }

        # Use OpenAI to analyze the query if it's complex enough
        if len(query.split()) > 3:  # Only analyze non-trivial queries
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # Prepare the prompt
            formatted_prompt = QUERY_ANALYSIS_PROMPT.format(query=query)

            # Call the model
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": QUERY_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            # Extract analysis from response
            analysis_text = response.choices[0].message.content

            # Parse the analysis (simple keyword-based parsing)
            if "SEMANTIC" in analysis_text:
                result["retrieval_strategy"] = RetrievalStrategy.SEMANTIC
            elif "KEYWORD" in analysis_text:
                result["retrieval_strategy"] = RetrievalStrategy.KEYWORD
            elif "CONCEPT" in analysis_text:
                result["retrieval_strategy"] = RetrievalStrategy.CONCEPT
            elif "TABLE" in analysis_text:
                result["retrieval_strategy"] = RetrievalStrategy.TABLE
                result["focused_elements"] = ["table"]
            elif "IMAGE" in analysis_text:
                result["retrieval_strategy"] = RetrievalStrategy.IMAGE
                result["focused_elements"] = ["figure"]
            elif "COMBINED" in analysis_text:
                result["retrieval_strategy"] = RetrievalStrategy.COMBINED

            # Extract query type if present
            if "query_type" in analysis_text.lower():
                query_type_line = [line for line in analysis_text.split('\n') if "query_type" in line.lower()]
                if query_type_line:
                    # Simple extraction, can be improved for robustness
                    query_type = query_type_line[0].split(":")[-1].strip().strip('"').lower()
                    result["query_type"] = query_type

            # Extract focused elements if present
            if "focused_elements" in analysis_text.lower():
                elements_line = [line for line in analysis_text.split('\n') if "focused_elements" in line.lower()]
                if elements_line:
                    # Simple extraction, can be improved for robustness
                    elements_text = elements_line[0].split(":")[-1].strip().lower()
                    elements = [e.strip().strip('"').strip("'").strip(",") for e in elements_text.split()]
                    if elements:
                        result["focused_elements"] = elements

        logger.info(f"Query analysis result: strategy={result['retrieval_strategy']}, type={result['query_type']}")
        return result

    except Exception as e:
        logger.error(f"Error in query analysis: {str(e)}")
        # Fall back to default analysis
        return {
            "query": query,
            "pdf_ids": pdf_ids or [],
            "retrieval_strategy": RetrievalStrategy.HYBRID,
            "query_type": "general",
            "focused_elements": ["text"],
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

def analyze(state: GraphState) -> GraphState:
    """
    Analyze a user query to determine the optimal retrieval strategy.

    Args:
        state: Current graph state

    Returns:
        Updated graph state
    """
    if not state.query_state:
        logger.error("Query state is required for query analysis")
        raise ValueError("Query state is required")

    # For now, use basic heuristics to determine retrieval strategy
    # In a production system, this would use a more sophisticated ML-based approach
    query = state.query_state.query.lower() if state.query_state.query else ""

    # Detect specific content types
    focused_elements = []
    if any(term in query for term in ["table", "row", "column", "cell", "tabular"]):
        focused_elements.append(ContentType.TABLE)
        state.query_state.retrieval_strategy = RetrievalStrategy.TABLE

    elif any(term in query for term in ["image", "figure", "diagram", "chart", "picture", "photo"]):
        focused_elements.append(ContentType.FIGURE)
        state.query_state.retrieval_strategy = RetrievalStrategy.IMAGE

    elif any(term in query for term in ["relate", "relationship", "concept", "connection", "between"]):
        state.query_state.retrieval_strategy = RetrievalStrategy.CONCEPT

    elif any(term in query for term in ["exact", "specific", "find", "locate", "where"]):
        state.query_state.retrieval_strategy = RetrievalStrategy.KEYWORD

    else:
        # Default to hybrid retrieval
        state.query_state.retrieval_strategy = RetrievalStrategy.HYBRID

    # Set focused elements
    state.query_state.focused_elements = focused_elements

    # Extract simple keywords (could be enhanced with NLP)
    import re
    words = re.findall(r'\b\w+\b', query)
    state.query_state.keywords = [word for word in words if len(word) > 3 and word not in [
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "about", "does", "this", "that", "these", "those", "have", "from"
    ]]

    # Set query type based on interrogative words
    if "how" in query:
        state.query_state.query_type = "procedural"
    elif "why" in query:
        state.query_state.query_type = "explanation"
    elif "compare" in query or "difference" in query or "versus" in query or "vs" in query:
        state.query_state.query_type = "comparison"
    elif any(w in query for w in ["what", "who", "when", "where"]):
        state.query_state.query_type = "factual"
    else:
        state.query_state.query_type = "general"

    # Log analysis results
    logger.info(f"Query analysis: strategy={state.query_state.retrieval_strategy}, " +
               f"type={state.query_state.query_type}, elements={focused_elements}")

    return state
