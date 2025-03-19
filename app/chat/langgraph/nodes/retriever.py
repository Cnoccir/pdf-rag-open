"""
Retriever node for LangGraph-based PDF RAG system.
Simplified implementation with improved async/sync handling.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType
from app.chat.vector_stores import get_vector_store
from app.chat.types import ContentElement

logger = logging.getLogger(__name__)

def retrieve_content(state: GraphState) -> GraphState:
    """
    Retrieve content based on query analysis.
    Simplified implementation with proper connection handling.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with retrieved content
    """
    # Validate state
    if not state.query_state:
        logger.error("Query state is required for retrieval")
        return state

    query = state.query_state.query
    pdf_ids = state.query_state.pdf_ids or []
    retrieval_strategy = state.query_state.retrieval_strategy
    focused_elements = state.query_state.focused_elements

    logger.info(f"Retrieval started: strategy={retrieval_strategy}, pdf_ids={pdf_ids}")

    try:
        # Get content filter types if specified
        content_types = None
        if focused_elements:
            # Convert element strings to proper content types
            content_types = [elem for elem in focused_elements if isinstance(elem, str)]

        # Get vector store instance
        vector_store = get_vector_store()

        # Validate PDF IDs
        if not pdf_ids or len(pdf_ids) == 0:
            logger.warning("No PDF IDs provided for retrieval")
            state.retrieval_state = RetrievalState(
                elements=[],
                sources=[],
                metadata={"error": "No PDF IDs provided for retrieval"},
                strategies_used=[retrieval_strategy.value]
            )
            return state

        # Determine if we have multiple PDFs (research mode)
        is_research_mode = len(pdf_ids) > 1

        # Set up retrieval parameters
        k = 10  # Number of results to return

        # Prepare to collect results
        all_results = []

        # Process each PDF ID
        for pdf_id in pdf_ids:
            # Choose retrieval method based on strategy
            if retrieval_strategy == RetrievalStrategy.SEMANTIC:
                logger.info(f"Using semantic search for {pdf_id}")
                results = vector_store.semantic_search(
                    query=query,
                    k=k,
                    pdf_id=pdf_id,
                    content_types=content_types
                )
            elif retrieval_strategy == RetrievalStrategy.KEYWORD:
                logger.info(f"Using keyword search for {pdf_id}")
                results = vector_store.keyword_search(
                    query=query,
                    k=k,
                    pdf_id=pdf_id
                )
            else:
                # Default to hybrid search
                logger.info(f"Using hybrid search for {pdf_id}")
                results = vector_store.hybrid_search(
                    query=query,
                    k=k,
                    pdf_id=pdf_id,
                    content_types=content_types
                )

            # Add document title if available
            for doc in results:
                # Ensure pdf_id is in metadata
                if not doc.metadata.get("pdf_id"):
                    doc.metadata["pdf_id"] = pdf_id

            # Add to combined results
            all_results.extend(results)

        # Sort results by score if we have multiple documents
        if is_research_mode and len(all_results) > 0:
            all_results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

        # Limit results
        all_results = all_results[:k]

        # Transform results into standard ContentElement format
        elements = []
        sources = []

        for doc in all_results:
            # Extract metadata
            metadata = doc.metadata
            doc_pdf_id = metadata.get("pdf_id", "")

            # Determine content type
            content_type_str = metadata.get("content_type", "text")

            try:
                content_type = ContentType(content_type_str)
            except:
                content_type = ContentType.TEXT

            # Create element dictionary (simpler than full ContentElement)
            element = {
                "id": metadata.get("id", ""),
                "element_id": metadata.get("id", ""),
                "content": doc.page_content,
                "content_type": content_type.value,
                "pdf_id": doc_pdf_id,
                "page": metadata.get("page_number", 0),
                "metadata": {
                    "score": metadata.get("score", 0.0),
                    "section": metadata.get("section", ""),
                    "content_type": content_type_str,
                    "document_title": metadata.get("document_title", f"Document {doc_pdf_id}")
                }
            }

            elements.append(element)

            # Add to sources for citation tracking
            sources.append({
                "id": metadata.get("id", ""),
                "pdf_id": doc_pdf_id,
                "page_number": metadata.get("page_number", 0),
                "section": metadata.get("section", ""),
                "score": metadata.get("score", 0.0),
                "document_title": metadata.get("document_title", f"Document {doc_pdf_id}")
            })

        # Create retrieval state
        state.retrieval_state = RetrievalState(
            elements=elements,
            sources=sources,
            strategies_used=[retrieval_strategy.value],
            metadata={
                "retrieval_time": datetime.now().isoformat(),
                "total_results": len(elements),
                "strategy": retrieval_strategy.value,
                "research_mode": is_research_mode
            }
        )

        logger.info(f"Retrieval complete: found {len(elements)} elements")

    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}", exc_info=True)

        # Create a minimal retrieval state with error information
        state.retrieval_state = RetrievalState(
            elements=[],
            sources=[],
            strategies_used=[retrieval_strategy.value] if retrieval_strategy else [],
            metadata={"error": str(e)}
        )

    return state
