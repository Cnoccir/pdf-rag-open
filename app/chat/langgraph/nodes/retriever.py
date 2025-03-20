"""
Retriever node for LangGraph-based PDF RAG system.
Updated to use UnifiedVectorStore with multi-level chunking and multi-embedding.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType, RetrievalState
from app.chat.types import ChunkLevel, EmbeddingType, ContentElement
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

def retrieve_content(state: GraphState) -> GraphState:
    """
    Retrieve content based on query analysis.
    Enhanced to use multi-level chunking and multi-embedding strategies.

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
    preferred_chunk_level = state.query_state.preferred_chunk_level
    preferred_embedding_type = state.query_state.preferred_embedding_type
    procedure_focused = state.query_state.procedure_focused
    parameter_query = state.query_state.parameter_query

    logger.info(f"Retrieval started: strategy={retrieval_strategy}, pdf_ids={pdf_ids}, "
                f"chunk_level={preferred_chunk_level}, embedding_type={preferred_embedding_type}")

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
                strategies_used=[retrieval_strategy.value],
                chunk_levels_used=[],
                embedding_types_used=[]
            )
            return state

        # Determine if we have multiple PDFs (research mode)
        is_research_mode = len(pdf_ids) > 1

        # Set up retrieval parameters
        k = 10  # Number of results to return

        # Build filter dictionary for MongoDB + Qdrant
        filter_dict = {}

        # Add content type filter
        if content_types:
            filter_dict["content_type"] = content_types

        # Add chunk level filter if specified
        if preferred_chunk_level:
            filter_dict["chunk_level"] = str(preferred_chunk_level)

        # Add embedding type filter if specified
        if preferred_embedding_type:
            filter_dict["embedding_type"] = str(preferred_embedding_type)

        # Handle procedure and parameter specific filters
        if procedure_focused:
            filter_dict["content_type"] = ["procedure"]
        elif parameter_query:
            filter_dict["content_type"] = ["parameter"]

        # Prepare to collect results
        all_results = []
        chunk_levels_used = set()
        embedding_types_used = set()

        # Process each PDF ID
        for pdf_id in pdf_ids:
            filter_dict["pdf_id"] = pdf_id

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
            elif retrieval_strategy == RetrievalStrategy.PROCEDURE:
                logger.info(f"Using procedure search for {pdf_id}")
                # For procedure search, prioritize procedure content and task embeddings
                procedure_filter = filter_dict.copy()
                procedure_filter["content_type"] = ["procedure"]
                procedure_filter["embedding_type"] = str(EmbeddingType.TASK)

                results = vector_store.semantic_search(
                    query=query,
                    k=k,
                    filter_dict=procedure_filter
                )

                # If not enough procedure results, fall back to hybrid
                if len(results) < 3:
                    logger.info(f"Insufficient procedure results, using hybrid search")
                    results = vector_store.hybrid_search(
                        query=query,
                        k=k,
                        filter_dict=filter_dict
                    )
            else:
                # Default to hybrid search
                logger.info(f"Using hybrid search for {pdf_id}")
                results = vector_store.hybrid_search(
                    query=query,
                    k=k,
                    filter_dict=filter_dict
                )

            # Track which chunk levels and embedding types were retrieved
            for doc in results:
                metadata = doc.metadata

                # Track chunk level if present
                if "chunk_level" in metadata:
                    chunk_levels_used.add(metadata["chunk_level"])

                # Track embedding type if present
                if "embedding_type" in metadata:
                    embedding_types_used.add(metadata["embedding_type"])

            # Add to combined results
            all_results.extend(results)

        # Sort results by score if we have multiple documents
        if is_research_mode and len(all_results) > 0:
            all_results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

        # Limit results
        all_results = all_results[:k]

        # Transform results into standard format for the LangGraph
        elements = []
        sources = []
        procedures_retrieved = []
        parameters_retrieved = []

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

            # Create element dictionary for the GraphState
            element = {
                "id": metadata.get("element_id", ""),
                "element_id": metadata.get("element_id", ""),
                "content": doc.page_content,
                "content_type": content_type_str,
                "pdf_id": doc_pdf_id,
                "page": metadata.get("page_number", 0),
                "metadata": {
                    "score": metadata.get("score", 0.0),
                    "section": metadata.get("section", ""),
                    "content_type": content_type_str,
                    "document_title": metadata.get("document_title", f"Document {doc_pdf_id}"),
                    "chunk_level": metadata.get("chunk_level", ""),
                    "embedding_type": metadata.get("embedding_type", "")
                }
            }

            elements.append(element)

            # Add to sources for citation tracking
            sources.append({
                "id": metadata.get("element_id", ""),
                "pdf_id": doc_pdf_id,
                "page_number": metadata.get("page_number", 0),
                "section": metadata.get("section", ""),
                "score": metadata.get("score", 0.0),
                "document_title": metadata.get("document_title", f"Document {doc_pdf_id}"),
                "content_type": content_type_str
            })

            # Keep track of procedure and parameter results separately
            if content_type_str == "procedure":
                procedures_retrieved.append(element)
            elif content_type_str == "parameter":
                parameters_retrieved.append(element)

        # Create retrieval state with enhanced metadata
        state.retrieval_state = RetrievalState(
            elements=elements,
            sources=sources,
            strategies_used=[retrieval_strategy.value],
            metadata={
                "retrieval_time": datetime.now().isoformat(),
                "total_results": len(elements),
                "strategy": retrieval_strategy.value,
                "research_mode": is_research_mode,
                "preferred_chunk_level": str(preferred_chunk_level) if preferred_chunk_level else None,
                "preferred_embedding_type": str(preferred_embedding_type) if preferred_embedding_type else None,
                "procedure_focused": procedure_focused,
                "parameter_query": parameter_query
            },
            chunk_levels_used=list(chunk_levels_used),
            embedding_types_used=list(embedding_types_used),
            procedures_retrieved=procedures_retrieved,
            parameters_retrieved=parameters_retrieved
        )

        logger.info(f"Retrieval complete: found {len(elements)} elements with "
                   f"chunk levels {chunk_levels_used} and embedding types {embedding_types_used}")

    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}", exc_info=True)

        # Create a minimal retrieval state with error information
        state.retrieval_state = RetrievalState(
            elements=[],
            sources=[],
            strategies_used=[retrieval_strategy.value] if retrieval_strategy else [],
            metadata={"error": str(e)},
            chunk_levels_used=[],
            embedding_types_used=[]
        )

    return state
