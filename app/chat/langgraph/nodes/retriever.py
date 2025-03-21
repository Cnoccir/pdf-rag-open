"""
Retriever node for LangGraph-based PDF RAG system.
Enhanced to use UnifiedVectorStore with multi-level chunking and multi-embedding.
Supports both single document and research modes.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from app.chat.langgraph.state import GraphState, RetrievalStrategy, ContentType, RetrievalState
from app.chat.types import ChunkLevel, EmbeddingType
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

# Update to app/chat/langgraph/nodes/retriever.py

def retrieve_content(state: GraphState) -> GraphState:
    """
    Retrieve content based on query analysis.
    Enhanced to handle errors and empty results gracefully.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with retrieved content
    """
    # Validate state
    if not state.query_state:
        logger.error("Query state is required for retrieval")
        state.retrieval_state = RetrievalState(
            elements=[],
            sources=[],
            metadata={"error": "Missing query state"},
            strategies_used=[],
            chunk_levels_used=[],
            embedding_types_used=[]
        )
        return state

    # Extract query parameters
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
        # Validate PDF IDs with better error handling
        if not pdf_ids or len(pdf_ids) == 0:
            logger.warning("No PDF IDs provided for retrieval")

            # Try to get PDF ID from conversation state as fallback
            if state.conversation_state and state.conversation_state.pdf_id:
                pdf_ids = [state.conversation_state.pdf_id]
                logger.info(f"Using PDF ID from conversation state: {pdf_ids}")
            else:
                state.retrieval_state = RetrievalState(
                    elements=[],
                    sources=[],
                    metadata={"error": "No PDF IDs provided for retrieval"},
                    strategies_used=[retrieval_strategy.value] if retrieval_strategy else [],
                    chunk_levels_used=[],
                    embedding_types_used=[]
                )
                return state

        # Get content filter types if specified
        content_types = None
        if focused_elements:
            # Convert element strings to proper content types
            content_types = [elem for elem in focused_elements if isinstance(elem, str)]

        # Check for procedure and parameter specific filters
        if procedure_focused:
            content_types = ["procedure"]
        elif parameter_query:
            content_types = ["parameter"]

        # Get vector store instance with error handling
        try:
            vector_store = get_vector_store()

            # Check if vector store is initialized
            if not vector_store._initialized:
                success = vector_store.initialize()
                if not success:
                    raise Exception("Failed to initialize vector store")
        except Exception as vs_error:
            logger.error(f"Vector store initialization error: {str(vs_error)}")
            state.retrieval_state = RetrievalState(
                elements=[],
                sources=[],
                metadata={"error": f"Vector store error: {str(vs_error)}"},
                strategies_used=[retrieval_strategy.value] if retrieval_strategy else [],
                chunk_levels_used=[],
                embedding_types_used=[]
            )
            return state

        # Determine if we're in research mode (multiple PDFs)
        is_research_mode = len(pdf_ids) > 1

        # Set up retrieval parameters
        k = 10  # Standard number of results to return
        if is_research_mode:
            # In research mode, get fewer results per document
            k = max(5, 20 // len(pdf_ids))

        # Execute retrieval based on strategy with retry mechanism
        all_results = []
        procedures_retrieved = []
        parameters_retrieved = []
        chunk_levels_used = set()
        embedding_types_used = set()

        # Track retrieval attempts
        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries and not all_results:
            try:
                # Perform retrieval based on strategy
                if retrieval_strategy == RetrievalStrategy.SEMANTIC:
                    for pdf_id in pdf_ids:
                        results = vector_store.semantic_search(
                            query=query,
                            k=k,
                            pdf_id=pdf_id,
                            content_types=content_types,
                            chunk_level=preferred_chunk_level,
                            embedding_type=preferred_embedding_type
                        )
                        all_results.extend(results)

                        # Track chunk levels and embedding types
                        for doc in results:
                            metadata = doc.metadata
                            if "chunk_level" in metadata:
                                chunk_levels_used.add(metadata["chunk_level"])
                            if "embedding_type" in metadata:
                                embedding_types_used.add(metadata["embedding_type"])

                elif retrieval_strategy == RetrievalStrategy.KEYWORD:
                    for pdf_id in pdf_ids:
                        results = vector_store.mongo_store.keyword_search(
                            query=query,
                            pdf_id=pdf_id,
                            content_types=content_types,
                            limit=k
                        )

                        # Convert to Document format for consistency
                        for result in results:
                            doc = Document(
                                page_content=result.get("content", ""),
                                metadata={k: v for k, v in result.items() if k != "content"}
                            )
                            all_results.append(doc)

                elif retrieval_strategy == RetrievalStrategy.PROCEDURE:
                    # For procedure search, first try to get procedure content
                    for pdf_id in pdf_ids:
                        # Get procedure results from vector store
                        results = vector_store.semantic_search(
                            query=query,
                            k=k,
                            pdf_id=pdf_id,
                            content_types=["procedure"],
                            embedding_type=EmbeddingType.TASK
                        )
                        all_results.extend(results)

                        # Get procedure details from MongoDB
                        procedures = vector_store.get_procedures_by_pdf_id(pdf_id)
                        if procedures:
                            procedures_retrieved.extend(procedures)

                            # Get related parameters for these procedures
                            for proc in procedures:
                                proc_id = proc.get("procedure_id")
                                if proc_id:
                                    params = vector_store.get_parameters_by_pdf_id(
                                        pdf_id=pdf_id,
                                        procedure_id=proc_id
                                    )
                                    parameters_retrieved.extend(params)

                        # Track chunk levels and embedding types
                        for doc in results:
                            metadata = doc.metadata
                            if "chunk_level" in metadata:
                                chunk_levels_used.add(metadata["chunk_level"])
                            if "embedding_type" in metadata:
                                embedding_types_used.add(metadata["embedding_type"])

                    # If we don't have enough results, fall back to hybrid search
                    if len(all_results) < 3:
                        additional_results = []
                        for pdf_id in pdf_ids:
                            results = vector_store.hybrid_search(
                                query=query,
                                k=k,
                                pdf_ids=[pdf_id],
                                content_types=content_types,
                                chunk_level=ChunkLevel.PROCEDURE
                            )
                            additional_results.extend(results)

                        # Add any new results
                        existing_ids = set(doc.metadata.get("element_id") for doc in all_results)
                        for doc in additional_results:
                            if doc.metadata.get("element_id") not in existing_ids:
                                all_results.append(doc)

                                # Track additional chunk levels and embedding types
                                metadata = doc.metadata
                                if "chunk_level" in metadata:
                                    chunk_levels_used.add(metadata["chunk_level"])
                                if "embedding_type" in metadata:
                                    embedding_types_used.add(metadata["embedding_type"])

                else:
                    # Default to hybrid search
                    all_results = vector_store.hybrid_search(
                        query=query,
                        k=k * len(pdf_ids),  # Get more results for hybrid search
                        pdf_ids=pdf_ids,
                        content_types=content_types,
                        chunk_level=preferred_chunk_level,
                        embedding_type=preferred_embedding_type
                    )

                    # Track chunk levels and embedding types from hybrid results
                    for doc in all_results:
                        metadata = doc.metadata
                        if "chunk_level" in metadata:
                            chunk_levels_used.add(metadata["chunk_level"])
                        if "embedding_type" in metadata:
                            embedding_types_used.add(metadata["embedding_type"])

                # For parameter queries, fetch parameters directly
                if parameter_query and not parameters_retrieved:
                    for pdf_id in pdf_ids:
                        params = vector_store.get_parameters_by_pdf_id(pdf_id)
                        parameters_retrieved.extend(params)

                # If we got results, break retry loop
                if all_results or procedures_retrieved or parameters_retrieved:
                    break

                # If no results on first attempt, try broadening search on next attempt
                if retry_count == 0:
                    logger.warning(f"No results found, trying broader search (retry {retry_count+1})")
                    # Remove content type filter
                    content_types = None
                    # Try different chunk level
                    if preferred_chunk_level == ChunkLevel.STEP:
                        preferred_chunk_level = ChunkLevel.PROCEDURE
                    elif preferred_chunk_level == ChunkLevel.PROCEDURE:
                        preferred_chunk_level = ChunkLevel.SECTION
                    # Try general embedding type
                    preferred_embedding_type = EmbeddingType.GENERAL
                elif retry_count == 1:
                    logger.warning(f"Still no results, trying keyword fallback (retry {retry_count+1})")
                    # Fall back to keyword search
                    retrieval_strategy = RetrievalStrategy.KEYWORD

                retry_count += 1

            except Exception as retrieval_error:
                logger.error(f"Error in {retrieval_strategy} retrieval: {str(retrieval_error)}")
                retry_count += 1

                # If this is the last attempt, use keyword search as fallback
                if retry_count == max_retries and not all_results:
                    logger.warning("Trying keyword search as final fallback")
                    try:
                        # Use keyword search as last resort
                        for pdf_id in pdf_ids:
                            results = vector_store.mongo_store.keyword_search(
                                query=query,
                                pdf_id=pdf_id,
                                content_types=None,  # No content type filter for broader results
                                limit=k
                            )

                            # Convert to Document format
                            for result in results:
                                doc = Document(
                                    page_content=result.get("content", ""),
                                    metadata={k: v for k, v in result.items() if k != "content"}
                                )
                                all_results.append(doc)
                    except Exception as keyword_error:
                        logger.error(f"Error in keyword fallback: {str(keyword_error)}")

        # If we still have no results at all
        if not all_results and not procedures_retrieved and not parameters_retrieved:
            logger.warning(f"No results found for query: {query[:50]}...")

            # Create minimal retrieval state with warning
            state.retrieval_state = RetrievalState(
                elements=[],
                sources=[],
                strategies_used=[retrieval_strategy.value] if retrieval_strategy else ["hybrid"],
                metadata={
                    "warning": "No relevant content found for this query",
                    "retrieval_time": datetime.now().isoformat(),
                    "pdf_ids": pdf_ids,
                    "content_types": content_types,
                },
                chunk_levels_used=list(chunk_levels_used),
                embedding_types_used=list(embedding_types_used),
                procedures_retrieved=procedures_retrieved,
                parameters_retrieved=parameters_retrieved
            )
            return state

        # Sort results by score if we have multiple documents
        if is_research_mode and len(all_results) > 0:
            all_results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

        # Limit results
        all_results = all_results[:15]  # Cap at 15 results total

        # Transform results into standard format for LangGraph
        elements = []
        sources = []

        for doc in all_results:
            # Extract metadata
            metadata = doc.metadata
            doc_pdf_id = metadata.get("pdf_id", "")

            # Determine content type
            content_type_str = metadata.get("content_type", "text")

            # Create element dictionary for the GraphState
            element = {
                "id": metadata.get("element_id", ""),
                "element_id": metadata.get("element_id", ""),
                "content": doc.page_content,
                "content_type": content_type_str,
                "pdf_id": doc_pdf_id,
                "page": metadata.get("page_number", 0) or metadata.get("page", 0),
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
                "page_number": metadata.get("page_number", 0) or metadata.get("page", 0),
                "section": metadata.get("section", ""),
                "score": metadata.get("score", 0.0),
                "document_title": metadata.get("document_title", f"Document {doc_pdf_id}"),
                "content_type": content_type_str
            })

        # If we're in research mode, add cross-document information
        cross_document_info = {}
        if is_research_mode:
            shared_concepts = vector_store.find_shared_concepts(pdf_ids)
            if shared_concepts:
                cross_document_info["shared_concepts"] = shared_concepts

            # Get document titles
            document_titles = {}
            for pdf_id in pdf_ids:
                doc_metadata = vector_store.get_document_metadata(pdf_id)
                if doc_metadata:
                    document_titles[pdf_id] = doc_metadata.get("title", f"Document {pdf_id}")
                else:
                    document_titles[pdf_id] = f"Document {pdf_id}"

            cross_document_info["document_titles"] = document_titles

        # Create retrieval state with enhanced metadata
        state.retrieval_state = RetrievalState(
            elements=elements,
            sources=sources,
            strategies_used=[retrieval_strategy.value] if retrieval_strategy else ["hybrid"],
            metadata={
                "retrieval_time": datetime.now().isoformat(),
                "total_results": len(elements),
                "strategy": retrieval_strategy.value if retrieval_strategy else "hybrid",
                "research_mode": is_research_mode,
                "retry_count": retry_count,
                "preferred_chunk_level": str(preferred_chunk_level) if preferred_chunk_level else None,
                "preferred_embedding_type": str(preferred_embedding_type) if preferred_embedding_type else None,
                "procedure_focused": procedure_focused,
                "parameter_query": parameter_query,
                "cross_document_info": cross_document_info if is_research_mode else {}
            },
            chunk_levels_used=list(chunk_levels_used),
            embedding_types_used=list(embedding_types_used),
            procedures_retrieved=procedures_retrieved,
            parameters_retrieved=parameters_retrieved
        )

        logger.info(f"Retrieval complete: found {len(elements)} elements, "
                   f"{len(procedures_retrieved)} procedures, {len(parameters_retrieved)} parameters "
                   f"with chunk levels {chunk_levels_used} and embedding types {embedding_types_used}")

    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}", exc_info=True)

        # Create a minimal retrieval state with error information
        state.retrieval_state = RetrievalState(
            elements=[],
            sources=[],
            strategies_used=[retrieval_strategy.value] if retrieval_strategy else [],
            metadata={"error": str(e)},
            chunk_levels_used=[],
            embedding_types_used=[],
            procedures_retrieved=[],
            parameters_retrieved=[]
        )

    return state
