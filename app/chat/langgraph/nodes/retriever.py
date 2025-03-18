"""
Retriever node for LangGraph-based PDF RAG system.
This node handles document retrieval using various strategies based on query analysis,
with optimized Neo4j vector store integration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.chat.langgraph.state import QueryState, RetrievalState, GraphState, RetrievalStrategy, ContentType
from app.chat.vector_stores import Neo4jVectorStore, get_vector_store
from app.chat.types import ContentElement, SearchQuery

logger = logging.getLogger(__name__)

async def _retrieve_content_async(query_state: QueryState) -> RetrievalState:
    """
    Asynchronous implementation of content retrieval with improved Neo4j integration.

    Args:
        query_state: Query state with analysis

    Returns:
        Retrieval state with results
    """
    logger.info(f"Beginning retrieval using strategy: {query_state.retrieval_strategy.value} with {len(query_state.pdf_ids)} documents")

    # Initialize retrieval state
    retrieval_state = RetrievalState(
        strategies_used=[query_state.retrieval_strategy]
    )

    try:
        # Get vector store instance
        vector_store = get_vector_store()

        if not vector_store:
            raise ValueError("Neo4j vector store not available")

        if not vector_store.initialized:
            logger.info("Vector store not initialized, initializing...")
            await vector_store.initialize_database()

        # Check if we have multiple PDFs (research mode)
        pdf_ids = query_state.pdf_ids
        is_research_mode = len(pdf_ids) > 1

        if is_research_mode:
            logger.info(f"Research mode: searching across {len(pdf_ids)} documents")

        # Determine which retrieval method to use based on strategy
        start_time = datetime.utcnow()

        # Set up retrieval parameters
        k = 10  # Number of results per document

        # Determine filter content types if any
        filter_content_types = None
        if hasattr(query_state, 'focused_elements') and query_state.focused_elements:
            # Convert element strings to proper content types
            try:
                filter_content_types = [elem for elem in query_state.focused_elements if isinstance(elem, str)]
            except:
                filter_content_types = None

        if query_state.retrieval_strategy == RetrievalStrategy.TABLE:
            filter_content_types = ["table"]
        elif query_state.retrieval_strategy == RetrievalStrategy.IMAGE:
            filter_content_types = ["figure", "image"]

        # Log retrieval parameters
        logger.info(f"Retrieval parameters: strategy={query_state.retrieval_strategy.value}, "
                   f"filter_types={filter_content_types}, research_mode={is_research_mode}")

        # Perform retrieval based on research mode
        if is_research_mode:
            # Multi-document retrieval for research mode
            combined_results = []

            # Create search tasks for each document
            search_tasks = []

            for pdf_id in pdf_ids:
                if query_state.retrieval_strategy == RetrievalStrategy.SEMANTIC:
                    # Pure semantic search
                    search_tasks.append(vector_store.semantic_search(
                        query=query_state.query,
                        k=k,
                        pdf_id=pdf_id,
                        content_types=filter_content_types
                    ))
                elif query_state.retrieval_strategy == RetrievalStrategy.KEYWORD:
                    # Keyword-based search
                    search_tasks.append(vector_store.keyword_search(
                        query=query_state.query,
                        k=k,
                        pdf_id=pdf_id
                    ))
                elif query_state.retrieval_strategy == RetrievalStrategy.CONCEPT:
                    # Concept-based search
                    search_tasks.append(vector_store.concept_search(
                        query=query_state.query,
                        k=k,
                        pdf_id=pdf_id
                    ))
                else:
                    # Default to hybrid search
                    search_tasks.append(vector_store.hybrid_search(
                        query=query_state.query,
                        k=k,
                        pdf_id=pdf_id,
                        content_types=filter_content_types
                    ))

            # Execute all search tasks in parallel
            results_by_doc = await asyncio.gather(*search_tasks)

            # Process results from each document
            for i, results in enumerate(results_by_doc):
                curr_pdf_id = pdf_ids[i]

                # Add PDF ID to results metadata if not present
                for doc in results:
                    # Ensure pdf_id is in metadata
                    if not doc.metadata.get("pdf_id"):
                        doc.metadata["pdf_id"] = curr_pdf_id

                    # Add document name if available
                    if not doc.metadata.get("document_title"):
                        try:
                            async with vector_store.driver.session() as session:
                                result = await session.run(
                                    "MATCH (d:Document {pdf_id: $pdf_id}) RETURN d.title AS title",
                                    {"pdf_id": curr_pdf_id}
                                )
                                record = await result.single()
                                if record and "title" in record:
                                    doc.metadata["document_title"] = record["title"]
                        except Exception as e:
                            logger.error(f"Error fetching document title: {str(e)}")

                combined_results.extend(results)

            # Sort all results by score
            combined_results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

            # Limit to top K results overall
            results = combined_results[:k]
        else:
            # Single document search
            pdf_id = pdf_ids[0] if pdf_ids else None

            if query_state.retrieval_strategy == RetrievalStrategy.SEMANTIC:
                # Pure semantic search
                results = await vector_store.semantic_search(
                    query=query_state.query,
                    k=k,
                    pdf_id=pdf_id,
                    content_types=filter_content_types
                )
            elif query_state.retrieval_strategy == RetrievalStrategy.KEYWORD:
                # Keyword-based search
                results = await vector_store.keyword_search(
                    query=query_state.query,
                    k=k,
                    pdf_id=pdf_id
                )
            elif query_state.retrieval_strategy == RetrievalStrategy.CONCEPT:
                # Concept-based search
                results = await vector_store.concept_search(
                    query=query_state.query,
                    k=k,
                    pdf_id=pdf_id
                )
            else:
                # Default to hybrid search
                results = await vector_store.hybrid_search(
                    query=query_state.query,
                    k=k,
                    pdf_id=pdf_id,
                    content_types=filter_content_types
                )

        # Record timing metrics
        retrieval_time = (datetime.utcnow() - start_time).total_seconds()

        # Transform results into standard ContentElement format
        elements = []
        sources = []

        for doc in results:
            # Extract metadata
            metadata = doc.metadata
            pdf_id = metadata.get("pdf_id", "")

            # Determine content type - default to text if not specified
            content_type_str = metadata.get("content_type", "text")
            try:
                content_type = ContentType(content_type_str)
            except:
                content_type = ContentType.TEXT

            # Create ContentElement
            element = ContentElement(
                id=metadata.get("id", ""),
                element_id=metadata.get("id", ""),
                content=doc.page_content,
                content_type=content_type,
                pdf_id=pdf_id,
                page=metadata.get("page_number", 0),
                metadata={
                    "score": metadata.get("score", 0.0),
                    "section": metadata.get("section", ""),
                    "section_title": metadata.get("section_title", ""),
                    "document_title": metadata.get("document_title", f"Document {pdf_id}"),
                    "chunk_id": metadata.get("chunk_id", metadata.get("id", "")),
                }
            )
            elements.append(element)

            # Add to sources for citation tracking
            sources.append({
                "id": metadata.get("id", ""),
                "pdf_id": pdf_id,
                "page_number": metadata.get("page_number", 0),
                "section": metadata.get("section", ""),
                "section_title": metadata.get("section_title", ""),
                "score": metadata.get("score", 0.0),
                "document_title": metadata.get("document_title", f"Document {pdf_id}")
            })

        # Update retrieval state
        retrieval_state.elements = elements
        retrieval_state.sources = sources
        retrieval_state.metadata["retrieval_time"] = retrieval_time
        retrieval_state.metadata["total_results"] = len(elements)
        retrieval_state.metadata["strategy"] = query_state.retrieval_strategy.value
        retrieval_state.metadata["research_mode"] = is_research_mode

        logger.info(f"Retrieval complete: found {len(elements)} relevant elements in {retrieval_time:.2f}s")

    except Exception as e:
        error_msg = f"Error during retrieval: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        retrieval_state.metadata["error"] = error_msg

    return retrieval_state

def retrieve_content(state: GraphState) -> GraphState:
    """
    Retrieve content based on query analysis.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with retrieved content
    """
    # Validate state
    if not state.query_state:
        raise ValueError("Query state is required for retrieval")

    # Create a new event loop to run the async retrieval
    try:
        retrieval_state = asyncio.run(_retrieve_content_async(state.query_state))
    except RuntimeError:
        # If there's already an event loop running, create and use a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            retrieval_state = loop.run_until_complete(_retrieve_content_async(state.query_state))
        finally:
            loop.close()

    # Update the graph state
    updated_state = state.model_copy()
    updated_state.retrieval_state = retrieval_state

    return updated_state
