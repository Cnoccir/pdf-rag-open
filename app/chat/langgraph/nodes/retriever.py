"""
Retriever node for LangGraph-based PDF RAG system.
This node handles document retrieval using various strategies based on query analysis.
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from app.chat.langgraph.state import QueryState, RetrievalState, GraphState, RetrievalStrategy
from app.chat.vector_stores import Neo4jVectorStore, get_vector_store
from app.chat.types import ContentElement, SearchQuery

logger = logging.getLogger(__name__)

async def _retrieve_content_async(query_state: QueryState) -> RetrievalState:
    """
    Asynchronous implementation of content retrieval.
    
    Args:
        query_state: Query state with analysis
        
    Returns:
        Retrieval state with results
    """
    logger.info(f"Beginning retrieval using strategy: {query_state.retrieval_strategy.value}")
    
    # Initialize retrieval state
    retrieval_state = RetrievalState(
        query_state=query_state,
        strategies_used=[query_state.retrieval_strategy]
    )
    
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        if not vector_store or not vector_store.initialized:
            raise ValueError("Vector store not available or not initialized")
        
        # Create search parameters
        pdf_id = query_state.pdf_ids[0] if query_state.pdf_ids else None
        k = 10  # Number of results to return
        
        # Perform retrieval based on strategy
        start_time = datetime.utcnow()
        
        if query_state.retrieval_strategy == RetrievalStrategy.SEMANTIC:
            # Pure semantic search
            result = vector_store.retrieve(query_state.query, k=k, pdf_id=pdf_id)
            
        elif query_state.retrieval_strategy == RetrievalStrategy.KEYWORD:
            # Keyword-based search
            # For now, implement as semantic search until we add dedicated keyword search
            result = vector_store.retrieve(query_state.query, k=k, pdf_id=pdf_id)
            
        elif query_state.retrieval_strategy == RetrievalStrategy.TABLE:
            # Table-specific search with filter
            result = vector_store.retrieve(
                query_state.query, 
                k=k, 
                pdf_id=pdf_id,
                filter_content_types=["table"]
            )
            
        elif query_state.retrieval_strategy == RetrievalStrategy.IMAGE:
            # Image-specific search with filter
            result = vector_store.retrieve(
                query_state.query, 
                k=k, 
                pdf_id=pdf_id,
                filter_content_types=["figure", "image"]
            )
            
        else:
            # Default to hybrid approach (most common case)
            result = vector_store.retrieve(query_state.query, k=k, pdf_id=pdf_id)
        
        # Record timing metrics
        retrieval_time = (datetime.utcnow() - start_time).total_seconds()
        vector_store.metrics.record_query_time(retrieval_time)
        vector_store.metrics.record_retrieval()
        
        # Transform results into standard ContentElement format
        elements = []
        sources = []
        
        for doc in result.documents:
            # Extract metadata
            metadata = doc.metadata
            
            # Create ContentElement
            element = ContentElement(
                id=metadata.get("id", ""),
                content=doc.page_content,
                content_type=metadata.get("content_type", "text"),
                pdf_id=metadata.get("pdf_id", ""),
                page_number=metadata.get("page_number", 0),
                section=metadata.get("section", ""),
                metadata={
                    "score": metadata.get("score", 0.0),
                    "section_title": metadata.get("section_title", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                }
            )
            elements.append(element)
            
            # Add to sources for citation tracking
            sources.append({
                "id": metadata.get("id", ""),
                "pdf_id": metadata.get("pdf_id", ""),
                "page_number": metadata.get("page_number", 0),
                "section": metadata.get("section", ""),
                "section_title": metadata.get("section_title", ""),
                "score": metadata.get("score", 0.0)
            })
        
        # Update retrieval state
        retrieval_state.elements = elements
        retrieval_state.sources = sources
        retrieval_state.metadata["retrieval_time"] = retrieval_time
        retrieval_state.metadata["total_results"] = len(elements)
        
        logger.info(f"Retrieval complete: found {len(elements)} relevant elements in {retrieval_time:.2f}s")
        
    except Exception as e:
        error_msg = f"Error during retrieval: {str(e)}"
        logger.error(error_msg)
        retrieval_state.metadata["error"] = error_msg
        
        # Record error in metrics
        if vector_store and hasattr(vector_store, "metrics"):
            vector_store.metrics.record_error(str(e))
    
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
    
    # Run the async retrieval in a new event loop
    retrieval_state = asyncio.run(_retrieve_content_async(state.query_state))
    
    # Update the graph state
    updated_state = state.model_copy()
    updated_state.retrieval_state = retrieval_state
    
    return updated_state
