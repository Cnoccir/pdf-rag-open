# app/chat/langgraph/nodes/research_synthesizer.py
"""
Research synthesis node for LangGraph architecture.
Leverages Neo4j graph capabilities for cross-document analysis.
"""

import logging
import os
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from app.chat.langgraph.state import GraphState, ResearchState

logger = logging.getLogger(__name__)

async def find_cross_document_connections(query: str, pdf_ids: List[str]) -> Dict[str, Any]:
    """
    Find connections between documents using Neo4j's graph capabilities.
    
    Args:
        query: User query
        pdf_ids: Document IDs to analyze
        
    Returns:
        Dictionary with cross-document connections
    """
    # Skip if only one document
    if not pdf_ids or len(pdf_ids) < 2:
        return {
            "common_concepts": [],
            "concept_paths": [],
            "insights": ["No cross-document analysis available with a single document."]
        }
    
    try:
        # Get Neo4j vector store
        from app.chat.vector_stores import get_vector_store
        vector_store = get_vector_store()
        
        # Extract key concepts from query
        from app.chat.utils.extraction import extract_technical_terms
        query_concepts = extract_technical_terms(query)
        
        # 1. Find common concepts across documents
        common_concepts = await vector_store.find_common_concepts(pdf_ids)
        
        # 2. Find concept paths between documents
        # Use the main query concept if available, otherwise use top common concept
        start_concept = None
        if query_concepts:
            start_concept = query_concepts[0]
        elif common_concepts:
            start_concept = common_concepts[0]["name"]
            
        concept_paths = []
        if start_concept:
            concept_paths = await vector_store.find_concept_paths(
                start_concept=start_concept,
                pdf_ids=pdf_ids
            )
        
        # 3. Generate insights from the connections
        insights = []
        
        # Add insight about document coverage
        document_meta = {}
        for pdf_id in pdf_ids:
            try:
                # Get document metadata from Neo4j
                async with vector_store.driver.session() as session:
                    result = await session.run(
                        "MATCH (d:Document {pdf_id: $pdf_id}) RETURN d.title AS title, d.metadata AS metadata",
                        {"pdf_id": pdf_id}
                    )
                    record = await result.single()
                    if record:
                        document_meta[pdf_id] = {
                            "title": record["title"],
                            "metadata": record["metadata"]
                        }
            except Exception as e:
                logger.error(f"Error getting document metadata: {str(e)}")
                document_meta[pdf_id] = {"title": f"Document {pdf_id}"}
        
        # Add insight about common concepts
        if common_concepts:
            concepts_list = ", ".join([c["name"] for c in common_concepts[:5]])
            insights.append(f"Found {len(common_concepts)} concepts shared across documents. Top concepts: {concepts_list}.")
            
            # Add specific insights for top concepts
            for concept in common_concepts[:3]:
                doc_titles = [document_meta.get(doc_id, {}).get("title", f"Document {doc_id}") 
                             for doc_id in concept["documents"]]
                doc_titles_str = ", ".join(doc_titles)
                insights.append(f"The concept '{concept['name']}' appears across multiple documents: {doc_titles_str}")
        else:
            insights.append("No significant common concepts found across these documents.")
        
        # Add insight about concept paths
        if concept_paths:
            insights.append(f"Found {len(concept_paths)} conceptual connections between documents.")
            
            # Add specific path insights
            for path in concept_paths[:2]:
                if 'path' in path and path['path']:
                    path_str = " → ".join(path['path'])
                    insights.append(f"Connection path: {path_str}")
                    
                    # Add document context for this path
                    doc_context = []
                    for doc_id in pdf_ids:
                        doc_title = document_meta.get(doc_id, {}).get("title", f"Document {doc_id}")
                        doc_context.append(f"{doc_title} ({doc_id})")
                    
                    insights.append(f"This connection spans across: {', '.join(doc_context)}")
        
        return {
            "common_concepts": common_concepts,
            "concept_paths": concept_paths,
            "insights": insights,
            "document_meta": document_meta
        }
    
    except Exception as e:
        logger.error(f"Error in cross-document analysis: {str(e)}", exc_info=True)
        return {
            "common_concepts": [],
            "concept_paths": [],
            "insights": [f"Error in cross-document analysis: {str(e)}"],
            "error": str(e)
        }

async def research_synthesize(state: GraphState) -> GraphState:
    """
    Research synthesis node for LangGraph.
    Uses Neo4j to perform cross-document analysis and synthesize insights.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with research insights
    """
    # Skip if no query state or retrieval state
    if not state.query_state or not state.retrieval_state:
        logger.info("Missing query state or retrieval state, skipping research synthesis")
        return state
    
    try:
        # Get the PDF IDs to analyze
        pdf_ids = state.query_state.pdf_ids
        
        # If no explicit PDF IDs in query, extract from retrieval results
        if not pdf_ids:
            pdf_ids = []
            for element in state.retrieval_state.elements:
                if element.pdf_id and element.pdf_id not in pdf_ids:
                    pdf_ids.append(element.pdf_id)
        
        # Skip if only one document
        if len(pdf_ids) <= 1:
            logger.info("Only one document in analysis, skipping research synthesis")
            if not state.research_state:
                state.research_state = ResearchState(
                    query_state=state.query_state,
                    insights=["Research mode requires multiple documents for cross-document analysis."]
                )
            return state
        
        logger.info(f"Performing research synthesis across {len(pdf_ids)} documents")
        
        # Find cross-document connections
        connections = await find_cross_document_connections(
            query=state.query_state.query,
            pdf_ids=pdf_ids
        )
        
        # Create or update research state
        if not state.research_state:
            state.research_state = ResearchState(query_state=state.query_state)
        
        # Update research state with connections
        state.research_state.cross_references = connections.get("common_concepts", [])
        state.research_state.insights = connections.get("insights", [])
        
        # Add concept paths to metadata
        if not state.research_state.metadata:
            state.research_state.metadata = {}
            
        state.research_state.metadata["concept_paths"] = connections.get("concept_paths", [])
        state.research_state.metadata["document_meta"] = connections.get("document_meta", {})
        state.research_state.metadata["analysis_timestamp"] = datetime.utcnow().isoformat()
        
        logger.info(f"Research synthesis complete with {len(state.research_state.insights)} insights")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in research synthesis: {str(e)}", exc_info=True)
        
        # Create research state with error if it doesn't exist
        if not state.research_state:
            state.research_state = ResearchState(
                query_state=state.query_state,
                insights=[f"Error in research synthesis: {str(e)}"],
                metadata={"error": str(e)}
            )
        
        return state