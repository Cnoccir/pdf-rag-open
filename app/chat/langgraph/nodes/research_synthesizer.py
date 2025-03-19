"""
Research synthesizer node for LangGraph architecture.
Handles cross-document analysis to identify shared concepts and insights.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from app.chat.langgraph.state import GraphState, ResearchState
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

def synthesize_research(state: GraphState) -> GraphState:
    """
    Synthesize research from multiple documents.
    Identifies connections between documents and extracts insights.

    Args:
        state: Current graph state

    Returns:
        Updated graph state with research synthesis
    """
    # Skip if no query state or retrieval state
    if not state.query_state or not state.retrieval_state:
        logger.warning("Missing query state or retrieval state, skipping research synthesis")
        return state

    # Get the PDF IDs to analyze
    pdf_ids = state.query_state.pdf_ids

    # If no explicit PDF IDs in query, extract from retrieval results
    if not pdf_ids:
        pdf_ids = []
        for element in state.retrieval_state.elements:
            if element.get("pdf_id") and element.get("pdf_id") not in pdf_ids:
                pdf_ids.append(element.get("pdf_id"))

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

    try:
        # Get Neo4j vector store
        vector_store = get_vector_store()

        # Find common concepts across documents
        common_concepts = _find_common_concepts(pdf_ids, vector_store)

        # Generate insights based on common concepts
        insights = _generate_insights(pdf_ids, common_concepts, vector_store)

        # Get document metadata
        document_meta = _get_document_metadata(pdf_ids, vector_store)

        # Create research state
        if not state.research_state:
            state.research_state = ResearchState(query_state=state.query_state)

        # Update research state
        state.research_state.cross_references = common_concepts
        state.research_state.insights = insights

        # Add metadata
        if not state.research_state.metadata:
            state.research_state.metadata = {}

        state.research_state.metadata["document_meta"] = document_meta
        state.research_state.metadata["analysis_timestamp"] = datetime.now().isoformat()

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

def _find_common_concepts(pdf_ids: List[str], vector_store) -> List[Dict[str, Any]]:
    """
    Find common concepts across multiple documents using Neo4j.

    Args:
        pdf_ids: List of PDF IDs to analyze
        vector_store: Neo4j vector store

    Returns:
        List of common concepts
    """
    try:
        # Use vector store's driver to execute Cypher query
        if not vector_store.driver or not vector_store._initialized:
            logger.warning("Vector store not initialized, cannot find common concepts")
            return []

        with vector_store.driver.session() as session:
            # Query to find common technical terms across documents
            query = """
            MATCH (d:Document)-[:CONTAINS]->(e:ContentElement)
            WHERE d.pdf_id IN $pdf_ids AND e.technical_terms IS NOT NULL
            UNWIND e.technical_terms AS term
            WITH term, collect(DISTINCT d.pdf_id) AS documents
            WHERE size(documents) > 1
            RETURN term AS name, documents, size(documents) AS doc_count
            ORDER BY doc_count DESC, name
            LIMIT 15
            """

            result = session.run(query, {"pdf_ids": pdf_ids})

            # Process results
            common_concepts = []
            for record in result:
                common_concepts.append({
                    "name": record["name"],
                    "documents": record["documents"],
                    "document_count": record["doc_count"]
                })

            return common_concepts

    except Exception as e:
        logger.error(f"Error finding common concepts: {str(e)}", exc_info=True)
        return []

def _generate_insights(pdf_ids: List[str], common_concepts: List[Dict[str, Any]], vector_store) -> List[str]:
    """
    Generate insights based on common concepts across documents.

    Args:
        pdf_ids: List of PDF IDs to analyze
        common_concepts: List of common concepts
        vector_store: Neo4j vector store

    Returns:
        List of insight statements
    """
    insights = []

    # Add insight about document coverage
    insights.append(f"Analysis includes {len(pdf_ids)} documents with {len(common_concepts)} shared concepts.")

    # Add insights about common concepts
    if common_concepts:
        concept_names = [concept["name"] for concept in common_concepts[:5]]
        concepts_str = ", ".join(concept_names)
        insights.append(f"Key concepts shared across documents: {concepts_str}.")

        # Add specific insights for top concepts
        for concept in common_concepts[:3]:
            document_count = len(concept["documents"])
            concept_name = concept["name"]
            insights.append(f"The concept '{concept_name}' appears in {document_count} different documents.")
    else:
        insights.append("No significant common concepts found across these documents.")

    # Try to get more specific insights using Neo4j
    try:
        if vector_store.driver and vector_store._initialized:
            with vector_store.driver.session() as session:
                # Query to find sections containing common concepts
                if common_concepts:
                    top_concept = common_concepts[0]["name"]

                    query = """
                    MATCH (d:Document)-[:CONTAINS]->(e:ContentElement)
                    WHERE d.pdf_id IN $pdf_ids
                    AND e.technical_terms IS NOT NULL
                    AND $concept IN e.technical_terms
                    RETURN d.pdf_id AS pdf_id, d.title AS title,
                           e.section AS section, e.content_type AS content_type
                    LIMIT 10
                    """

                    result = session.run(query, {"pdf_ids": pdf_ids, "concept": top_concept})

                    sections = []
                    for record in result:
                        if record["section"]:
                            sections.append(f"{record['title'] or 'Document ' + record['pdf_id']}: {record['section']}")

                    if sections:
                        insights.append(f"The concept '{top_concept}' appears in these contexts: {'; '.join(sections[:3])}.")
    except Exception as e:
        logger.error(f"Error generating detailed insights: {str(e)}", exc_info=True)

    return insights

def _get_document_metadata(pdf_ids: List[str], vector_store) -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for each document.

    Args:
        pdf_ids: List of PDF IDs
        vector_store: Neo4j vector store

    Returns:
        Dictionary mapping PDF IDs to metadata
    """
    document_meta = {}

    try:
        if vector_store.driver and vector_store._initialized:
            with vector_store.driver.session() as session:
                for pdf_id in pdf_ids:
                    query = """
                    MATCH (d:Document {pdf_id: $pdf_id})
                    RETURN d.title AS title,
                           d.pdf_id AS pdf_id
                    """

                    result = session.run(query, {"pdf_id": pdf_id})
                    record = result.single()

                    if record:
                        document_meta[pdf_id] = {
                            "title": record["title"] or f"Document {pdf_id}",
                            "pdf_id": pdf_id
                        }
                    else:
                        document_meta[pdf_id] = {
                            "title": f"Document {pdf_id}",
                            "pdf_id": pdf_id
                        }
    except Exception as e:
        logger.error(f"Error getting document metadata: {str(e)}", exc_info=True)

        # Provide basic metadata as fallback
        for pdf_id in pdf_ids:
            document_meta[pdf_id] = {
                "title": f"Document {pdf_id}",
                "pdf_id": pdf_id
            }

    return document_meta
