"""
Research synthesizer node for LangGraph architecture.
Handles cross-document analysis to identify shared concepts and insights.
Enhanced to work with MongoDB + Qdrant storage.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.chat.langgraph.state import GraphState, ResearchState
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

def synthesize_research(state: GraphState) -> GraphState:
    """
    Synthesize research from multiple documents.
    Identifies connections between documents and extracts insights.
    Enhanced to work with MongoDB + Qdrant storage.

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
    if not pdf_ids or len(pdf_ids) == 0:
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
        # Get unified vector store
        vector_store = get_vector_store()

        # Get cross-document information from retrieval state if available
        cross_document_info = state.retrieval_state.metadata.get("cross_document_info", {})

        # Find common concepts across documents
        shared_concepts = cross_document_info.get("shared_concepts", [])

        # If no shared concepts in metadata, query directly
        if not shared_concepts:
            shared_concepts = vector_store.find_shared_concepts(pdf_ids)

        # Generate insights based on retrieved content and shared concepts
        insights = generate_cross_document_insights(
            pdf_ids=pdf_ids,
            shared_concepts=shared_concepts,
            elements=state.retrieval_state.elements,
            query=state.query_state.query
        )

        # Get document metadata
        document_meta = cross_document_info.get("document_titles", {})

        # If document titles not in metadata, get them directly
        if not document_meta:
            document_meta = {}
            for pdf_id in pdf_ids:
                doc_metadata = vector_store.get_document_metadata(pdf_id)
                if doc_metadata:
                    document_meta[pdf_id] = {
                        "title": doc_metadata.get("title", f"Document {pdf_id}"),
                        "pdf_id": pdf_id
                    }
                else:
                    document_meta[pdf_id] = {
                        "title": f"Document {pdf_id}",
                        "pdf_id": pdf_id
                    }

        # Create research state
        if not state.research_state:
            state.research_state = ResearchState(query_state=state.query_state)

        # Update research state
        state.research_state.cross_references = shared_concepts
        state.research_state.insights = insights

        # Add metadata
        if not state.research_state.metadata:
            state.research_state.metadata = {}

        state.research_state.metadata["document_meta"] = document_meta
        state.research_state.metadata["analysis_timestamp"] = datetime.now().isoformat()
        state.research_state.metadata["document_count"] = len(pdf_ids)

        # Create research context for generation
        from app.chat.types import ResearchContext
        research_context = ResearchContext(
            primary_pdf_id=pdf_ids[0]  # First PDF is primary
        )

        # Add all PDFs to research context
        for pdf_id in pdf_ids:
            title = document_meta.get(pdf_id, {}).get("title", f"Document {pdf_id}")
            if isinstance(document_meta.get(pdf_id), dict):
                title = document_meta[pdf_id].get("title", f"Document {pdf_id}")
            elif isinstance(document_meta.get(pdf_id), str):
                title = document_meta[pdf_id]
            research_context.add_document(pdf_id, title)

        # Add insights to research context if possible
        if hasattr(research_context, "insights"):
            research_context.insights = insights

        # Add shared concepts if possible
        if hasattr(research_context, "shared_concepts"):
            research_context.shared_concepts = [c.get("name") for c in shared_concepts if isinstance(c, dict)]

        # Set research context in state
        state.research_state.research_context = research_context

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

def generate_cross_document_insights(
    pdf_ids: List[str],
    shared_concepts: List[Dict[str, Any]],
    elements: List[Dict[str, Any]],
    query: str
) -> List[str]:
    """
    Generate insights by analyzing cross-document patterns.

    Args:
        pdf_ids: List of PDF IDs to analyze
        shared_concepts: List of concepts shared across documents
        elements: Retrieved content elements
        query: Original user query

    Returns:
        List of insight statements
    """
    insights = []

    # 1. Add basic insight about document coverage
    insights.append(f"Analysis includes {len(pdf_ids)} documents with {len(shared_concepts)} shared concepts.")

    # 2. Add insights about common concepts if available
    if shared_concepts:
        # Get top concept names
        top_concepts = []
        for concept in shared_concepts[:5]:
            if isinstance(concept, dict) and "name" in concept:
                top_concepts.append(concept["name"])
            elif isinstance(concept, str):
                top_concepts.append(concept)

        if top_concepts:
            concepts_str = ", ".join(top_concepts)
            insights.append(f"Key concepts shared across documents: {concepts_str}.")

            # Add detailed insights for top concepts
            for i, concept in enumerate(shared_concepts[:3]):
                if isinstance(concept, dict):
                    concept_name = concept.get("name", "")
                    doc_count = concept.get("document_count", 0)
                    if concept_name and doc_count > 1:
                        insights.append(f"The concept '{concept_name}' appears in {doc_count} different documents.")
    else:
        insights.append("No significant common concepts found across these documents.")

    # 3. Add document-specific insights based on retrieved elements
    doc_element_counts = {}
    doc_content_types = {}

    for element in elements:
        pdf_id = element.get("pdf_id")
        if pdf_id:
            # Count elements per document
            doc_element_counts[pdf_id] = doc_element_counts.get(pdf_id, 0) + 1

            # Track content types per document
            content_type = element.get("content_type")
            if content_type:
                if pdf_id not in doc_content_types:
                    doc_content_types[pdf_id] = set()
                doc_content_types[pdf_id].add(content_type)

    # Find documents with unique content types
    for pdf_id, content_types in doc_content_types.items():
        for content_type in content_types:
            # Check if this content type is unique to this document
            is_unique = True
            for other_id, other_types in doc_content_types.items():
                if other_id != pdf_id and content_type in other_types:
                    is_unique = False
                    break

            if is_unique and content_type in ['table', 'image', 'procedure', 'parameter']:
                insights.append(f"Document {pdf_id} uniquely contains {content_type} content relevant to your query.")

    # 4. Add insights based on element distribution
    if len(doc_element_counts) > 1:
        # Find document with most results
        most_relevant_id = max(doc_element_counts.items(), key=lambda x: x[1])[0]
        insights.append(f"Document {most_relevant_id} has the most relevant content for your query.")

    # 5. Add query-specific insights if possible
    query_lower = query.lower()

    # Check for comparison queries
    if any(term in query_lower for term in ["compare", "difference", "versus", "vs"]):
        insights.append("Your comparison query will benefit from content across multiple documents.")

    # Check for specific content type requests
    if any(term in query_lower for term in ["procedure", "how to", "steps"]):
        procedure_docs = []
        for pdf_id, types in doc_content_types.items():
            if "procedure" in types:
                procedure_docs.append(pdf_id)

        if procedure_docs:
            if len(procedure_docs) == 1:
                insights.append(f"For procedure-related information, focus on document {procedure_docs[0]}.")
            else:
                insights.append(f"Procedures were found in {len(procedure_docs)} different documents.")

    return insights
