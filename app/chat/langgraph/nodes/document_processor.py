"""
Document processor node for LangGraph-based PDF RAG system.
Simplified implementation for document extraction and ingestion.
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

from app.chat.langgraph.state import GraphState
from app.chat.types import ProcessingConfig
from app.chat.vector_stores import get_vector_store
from app.chat.document_fetcher import process_technical_document

logger = logging.getLogger(__name__)

def process_document(state: GraphState) -> GraphState:
    """
    Process a document and extract content for Neo4j vector store.

    Args:
        state: Current graph state containing document information

    Returns:
        Updated graph state with processing results
    """
    # Validate state
    if not state.document_state or "pdf_id" not in state.document_state:
        logger.error("Document state with pdf_id is required")
        if not state.document_state:
            state.document_state = {"error": "Missing document state"}
        else:
            state.document_state["error"] = "Missing pdf_id in document state"
        return state

    # Get PDF ID
    pdf_id = state.document_state["pdf_id"]
    logger.info(f"Processing document: {pdf_id}")

    try:
        # Create processing configuration
        config = ProcessingConfig(
            pdf_id=pdf_id,
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="text-embedding-3-small",
            process_images=True,
            process_tables=True,
            extract_technical_terms=True,
            extract_relationships=True
        )

        # Get OpenAI client
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Process document using document_fetcher
        processing_result = process_technical_document(
            pdf_id=pdf_id,
            config=config,
            openai_client=openai_client
        )

        # Update database record
        try:
            from app.web.db.models import Pdf
            from app.web.db import db

            # Find PDF record
            pdf = db.session.execute(
                db.select(Pdf).filter_by(id=pdf_id)
            ).scalar_one_or_none()

            if pdf:
                # Update processing status
                pdf.processed = True
                pdf.error = None

                # Update metadata with document summary
                if hasattr(processing_result, 'document_summary') and processing_result.document_summary:
                    document_summary = processing_result.document_summary
                    if not isinstance(document_summary, dict):
                        document_summary = processing_result.document_summary.dict() if hasattr(processing_result.document_summary, 'dict') else processing_result.document_summary.__dict__

                    # Add predicted category if available
                    if document_summary and document_summary.get('primary_concepts'):
                        primary_concepts = document_summary.get('primary_concepts', [])
                        primary_category = _predict_category_from_concepts(primary_concepts)
                        document_summary['predicted_category'] = primary_category

                    # Update metadata
                    pdf.update_metadata({
                        "document_summary": document_summary,
                        "processing_complete": True,
                        "processing_time": datetime.now().isoformat()
                    })

                # Commit changes
                db.session.commit()
                logger.info(f"Updated database record for {pdf_id}")
        except Exception as db_error:
            logger.error(f"Error updating database record: {str(db_error)}")

        # Update state
        element_count = len(processing_result.elements) if hasattr(processing_result, 'elements') else 0

        state.document_state.update({
            "status": "success",
            "element_count": element_count,
            "processing_time": datetime.now().isoformat(),
            "document_title": processing_result.document_summary.get('title', f"Document {pdf_id}") if hasattr(processing_result, 'document_summary') else f"Document {pdf_id}"
        })

        # Only store summary in state to avoid excessive state size
        if hasattr(processing_result, 'document_summary'):
            state.document_state["document_summary"] = {
                "title": processing_result.document_summary.get('title', f"Document {pdf_id}"),
                "primary_concepts": processing_result.document_summary.get('primary_concepts', [])[:5],
                "document_type": processing_result.document_summary.get('document_type', "Technical Document")
            }

        logger.info(f"Document processing complete: {pdf_id}")
        return state

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}", exc_info=True)

        # Update database record with error
        try:
            from app.web.db.models import Pdf
            from app.web.db import db

            pdf = db.session.execute(
                db.select(Pdf).filter_by(id=pdf_id)
            ).scalar_one_or_none()

            if pdf:
                pdf.processed = False
                pdf.error = str(e)
                db.session.commit()
        except Exception as db_error:
            logger.error(f"Error updating database record with error: {str(db_error)}")

        # Update state with error
        state.document_state["status"] = "error"
        state.document_state["error"] = str(e)
        state.document_state["processing_time"] = datetime.now().isoformat()

        return state

def _predict_category_from_concepts(concepts: list) -> str:
    """
    Predict document category based on primary concepts.

    Args:
        concepts: List of primary concepts

    Returns:
        Predicted category
    """
    # Define concept keywords for each category
    category_keywords = {
        "tridium": {"niagara", "jace", "tridium", "fox", "workbench", "supervisor", "station",
                   "driver", "controller", "baja", "hierarchy", "wiresheet", "ord"},

        "honeywell": {"honeywell", "webs", "excel", "wpa", "spyder", "lyric", "vista", "notifier",
                     "comfort", "controlpoint", "tps"},

        "johnson_controls": {"johnson", "metasys", "fec", "nae", "ncm", "vav", "bacnet", "n2",
                            "fac", "facility explorer", "verasys"}
    }

    # Count matches for each category
    category_scores = {category: 0 for category in category_keywords}

    for concept in concepts:
        concept_lower = concept.lower()
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in concept_lower or concept_lower in keyword:
                    category_scores[category] += 1
                    break

    # Find category with highest score
    best_category = max(category_scores.items(), key=lambda x: x[1])

    # Return best category if score > 0, otherwise "general"
    return best_category[0] if best_category[1] > 0 else "general"
