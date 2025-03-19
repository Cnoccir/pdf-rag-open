"""
PDF status check utilities for monitoring document processing.
Helps diagnose issues in the document processing pipeline.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)

async def check_pdf_status(pdf_id: str) -> Dict[str, Any]:
    """
    Check the status of a PDF in the Neo4j vector store.

    Args:
        pdf_id: PDF ID to check

    Returns:
        Dictionary with PDF status information
    """
    result = {
        "pdf_id": pdf_id,
        "exists": False,
        "indexed": False,
        "element_count": 0,
        "concept_count": 0,
        "status": "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        # Get the vector store
        vector_store = get_vector_store()
        if not vector_store.initialized:
            result["error"] = "Vector store not initialized"
            result["status"] = "error"
            return result

        # Check if the document exists in Neo4j
        with vector_store.driver.session() as session:
            # Check for Document node
            query = """
            MATCH (d:Document {pdf_id: $pdf_id})
            RETURN d.title AS title, d.created_at AS created_at
            """

            doc_result = session.run(query, {"pdf_id": pdf_id})
            doc_record = doc_result.single()

            if doc_record:
                result["exists"] = True
                result["title"] = doc_record.get("title", "Untitled")
                result["created_at"] = doc_record.get("created_at", None)

                # Check for content elements
                element_query = """
                MATCH (d:Document {pdf_id: $pdf_id})-[:CONTAINS]->(e:ContentElement)
                RETURN count(e) AS element_count
                """

                element_result = session.run(element_query, {"pdf_id": pdf_id})
                element_record = element_result.single()

                if element_record:
                    result["element_count"] = element_record["element_count"]
                    result["indexed"] = result["element_count"] > 0

                # Check for concepts
                concept_query = """
                MATCH (d:Document {pdf_id: $pdf_id})-[:HAS_CONCEPT]->(c:Concept)
                RETURN count(c) AS concept_count
                """

                concept_result = session.run(concept_query, {"pdf_id": pdf_id})
                concept_record = concept_result.single()

                if concept_record:
                    result["concept_count"] = concept_record["concept_count"]

            # Determine status
            if not result["exists"]:
                result["status"] = "not_found"
            elif not result["indexed"]:
                result["status"] = "exists_but_not_indexed"
            else:
                result["status"] = "indexed"

        # Get DB status
        try:
            from app.web.db.models import Pdf
            from app.web.db import db

            pdf = db.session.execute(
                db.select(Pdf).filter_by(id=pdf_id)
            ).scalar_one_or_none()

            if pdf:
                result["db_record"] = {
                    "processed": pdf.processed,
                    "error": pdf.error,
                    "category": pdf.category,
                    "name": pdf.name,
                    "created_at": pdf.created_at.isoformat() if hasattr(pdf.created_at, "isoformat") else str(pdf.created_at),
                    "updated_at": pdf.updated_at.isoformat() if hasattr(pdf.updated_at, "isoformat") else str(pdf.updated_at)
                }
        except Exception as db_error:
            logger.warning(f"Error getting PDF database status: {str(db_error)}")
            result["db_error_message"] = str(db_error)

        return result

    except Exception as e:
        logger.error(f"Error checking PDF status: {str(e)}", exc_info=True)
        result["error"] = str(e)
        result["status"] = "error"
        return result

async def get_recent_pdfs(limit: int = 5) -> Dict[str, Any]:
    """
    Get status information for the most recent PDFs.

    Args:
        limit: Maximum number of PDFs to return

    Returns:
        Dictionary with PDF status information
    """
    result = {
        "pdfs": [],
        "count": 0,
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        # Get recent PDFs from the database
        try:
            from app.web.db.models import Pdf
            from app.web.db import db

            pdfs = db.session.execute(
                db.select(Pdf).filter_by(is_deleted=False).order_by(Pdf.updated_at.desc()).limit(limit)
            ).scalars().all()

            pdf_ids = [pdf.id for pdf in pdfs]

            # Check each PDF in Neo4j
            pdf_statuses = []
            for pdf_id in pdf_ids:
                pdf_status = await check_pdf_status(pdf_id)
                pdf_statuses.append(pdf_status)

            result["pdfs"] = pdf_statuses
            result["count"] = len(pdf_statuses)

        except Exception as db_error:
            logger.warning(f"Error getting recent PDFs from database: {str(db_error)}")
            result["error"] = str(db_error)
            result["status"] = "error"

        return result

    except Exception as e:
        logger.error(f"Error getting recent PDFs: {str(e)}", exc_info=True)
        result["error"] = str(e)
        result["status"] = "error"
        return result
