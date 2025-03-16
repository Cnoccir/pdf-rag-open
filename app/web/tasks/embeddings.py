import asyncio
import logging
import os
from celery import shared_task
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import current_app
from openai import AsyncOpenAI

from app.chat.types import (
    ProcessingConfig,
    ProcessingResult,
    ResearchManager,
    ResearchMode,
    ConceptNetwork,
    DocumentSummary
)
from app.web.db.models import Pdf
from app.web.db import db
from app.chat.document_fetcher import process_technical_document
from app.chat.vector_stores.vector_store import get_vector_store
from app.chat.utils import setup_logging, normalize_pdf_id

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_document(self, pdf_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced Celery task for document processing with Neo4j integration.
    Processes document content and stores in Neo4j graph database.

    Args:
        pdf_id: Document identifier
        config: Optional processing configuration

    Returns:
        Processing results metadata
    """
    processing_start = datetime.utcnow()
    normalized_pdf_id = normalize_pdf_id(pdf_id)

    try:
        # Create ProcessingConfig with optimized defaults for technical documents
        processing_config = ProcessingConfig(
            pdf_id=normalized_pdf_id,
            **(config or {}),
            # Enhanced defaults for technical content
            chunk_size=500,          # Optimal for technical content
            chunk_overlap=100,       # Better context preservation
            embedding_model="text-embedding-3-small",
            process_images=True,
            process_tables=True,
            extract_technical_terms=True,
            extract_relationships=True,
            max_concepts_per_document=200
        )

        # Setup logging for this processing job
        setup_logging(normalized_pdf_id)
        logger.info(f"Starting Neo4j-integrated document processing for {normalized_pdf_id}")

        async def async_process():
            try:
                # Initialize OpenAI client with appropriate configuration
                openai_client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=60.0  # Increased timeout for comprehensive processing
                )

                # Process the document using Neo4j-compatible document processor
                logger.info(f"Processing document {normalized_pdf_id} with Neo4j compatibility")
                
                # Use the enhanced document processor from document_fetcher.py
                from app.chat.document_fetcher import process_technical_document
                result = await process_technical_document(
                    pdf_id=normalized_pdf_id,
                    config=processing_config,
                    openai_client=openai_client
                )

                # Neo4j vector store integration is now handled directly in process_technical_document
                # No separate vector store ingestion is needed

                # Update PDF status in database
                pdf = db.session.execute(
                    db.select(Pdf).filter_by(id=pdf_id)
                ).scalar_one()

                pdf.processed = True
                pdf.error = None
                pdf.processed_at = datetime.utcnow()

                # Store comprehensive processing metadata
                concept_network_data = {}
                if hasattr(result, 'concept_network') and result.concept_network:
                    concept_network_data = {
                        "concept_count": len(result.concept_network.concepts),
                        "relationship_count": len(result.concept_network.relationships),
                        "primary_concepts": [c.name for c in result.concept_network.concepts[:5] if hasattr(c, 'name')]
                    }
                
                document_summary_data = {}
                if hasattr(result, 'document_summary') and result.document_summary:
                    if isinstance(result.document_summary, dict):
                        document_summary_data = result.document_summary
                    else:
                        document_summary_data = result.document_summary.dict()

                pdf.update_metadata({
                    "processing_stats": {
                        "element_count": len(result.elements),
                        "processing_duration": (datetime.utcnow() - processing_start).total_seconds()
                    },
                    "config_used": processing_config.dict(exclude_unset=True),
                    "concept_network": concept_network_data,
                    "document_summary": document_summary_data,
                    "neo4j_ready": True,  # Mark as ready in Neo4j
                    "langgraph_ready": True,  # Mark as ready for LangGraph
                    "completed_at": datetime.utcnow().isoformat()
                })

                # Commit changes to database
                db.session.commit()
                logger.info(f"Database updated with processing results for {normalized_pdf_id}")

                return {
                    "status": "success",
                    "pdf_id": normalized_pdf_id,
                    "processed_at": datetime.utcnow().isoformat(),
                    "neo4j_ready": True,
                    "langgraph_ready": True,
                    "element_count": len(result.elements)
                }

            except Exception as e:
                logger.error(
                    f"Processing failure for PDF {normalized_pdf_id}",
                    exc_info=True
                )

                # Update error status
                try:
                    pdf = db.session.execute(
                        db.select(Pdf).filter_by(id=pdf_id)
                    ).scalar_one()

                    pdf.processed = False
                    pdf.error = str(e)
                    pdf.processed_at = datetime.utcnow()
                    db.session.commit()
                    logger.info(f"Database updated with error status for {normalized_pdf_id}")
                except Exception as db_error:
                    logger.error(f"Failed to update PDF status: {str(db_error)}")

                # Retry logic for specific errors
                if isinstance(e, (TimeoutError, ConnectionError)):
                    raise self.retry(exc=e, countdown=30)

                return {
                    "status": "error",
                    "error": str(e),
                    "pdf_id": normalized_pdf_id
                }

        # Set up and run the async loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            with current_app.app_context():
                return loop.run_until_complete(async_process())
        finally:
            loop.close()

    except Exception as e:
        logger.error(
            f"Task execution failed for PDF {normalized_pdf_id}",
            exc_info=True
        )
        return {
            "status": "error",
            "error": str(e),
            "pdf_id": normalized_pdf_id,
            "neo4j_ready": False
        }