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
    Enhanced Celery task for document processing with LangGraph integration.
    Processes document content through extraction, chunking, and embedding steps.

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
            process_tables=True
        )

        # Setup logging for this processing job
        setup_logging(normalized_pdf_id)
        logger.info(f"Starting LangGraph-integrated document processing for {normalized_pdf_id}")

        async def async_process():
            try:
                # Initialize OpenAI client with appropriate configuration
                openai_client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=60.0  # Increased timeout for comprehensive processing
                )

                # Step 1: Process the document with enhanced extraction
                # Leveraging document_fetcher.py's effective PDF processing
                logger.info(f"Processing document content for {normalized_pdf_id}")
                result = await process_technical_document(
                    pdf_id=normalized_pdf_id,
                    config=processing_config,
                    openai_client=openai_client
                )

                # Step 2: Get document statistics for reporting
                extracted_elements = len(result.elements)
                content_types = {}
                for elem in result.elements:
                    content_types[elem.type.value] = content_types.get(elem.type.value, 0) + 1
                
                processing_stats = {
                    "element_count": extracted_elements,
                    "content_types": content_types,
                    "processing_duration": (datetime.utcnow() - processing_start).total_seconds()
                }
                
                logger.info(f"Document processing complete: {processing_stats}")

                # Step 3: Ingest processed content with LangGraph-compatible format
                # Get storage instance directly without creating a new TechnicalDocumentStore
                logger.info(f"Ingesting content into vector store for {normalized_pdf_id}")
                storage = get_vector_store(normalized_pdf_id)
                
                # Use the storage's ingest method to store document content
                await storage.ingest_processed_content(result)
                logger.info(f"Content ingestion complete for {normalized_pdf_id}")

                # Step 4: Update PDF status with enhanced metadata
                pdf = db.session.execute(
                    db.select(Pdf).filter_by(id=pdf_id)
                ).scalar_one()

                pdf.processed = True
                pdf.error = None
                pdf.processed_at = datetime.utcnow()

                # Store comprehensive processing metadata with LangGraph integration info
                concept_network_data = {}
                if hasattr(result, 'concept_network') and result.concept_network:
                    concept_network_data = {
                        "concept_count": len(result.concept_network.concepts),
                        "relationship_count": len(result.concept_network.relationships),
                        "primary_concepts": [c.name for c in result.concept_network.concepts[:5]] 
                    }
                
                document_summary_data = {}
                if hasattr(result, 'document_summary') and result.document_summary:
                    document_summary_data = {
                        "title": result.document_summary.title,
                        "primary_concepts": result.document_summary.primary_concepts,
                        "key_insights": result.document_summary.key_insights
                    }

                pdf.metadata = {
                    "processing_stats": processing_stats,
                    "config_used": processing_config.dict(exclude_unset=True),
                    "processing_duration": (datetime.utcnow() - processing_start).total_seconds(),
                    "element_counts": content_types,
                    "concept_network": concept_network_data,
                    "document_summary": document_summary_data,
                    "langgraph_ready": True,
                    "completed_at": datetime.utcnow().isoformat()
                }

                # Commit changes to database
                db.session.commit()
                logger.info(f"Database updated with processing results for {normalized_pdf_id}")

                return {
                    "status": "success",
                    "pdf_id": normalized_pdf_id,
                    "statistics": processing_stats,
                    "processed_at": datetime.utcnow().isoformat(),
                    "langgraph_ready": True
                }

            except Exception as e:
                logger.error(
                    f"Processing failure for PDF {normalized_pdf_id}",
                    exc_info=True,
                    extra={"pdf_id": normalized_pdf_id}
                )

                # Update error status with detailed information
                try:
                    pdf = db.session.execute(
                        db.select(Pdf).filter_by(id=pdf_id)
                    ).scalar_one()

                    error_detail = {
                        "message": str(e),
                        "type": type(e).__name__,
                        "timestamp": datetime.utcnow().isoformat()
                    }

                    pdf.processed = False
                    pdf.error = str(error_detail)
                    pdf.processed_at = datetime.utcnow()
                    db.session.commit()
                    logger.info(f"Database updated with error status for {normalized_pdf_id}")
                except Exception as db_error:
                    logger.error(
                        f"Failed to update PDF status for {normalized_pdf_id}",
                        exc_info=True
                    )

                # Retry logic for specific errors
                if isinstance(e, (TimeoutError, ConnectionError)):
                    raise self.retry(exc=e, countdown=30)

                return {
                    "status": "error",
                    "error": str(e),
                    "pdf_id": normalized_pdf_id,
                    "error_detail": error_detail
                }

        # Set up and run the async loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with current_app.app_context():
                return loop.run_until_complete(async_process())
        finally:
            loop.close()

    except Exception as e:
        logger.error(
            f"Task execution failed for PDF {normalized_pdf_id}",
            exc_info=True,
            extra={"pdf_id": normalized_pdf_id}
        )
        return {
            "status": "error",
            "error": str(e),
            "pdf_id": normalized_pdf_id
        }


@shared_task(bind=True)
def process_multiple_documents(self, pdf_ids: List[str], enable_research_mode: bool = True) -> Dict[str, Any]:
    """
    Process multiple documents and enable research mode for cross-document analysis
    with LangGraph integration for improved research capabilities.

    Args:
        pdf_ids: List of PDF IDs to process
        enable_research_mode: Whether to enable research mode for cross-document analysis

    Returns:
        Processing results summary
    """
    processing_start = datetime.utcnow()
    results = {}
    errors = {}
    
    # Normalize all PDF IDs
    normalized_pdf_ids = [normalize_pdf_id(pdf_id) for pdf_id in pdf_ids]
    
    try:
        logger.info(f"Processing multiple documents: {normalized_pdf_ids}")
        
        # Process each document individually
        for pdf_id in normalized_pdf_ids:
            try:
                logger.info(f"Starting processing for document {pdf_id}")
                result = process_document(pdf_id)
                results[pdf_id] = result
                logger.info(f"Completed processing for document {pdf_id}")
            except Exception as e:
                logger.error(
                    f"Failed to process document {pdf_id}",
                    exc_info=True,
                    extra={"pdf_id": pdf_id}
                )
                errors[pdf_id] = str(e)
                
        # Only enable research mode if specifically requested and multiple documents were processed
        if enable_research_mode and len(normalized_pdf_ids) > 1:
            try:
                logger.info(f"Enabling research mode for documents: {normalized_pdf_ids}")
                
                # Initialize research context for LangGraph integration
                async def setup_research():
                    openai_client = AsyncOpenAI(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        timeout=60.0
                    )
                    
                    # Create a research manager that can be used by LangGraph nodes
                    research_manager = ResearchManager(
                        primary_pdf_id=normalized_pdf_ids[0],
                        research_mode=ResearchMode.MULTI
                    )
                    
                    # Activate all documents in the research context
                    for pdf_id in normalized_pdf_ids:
                        await research_manager.activate_document(pdf_id)
                    
                    # Build cross-document relationships for LangGraph to leverage
                    await research_manager.build_cross_document_relationships()
                    
                    # Get the research context which will be used by LangGraph nodes
                    research_context = await research_manager.get_research_context()
                    
                    return {
                        "status": "success",
                        "research_mode": "enabled",
                        "document_count": len(normalized_pdf_ids),
                        "cross_references": len(research_context.cross_document_evidence)
                    }
                
                # Set up and run the async loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    with current_app.app_context():
                        research_result = loop.run_until_complete(setup_research())
                        logger.info(f"Research mode enabled: {research_result}")
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(
                    "Failed to enable research mode",
                    exc_info=True
                )
                errors["research_mode"] = str(e)
        
        return {
            "status": "completed",
            "processed_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
            "processing_duration": (datetime.utcnow() - processing_start).total_seconds(),
            "langgraph_ready": True
        }
        
    except Exception as e:
        logger.error(
            "Failed to process multiple documents",
            exc_info=True
        )
        return {
            "status": "error",
            "error": str(e),
            "processing_duration": (datetime.utcnow() - processing_start).total_seconds()
        }
