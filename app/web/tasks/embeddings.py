import asyncio
import logging
import os
from celery import shared_task
from datetime import datetime
from typing import Dict, Any, Optional
from flask import current_app
from openai import AsyncOpenAI

from app.chat.types import ProcessingConfig
from app.web.db.models import Pdf
from app.web.db import db
from app.chat.document_fetcher import process_technical_document
from app.chat.utils.processing import normalize_pdf_id
from app.chat.vector_stores import get_vector_store

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_document(self, pdf_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Celery task for document processing with Neo4j + LangGraph integration.

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

        logger.info(f"Starting document processing for {normalized_pdf_id}")

        async def async_process():
            try:
                # Initialize OpenAI client
                openai_client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=60.0  # Increased timeout for comprehensive processing
                )

                # Initialize Neo4j vector store with robust checking
                vector_store = get_vector_store()
                vector_store_ready = False

                # Check if the vector store is initialized
                if hasattr(vector_store, 'initialized'):
                    vector_store_ready = vector_store.initialized
                elif hasattr(vector_store, '_initialized'):
                    vector_store_ready = vector_store._initialized

                if not vector_store_ready:
                    logger.warning(f"Vector store not initialized for {normalized_pdf_id}, attempting to initialize")

                    # Method 1: Try initialize_database() if it exists
                    if hasattr(vector_store, 'initialize_database') and callable(getattr(vector_store, 'initialize_database')):
                        try:
                            vector_store_ready = await vector_store.initialize_database()
                        except Exception as init_error:
                            logger.error(f"Error calling initialize_database: {str(init_error)}")

                    # Method 2: Try initialize() if it exists
                    elif hasattr(vector_store, 'initialize') and callable(getattr(vector_store, 'initialize')):
                        try:
                            vector_store_ready = vector_store.initialize()
                        except Exception as init_error:
                            logger.error(f"Error calling initialize: {str(init_error)}")

                    # Method 3: Manually try to connect and initialize
                    else:
                        try:
                            # Attempt to access driver to trigger connection
                            if hasattr(vector_store, 'driver') and vector_store.driver:
                                # Try to run a test query to verify connection
                                with vector_store.driver.session() as session:
                                    result = session.run("RETURN 1 as test")
                                    test_value = result.single()["test"]
                                    if test_value == 1:
                                        logger.info("Manually verified Neo4j connection")
                                        vector_store_ready = True
                        except Exception as manual_error:
                            logger.error(f"Manual connection verification failed: {str(manual_error)}")

                # Check if we were able to initialize the vector store
                if not vector_store_ready:
                    raise Exception("Vector store initialization failed, cannot proceed with document processing")

                logger.info(f"Neo4j vector store ready for document processing")

                # Process the document using document_fetcher
                result = await process_technical_document(
                    pdf_id=normalized_pdf_id,
                    config=processing_config,
                    openai_client=openai_client
                )

                # Update PDF status in database
                pdf = db.session.execute(
                    db.select(Pdf).filter_by(id=pdf_id)
                ).scalar_one()

                pdf.processed = True
                pdf.error = None

                # Store processing timestamp
                pdf.processed_at = datetime.utcnow()

                doc_summary_dict = None
                if hasattr(result, 'document_summary') and result.document_summary:
                    if hasattr(result.document_summary, 'dict'):
                        doc_summary_dict = result.document_summary.dict()
                    elif hasattr(result.document_summary, '__dict__'):
                        doc_summary_dict = result.document_summary.__dict__
                    else:
                        doc_summary_dict = {"title": str(result.document_summary)}

                pdf.update_metadata({
                    "document_summary": doc_summary_dict,
                    "processed_at": datetime.utcnow().isoformat(),
                    "neo4j_ready": True,
                    "langgraph_ready": True
                })

                # Commit changes
                db.session.commit()
                logger.info(f"Successfully processed document {normalized_pdf_id}")

                return {
                    "status": "success",
                    "pdf_id": normalized_pdf_id,
                    "processed_at": datetime.utcnow().isoformat(),
                    "element_count": len(result.elements) if hasattr(result, 'elements') else 0
                }

            except Exception as e:
                logger.error(f"Processing error for PDF {normalized_pdf_id}: {str(e)}", exc_info=True)

                # Update error status
                try:
                    pdf = db.session.execute(
                        db.select(Pdf).filter_by(id=pdf_id)
                    ).scalar_one()

                    pdf.processed = False
                    pdf.error = str(e)
                    pdf.processed_at = datetime.utcnow()
                    db.session.commit()
                    logger.info(f"Updated error status for {normalized_pdf_id}")
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
        logger.error(f"Task execution failed for PDF {normalized_pdf_id}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "pdf_id": normalized_pdf_id
        }
