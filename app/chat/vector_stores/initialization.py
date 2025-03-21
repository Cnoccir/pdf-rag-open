import logging
import time
from typing import Dict, Any

from app.chat.vector_stores import get_vector_store, get_mongo_store, get_qdrant_store

logger = logging.getLogger(__name__)

def initialize_and_verify_stores(max_retries: int = 3, retry_delay: int = 5) -> Dict[str, Any]:
    """
    Initialize and verify all vector stores with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Dictionary with initialization status
    """
    results = {
        "success": False,
        "mongo_initialized": False,
        "qdrant_initialized": False,
        "unified_initialized": False,
        "errors": []
    }

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to initialize vector stores (attempt {attempt+1}/{max_retries})")

            # Initialize MongoDB
            mongo_store = get_mongo_store()
            mongo_success = mongo_store.initialize()
            results["mongo_initialized"] = mongo_success

            if not mongo_success:
                results["errors"].append(f"MongoDB initialization failed: {mongo_store.error}")
                logger.error(f"MongoDB initialization failed: {mongo_store.error}")

            # Initialize Qdrant
            qdrant_store = get_qdrant_store()
            qdrant_success = qdrant_store.initialize()
            results["qdrant_initialized"] = qdrant_success

            if not qdrant_success:
                results["errors"].append(f"Qdrant initialization failed: {qdrant_store.error}")
                logger.error(f"Qdrant initialization failed: {qdrant_store.error}")

            # Initialize unified store
            unified_store = get_vector_store()
            unified_success = unified_store.initialize()
            results["unified_initialized"] = unified_success

            if not unified_success:
                results["errors"].append("Unified store initialization failed")
                logger.error("Unified store initialization failed")

            # Check overall success
            if mongo_success and qdrant_success and unified_success:
                results["success"] = True
                logger.info("All vector stores initialized successfully")

                # Log collection statistics
                try:
                    mongo_stats = mongo_store.get_stats()
                    logger.info(f"MongoDB collections: {mongo_stats.get('collection_counts', {})}")

                    # Log a sample of document IDs
                    if mongo_store.db and hasattr(mongo_store.db, "documents"):
                        sample_docs = list(mongo_store.db.documents.find({}, {"pdf_id": 1, "title": 1}).limit(5))
                        if sample_docs:
                            logger.info(f"Sample documents: {sample_docs}")
                except Exception as stats_error:
                    logger.warning(f"Error getting MongoDB stats: {str(stats_error)}")

                return results

            # If not successful, retry after delay
            logger.warning(f"Vector store initialization incomplete, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        except Exception as e:
            error_msg = f"Error during vector store initialization: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
            time.sleep(retry_delay)

    # If we get here, all retries failed
    logger.error(f"Failed to initialize vector stores after {max_retries} attempts")
    return results

# Function to check if a given PDF exists in the vector stores
def verify_pdf_exists(pdf_id: str) -> Dict[str, Any]:
    """
    Verify if a PDF exists in both MongoDB and Qdrant.

    Args:
        pdf_id: PDF ID to check

    Returns:
        Dictionary with verification results
    """
    results = {
        "pdf_id": pdf_id,
        "exists_in_mongo": False,
        "exists_in_qdrant": False,
        "document_title": None,
        "element_count": 0,
        "embedding_count": 0
    }

    try:
        # Check MongoDB
        mongo_store = get_mongo_store()
        if mongo_store._initialized:
            # Check document
            doc = mongo_store.get_document(pdf_id)
            if doc:
                results["exists_in_mongo"] = True
                results["document_title"] = doc.get("title")

                # Count elements
                elements = mongo_store.get_elements_by_pdf_id(pdf_id, limit=1000)
                results["element_count"] = len(elements)

                logger.info(f"Found PDF {pdf_id} in MongoDB with {len(elements)} elements")

        # Check Qdrant
        qdrant_store = get_qdrant_store()
        if qdrant_store._initialized and qdrant_store.client:
            try:
                # Count vectors for this PDF
                count_result = qdrant_store.client.count(
                    collection_name=qdrant_store.collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="pdf_id",
                                match=models.MatchValue(value=pdf_id)
                            )
                        ]
                    )
                )

                results["embedding_count"] = count_result.count
                results["exists_in_qdrant"] = count_result.count > 0

                logger.info(f"Found PDF {pdf_id} in Qdrant with {count_result.count} embeddings")
            except Exception as qdrant_error:
                logger.error(f"Error checking Qdrant for PDF {pdf_id}: {str(qdrant_error)}")

        return results

    except Exception as e:
        logger.error(f"Error verifying PDF {pdf_id}: {str(e)}", exc_info=True)
        results["error"] = str(e)
        return results
