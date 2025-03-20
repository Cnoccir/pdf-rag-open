"""
Unified retrieval interface coordinating MongoDB and Qdrant stores.
Provides a seamless interface similar to the previous Neo4j implementation.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
import uuid
import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.chat.vector_stores.mongo_store import get_mongo_store
from app.chat.vector_stores.qdrant_store import get_qdrant_store
from app.chat.types import ContentType, ContentElement

logger = logging.getLogger(__name__)

class UnifiedVectorStore:
    """
    Unified interface for MongoDB and Qdrant coordination.
    Provides a backward-compatible interface with Neo4j implementation.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for connection management"""
        if cls._instance is None:
            cls._instance = super(UnifiedVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        embedding_dimension: int = 1536,
        embedding_model: str = "text-embedding-3-small"
    ):
        # Skip initialization if already done (singleton pattern)
        if self._initialized:
            return

        # Embedding settings
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model

        # Initialize MongoDB and Qdrant stores
        self.mongo_store = get_mongo_store()
        self.qdrant_store = get_qdrant_store()

        # Track initialization state
        self._initialized = self.mongo_store._initialized and self.qdrant_store._initialized

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Initialize metrics
        self.metrics = {
            "queries": 0,
            "errors": 0,
            "avg_query_time": 0,
            "total_query_time": 0
        }

        logger.info(f"Unified Vector Store initialized with {self.embedding_dimension} dimensions")

    def initialize(self) -> bool:
        """
        Initialize both MongoDB and Qdrant stores.
        Returns: Success status
        """
        # Try initializing MongoDB if not already initialized
        if not self.mongo_store._initialized:
            if not self.mongo_store.initialize():
                logger.error("Failed to initialize MongoDB store")
                return False

        # Try initializing Qdrant if not already initialized
        if not self.qdrant_store._initialized:
            if not self.qdrant_store.initialize():
                logger.error("Failed to initialize Qdrant store")
                return False

        # Mark as initialized
        self._initialized = True
        return True

    def create_document_node(
        self,
        pdf_id: str,
        title: str = "Untitled Document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a document entry in the metadata store.

        Args:
            pdf_id: Document ID
            title: Document title
            metadata: Additional metadata

        Returns:
            Success status
        """
        # Forward to MongoDB store
        return self.mongo_store.create_document_node(pdf_id, title, metadata)

    def add_content_element(
        self,
        element: Union[ContentElement, Dict[str, Any]],
        pdf_id: str
    ) -> bool:
        """
        Add content element to both MongoDB and Qdrant.

        Args:
            element: Content element object or dictionary
            pdf_id: PDF ID

        Returns:
            Success status
        """
        if not self._initialized:
            if not self.initialize():
                return False

        try:
            # Add to MongoDB
            mongo_success = self.mongo_store.add_content_element(element, pdf_id)

            # Convert element for Qdrant if needed
            if hasattr(element, "dict") and callable(getattr(element, "dict")):
                qdrant_element = element.dict()
            elif isinstance(element, dict):
                qdrant_element = element
            else:
                # Create element dictionary
                qdrant_element = {
                    "element_id": getattr(element, "element_id", f"elem_{pdf_id}_{uuid.uuid4()}"),
                    "content": getattr(element, "content", ""),
                    "content_type": str(getattr(element, "content_type", "")),
                    "pdf_id": pdf_id,
                    "page": getattr(element, "page", 0)
                }

                # Get metadata
                if hasattr(element, "metadata"):
                    # Handle different metadata formats
                    if hasattr(element.metadata, "dict") and callable(getattr(element.metadata, "dict")):
                        qdrant_element.update(element.metadata.dict())
                    elif isinstance(element.metadata, dict):
                        qdrant_element.update(element.metadata)

            # Add metadata from element
            if not isinstance(qdrant_element, dict):
                logger.error(f"Invalid element type for Qdrant: {type(qdrant_element)}")
                return False

            # Ensure content is good for embedding
            if not qdrant_element.get("content"):
                logger.warning(f"Empty content for element {qdrant_element.get('element_id')}, skipping Qdrant storage")
                return mongo_success

            # Add to Qdrant
            qdrant_success = len(self.qdrant_store.add_elements([qdrant_element])) > 0

            # Return combined success
            return mongo_success and qdrant_success

        except Exception as e:
            logger.error(f"Error adding content element: {str(e)}")
            return False

    def add_concept(
        self,
        concept_name: str,
        pdf_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a concept to MongoDB.

        Args:
            concept_name: Concept name/term
            pdf_id: PDF ID
            metadata: Concept metadata

        Returns:
            Success status
        """
        # Forward to MongoDB store
        return self.mongo_store.add_concept(concept_name, pdf_id, metadata)

    def add_concept_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        pdf_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add relationship between concepts to MongoDB.

        Args:
            source: Source concept name
            target: Target concept name
            rel_type: Relationship type
            pdf_id: PDF ID
            metadata: Relationship metadata

        Returns:
            Success status
        """
        # Forward to MongoDB store
        return self.mongo_store.add_concept_relationship(source, target, rel_type, pdf_id, metadata)

    def add_section_concept_relation(
        self,
        section: str,
        concept: str,
        pdf_id: str
    ) -> bool:
        """
        Add section-concept relationship to MongoDB.

        Args:
            section: Section path or title
            concept: Concept name
            pdf_id: PDF ID

        Returns:
            Success status
        """
        # Forward to MongoDB store
        return self.mongo_store.add_section_concept_relation(section, concept, pdf_id)

    def semantic_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Query text
            k: Number of results to return
            pdf_id: Optional PDF ID to filter by
            content_types: Optional content types to filter by

        Returns:
            List of document results
        """
        if not self._initialized:
            if not self.initialize():
                return []

        start_time = datetime.utcnow()

        try:
            # Build filter for Qdrant
            filter_dict = {}

            if pdf_id:
                filter_dict["pdf_id"] = pdf_id

            if content_types and len(content_types) > 0:
                filter_dict["content_type"] = content_types

            # Execute search
            search_results = self.qdrant_store.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict
            )

            # Update metrics
            self.metrics["queries"] += 1
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["total_query_time"] += query_time
            self.metrics["avg_query_time"] = self.metrics["total_query_time"] / self.metrics["queries"]

            logger.info(f"Semantic search completed in {query_time:.2f}s, found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            self.metrics["errors"] += 1
            return []

    def keyword_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None
    ) -> List[Document]:
        """
        Perform keyword search using standard index matching.

        Args:
            query: Query text
            k: Number of results to return
            pdf_id: Optional PDF ID to filter by

        Returns:
            List of document results
        """
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            # Use MongoDB for keyword search
            search_results = self.mongo_store.keyword_search(
                query=query,
                pdf_id=pdf_id,
                limit=k
            )

            # Convert to Document objects
            documents = []
            for result in search_results:
                metadata = result.copy()
                content = metadata.pop("content", "")

                doc = Document(
                    page_content=content,
                    metadata=metadata
                )

                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            self.metrics["errors"] += 1
            return []

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Query text
            k: Number of results to return
            pdf_id: Optional PDF ID to filter by
            content_types: Optional content types to filter by

        Returns:
            List of document results
        """
        if not self._initialized:
            if not self.initialize():
                return []

        start_time = datetime.utcnow()

        try:
            # Perform semantic search
            semantic_results = self.semantic_search(query, k, pdf_id, content_types)

            # Perform keyword search
            keyword_results = self.keyword_search(query, k, pdf_id)

            # Combine results
            combined_results = {}

            # Add semantic results
            for doc in semantic_results:
                doc_id = doc.metadata.get("element_id")
                if not doc_id:
                    continue

                combined_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": doc.metadata.get("score", 0.0),
                    "keyword_score": 0.0
                }

            # Add keyword results
            for doc in keyword_results:
                doc_id = doc.metadata.get("element_id")
                if not doc_id:
                    continue

                if doc_id in combined_results:
                    combined_results[doc_id]["keyword_score"] = doc.metadata.get("score", 0.0)
                else:
                    combined_results[doc_id] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "keyword_score": doc.metadata.get("score", 0.0)
                    }

            # Calculate combined scores
            for doc_id, data in combined_results.items():
                combined_score = (data["semantic_score"] * 0.7) + (data["keyword_score"] * 0.3)
                data["combined_score"] = combined_score
                data["doc"].metadata["score"] = combined_score

            # Sort by combined score
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )

            # Return top k results
            final_results = [data["doc"] for data in sorted_results[:k]]

            # Update metrics
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["queries"] += 1
            self.metrics["total_query_time"] += query_time
            self.metrics["avg_query_time"] = self.metrics["total_query_time"] / self.metrics["queries"]

            logger.info(f"Hybrid search completed in {query_time:.2f}s, found {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            self.metrics["errors"] += 1
            return []

    def delete_document(self, pdf_id: str) -> bool:
        """
        Delete a document and its content from both MongoDB and Qdrant.

        Args:
            pdf_id: PDF ID

        Returns:
            Success status
        """
        if not self._initialized:
            if not self.initialize():
                return False

        try:
            # Delete from MongoDB
            mongo_success = self.mongo_store.delete_document(pdf_id)

            # Delete from Qdrant
            qdrant_success = self.qdrant_store.delete_by_pdf_id(pdf_id)

            logger.info(f"Deleted document {pdf_id} from unified store")
            return mongo_success and qdrant_success

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def close(self) -> None:
        """Close all connections"""
        try:
            if self.mongo_store:
                self.mongo_store.close()

            if self.qdrant_store:
                self.qdrant_store.close()

            self._initialized = False
            logger.info("Unified store connections closed")

        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")

    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of both MongoDB and Qdrant.

        Returns:
            Dictionary with health status information
        """
        health_info = {
            "status": "error",
            "mongo_ready": False,
            "qdrant_ready": False,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self._initialized:
            health_info["error"] = "Unified store not initialized"
            return health_info

        try:
            # Check MongoDB health
            mongo_health = await self.mongo_store.check_health()
            health_info["mongo_ready"] = mongo_health.get("database_ready", False)
            health_info["mongo_status"] = mongo_health

            # Check Qdrant health
            qdrant_health = await self.qdrant_store.check_health()
            health_info["qdrant_ready"] = qdrant_health.get("database_ready", False)
            health_info["qdrant_status"] = qdrant_health

            # Determine overall status
            if health_info["mongo_ready"] and health_info["qdrant_ready"]:
                health_info["status"] = "ok"
            else:
                health_info["status"] = "degraded"

            return health_info

        except Exception as e:
            health_info["error"] = str(e)
            return health_info

# Create an alias for backward compatibility with previous Neo4j implementation
TechDocVectorStore = UnifiedVectorStore

# Singleton instance getter
def get_vector_store() -> UnifiedVectorStore:
    """Get or create unified vector store instance"""
    return UnifiedVectorStore()
