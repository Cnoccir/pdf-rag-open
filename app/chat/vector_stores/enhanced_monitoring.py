"""
Enhanced monitoring capabilities for vector stores and conversation management.
This module provides additional methods for collecting statistics and health checks.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EnhancedMonitoring:
    """
    Enhanced monitoring capabilities that can be added to existing classes.
    """

    @staticmethod
    async def get_embedding_stats(self, pdf_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get detailed embedding statistics for documents.

        Args:
            pdf_ids: Optional list of PDF IDs to check (default: all documents)

        Returns:
            Dictionary with embedding statistics
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Vector store not initialized"}

        try:
            # Initialize statistics
            stats = {
                "total": 0,
                "documents": {},
                "content_types": {},
                "chunk_levels": {},
                "embedding_types": {}
            }

            # Get counts from Qdrant
            if hasattr(self.qdrant_store, "client") and self.qdrant_store.client:
                # First get overall collection stats
                collection_info = self.qdrant_store.client.get_collection(
                    collection_name=self.qdrant_store.collection_name
                )

                # Get total vector count
                collection_count = self.qdrant_store.client.count(
                    collection_name=self.qdrant_store.collection_name,
                    count_filter=None
                )

                stats["total"] = collection_count.count

                # If we have specific PDF IDs, get counts for each
                if pdf_ids:
                    for pdf_id in pdf_ids:
                        # Count vectors for this PDF
                        pdf_count = self.qdrant_store.client.count(
                            collection_name=self.qdrant_store.collection_name,
                            count_filter={
                                "must": [
                                    {"key": "pdf_id", "match": {"value": pdf_id}}
                                ]
                            }
                        )

                        stats["documents"][pdf_id] = pdf_count.count

                        # Get content type breakdown
                        content_types = await self._get_content_type_counts(pdf_id)
                        for content_type, count in content_types.items():
                            # Add to global content type counts
                            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + count

                        # Get chunk level breakdown
                        chunk_levels = await self._get_chunk_level_counts(pdf_id)
                        for chunk_level, count in chunk_levels.items():
                            # Add to global chunk level counts
                            stats["chunk_levels"][chunk_level] = stats["chunk_levels"].get(chunk_level, 0) + count

                        # Get embedding type breakdown
                        embedding_types = await self._get_embedding_type_counts(pdf_id)
                        for embedding_type, count in embedding_types.items():
                            # Add to global embedding type counts
                            stats["embedding_types"][embedding_type] = stats["embedding_types"].get(embedding_type, 0) + count

            return stats

        except Exception as e:
            logger.error(f"Error getting embedding stats: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    async def get_embedding_stats_for_pdf(self, pdf_id: str) -> Dict[str, Any]:
        """
        Get detailed embedding statistics for a specific PDF.

        Args:
            pdf_id: PDF ID to check

        Returns:
            Dictionary with embedding statistics for the PDF
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "Vector store not initialized"}

        try:
            # Initialize statistics
            stats = {
                "total": 0,
                "content_types": {},
                "chunk_levels": {},
                "embedding_types": {}
            }

            # Get counts from Qdrant
            if hasattr(self.qdrant_store, "client") and self.qdrant_store.client:
                # Count vectors for this PDF
                pdf_count = self.qdrant_store.client.count(
                    collection_name=self.qdrant_store.collection_name,
                    count_filter={
                        "must": [
                            {"key": "pdf_id", "match": {"value": pdf_id}}
                        ]
                    }
                )

                stats["total"] = pdf_count.count

                # Get content type breakdown
                stats["content_types"] = await self._get_content_type_counts(pdf_id)

                # Get chunk level breakdown
                stats["chunk_levels"] = await self._get_chunk_level_counts(pdf_id)

                # Get embedding type breakdown
                stats["embedding_types"] = await self._get_embedding_type_counts(pdf_id)

            return stats

        except Exception as e:
            logger.error(f"Error getting embedding stats for PDF {pdf_id}: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    async def _get_content_type_counts(self, pdf_id: str) -> Dict[str, int]:
        """
        Get counts of embeddings by content type for a PDF.

        Args:
            pdf_id: PDF ID to check

        Returns:
            Dictionary mapping content types to counts
        """
        content_counts = {}

        try:
            # Get distinct content types for this PDF
            search_result = self.qdrant_store.client.scroll(
                collection_name=self.qdrant_store.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "pdf_id", "match": {"value": pdf_id}}
                    ]
                },
                limit=100,
                with_payload=["content_type"],
                with_vectors=False
            )

            # Count occurrences of each content type
            points = search_result[0]
            for point in points:
                if point.payload and "content_type" in point.payload:
                    content_type = point.payload["content_type"]
                    content_counts[content_type] = content_counts.get(content_type, 0) + 1

        except Exception as e:
            logger.error(f"Error getting content type counts: {str(e)}")

        return content_counts

    @staticmethod
    async def _get_chunk_level_counts(self, pdf_id: str) -> Dict[str, int]:
        """
        Get counts of embeddings by chunk level for a PDF.

        Args:
            pdf_id: PDF ID to check

        Returns:
            Dictionary mapping chunk levels to counts
        """
        chunk_counts = {}

        try:
            # Get distinct chunk levels for this PDF
            search_result = self.qdrant_store.client.scroll(
                collection_name=self.qdrant_store.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "pdf_id", "match": {"value": pdf_id}}
                    ]
                },
                limit=100,
                with_payload=["chunk_level"],
                with_vectors=False
            )

            # Count occurrences of each chunk level
            points = search_result[0]
            for point in points:
                if point.payload and "chunk_level" in point.payload:
                    chunk_level = point.payload["chunk_level"]
                    chunk_counts[chunk_level] = chunk_counts.get(chunk_level, 0) + 1

        except Exception as e:
            logger.error(f"Error getting chunk level counts: {str(e)}")

        return chunk_counts

    @staticmethod
    async def _get_embedding_type_counts(self, pdf_id: str) -> Dict[str, int]:
        """
        Get counts of embeddings by embedding type for a PDF.

        Args:
            pdf_id: PDF ID to check

        Returns:
            Dictionary mapping embedding types to counts
        """
        embedding_counts = {}

        try:
            # Get distinct embedding types for this PDF
            search_result = self.qdrant_store.client.scroll(
                collection_name=self.qdrant_store.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "pdf_id", "match": {"value": pdf_id}}
                    ]
                },
                limit=100,
                with_payload=["embedding_type"],
                with_vectors=False
            )

            # Count occurrences of each embedding type
            points = search_result[0]
            for point in points:
                if point.payload and "embedding_type" in point.payload:
                    embedding_type = point.payload["embedding_type"]
                    embedding_counts[embedding_type] = embedding_counts.get(embedding_type, 0) + 1

        except Exception as e:
            logger.error(f"Error getting embedding type counts: {str(e)}")

        return embedding_counts

    @staticmethod
    async def get_counts_by_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, int]:
        """
        Get counts of vectors matching a filter.

        Args:
            filter_dict: Filter dictionary

        Returns:
            Dictionary with counts
        """
        if not self._initialized:
            if not self.initialize():
                return {"total": 0}

        try:
            # Convert filter_dict to Qdrant filter
            filter_condition = None
            if filter_dict:
                from qdrant_client.http import models
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

                if conditions:
                    filter_condition = models.Filter(
                        must=conditions
                    )

            # Get count
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_condition
            )

            return {"total": count_result.count}

        except Exception as e:
            logger.error(f"Error getting counts by filter: {str(e)}")
            return {"total": 0, "error": str(e)}

    @staticmethod
    def get_mongo_stats(self) -> Dict[str, Any]:
        """
        Get enhanced statistics about the MongoDB store.

        Returns:
            Dictionary with MongoDB statistics
        """
        if not self._initialized:
            if not self.initialize():
                return {"error": "MongoDB not initialized"}

        try:
            stats = {
                "collection_counts": {},
                "document_stats": {},
                "timestamp": datetime.utcnow().isoformat()
            }

            # Get collection counts
            for collection_name in ["documents", "content_elements", "concepts", "relationships", "procedures", "parameters"]:
                try:
                    if hasattr(self.db, collection_name):
                        collection = getattr(self.db, collection_name)
                        stats["collection_counts"][collection_name] = collection.count_documents({})
                except Exception as coll_err:
                    stats["collection_counts"][collection_name] = f"error: {str(coll_err)}"

            # Get document stats
            if hasattr(self.db, "documents"):
                # Count documents by metadata fields
                try:
                    # Count by category
                    category_pipeline = [
                        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ]
                    category_counts = list(self.db.documents.aggregate(category_pipeline))
                    stats["document_stats"]["categories"] = {item["_id"] or "unknown": item["count"] for item in category_counts}

                    # Count by processing status (if field exists)
                    status_pipeline = [
                        {"$group": {"_id": "$processed", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ]
                    status_counts = list(self.db.documents.aggregate(status_pipeline))
                    stats["document_stats"]["processed_status"] = {str(item["_id"]): item["count"] for item in status_counts}

                except Exception as doc_err:
                    stats["document_stats"]["error"] = str(doc_err)

            # Get concept stats
            if hasattr(self.db, "concepts"):
                try:
                    # Count by primary vs regular concepts
                    primary_pipeline = [
                        {"$group": {"_id": "$is_primary", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ]
                    primary_counts = list(self.db.concepts.aggregate(primary_pipeline))
                    stats["concept_stats"] = {
                        "primary_vs_regular": {str(item["_id"]): item["count"] for item in primary_counts},
                        "total": sum(item["count"] for item in primary_counts)
                    }

                    # Get document with most concepts
                    doc_concept_pipeline = [
                        {"$group": {"_id": "$pdf_id", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": 5}
                    ]
                    doc_concept_counts = list(self.db.concepts.aggregate(doc_concept_pipeline))
                    stats["concept_stats"]["top_documents"] = {item["_id"]: item["count"] for item in doc_concept_counts}

                except Exception as concept_err:
                    stats["concept_stats"]["error"] = str(concept_err)

            return stats

        except Exception as e:
            logger.error(f"Error getting MongoDB stats: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory management system.

        Returns:
            Dictionary with memory manager statistics
        """
        stats = {
            "conversation_count": 0,
            "pdf_distribution": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            # Get all conversations
            conversations = self.list_conversations()
            stats["conversation_count"] = len(conversations)

            # Count conversations by PDF ID
            pdf_counts = {}
            for conv in conversations:
                pdf_id = conv.pdf_id
                if pdf_id:
                    pdf_counts[pdf_id] = pdf_counts.get(pdf_id, 0) + 1

            # Get top PDFs by conversation count
            sorted_pdfs = sorted(pdf_counts.items(), key=lambda x: x[1], reverse=True)
            stats["pdf_distribution"] = {pdf_id: count for pdf_id, count in sorted_pdfs[:10]}

            # Count messages
            message_counts = {"total": 0, "by_type": {}}
            for conv in conversations:
                for msg in conv.messages:
                    message_counts["total"] += 1
                    msg_type = msg.type.value if hasattr(msg.type, "value") else str(msg.type)
                    message_counts["by_type"][msg_type] = message_counts["by_type"].get(msg_type, 0) + 1

            stats["message_counts"] = message_counts

            # Get average messages per conversation
            if stats["conversation_count"] > 0:
                stats["avg_messages_per_conversation"] = message_counts["total"] / stats["conversation_count"]

            return stats

        except Exception as e:
            logger.error(f"Error getting memory manager stats: {str(e)}")
            return {"error": str(e)}

def enhance_vector_stores():
    """Connect the enhancement methods to their respective classes."""

    # Import the necessary classes
    from app.chat.vector_stores.unified_store import UnifiedVectorStore
    from app.chat.vector_stores.qdrant_store import QdrantStore
    from app.chat.vector_stores.mongo_store import MongoStore
    from app.chat.memories.memory_manager import MemoryManager

    # Add methods to UnifiedVectorStore
    UnifiedVectorStore.get_embedding_stats = EnhancedMonitoring.get_embedding_stats
    UnifiedVectorStore.get_embedding_stats_for_pdf = EnhancedMonitoring.get_embedding_stats_for_pdf
    UnifiedVectorStore._get_content_type_counts = EnhancedMonitoring._get_content_type_counts
    UnifiedVectorStore._get_chunk_level_counts = EnhancedMonitoring._get_chunk_level_counts
    UnifiedVectorStore._get_embedding_type_counts = EnhancedMonitoring._get_embedding_type_counts

    # Add methods to QdrantStore
    QdrantStore.get_counts_by_filter = EnhancedMonitoring.get_counts_by_filter

    # Add methods to MongoStore
    MongoStore.get_stats = EnhancedMonitoring.get_mongo_stats

    # Add methods to MemoryManager
    MemoryManager.get_stats = EnhancedMonitoring.get_memory_stats

    logger.info("Enhanced vector stores and memory manager with monitoring capabilities")
