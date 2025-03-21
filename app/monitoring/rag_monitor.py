# app/monitoring/rag_monitor.py

"""
Enhanced RAG monitoring system.
Provides comprehensive metrics collection and performance analysis.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import threading
import uuid

logger = logging.getLogger(__name__)

class RAGMonitor:
    """
    Enhanced monitoring solution for RAG operations.
    Tracks metrics, performance, and system health for analysis.
    """

    def __init__(self, mongo_client=None, db_name=None):
        """
        Initialize the RAG monitor.

        Args:
            mongo_client: Optional MongoDB client
            db_name: Optional MongoDB database name
        """
        self.mongo_client = mongo_client
        self.db_name = db_name or "rag_monitoring"
        self.collection_name = "metrics"
        self._initialized = False
        self._lock = threading.Lock()
        self._buffer = []  # Buffer metrics before writing to DB
        self._buffer_size = 10  # Write to DB every N metrics
        self._init_db()

        # In-memory stats - updated even if DB connection fails
        self._stats = {
            "query_count": 0,
            "processing_count": 0,
            "error_count": 0,
            "avg_query_time_ms": 0,
            "total_query_time_ms": 0,
            "avg_processing_time_ms": 0,
            "total_processing_time_ms": 0,
            "pdf_stats": {}  # PDF ID -> stats
        }

        logger.info("RAG Monitor initialized")

    def _init_db(self):
        """Initialize MongoDB connection and create indexes."""
        if self._initialized:
            return True

        with self._lock:
            if self._initialized:
                return True

            try:
                if not self.mongo_client:
                    from pymongo import MongoClient
                    import os

                    # Use environment variables or default
                    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
                    self.mongo_client = MongoClient(mongo_uri)

                self.db = self.mongo_client[self.db_name]
                self.metrics = self.db[self.collection_name]

                # Create indexes
                self.metrics.create_index("timestamp")
                self.metrics.create_index("operation")
                self.metrics.create_index("pdf_id")

                self._initialized = True
                logger.info("RAG Monitor connected to MongoDB")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize RAG Monitor: {str(e)}")
                return False

    def record_operation(self, operation: str, details: Optional[Dict[str, Any]] = None,
                        pdf_id: Optional[str] = None, duration_ms: Optional[float] = None,
                        user_id: Optional[str] = None):
        """
        Record an operation for monitoring.

        Args:
            operation: Operation type (e.g., 'query', 'processing')
            details: Operation details
            pdf_id: PDF ID related to the operation
            duration_ms: Operation duration in milliseconds
            user_id: User ID who performed the operation
        """
        # Create metric record
        metric = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow(),
            "operation": operation,
            "pdf_id": pdf_id,
            "user_id": user_id,
            "duration_ms": duration_ms,
            "details": details or {}
        }

        # Update in-memory stats
        self._update_stats(metric)

        # Add to buffer
        with self._lock:
            self._buffer.append(metric)

            # If buffer is full, write to DB
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def _update_stats(self, metric: Dict[str, Any]):
        """Update in-memory stats with the new metric."""
        with self._lock:
            operation = metric["operation"]
            duration = metric.get("duration_ms", 0)
            pdf_id = metric.get("pdf_id")

            # Update operation counts
            if operation == "query":
                self._stats["query_count"] += 1
                self._stats["total_query_time_ms"] += duration
                self._stats["avg_query_time_ms"] = self._stats["total_query_time_ms"] / self._stats["query_count"]
            elif operation == "processing":
                self._stats["processing_count"] += 1
                self._stats["total_processing_time_ms"] += duration
                self._stats["avg_processing_time_ms"] = self._stats["total_processing_time_ms"] / self._stats["processing_count"]
            elif operation == "error":
                self._stats["error_count"] += 1

            # Update PDF-specific stats
            if pdf_id:
                if pdf_id not in self._stats["pdf_stats"]:
                    self._stats["pdf_stats"][pdf_id] = {
                        "query_count": 0,
                        "processing_count": 0,
                        "error_count": 0,
                        "total_duration_ms": 0
                    }

                pdf_stats = self._stats["pdf_stats"][pdf_id]

                if operation == "query":
                    pdf_stats["query_count"] += 1
                elif operation == "processing":
                    pdf_stats["processing_count"] += 1
                elif operation == "error":
                    pdf_stats["error_count"] += 1

                pdf_stats["total_duration_ms"] += duration

    def _flush_buffer(self):
        """Write buffered metrics to MongoDB."""
        if not self._buffer:
            return

        if not self._initialized and not self._init_db():
            logger.warning("Failed to write metrics: MongoDB not initialized")
            return

        try:
            # Convert timestamps to make them MongoDB-compatible
            metrics_to_write = []
            for metric in self._buffer:
                metric_copy = metric.copy()
                if isinstance(metric_copy["timestamp"], datetime):
                    # No change needed
                    pass
                metrics_to_write.append(metric_copy)

            # Insert metrics
            if metrics_to_write:
                self.metrics.insert_many(metrics_to_write)

            # Clear buffer
            self._buffer = []

        except Exception as e:
            logger.error(f"Failed to write metrics to MongoDB: {str(e)}")

    def record_query(self, query: str, pdf_ids: List[str], results_count: int,
                    duration_ms: float, strategy: str, user_id: Optional[str] = None):
        """
        Record a query operation with enhanced details.

        Args:
            query: The user's query
            pdf_ids: List of PDF IDs queried
            results_count: Number of results returned
            duration_ms: Query duration in milliseconds
            strategy: Search strategy used
            user_id: User ID who performed the query
        """
        details = {
            "query": query,
            "pdf_ids": pdf_ids,
            "results_count": results_count,
            "strategy": strategy,
            "query_length": len(query),
            "success": results_count > 0
        }

        self.record_operation(
            operation="query",
            details=details,
            pdf_id=pdf_ids[0] if pdf_ids else None,
            duration_ms=duration_ms,
            user_id=user_id
        )

    def record_processing(self, pdf_id: str, element_count: int, chunk_count: int,
                         duration_ms: float, user_id: Optional[str] = None):
        """
        Record document processing metrics.

        Args:
            pdf_id: PDF ID being processed
            element_count: Number of content elements extracted
            chunk_count: Number of chunks created
            duration_ms: Processing duration in milliseconds
            user_id: User ID who initiated processing
        """
        details = {
            "element_count": element_count,
            "chunk_count": chunk_count,
            "elements_per_second": (element_count / (duration_ms / 1000)) if duration_ms > 0 else 0,
            "chunks_per_second": (chunk_count / (duration_ms / 1000)) if duration_ms > 0 else 0
        }

        self.record_operation(
            operation="processing",
            details=details,
            pdf_id=pdf_id,
            duration_ms=duration_ms,
            user_id=user_id
        )

    def record_error(self, operation: str, error_message: str, pdf_id: Optional[str] = None,
                    user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Record an error event.

        Args:
            operation: Operation that encountered the error
            error_message: Error message
            pdf_id: PDF ID related to the error
            user_id: User ID related to the error
            details: Additional error details
        """
        error_details = details or {}
        error_details["error_message"] = error_message
        error_details["error_operation"] = operation

        self.record_operation(
            operation="error",
            details=error_details,
            pdf_id=pdf_id,
            user_id=user_id
        )

    def get_recent_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent operations for dashboard display.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of recent operations
        """
        try:
            # Make sure to flush any pending operations
            self._flush_buffer()

            if not self._initialized and not self._init_db():
                logger.warning("Failed to get operations: MongoDB not initialized")
                return []

            return list(self.metrics.find().sort("timestamp", -1).limit(limit))

        except Exception as e:
            logger.error(f"Failed to get recent operations: {str(e)}")
            return []

    def get_pdf_metrics(self, pdf_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific PDF.

        Args:
            pdf_id: PDF ID to get metrics for

        Returns:
            Dictionary with PDF metrics
        """
        try:
            # Make sure to flush any pending operations
            self._flush_buffer()

            if not self._initialized and not self._init_db():
                # Return in-memory stats if available
                if pdf_id in self._stats["pdf_stats"]:
                    return self._stats["pdf_stats"][pdf_id]
                return {}

            # Get metrics for the PDF
            metrics = list(self.metrics.find({"pdf_id": pdf_id}).sort("timestamp", -1))

            # Compute aggregated metrics
            result = {
                "pdf_id": pdf_id,
                "total_operations": len(metrics),
                "operation_counts": {},
                "recent_operations": [],
                "performance": {}
            }

            # Count operations by type
            op_counts = {}
            op_durations = {}
            for metric in metrics:
                op_type = metric.get("operation", "unknown")
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

                duration = metric.get("duration_ms", 0)
                if op_type not in op_durations:
                    op_durations[op_type] = []
                op_durations[op_type].append(duration)

            result["operation_counts"] = op_counts

            # Calculate performance metrics
            for op_type, durations in op_durations.items():
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    max_duration = max(durations)
                    min_duration = min(durations)

                    result["performance"][op_type] = {
                        "avg_duration_ms": avg_duration,
                        "max_duration_ms": max_duration,
                        "min_duration_ms": min_duration,
                        "total_duration_ms": sum(durations)
                    }

            # Get 10 most recent operations
            result["recent_operations"] = metrics[:10]

            return result

        except Exception as e:
            logger.error(f"Failed to get PDF metrics: {str(e)}")

            # Return in-memory stats if available
            if pdf_id in self._stats["pdf_stats"]:
                return self._stats["pdf_stats"][pdf_id]
            return {}

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics for overall system performance.

        Returns:
            Dictionary with summary metrics
        """
        try:
            # Make sure to flush any pending operations
            self._flush_buffer()

            if not self._initialized and not self._init_db():
                # Return in-memory stats
                return self._stats

            # Get last 24 hours of metrics
            start_time = datetime.utcnow() - timedelta(days=1)
            recent_metrics = list(self.metrics.find({"timestamp": {"$gte": start_time}}))

            # Compute summary metrics
            summary = {
                "last_24h": {
                    "total_operations": len(recent_metrics),
                    "operation_counts": {},
                    "avg_durations": {},
                    "error_rate": 0
                },
                "all_time": self._stats
            }

            # Count operations by type
            op_counts = {}
            op_durations = {}
            error_count = 0

            for metric in recent_metrics:
                op_type = metric.get("operation", "unknown")
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

                if op_type == "error":
                    error_count += 1

                duration = metric.get("duration_ms", 0)
                if op_type not in op_durations:
                    op_durations[op_type] = []
                op_durations[op_type].append(duration)

            summary["last_24h"]["operation_counts"] = op_counts

            # Calculate average durations
            for op_type, durations in op_durations.items():
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    summary["last_24h"]["avg_durations"][op_type] = avg_duration

            # Calculate error rate
            if recent_metrics:
                summary["last_24h"]["error_rate"] = error_count / len(recent_metrics)

            return summary

        except Exception as e:
            logger.error(f"Failed to get summary metrics: {str(e)}")

            # Return in-memory stats
            return self._stats

    def close(self):
        """Close MongoDB connection and flush any remaining metrics."""
        try:
            # Flush buffer
            self._flush_buffer()

            # Close MongoDB connection
            if self.mongo_client:
                self.mongo_client.close()
                self.mongo_client = None
                self._initialized = False

        except Exception as e:
            logger.error(f"Error closing RAG Monitor: {str(e)}")
