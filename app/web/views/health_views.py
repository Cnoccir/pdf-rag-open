"""
Enhanced health monitoring for the application.
Provides comprehensive monitoring for databases, vector stores, and conversation history.
"""

from flask import Blueprint, jsonify, current_app, request
import logging
import os
import platform
import sys
import traceback
from datetime import datetime, timedelta
import psutil
from typing import Dict, Any, List, Optional
import json
import time

from app.chat.vector_stores import get_vector_store, get_mongo_store, get_qdrant_store
from app.web.db import db
from app.web.async_wrapper import async_handler
from app.chat.utils.pdf_status import check_pdf_status, get_recent_pdfs
from app.chat.memories.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

bp = Blueprint("health", __name__, url_prefix="/api/health")

@bp.route("/", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "app_name": "PDF-RAG-App",
        "version": "1.0.0"
    })

@bp.route("/system", methods=["GET"])
def system_info():
    """Get detailed system information."""
    memory = psutil.virtual_memory()

    # Get CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_times = psutil.cpu_times_percent(interval=0.5)

    # Get disk metrics for all mounts
    disk_info = {}
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_info[partition.mountpoint] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
                "filesystem": partition.fstype
            }
        except (PermissionError, FileNotFoundError):
            # Some mounts might not be accessible
            pass

    # Get network metrics
    net_io = psutil.net_io_counters()

    return jsonify({
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version()
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "path": sys.executable
        },
        "cpu": {
            "count_physical": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
            "percent": cpu_percent,
            "user": cpu_times.user,
            "system": cpu_times.system,
            "idle": cpu_times.idle
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free,
            "percent": memory.percent,
            "human_total": f"{memory.total / (1024**3):.2f} GB",
            "human_available": f"{memory.available / (1024**3):.2f} GB"
        },
        "disk": disk_info,
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "human_sent": f"{net_io.bytes_sent / (1024**2):.2f} MB",
            "human_recv": f"{net_io.bytes_recv / (1024**2):.2f} MB"
        },
        "timestamp": datetime.utcnow().isoformat()
    })

@bp.route("/database", methods=["GET"])
def database_health():
    """Check database health with enhanced metrics."""
    try:
        # Run a simple query to check database connection
        result = db.session.execute(db.text("SELECT 1")).fetchone()
        db_status = "ok" if result[0] == 1 else "error"

        # Get table counts with additional metrics
        table_metrics = {}
        for table_name in ["user", "pdf", "conversation", "message"]:
            try:
                # Get count
                count = db.session.execute(
                    db.text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()

                # Get additional metrics based on table type
                metrics = {"count": count}

                if table_name == "pdf":
                    # Get processed vs unprocessed counts
                    processed = db.session.execute(
                        db.text(f"SELECT COUNT(*) FROM {table_name} WHERE processed = True")
                    ).scalar()

                    metrics.update({
                        "processed": processed,
                        "unprocessed": count - processed,
                        "processing_rate": f"{(processed / max(1, count)) * 100:.2f}%"
                    })

                elif table_name == "conversation":
                    # Get active vs deleted counts
                    deleted = db.session.execute(
                        db.text(f"SELECT COUNT(*) FROM {table_name} WHERE is_deleted = True")
                    ).scalar()

                    metrics.update({
                        "active": count - deleted,
                        "deleted": deleted
                    })

                elif table_name == "message":
                    # Get counts by role
                    for role in ["user", "assistant", "system"]:
                        role_count = db.session.execute(
                            db.text(f"SELECT COUNT(*) FROM {table_name} WHERE role = '{role}'")
                        ).scalar()
                        metrics[f"{role}_count"] = role_count

                table_metrics[table_name] = metrics

            except Exception as e:
                table_metrics[table_name] = {"error": str(e)}

        # Get database size if possible
        db_size = None
        try:
            # This works for PostgreSQL
            if "postgresql" in current_app.config.get("SQLALCHEMY_DATABASE_URI", ""):
                size_query = """
                SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                       pg_database_size(current_database()) as bytes
                """
                size_result = db.session.execute(db.text(size_query)).fetchone()
                db_size = {
                    "pretty": size_result[0],
                    "bytes": size_result[1]
                }
        except Exception:
            pass

        return jsonify({
            "status": db_status,
            "type": current_app.config.get("SQLALCHEMY_DATABASE_URI", "").split("://")[0],
            "tables": table_metrics,
            "size": db_size,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/vector-store", methods=["GET"])
@async_handler
async def vector_store_health():
    """
    Check vector store health with detailed metrics for MongoDB and Qdrant.
    Provides insights into embedding counts, database sizes, and performance.
    """
    try:
        # Initialize results dictionary
        result = {
            "status": "checking",
            "unified_status": None,
            "mongodb_status": None,
            "qdrant_status": None,
            "stats": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Get the vector stores
        unified_store = get_vector_store()
        mongo_store = get_mongo_store()
        qdrant_store = get_qdrant_store()

        # Check unified store health
        try:
            start_time = time.time()
            unified_health = await unified_store.check_health()
            response_time = time.time() - start_time

            result["unified_status"] = unified_health.get("status")
            result["unified_response_time"] = f"{response_time:.3f}s"

            # Include vector count if available
            if "qdrant_status" in unified_health and "vector_count" in unified_health["qdrant_status"]:
                result["stats"]["vector_count"] = unified_health["qdrant_status"]["vector_count"]
        except Exception as e:
            result["unified_error"] = str(e)

        # Check MongoDB health
        try:
            start_time = time.time()
            mongo_health = await mongo_store.check_health()
            response_time = time.time() - start_time

            result["mongodb_status"] = mongo_health.get("status")
            result["mongodb_response_time"] = f"{response_time:.3f}s"
            result["mongodb_details"] = mongo_health

            # Include MongoDB stats
            if "stats" in mongo_health:
                result["stats"]["mongodb"] = mongo_health["stats"]
        except Exception as e:
            result["mongodb_error"] = str(e)

        # Check Qdrant health
        try:
            start_time = time.time()
            qdrant_health = await qdrant_store.check_health()
            response_time = time.time() - start_time

            result["qdrant_status"] = qdrant_health.get("status")
            result["qdrant_response_time"] = f"{response_time:.3f}s"
            result["qdrant_details"] = qdrant_health

            # Include Qdrant stats
            if "vector_count" in qdrant_health:
                result["stats"]["qdrant"] = {
                    "vector_count": qdrant_health["vector_count"],
                    "dimension": qdrant_health.get("dimension"),
                }
        except Exception as e:
            result["qdrant_error"] = str(e)

        # Determine overall status
        if all(s == "ok" for s in [result["mongodb_status"], result["qdrant_status"]] if s):
            result["status"] = "ok"
        elif any(s == "ok" for s in [result["mongodb_status"], result["qdrant_status"]] if s):
            result["status"] = "degraded"
        else:
            result["status"] = "error"

        return jsonify(result)

    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/embeddings", methods=["GET"])
@async_handler
async def embeddings_stats():
    """
    Get detailed statistics about embeddings in the vector store.
    Provides counts by document, type, and other dimensions.
    """
    try:
        # Get the vector store
        vector_store = get_vector_store()

        # Initialize response
        response = {
            "status": "checking",
            "counts": {},
            "types": {},
            "documents": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Get document IDs to query - either from request or recent docs
        pdf_ids = request.args.getlist("pdf_id")
        limit = int(request.args.get("limit", "10"))

        if not pdf_ids:
            # Get recent PDFs
            recent_pdfs_result = await get_recent_pdfs(limit)
            pdf_ids = [pdf["pdf_id"] for pdf in recent_pdfs_result.get("pdfs", [])]

        # Query embedding statistics per document
        total_vectors = 0
        document_counts = {}
        content_type_counts = {}
        chunk_level_counts = {}
        embedding_type_counts = {}

        # Check if we have the necessary methods in our vector stores
        has_detailed_stats = hasattr(vector_store, "get_embedding_stats")

        if has_detailed_stats:
            # Use the detailed stats method if available
            stats = await vector_store.get_embedding_stats(pdf_ids)

            # Process stats
            total_vectors = stats.get("total", 0)
            document_counts = stats.get("documents", {})
            content_type_counts = stats.get("content_types", {})
            chunk_level_counts = stats.get("chunk_levels", {})
            embedding_type_counts = stats.get("embedding_types", {})
        else:
            # Fallback to basic stats from health check
            health = await vector_store.check_health()
            if "qdrant_status" in health and "vector_count" in health["qdrant_status"]:
                total_vectors = health["qdrant_status"]["vector_count"]

        # Fill response
        response["counts"]["total"] = total_vectors
        response["documents"] = document_counts
        response["types"]["content_types"] = content_type_counts
        response["types"]["chunk_levels"] = chunk_level_counts
        response["types"]["embedding_types"] = embedding_type_counts

        # Set status
        response["status"] = "ok" if total_vectors > 0 else "no_data"

        return jsonify(response)

    except Exception as e:
        logger.error(f"Embeddings stats check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/pdf/<string:pdf_id>", methods=["GET"])
@async_handler
async def check_pdf(pdf_id):
    """
    Check status of a specific PDF with enhanced metrics.
    Includes embedding counts, processing status, and document metadata.
    """
    try:
        # Get basic PDF status
        result = await check_pdf_status(pdf_id)

        # Try to get embedding counts by content type
        try:
            # Get vector store
            vector_store = get_vector_store()

            # Check if we have the method to get detailed embedding stats for a PDF
            if hasattr(vector_store, "get_embedding_stats_for_pdf"):
                embedding_stats = await vector_store.get_embedding_stats_for_pdf(pdf_id)
                result["embeddings"] = embedding_stats
            elif hasattr(vector_store.qdrant_store, "get_counts_by_filter"):
                # Fallback to direct Qdrant query
                filter_dict = {"pdf_id": pdf_id}
                counts = await vector_store.qdrant_store.get_counts_by_filter(filter_dict)

                # Add to result
                result["embeddings"] = {
                    "total": counts.get("total", 0)
                }
        except Exception as embedding_err:
            logger.warning(f"Could not retrieve embedding stats: {str(embedding_err)}")
            result["embeddings_error"] = str(embedding_err)

        # Try to get document content metrics
        try:
            from app.chat.vector_stores.mongo_store import get_mongo_store
            mongo_store = get_mongo_store()

            if hasattr(mongo_store, "get_elements_by_pdf_id"):
                # Count elements by content type
                elements = mongo_store.get_elements_by_pdf_id(pdf_id, limit=1000)

                content_types = {}
                for element in elements:
                    content_type = element.get("content_type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

                result["content_metrics"] = {
                    "total_elements": len(elements),
                    "by_content_type": content_types
                }
        except Exception as content_err:
            logger.warning(f"Could not retrieve content metrics: {str(content_err)}")
            result["content_metrics_error"] = str(content_err)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error checking PDF status: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "pdf_id": pdf_id,
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/pdfs", methods=["GET"])
@async_handler
async def list_pdfs():
    """
    Get a list of all PDFs with their processing status.
    Supports filtering and pagination.
    """
    try:
        # Get query parameters
        limit = int(request.args.get("limit", "20"))
        offset = int(request.args.get("offset", "0"))
        processed = request.args.get("processed", None)  # None, "true", "false"
        category = request.args.get("category", None)

        # Build query
        from app.web.db.models import Pdf
        query = db.select(Pdf).filter_by(is_deleted=False)

        # Apply filters
        if processed is not None:
            if processed.lower() == "true":
                query = query.filter_by(processed=True)
            elif processed.lower() == "false":
                query = query.filter_by(processed=False)

        if category:
            query = query.filter_by(category=category)

        # Apply pagination and order
        query = query.order_by(Pdf.created_at.desc()).offset(offset).limit(limit)

        # Execute query
        pdfs = db.session.execute(query).scalars().all()

        # Format results
        pdf_list = []
        for pdf in pdfs:
            # Check if vectors exist for this PDF
            vector_status = "unknown"
            try:
                vector_store = get_vector_store()
                health = await vector_store.check_health()
                if health.get("status") in ["ok", "degraded"]:
                    vector_status = "available"
            except:
                pass

            # Format PDF info
            pdf_info = {
                "id": pdf.id,
                "name": pdf.name,
                "processed": pdf.processed,
                "error": pdf.error,
                "category": pdf.category,
                "created_at": pdf.created_at.isoformat() if hasattr(pdf.created_at, "isoformat") else str(pdf.created_at),
                "metadata": pdf.get_metadata(),
                "vector_status": vector_status
            }
            pdf_list.append(pdf_info)

        # Get total count
        total_count = db.session.execute(
            db.select(db.func.count()).select_from(Pdf).filter_by(is_deleted=False)
        ).scalar()

        return jsonify({
            "pdfs": pdf_list,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/conversations", methods=["GET"])
@async_handler
async def conversation_stats():
    """
    Get statistics about conversations in the system.
    Includes counts, message distribution, and active conversations.
    """
    try:
        # Get query parameters
        days = int(request.args.get("days", "7"))  # Default to last 7 days

        # Calculate date threshold
        threshold_date = datetime.utcnow() - timedelta(days=days)

        # Get conversation counts
        from app.web.db.models import Conversation, Message

        # Total conversations
        total_convos = db.session.execute(
            db.select(db.func.count()).select_from(Conversation)
        ).scalar()

        # Active conversations (not deleted)
        active_convos = db.session.execute(
            db.select(db.func.count()).select_from(Conversation).filter_by(is_deleted=False)
        ).scalar()

        # Recent conversations (created in the last N days)
        recent_convos = db.session.execute(
            db.select(db.func.count()).select_from(Conversation)
            .filter(Conversation.created_on >= threshold_date)
        ).scalar()

        # Total messages
        total_messages = db.session.execute(
            db.select(db.func.count()).select_from(Message)
        ).scalar()

        # Messages by role
        message_counts = {}
        for role in ["user", "assistant", "system", "tool"]:
            count = db.session.execute(
                db.select(db.func.count()).select_from(Message).filter_by(role=role)
            ).scalar()
            message_counts[role] = count

        # Recent messages (last N days)
        recent_messages = db.session.execute(
            db.select(db.func.count()).select_from(Message)
            .filter(Message.created_on >= threshold_date)
        ).scalar()

        # Get PDF IDs with most conversations
        pdf_conversation_counts = db.session.execute(
            db.text("""
                SELECT pdf_id, COUNT(*) as count
                FROM conversation
                WHERE is_deleted = FALSE
                GROUP BY pdf_id
                ORDER BY count DESC
                LIMIT 10
            """)
        ).fetchall()

        top_pdfs = [{"pdf_id": row[0], "conversation_count": row[1]} for row in pdf_conversation_counts]

        # Try to get PDF names for these IDs
        from app.web.db.models import Pdf
        for pdf_data in top_pdfs:
            pdf_id = pdf_data["pdf_id"]
            pdf = db.session.execute(
                db.select(Pdf).filter_by(id=pdf_id)
            ).scalar_one_or_none()

            if pdf:
                pdf_data["name"] = pdf.name

        # Activity over time (messages per day for last N days)
        daily_activity = []
        for i in range(days):
            day = datetime.utcnow() - timedelta(days=i)
            day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
            day_end = datetime(day.year, day.month, day.day, 23, 59, 59)

            # Count messages
            message_count = db.session.execute(
                db.select(db.func.count()).select_from(Message)
                .filter(Message.created_on >= day_start)
                .filter(Message.created_on <= day_end)
            ).scalar()

            # Count conversations
            convo_count = db.session.execute(
                db.select(db.func.count()).select_from(Conversation)
                .filter(Conversation.created_on >= day_start)
                .filter(Conversation.created_on <= day_end)
            ).scalar()

            daily_activity.append({
                "date": day_start.isoformat(),
                "message_count": message_count,
                "conversation_count": convo_count
            })

        # Conversation length distribution
        conversation_lengths = db.session.execute(
            db.text("""
                SELECT c.id, COUNT(m.id) as message_count
                FROM conversation c
                LEFT JOIN message m ON c.id = m.conversation_id
                WHERE c.is_deleted = FALSE
                GROUP BY c.id
            """)
        ).fetchall()

        # Calculate distribution
        length_distribution = {}
        for _, count in conversation_lengths:
            # Group into ranges: 1-5, 6-10, 11-20, 21-50, 50+
            if count <= 5:
                key = "1-5"
            elif count <= 10:
                key = "6-10"
            elif count <= 20:
                key = "11-20"
            elif count <= 50:
                key = "21-50"
            else:
                key = "50+"

            length_distribution[key] = length_distribution.get(key, 0) + 1

        return jsonify({
            "summary": {
                "total_conversations": total_convos,
                "active_conversations": active_convos,
                "recent_conversations": recent_convos,
                "total_messages": total_messages,
                "recent_messages": recent_messages,
                "avg_messages_per_conversation": round(total_messages / max(1, total_convos), 2)
            },
            "messages_by_role": message_counts,
            "top_pdfs_by_conversations": top_pdfs,
            "daily_activity": daily_activity,
            "conversation_length_distribution": length_distribution,
            "time_period_days": days,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting conversation stats: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/memory", methods=["GET"])
@async_handler
async def memory_stats():
    """
    Get statistics about the memory management system.
    Provides insights into conversation persistence and memory usage.
    """
    try:
        # Create memory manager
        memory_manager = MemoryManager()

        # Get stats if possible
        memory_stats = {}
        if hasattr(memory_manager, "get_stats"):
            memory_stats = memory_manager.get_stats()

        # Count active conversations in the memory system
        conversation_count = 0
        if hasattr(memory_manager, "list_conversations"):
            conversations = memory_manager.list_conversations()
            conversation_count = len(conversations)

        # Get memory system type
        memory_type = "SQL"  # Default
        if hasattr(memory_manager, "memory_type"):
            memory_type = memory_manager.memory_type

        # Get other database metrics if available
        from app.web.db.models import Conversation, Message

        # Total conversations in DB
        total_convos = db.session.execute(
            db.select(db.func.count()).select_from(Conversation)
        ).scalar()

        # Total messages in DB
        total_messages = db.session.execute(
            db.select(db.func.count()).select_from(Message)
        ).scalar()

        # Check if conversation and message counts match
        consistency_status = "unknown"
        if conversation_count > 0:
            consistency_status = "consistent" if conversation_count == total_convos else "inconsistent"

        return jsonify({
            "memory_system": {
                "type": memory_type,
                "active_conversations": conversation_count,
                "stats": memory_stats
            },
            "database": {
                "total_conversations": total_convos,
                "total_messages": total_messages,
                "consistency_status": consistency_status
            },
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/dashboard", methods=["GET"])
@async_handler
async def dashboard_summary():
    """
    Get a comprehensive system summary for the dashboard.
    Aggregates key metrics from all health endpoints.
    """
    try:
        # Initialize response
        response = {
            "status": "ok",
            "system": {},
            "database": {},
            "vector_stores": {},
            "pdf_processing": {},
            "conversations": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Get basic system metrics
        memory = psutil.virtual_memory()
        response["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2)
        }

        # Get database metrics
        try:
            # Run a simple query to check database connection
            result = db.session.execute(db.text("SELECT 1")).fetchone()
            db_status = "ok" if result[0] == 1 else "error"

            # Get table counts
            table_counts = {}
            for table_name in ["user", "pdf", "conversation", "message"]:
                count = db.session.execute(
                    db.text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
                table_counts[table_name] = count

            response["database"] = {
                "status": db_status,
                "table_counts": table_counts
            }
        except Exception as db_err:
            response["database"] = {
                "status": "error",
                "error": str(db_err)
            }

        # Get vector store metrics
        try:
            # Get the vector store
            vector_store = get_vector_store()
            health = await vector_store.check_health()

            response["vector_stores"] = {
                "status": health.get("status", "unknown"),
                "mongodb_ready": health.get("mongo_ready", False),
                "qdrant_ready": health.get("qdrant_ready", False)
            }

            # Get vector counts if available
            if "qdrant_status" in health and "vector_count" in health["qdrant_status"]:
                response["vector_stores"]["vector_count"] = health["qdrant_status"]["vector_count"]
        except Exception as vs_err:
            response["vector_stores"] = {
                "status": "error",
                "error": str(vs_err)
            }

        # Get PDF processing metrics
        try:
            from app.web.db.models import Pdf

            # Get total PDFs
            total_pdfs = db.session.execute(
                db.select(db.func.count()).select_from(Pdf).filter_by(is_deleted=False)
            ).scalar()

            # Get processed PDFs
            processed_pdfs = db.session.execute(
                db.select(db.func.count()).select_from(Pdf).filter_by(is_deleted=False, processed=True)
            ).scalar()

            # Get error PDFs
            error_pdfs = db.session.execute(
                db.select(db.func.count()).select_from(Pdf)
                .filter_by(is_deleted=False, processed=False)
                .filter(Pdf.error != None)
            ).scalar()

            response["pdf_processing"] = {
                "total": total_pdfs,
                "processed": processed_pdfs,
                "errors": error_pdfs,
                "processing_rate": f"{(processed_pdfs / max(1, total_pdfs)) * 100:.2f}%"
            }
        except Exception as pdf_err:
            response["pdf_processing"] = {
                "status": "error",
                "error": str(pdf_err)
            }

        # Get conversation metrics
        try:
            from app.web.db.models import Conversation, Message

            # Total active conversations
            active_convos = db.session.execute(
                db.select(db.func.count()).select_from(Conversation).filter_by(is_deleted=False)
            ).scalar()

            # Recent conversations (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_convos = db.session.execute(
                db.select(db.func.count()).select_from(Conversation)
                .filter(Conversation.created_on >= yesterday)
            ).scalar()

            # Total messages
            total_messages = db.session.execute(
                db.select(db.func.count()).select_from(Message)
            ).scalar()

            # Message counts by role
            user_messages = db.session.execute(
                db.select(db.func.count()).select_from(Message).filter_by(role="user")
            ).scalar()

            assistant_messages = db.session.execute(
                db.select(db.func.count()).select_from(Message).filter_by(role="assistant")
            ).scalar()

            response["conversations"] = {
                "active": active_convos,
                "recent": recent_convos,
                "messages": {
                    "total": total_messages,
                    "user": user_messages,
                    "assistant": assistant_messages
                },
                "avg_per_conversation": round(total_messages / max(1, active_convos), 2)
            }
        except Exception as conv_err:
            response["conversations"] = {
                "status": "error",
                "error": str(conv_err)
            }

        # Determine overall status
        status_values = [
            response["database"].get("status"),
            response["vector_stores"].get("status"),
        ]

        if all(s == "ok" for s in status_values if s):
            response["status"] = "ok"
        elif any(s == "error" for s in status_values if s):
            response["status"] = "error"
        else:
            response["status"] = "degraded"

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error generating dashboard summary: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500
