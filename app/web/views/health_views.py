"""
Health check endpoints for the application.
Used for monitoring system health and diagnosing issues.
"""

from flask import Blueprint, jsonify, current_app
import logging
import os
import platform
import sys
from datetime import datetime
import psutil
import asyncio
from typing import Dict, Any

from app.chat.vector_stores import get_vector_store
from app.web.db import db
from app.chat.memories.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

bp = Blueprint("health", __name__, url_prefix="/api/health")

@bp.route("/", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "app_name": "PDF-RAG-App"
    })

@bp.route("/system", methods=["GET"])
def system_info():
    """Get system information."""
    memory = psutil.virtual_memory()
    return jsonify({
        "os": platform.platform(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_total": memory.total,
        "memory_available": memory.available,
        "memory_percent": memory.percent,
        "disk_usage": {
            "/": psutil.disk_usage("/").percent
        }
    })

@bp.route("/database", methods=["GET"])
def database_health():
    """Check database health."""
    try:
        # Run a simple query to check database connection
        result = db.session.execute(db.text("SELECT 1")).fetchone()
        db_status = "ok" if result[0] == 1 else "error"

        # Get table counts
        table_counts = {}
        for table_name in ["user", "pdf", "conversation", "message"]:
            try:
                count = db.session.execute(
                    db.text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
                table_counts[table_name] = count
            except Exception as e:
                table_counts[table_name] = f"Error: {str(e)}"

        return jsonify({
            "status": db_status,
            "type": current_app.config.get("SQLALCHEMY_DATABASE_URI", "").split("://")[0],
            "table_counts": table_counts,
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
async def vector_store_health():
    """Check Neo4j vector store health."""
    try:
        # Get the vector store instance
        vector_store = get_vector_store()

        # Check health asynchronously
        health_info = await vector_store.check_health()

        return jsonify({
            "status": health_info["status"],
            "connection": health_info["connection"],
            "database_ready": health_info["database_ready"],
            "indexes_count": len(health_info["indexes"]),
            "vector_indexes_count": len(health_info["vector_indexes"]),
            "node_counts": health_info["node_counts"],
            "url": vector_store.url.split("@")[-1],  # Hide credentials
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/memory", methods=["GET"])
async def memory_health():
    """Check conversation memory health for SQL-based implementation."""
    try:
        memory_manager = MemoryManager()

        # Get conversation statistics from the database
        from app.web.db.models import Conversation, Message
        from app.web.db import db
        from sqlalchemy import func

        # Count conversations
        conversation_count = db.session.query(func.count(Conversation.id)).filter_by(is_deleted=False).scalar() or 0

        # Count deleted conversations
        deleted_count = db.session.query(func.count(Conversation.id)).filter_by(is_deleted=True).scalar() or 0

        # Count messages
        message_count = db.session.query(func.count(Message.id)).scalar() or 0

        # Get latest conversation
        latest_conv = db.session.query(Conversation).filter_by(is_deleted=False).order_by(Conversation.last_updated.desc()).first()

        # Check a random conversation for integrity
        sample_conversation = None
        sample_status = "no_conversations"

        if conversation_count > 0:
            # Get a random conversation ID
            import random

            conversation_ids = db.session.query(Conversation.id).filter_by(is_deleted=False).limit(10).all()
            if conversation_ids:
                sample_id = random.choice(conversation_ids)[0]
                try:
                    sample_conversation = await memory_manager.get_conversation(sample_id)
                    sample_status = "ok" if sample_conversation else "missing"
                except Exception as e:
                    sample_status = f"error: {str(e)}"

        return jsonify({
            "status": "ok",
            "storage_type": "sql",
            "conversation_count": conversation_count,
            "deleted_conversation_count": deleted_count,
            "message_count": message_count,
            "latest_conversation": latest_conv.id if latest_conv else None,
            "latest_update": latest_conv.last_updated.isoformat() if latest_conv else None,
            "sample_conversation_status": sample_status,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Memory health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500
