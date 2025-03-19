"""
Health check endpoints for the application.
Used for monitoring system health and diagnosing issues.
"""

from flask import Blueprint, jsonify, current_app, request
import logging
import os
import platform
import sys
import traceback
from datetime import datetime
import psutil
from typing import Dict, Any

from app.chat.vector_stores import get_vector_store
from app.web.db import db
from app.web.async_wrapper import async_handler
from app.chat.utils.pdf_status import check_pdf_status, get_recent_pdfs

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
@async_handler
async def vector_store_health():
    """Check Neo4j vector store health."""
    try:
        # Get the vector store instance
        vector_store = get_vector_store()

        # Check health asynchronously
        health_info = await vector_store.check_health()

        return jsonify(health_info)
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/pdf/<string:pdf_id>", methods=["GET"])
@async_handler
async def check_pdf(pdf_id):
    """Check status of a specific PDF in the system."""
    try:
        # Check PDF status in Neo4j
        result = await check_pdf_status(pdf_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error checking PDF status: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "pdf_id": pdf_id,
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/pdfs/recent", methods=["GET"])
@async_handler
async def recent_pdfs():
    """Get status information for recent PDFs."""
    try:
        # Get limit parameter
        limit = int(request.args.get("limit", "5"))

        # Get recent PDFs status
        result = await get_recent_pdfs(limit)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting recent PDFs: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500
