# app/web/views/health_views.py

from flask import Blueprint, g, request, jsonify, Response, current_app
import logging
import json
import time
import traceback
from datetime import datetime
import concurrent.futures
from typing import Dict, Any, List, Optional

from app.web.hooks import login_required
from app.web.db.models import Pdf, User, Conversation
from app.web.db import db
from app.chat.vector_stores import get_vector_store, get_mongo_store, get_qdrant_store
from app.web.async_wrapper import async_handler
from app.chat.memories.memory_manager import MemoryManager
from qdrant_client.http import models 

logger = logging.getLogger(__name__)

bp = Blueprint("health", __name__, url_prefix="/api/health")

@bp.route("/", methods=["GET"])
def check_health():
    """Basic health check for system status."""
    try:
        # Simple status check
        status = {
            "status": "ok",
            "time": datetime.utcnow().isoformat(),
            "message": "PDF RAG system is running"
        }

        # Add app version if available
        if hasattr(current_app, 'version'):
            status["version"] = current_app.version

        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "time": datetime.utcnow().isoformat(),
            "error": str(e)
        }), 500

@bp.route("/system", methods=["GET"])
@login_required
def check_system():
    """Comprehensive system health check."""
    try:
        health_data = {
            "status": "checking",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }

        # Check database connections
        db_status = check_databases()
        health_data["components"]["databases"] = db_status

        # Check memory manager
        try:
            memory_manager = MemoryManager()
            memory_stats = memory_manager.get_stats()
            health_data["components"]["memory_manager"] = {
                "status": "ok",
                "conversation_count": memory_stats.get("conversation_count", 0),
                "message_counts": memory_stats.get("message_counts", {"total": 0})
            }
        except Exception as memory_error:
            logger.error(f"Memory manager check failed: {str(memory_error)}")
            health_data["components"]["memory_manager"] = {
                "status": "error",
                "error": str(memory_error)
            }

        # Check RAG Monitor if available
        try:
            if hasattr(current_app, 'config') and 'RAG_MONITOR' in current_app.config:
                monitor = current_app.config['RAG_MONITOR']
                recent_ops = monitor.get_recent_operations(10)
                health_data["components"]["rag_monitor"] = {
                    "status": "ok",
                    "recent_operations": len(recent_ops)
                }
            else:
                health_data["components"]["rag_monitor"] = {
                    "status": "not_available"
                }
        except Exception as monitor_error:
            logger.error(f"RAG monitor check failed: {str(monitor_error)}")
            health_data["components"]["rag_monitor"] = {
                "status": "error",
                "error": str(monitor_error)
            }

        # Check user stats from database
        try:
            user_count = db.session.query(User).count()
            pdf_count = db.session.query(Pdf).filter_by(is_deleted=False).count()
            conversation_count = db.session.query(Conversation).filter_by(is_deleted=False).count()

            health_data["components"]["database_stats"] = {
                "status": "ok",
                "user_count": user_count,
                "pdf_count": pdf_count,
                "conversation_count": conversation_count
            }
        except Exception as db_stats_error:
            logger.error(f"Database stats check failed: {str(db_stats_error)}")
            health_data["components"]["database_stats"] = {
                "status": "error",
                "error": str(db_stats_error)
            }

        # Check file storage
        try:
            from app.web.files import get_s3_client
            s3_client = get_s3_client()
            if s3_client:
                health_data["components"]["storage"] = {
                    "status": "ok",
                    "type": "s3",
                    "bucket": current_app.config.get('AWS_BUCKET_NAME', 'unknown')
                }
            else:
                health_data["components"]["storage"] = {
                    "status": "error",
                    "message": "S3 client initialization failed"
                }
        except Exception as storage_error:
            logger.error(f"Storage check failed: {str(storage_error)}")
            health_data["components"]["storage"] = {
                "status": "error",
                "error": str(storage_error)
            }

        # Determine overall status
        component_statuses = [
            comp.get("status") for comp in health_data["components"].values()
        ]

        if all(status == "ok" for status in component_statuses):
            health_data["status"] = "ok"
        elif "error" in component_statuses:
            health_data["status"] = "error"
        else:
            health_data["status"] = "degraded"

        return jsonify(health_data)

    except Exception as e:
        logger.error(f"System health check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@bp.route("/databases", methods=["GET"])
@login_required
def check_databases():
    """Check database connections and status."""
    try:
        # Get current database status
        mongo_store = get_mongo_store()
        qdrant_store = get_qdrant_store()
        vector_store = get_vector_store()

        # Check initialization status
        db_status = {
            "mongo": {
                "initialized": mongo_store._initialized,
                "status": "ok" if mongo_store._initialized else "error"
            },
            "qdrant": {
                "initialized": qdrant_store._initialized,
                "status": "ok" if qdrant_store._initialized else "error"
            },
            "vector_store": {
                "initialized": vector_store._initialized,
                "status": "ok" if vector_store._initialized else "error"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Check if we should attempt initialization
        force_init = request.args.get("force_init", "").lower() == "true"
        if force_init or not all([
            mongo_store._initialized,
            qdrant_store._initialized,
            vector_store._initialized
        ]):
            logger.info("Attempting database initialization")

            # Try initializing MongoDB
            if not mongo_store._initialized:
                mongo_init_success = mongo_store.initialize()
                db_status["mongo"]["initialization_attempted"] = True
                db_status["mongo"]["initialization_success"] = mongo_init_success
                if mongo_init_success:
                    db_status["mongo"]["status"] = "ok"
                    db_status["mongo"]["initialized"] = True

            # Try initializing Qdrant
            if not qdrant_store._initialized:
                qdrant_init_success = qdrant_store.initialize()
                db_status["qdrant"]["initialization_attempted"] = True
                db_status["qdrant"]["initialization_success"] = qdrant_init_success
                if qdrant_init_success:
                    db_status["qdrant"]["status"] = "ok"
                    db_status["qdrant"]["initialized"] = True

            # Try initializing Vector Store
            if not vector_store._initialized:
                vector_init_success = vector_store.initialize()
                db_status["vector_store"]["initialization_attempted"] = True
                db_status["vector_store"]["initialization_success"] = vector_init_success
                if vector_init_success:
                    db_status["vector_store"]["status"] = "ok"
                    db_status["vector_store"]["initialized"] = True

        # Get MongoDB stats if initialized
        if mongo_store._initialized:
            try:
                mongo_stats = mongo_store.get_stats()
                db_status["mongo"]["stats"] = mongo_stats
            except Exception as mongo_error:
                logger.error(f"Error getting MongoDB stats: {str(mongo_error)}")
                db_status["mongo"]["stats_error"] = str(mongo_error)

        # Get vector counts from Qdrant if initialized
        if qdrant_store._initialized and qdrant_store.client:
            try:
                collection_count = qdrant_store.client.count(
                    collection_name=qdrant_store.collection_name,
                    count_filter=None
                )
                db_status["qdrant"]["vector_count"] = collection_count.count

                # Get collection info
                collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)
                db_status["qdrant"]["collection_info"] = {
                    "name": qdrant_store.collection_name,
                    "dimension": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance),
                }
            except Exception as qdrant_error:
                logger.error(f"Error getting Qdrant stats: {str(qdrant_error)}")
                db_status["qdrant"]["stats_error"] = str(qdrant_error)

        # Check overall database status
        all_initialized = all([
            db_status["mongo"]["initialized"],
            db_status["qdrant"]["initialized"],
            db_status["vector_store"]["initialized"]
        ])

        if request.path == "/api/health/databases":  # Only for direct endpoint access
            return jsonify({
                "status": "ok" if all_initialized else "degraded",
                "databases": db_status,
                "all_initialized": all_initialized,
            })

        return db_status

    except Exception as e:
        error_response = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }

        if request.path == "/api/health/databases":  # Only for direct endpoint access
            return jsonify(error_response), 500

        return error_response

@bp.route("/pdf/<string:pdf_id>", methods=["GET"])
@login_required
@async_handler
async def check_pdf(pdf_id):
    """Check PDF existence and health in the system."""
    try:
        result = {
            "pdf_id": pdf_id,
            "timestamp": datetime.utcnow().isoformat(),
            "exists": {
                "database": False,
                "mongodb": False,
                "qdrant": False,
                "s3": False
            },
            "metadata": {},
            "vector_info": {}
        }

        # Check in SQL database
        pdf_record = db.session.query(Pdf).filter_by(id=pdf_id).first()
        if pdf_record:
            result["exists"]["database"] = True
            result["metadata"] = {
                "name": pdf_record.name,
                "processed": pdf_record.processed,
                "error": pdf_record.error,
                "category": pdf_record.category,
                "description": pdf_record.description,
                "created_at": pdf_record.created_at.isoformat() if hasattr(pdf_record.created_at, "isoformat") else str(pdf_record.created_at),
                "updated_at": pdf_record.updated_at.isoformat() if hasattr(pdf_record.updated_at, "isoformat") else str(pdf_record.updated_at)
            }

            # Get additional metadata if available
            try:
                meta = pdf_record.get_metadata()
                if meta:
                    result["metadata"]["document_meta"] = meta
            except:
                pass

        # Check in MongoDB
        mongo_store = get_mongo_store()
        if mongo_store._initialized:
            doc = mongo_store.get_document(pdf_id)
            if doc:
                result["exists"]["mongodb"] = True
                result["mongodb_info"] = {
                    "title": doc.get("title", ""),
                    "created_at": str(doc.get("created_at", "")),
                    "updated_at": str(doc.get("updated_at", ""))
                }

                # Count elements
                try:
                    elements = mongo_store.get_elements_by_pdf_id(pdf_id, limit=5000)
                    result["mongodb_info"]["element_count"] = len(elements)

                    # Get content type breakdown
                    content_types = {}
                    for elem in elements:
                        ct = elem.get("content_type", "unknown")
                        content_types[ct] = content_types.get(ct, 0) + 1
                    result["mongodb_info"]["content_types"] = content_types

                    # Get concept count
                    concepts = mongo_store.get_concepts_by_pdf_id(pdf_id, limit=5000)
                    result["mongodb_info"]["concept_count"] = len(concepts)
                except Exception as count_error:
                    logger.error(f"Error counting elements: {str(count_error)}")
                    result["mongodb_info"]["error"] = str(count_error)

        # Check in Qdrant
        qdrant_store = get_qdrant_store()
        if qdrant_store._initialized and qdrant_store.client:
            try:
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

                result["exists"]["qdrant"] = count_result.count > 0
                result["vector_info"]["embedding_count"] = count_result.count

                if count_result.count > 0:
                    # Get a sample of vectors
                    sample_result = qdrant_store.client.scroll(
                        collection_name=qdrant_store.collection_name,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="pdf_id",
                                    match=models.MatchValue(value=pdf_id)
                                )
                            ]
                        ),
                        limit=5,
                        with_payload=True,
                        with_vectors=False
                    )

                    # Extract metadata from samples
                    samples = []
                    for point in sample_result[0]:
                        if point.payload:
                            samples.append({
                                "element_id": point.payload.get("element_id", ""),
                                "content_type": point.payload.get("content_type", ""),
                                "page_number": point.payload.get("page_number", 0),
                            })

                    result["vector_info"]["samples"] = samples
            except Exception as qdrant_error:
                logger.error(f"Error checking Qdrant for PDF {pdf_id}: {str(qdrant_error)}")
                result["vector_info"]["error"] = str(qdrant_error)

        # Check in S3
        try:
            from app.web.files import get_s3_client, get_s3_key
            s3_client = get_s3_client()
            if s3_client:
                s3_key = get_s3_key(pdf_id)
                try:
                    s3_client.head_object(
                        Bucket=current_app.config['AWS_BUCKET_NAME'],
                        Key=s3_key
                    )
                    result["exists"]["s3"] = True
                    result["s3_info"] = {
                        "bucket": current_app.config['AWS_BUCKET_NAME'],
                        "key": s3_key
                    }
                except Exception as s3_head_error:
                    # Object doesn't exist or other error
                    logger.info(f"S3 head check failed for {pdf_id}: {str(s3_head_error)}")
                    result["s3_info"] = {"error": str(s3_head_error)}
        except Exception as s3_error:
            logger.error(f"Error checking S3 for PDF {pdf_id}: {str(s3_error)}")
            result["s3_info"] = {"error": str(s3_error)}

        # Test a simple query if document exists in both MongoDB and Qdrant
        if result["exists"]["mongodb"] and result["exists"]["qdrant"]:
            try:
                vector_store = get_vector_store()
                if vector_store._initialized:
                    # Simple test query
                    test_results = vector_store.semantic_search(
                        query="what is this document about",
                        k=3,
                        pdf_id=pdf_id
                    )

                    result["query_test"] = {
                        "status": "ok",
                        "result_count": len(test_results),
                        "success": len(test_results) > 0
                    }
            except Exception as query_error:
                logger.error(f"Error testing query for PDF {pdf_id}: {str(query_error)}")
                result["query_test"] = {
                    "status": "error",
                    "error": str(query_error)
                }

        # Overall status
        any_exists = any(result["exists"].values())
        all_exists = all([
            result["exists"]["database"],
            result["exists"]["mongodb"],
            result["exists"]["qdrant"],
            result["exists"]["s3"]
        ])

        if all_exists:
            result["status"] = "ok"
        elif any_exists:
            result["status"] = "partial"

            # Identify specific misalignment
            if result["exists"]["database"] and not result["exists"]["mongodb"]:
                result["issue"] = "PDF exists in database but not in MongoDB"
            elif result["exists"]["database"] and not result["exists"]["qdrant"]:
                result["issue"] = "PDF exists in database but not in Qdrant"
            elif result["exists"]["database"] and not result["exists"]["s3"]:
                result["issue"] = "PDF exists in database but not in S3"
        else:
            result["status"] = "not_found"

        return jsonify(result)

    except Exception as e:
        logger.error(f"PDF check failed for {pdf_id}: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "pdf_id": pdf_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/query_test", methods=["POST"])
@login_required
def test_query():
    """Test a query against the vector store without going through the full workflow."""
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Query is required"}), 400

        query = data["query"]
        pdf_id = data.get("pdf_id")
        k = int(data.get("k", 5))

        # Get vector store
        vector_store = get_vector_store()
        if not vector_store._initialized:
            if not vector_store.initialize():
                return jsonify({"error": "Vector store not initialized"}), 500

        # Execute semantic search
        start_time = time.time()
        results = vector_store.semantic_search(
            query=query,
            k=k,
            pdf_id=pdf_id
        )
        query_time = time.time() - start_time

        # Format results
        formatted_results = []
        for i, doc in enumerate(results):
            # Truncate long content for the response
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_results.append({
                "index": i,
                "content": content,
                "score": doc.metadata.get("score", 0),
                "pdf_id": doc.metadata.get("pdf_id", ""),
                "content_type": doc.metadata.get("content_type", ""),
                "page": doc.metadata.get("page_number", 0) or doc.metadata.get("page", 0),
                "element_id": doc.metadata.get("element_id", ""),
            })

        # Create response
        response = {
            "query": query,
            "pdf_id": pdf_id,
            "results_count": len(results),
            "time_taken": query_time,
            "results": formatted_results
        }

        # Also perform a keyword search for comparison
        try:
            mongo_store = get_mongo_store()
            if mongo_store._initialized:
                keyword_start_time = time.time()
                keyword_results = mongo_store.keyword_search(
                    query=query,
                    pdf_id=pdf_id,
                    limit=k
                )
                keyword_time = time.time() - keyword_start_time

                # Format keyword results
                formatted_keyword_results = []
                for i, result in enumerate(keyword_results):
                    content = result.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."

                    formatted_keyword_results.append({
                        "index": i,
                        "content": content,
                        "score": result.get("score", 0),
                        "pdf_id": result.get("pdf_id", ""),
                        "content_type": result.get("content_type", ""),
                        "page": result.get("page_number", 0) or result.get("page", 0),
                        "element_id": result.get("element_id", ""),
                    })

                response["keyword_search"] = {
                    "results_count": len(keyword_results),
                    "time_taken": keyword_time,
                    "results": formatted_keyword_results
                }
        except Exception as keyword_error:
            logger.error(f"Keyword search error: {str(keyword_error)}")
            response["keyword_search_error"] = str(keyword_error)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Query test failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/metrics", methods=["GET"])
@login_required
def get_system_metrics():
    """Get system-wide metrics and stats."""
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": {},
            "vector_store": {},
            "pdfs": {},
            "conversations": {},
            "users": {}
        }

        # Get SQL database stats
        try:
            # PDF stats
            pdf_count = db.session.query(Pdf).count()
            active_pdf_count = db.session.query(Pdf).filter_by(is_deleted=False).count()
            processed_pdf_count = db.session.query(Pdf).filter_by(processed=True).count()
            error_pdf_count = db.session.query(Pdf).filter(Pdf.error.isnot(None)).count()

            metrics["pdfs"] = {
                "total": pdf_count,
                "active": active_pdf_count,
                "processed": processed_pdf_count,
                "with_errors": error_pdf_count,
                "percent_processed": round((processed_pdf_count / max(1, active_pdf_count)) * 100, 2)
            }

            # Conversation stats
            conversation_count = db.session.query(Conversation).count()
            active_conversation_count = db.session.query(Conversation).filter_by(is_deleted=False).count()

            # Try to get average messages per conversation
            try:
                from app.web.db.models.message import Message
                message_count = db.session.query(Message).count()
                avg_messages = message_count / max(1, conversation_count)
            except:
                message_count = 0
                avg_messages = 0

            metrics["conversations"] = {
                "total": conversation_count,
                "active": active_conversation_count,
                "messages": message_count,
                "avg_messages_per_conversation": round(avg_messages, 2)
            }

            # User stats
            user_count = db.session.query(User).count()
            metrics["users"] = {
                "total": user_count,
                "avg_pdfs_per_user": round(pdf_count / max(1, user_count), 2),
                "avg_conversations_per_user": round(conversation_count / max(1, user_count), 2)
            }
        except Exception as db_error:
            logger.error(f"Error getting database metrics: {str(db_error)}")
            metrics["database"]["error"] = str(db_error)

        # MongoDB stats
        mongo_store = get_mongo_store()
        if mongo_store._initialized:
            try:
                mongo_stats = mongo_store.get_stats()
                metrics["mongodb"] = mongo_stats
            except Exception as mongo_error:
                logger.error(f"Error getting MongoDB metrics: {str(mongo_error)}")
                metrics["mongodb"] = {"error": str(mongo_error)}

        # Qdrant stats
        qdrant_store = get_qdrant_store()
        if qdrant_store._initialized and qdrant_store.client:
            try:
                # Get collection count
                collection_count = qdrant_store.client.count(
                    collection_name=qdrant_store.collection_name,
                    count_filter=None
                )

                # Get collection info
                collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)

                metrics["qdrant"] = {
                    "vector_count": collection_count.count,
                    "collection": qdrant_store.collection_name,
                    "dimension": collection_info.config.params.vectors.size
                }

                # Get metrics for operations
                metrics["qdrant"]["metrics"] = qdrant_store.metrics

            except Exception as qdrant_error:
                logger.error(f"Error getting Qdrant metrics: {str(qdrant_error)}")
                metrics["qdrant"] = {"error": str(qdrant_error)}

        # Memory manager stats
        try:
            memory_manager = MemoryManager()
            memory_stats = memory_manager.get_stats()
            metrics["memory_manager"] = memory_stats
        except Exception as memory_error:
            logger.error(f"Error getting memory manager metrics: {str(memory_error)}")
            metrics["memory_manager"] = {"error": str(memory_error)}

        # RAG Monitor stats if available
        try:
            if hasattr(current_app, 'config') and 'RAG_MONITOR' in current_app.config:
                monitor = current_app.config['RAG_MONITOR']
                recent_ops = monitor.get_recent_operations(20)

                # Aggregate operations by type
                op_types = {}
                for op in recent_ops:
                    op_type = op.get("operation", "unknown")
                    op_types[op_type] = op_types.get(op_type, 0) + 1

                metrics["rag_monitor"] = {
                    "recent_operations": len(recent_ops),
                    "operation_types": op_types
                }
        except Exception as monitor_error:
            logger.error(f"Error getting RAG monitor metrics: {str(monitor_error)}")
            metrics["rag_monitor"] = {"error": str(monitor_error)}

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/vector_stores", methods=["GET"])
@login_required
@async_handler
async def check_vector_stores():
    """Detailed health check for vector stores."""
    try:
        # Get vector stores
        mongo_store = get_mongo_store()
        qdrant_store = get_qdrant_store()
        vector_store = get_vector_store()

        # Check health asynchronously
        health_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "mongo": {"status": "checking"},
            "qdrant": {"status": "checking"},
            "unified": {"status": "checking"}
        }

        # Use ThreadPoolExecutor for parallel health checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            mongo_future = executor.submit(async_check_mongo_health, mongo_store)
            qdrant_future = executor.submit(async_check_qdrant_health, qdrant_store)
            unified_future = executor.submit(async_check_unified_health, vector_store)

            # Get results
            health_info["mongo"] = mongo_future.result()
            health_info["qdrant"] = qdrant_future.result()
            health_info["unified"] = unified_future.result()

        # Determine overall status
        if (health_info["mongo"]["status"] == "ok" and
            health_info["qdrant"]["status"] == "ok" and
            health_info["unified"]["status"] == "ok"):
            health_info["status"] = "ok"
        elif (health_info["mongo"]["status"] == "error" or
              health_info["qdrant"]["status"] == "error" or
              health_info["unified"]["status"] == "error"):
            health_info["status"] = "error"
        else:
            health_info["status"] = "degraded"

        return jsonify(health_info)

    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

async def async_check_mongo_health(mongo_store):
    """Check MongoDB health asynchronously."""
    try:
        if not mongo_store._initialized:
            return {
                "status": "error",
                "error": "MongoDB not initialized",
                "initialized": False
            }

        # Check connection
        if not mongo_store.client:
            return {
                "status": "error",
                "error": "MongoDB client not available",
                "initialized": True
            }

        # Check if we can ping
        mongo_store.client.admin.command('ping')

        # Get collection stats
        collection_stats = {}
        for collection_name in ["documents", "content_elements", "concepts", "relationships"]:
            collection = getattr(mongo_store.db, collection_name)
            collection_stats[collection_name] = collection.count_documents({})

        return {
            "status": "ok",
            "initialized": True,
            "collections": collection_stats,
            "database": mongo_store.db_name
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "initialized": mongo_store._initialized
        }

async def async_check_qdrant_health(qdrant_store):
    """Check Qdrant health asynchronously."""
    try:
        if not qdrant_store._initialized:
            return {
                "status": "error",
                "error": "Qdrant not initialized",
                "initialized": False
            }

        # Check if client is available
        if not qdrant_store.client:
            return {
                "status": "error",
                "error": "Qdrant client not available",
                "initialized": True
            }

        # Check collection
        collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)

        # Count vectors
        count_result = qdrant_store.client.count(
            collection_name=qdrant_store.collection_name,
            count_filter=None
        )

        return {
            "status": "ok",
            "initialized": True,
            "collection": qdrant_store.collection_name,
            "vector_count": count_result.count,
            "dimension": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "initialized": qdrant_store._initialized
        }

async def async_check_unified_health(vector_store):
    """Check unified vector store health asynchronously."""
    try:
        if not vector_store._initialized:
            return {
                "status": "error",
                "error": "Unified vector store not initialized",
                "initialized": False
            }

        # Check component stores
        mongo_initialized = vector_store.mongo_store._initialized if hasattr(vector_store, 'mongo_store') else False
        qdrant_initialized = vector_store.qdrant_store._initialized if hasattr(vector_store, 'qdrant_store') else False

        # Use internal check_health method
        health = await vector_store.check_health()

        health_data = {
            "status": health.get("status", "error"),
            "initialized": vector_store._initialized,
            "components": {
                "mongo_ready": mongo_initialized,
                "qdrant_ready": qdrant_initialized
            },
            "embedding_model": vector_store.embedding_model,
            "embedding_dimension": vector_store.embedding_dimension
        }

        # Add more details if available
        if "mongo_status" in health:
            health_data["mongo_details"] = health["mongo_status"]

        if "qdrant_status" in health:
            health_data["qdrant_details"] = health["qdrant_status"]

        # Get metrics
        health_data["metrics"] = vector_store.metrics

        return health_data
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "initialized": vector_store._initialized
        }

@bp.route("/memory", methods=["GET"])
@login_required
def check_memory_manager():
    """Check memory manager health and stats."""
    try:
        memory_manager = MemoryManager()
        memory_stats = memory_manager.get_stats()

        memory_health = {
            "status": "ok",
            "stats": memory_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Try to get a sample of conversations
        try:
            conversation_sample = memory_manager.list_conversations()[:5]
            sample_data = []

            for conv in conversation_sample:
                sample_data.append({
                    "id": conv.conversation_id,
                    "title": conv.title,
                    "pdf_id": conv.pdf_id,
                    "message_count": len(conv.messages),
                    "updated_at": conv.updated_at.isoformat() if hasattr(conv.updated_at, "isoformat") else str(conv.updated_at)
                })

            memory_health["conversation_sample"] = sample_data
        except Exception as sample_error:
            logger.error(f"Error getting conversation sample: {str(sample_error)}")
            memory_health["sample_error"] = str(sample_error)

        return jsonify(memory_health)
    except Exception as e:
        logger.error(f"Memory manager health check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@bp.route("/diagnostic", methods=["GET"])
@login_required
def run_system_diagnostic():
    """Run a comprehensive system diagnostic."""
    try:
        # Start with basic diagnostics
        diagnostic = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {},
            "database": {},
            "vector_stores": {},
            "memory": {},
            "file_storage": {}
        }

        # Check package versions
        import pkg_resources
        python_version = sys.version

        try:
            packages = [
                "flask", "sqlalchemy", "pymongo", "qdrant_client",
                "openai", "langchain", "langgraph", "boto3"
            ]

            package_versions = {}
            for package in packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    package_versions[package] = version
                except:
                    package_versions[package] = "not found"

            diagnostic["system"]["python_version"] = python_version
            diagnostic["system"]["packages"] = package_versions
        except Exception as pkg_error:
            logger.error(f"Error checking package versions: {str(pkg_error)}")
            diagnostic["system"]["package_error"] = str(pkg_error)

        # Check database connections
        database_health = check_databases()
        diagnostic["database"] = database_health

        # Check vector stores health
        try:
            mongo_store = get_mongo_store()
            qdrant_store = get_qdrant_store()
            vector_store = get_vector_store()

            # Basic initialization check
            diagnostic["vector_stores"] = {
                "mongo_initialized": mongo_store._initialized,
                "qdrant_initialized": qdrant_store._initialized,
                "unified_initialized": vector_store._initialized
            }

            # Try a basic test query
            if vector_store._initialized:
                test_results = vector_store.semantic_search(
                    query="test",
                    k=1
                )

                diagnostic["vector_stores"]["test_query"] = {
                    "status": "ok",
                    "results": len(test_results)
                }
        except Exception as vs_error:
            logger.error(f"Error checking vector stores: {str(vs_error)}")
            diagnostic["vector_stores"]["error"] = str(vs_error)

        # Memory manager check
        try:
            memory_manager = MemoryManager()
            memory_stats = memory_manager.get_stats()
            diagnostic["memory"] = memory_stats
        except Exception as mem_error:
            logger.error(f"Error checking memory manager: {str(mem_error)}")
            diagnostic["memory"]["error"] = str(mem_error)

        # File storage check
        try:
            from app.web.files import get_s3_client
            s3_client = get_s3_client()

            if s3_client:
                bucket_name = current_app.config['AWS_BUCKET_NAME']

                # Try to list a few objects
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=5
                )

                diagnostic["file_storage"] = {
                    "status": "ok",
                    "type": "s3",
                    "bucket": bucket_name,
                    "sample_count": len(response.get('Contents', []))
                }
        except Exception as storage_error:
            logger.error(f"Error checking file storage: {str(storage_error)}")
            diagnostic["file_storage"]["error"] = str(storage_error)

        # Overall health assessment
        health_issues = []

        # Check database
        if isinstance(diagnostic["database"], dict) and diagnostic["database"].get("status") != "ok":
            health_issues.append("Database connectivity issues")

        # Check vector stores
        if not all([
            diagnostic["vector_stores"].get("mongo_initialized", False),
            diagnostic["vector_stores"].get("qdrant_initialized", False),
            diagnostic["vector_stores"].get("unified_initialized", False)
        ]):
            health_issues.append("Vector store initialization issues")

        # Check file storage
        if "error" in diagnostic.get("file_storage", {}):
            health_issues.append("File storage connectivity issues")

        if health_issues:
            diagnostic["health_assessment"] = {
                "status": "issues_detected",
                "issues": health_issues,
                "recommendations": [
                    "Check database connections",
                    "Verify environment variables",
                    "Ensure vector stores are properly initialized",
                    "Check S3 credentials and permissions"
                ]
            }
        else:
            diagnostic["health_assessment"] = {
                "status": "healthy",
                "message": "All systems appear to be functioning correctly"
            }

        return jsonify(diagnostic)

    except Exception as e:
        logger.error(f"System diagnostic failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500
