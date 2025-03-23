from flask import Blueprint, g, request, jsonify, Response, current_app
import inspect
import logging
import json
import time
import traceback
from datetime import datetime
import concurrent.futures
from typing import Dict, Any, List, Optional
import sys

from app.web.hooks import login_required
from app.web.db.models import Pdf, User, Conversation
from app.web.db import db
from app.chat.vector_stores import get_vector_store, get_mongo_store, get_qdrant_store
from app.chat.memories.memory_manager import MemoryManager
from qdrant_client.http import models

logger = logging.getLogger(__name__)

bp = Blueprint("health", __name__, url_prefix="/api/health")

def get_database_status():
    """Get database connections and status data."""
    try:
        mongo_store = get_mongo_store()
        qdrant_store = get_qdrant_store()
        vector_store = get_vector_store()

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

        force_init = False
        if request:
            force_init = request.args.get("force_init", "").lower() == "true"

        if force_init or not all([
            mongo_store._initialized,
            qdrant_store._initialized,
            vector_store._initialized
        ]):
            logger.info("Attempting database initialization")
            if not mongo_store._initialized:
                mongo_init_success = mongo_store.initialize()
                db_status["mongo"]["initialization_attempted"] = True
                db_status["mongo"]["initialization_success"] = mongo_init_success
                if mongo_init_success:
                    db_status["mongo"]["status"] = "ok"
                    db_status["mongo"]["initialized"] = True
            if not qdrant_store._initialized:
                qdrant_init_success = qdrant_store.initialize()
                db_status["qdrant"]["initialization_attempted"] = True
                db_status["qdrant"]["initialization_success"] = qdrant_init_success
                if qdrant_init_success:
                    db_status["qdrant"]["status"] = "ok"
                    db_status["qdrant"]["initialized"] = True
            if not vector_store._initialized:
                vector_init_success = vector_store.initialize()
                db_status["vector_store"]["initialization_attempted"] = True
                db_status["vector_store"]["initialization_success"] = vector_init_success
                if vector_init_success:
                    db_status["vector_store"]["status"] = "ok"
                    db_status["vector_store"]["initialized"] = True

        if mongo_store._initialized:
            try:
                mongo_stats = mongo_store.get_stats()
                db_status["mongo"]["stats"] = mongo_stats
            except Exception as mongo_error:
                logger.error(f"Error getting MongoDB stats: {str(mongo_error)}")
                db_status["mongo"]["stats_error"] = str(mongo_error)

        if qdrant_store._initialized and qdrant_store.client:
            try:
                collection_count = qdrant_store.client.count(
                    collection_name=qdrant_store.collection_name,
                    count_filter=None
                )
                db_status["qdrant"]["vector_count"] = collection_count.count
                collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)
                db_status["qdrant"]["collection_info"] = {
                    "name": qdrant_store.collection_name,
                    "dimension": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance),
                }
            except Exception as qdrant_error:
                logger.error(f"Error getting Qdrant stats: {str(qdrant_error)}")
                db_status["qdrant"]["stats_error"] = str(qdrant_error)

        all_initialized = all([
            db_status["mongo"]["initialized"],
            db_status["qdrant"]["initialized"],
            db_status["vector_store"]["initialized"]
        ])

        return {
            "status": "ok" if all_initialized else "degraded",
            "databases": db_status,
            "all_initialized": all_initialized,
        }

    except Exception as e:
        logger.error(f"Database status check failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }

@bp.route("/", methods=["GET"])
def check_health():
    try:
        status = {
            "status": "ok",
            "time": datetime.utcnow().isoformat(),
            "message": "PDF RAG system is running"
        }
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
    try:
        health_data = {
            "status": "checking",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        db_status = get_database_status()
        health_data["components"]["databases"] = db_status
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
        component_statuses = [comp.get("status") for comp in health_data["components"].values()]
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
    status_data = get_database_status()
    return jsonify(status_data)

@bp.route("/pdf/<string:pdf_id>", methods=["GET"])
@login_required
def check_pdf(pdf_id):
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
            try:
                meta = pdf_record.get_metadata()
                if meta:
                    result["metadata"]["document_meta"] = meta
            except Exception:
                pass
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
                try:
                    elements = mongo_store.get_elements_by_pdf_id(pdf_id, limit=5000)
                    result["mongodb_info"]["element_count"] = len(elements)
                    content_types = {}
                    for elem in elements:
                        ct = elem.get("content_type", "unknown")
                        content_types[ct] = content_types.get(ct, 0) + 1
                    result["mongodb_info"]["content_types"] = content_types
                    concepts = mongo_store.get_concepts_by_pdf_id(pdf_id, limit=5000)
                    result["mongodb_info"]["concept_count"] = len(concepts)
                except Exception as count_error:
                    logger.error(f"Error counting elements: {str(count_error)}")
                    result["mongodb_info"]["error"] = str(count_error)
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
                        "key": s3_key,
                        "message": "File found in S3"
                    }
                except Exception as s3_head_error:
                    logger.info(f"S3 head check failed for {pdf_id}: {str(s3_head_error)}")
                    result["s3_info"] = {"error": str(s3_head_error), "key": s3_key}
        except Exception as s3_error:
            logger.error(f"Error checking S3 for PDF {pdf_id}: {str(s3_error)}")
            result["s3_info"] = {"error": str(s3_error)}
        if result["exists"]["mongodb"] and result["exists"]["qdrant"]:
            try:
                vector_store = get_vector_store()
                if vector_store._initialized:
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
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Query is required"}), 400

        query = data["query"]
        pdf_id = data.get("pdf_id")
        k = int(data.get("k", 5))

        vector_store = get_vector_store()
        if not vector_store._initialized:
            if not vector_store.initialize():
                return jsonify({"error": "Vector store not initialized"}), 500

        start_time = time.time()
        results = vector_store.semantic_search(
            query=query,
            k=k,
            pdf_id=pdf_id
        )
        query_time = time.time() - start_time

        formatted_results = []
        for i, doc in enumerate(results):
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

        response = {
            "query": query,
            "pdf_id": pdf_id,
            "results_count": len(results),
            "time_taken": query_time,
            "results": formatted_results
        }

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
                formatted_keyword_results = []
                for i, res in enumerate(keyword_results):
                    content = res.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    formatted_keyword_results.append({
                        "index": i,
                        "content": content,
                        "score": res.get("score", 0),
                        "pdf_id": res.get("pdf_id", ""),
                        "content_type": res.get("content_type", ""),
                        "page": res.get("page_number", 0) or res.get("page", 0),
                        "element_id": res.get("element_id", ""),
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
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": {},
            "vector_store": {},
            "pdfs": {},
            "conversations": {},
            "users": {}
        }
        try:
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
            conversation_count = db.session.query(Conversation).count()
            active_conversation_count = db.session.query(Conversation).filter_by(is_deleted=False).count()
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
            user_count = db.session.query(User).count()
            metrics["users"] = {
                "total": user_count,
                "avg_pdfs_per_user": round(pdf_count / max(1, user_count), 2),
                "avg_conversations_per_user": round(conversation_count / max(1, user_count), 2)
            }
        except Exception as db_error:
            logger.error(f"Error getting database metrics: {str(db_error)}")
            metrics["database"]["error"] = str(db_error)
        mongo_store = get_mongo_store()
        if mongo_store._initialized:
            try:
                mongo_stats = mongo_store.get_stats()
                metrics["mongodb"] = mongo_stats
            except Exception as mongo_error:
                logger.error(f"Error getting MongoDB metrics: {str(mongo_error)}")
                metrics["mongodb"] = {"error": str(mongo_error)}
        qdrant_store = get_qdrant_store()
        if qdrant_store._initialized and qdrant_store.client:
            try:
                collection_count = qdrant_store.client.count(
                    collection_name=qdrant_store.collection_name,
                    count_filter=None
                )
                collection_info = qdrant_store.client.get_collection(qdrant_store.collection_name)
                metrics["qdrant"] = {
                    "vector_count": collection_count.count,
                    "collection": qdrant_store.collection_name,
                    "dimension": collection_info.config.params.vectors.size
                }
                metrics["qdrant"]["metrics"] = qdrant_store.metrics
            except Exception as qdrant_error:
                logger.error(f"Error getting Qdrant metrics: {str(qdrant_error)}")
                metrics["qdrant"] = {"error": str(qdrant_error)}
        try:
            memory_manager = MemoryManager()
            memory_stats = memory_manager.get_stats()
            metrics["memory_manager"] = memory_stats
        except Exception as memory_error:
            logger.error(f"Error getting memory manager metrics: {str(memory_error)}")
            metrics["memory_manager"] = {"error": str(memory_error)}
        try:
            if hasattr(current_app, 'config') and 'RAG_MONITOR' in current_app.config:
                monitor = current_app.config['RAG_MONITOR']
                recent_ops = monitor.get_recent_operations(20)
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
def check_vector_stores():
    try:
        mongo_store = get_mongo_store()
        qdrant_store = get_qdrant_store()
        vector_store = get_vector_store()

        health_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "mongo": {"status": "checking"},
            "qdrant": {"status": "checking"},
            "unified": {"status": "checking"}
        }

        if hasattr(mongo_store, "check_health") and callable(mongo_store.check_health):
            health_info["mongo"] = mongo_store.check_health()
        else:
            health_info["mongo"] = {"status": "ok" if mongo_store._initialized else "error"}

        if hasattr(qdrant_store, "check_health") and callable(qdrant_store.check_health):
            health_info["qdrant"] = qdrant_store.check_health()
        else:
            health_info["qdrant"] = {"status": "ok" if qdrant_store._initialized else "error"}

        if hasattr(vector_store, "check_health") and callable(vector_store.check_health):
            health_info["unified"] = vector_store.check_health()
        else:
            health_info["unified"] = {"status": "ok" if vector_store._initialized else "error"}

        if (health_info["mongo"].get("status") == "ok" and
            health_info["qdrant"].get("status") == "ok" and
            health_info["unified"].get("status") == "ok"):
            health_info["status"] = "ok"
        elif ("error" in [health_info["mongo"].get("status"), health_info["qdrant"].get("status"), health_info["unified"].get("status")]):
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

@bp.route("/memory", methods=["GET"])
@login_required
def check_memory_manager():
    try:
        memory_manager = MemoryManager()
        memory_stats = memory_manager.get_stats()
        memory_health = {
            "status": "ok",
            "stats": memory_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
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
    try:
        diagnostic = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {},
            "database": {},
            "vector_stores": {},
            "memory": {},
            "file_storage": {}
        }
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
        database_health = get_database_status()
        diagnostic["database"] = database_health
        try:
            mongo_store = get_mongo_store()
            qdrant_store = get_qdrant_store()
            vector_store = get_vector_store()
            diagnostic["vector_stores"] = {
                "mongo_initialized": mongo_store._initialized,
                "qdrant_initialized": qdrant_store._initialized,
                "unified_initialized": vector_store._initialized
            }
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
        try:
            memory_manager = MemoryManager()
            memory_stats = memory_manager.get_stats()
            diagnostic["memory"] = memory_stats
        except Exception as mem_error:
            logger.error(f"Error checking memory manager: {str(mem_error)}")
            diagnostic["memory"]["error"] = str(mem_error)
        try:
            from app.web.files import get_s3_client
            s3_client = get_s3_client()
            if s3_client:
                bucket_name = current_app.config['AWS_BUCKET_NAME']
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
        health_issues = []
        if isinstance(diagnostic["database"], dict) and diagnostic["database"].get("status") != "ok":
            health_issues.append("Database connectivity issues")
        if not all([
            diagnostic["vector_stores"].get("mongo_initialized", False),
            diagnostic["vector_stores"].get("qdrant_initialized", False),
            diagnostic["vector_stores"].get("unified_initialized", False)
        ]):
            health_issues.append("Vector store initialization issues")
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
