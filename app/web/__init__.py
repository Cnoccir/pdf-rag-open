"""
Enhanced application initialization with monitoring capabilities.
This patch adds the necessary initializations for health monitoring.
"""

from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from app.web.db import db, init_db_command
from app.web.db import models
from app.celery import celery_init_app
from app.web.config import Config
from app.web.hooks import load_logged_in_user, handle_error, add_headers
from app.web.views import (
    auth_views,
    pdf_views,
    client_views,
    conversation_views,
    stream_views,
    health_views,
)
# Import the async wrapper
from app.web.async_wrapper import async_handler, run_async
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from dotenv import load_dotenv
import asyncio
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure MongoDB and Qdrant (replacing Neo4j)
if not os.getenv("MONGODB_URI"):
    os.environ["MONGODB_URI"] = "mongodb://localhost:27017/"
    os.environ["MONGODB_DB_NAME"] = "tech_rag"

if not os.getenv("QDRANT_HOST"):
    os.environ["QDRANT_HOST"] = "localhost"
    os.environ["QDRANT_PORT"] = "6333"
    os.environ["QDRANT_COLLECTION"] = "content_vectors"

# Initialize nest_asyncio for better event loop handling
try:
    import nest_asyncio
    nest_asyncio.apply()
    logging.info("nest_asyncio applied successfully")
except ImportError:
    logging.warning("nest_asyncio not available - async nesting may have issues")
except Exception as e:
    logging.warning(f"Error applying nest_asyncio: {str(e)}")

migrate = Migrate()

class RAGMonitor:
    """
    Custom monitoring solution for RAG operations.
    Stores metrics in MongoDB for visualization and analysis.
    """

    def __init__(self, mongo_client=None):
        self.mongo_client = mongo_client
        self.db_name = os.getenv("MONITORING_DB_NAME", "rag_monitoring")
        self.collection_name = "metrics"
        self._initialized = False
        self._init_db()

    def _init_db(self):
        if not self.mongo_client:
            from pymongo import MongoClient
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
            self.mongo_client = MongoClient(mongo_uri)

        self.db = self.mongo_client[self.db_name]
        self.metrics = self.db[self.collection_name]

        # Create indexes
        self.metrics.create_index("timestamp")
        self.metrics.create_index("operation")
        self.metrics.create_index("pdf_id")

        self._initialized = True

    def record_operation(self, operation, details=None, pdf_id=None, duration_ms=None):
        """Record an operation for monitoring."""
        if not self._initialized:
            self._init_db()

        metric = {
            "timestamp": datetime.utcnow(),
            "operation": operation,
            "pdf_id": pdf_id,
            "duration_ms": duration_ms,
            "details": details or {}
        }

        self.metrics.insert_one(metric)

    def record_query(self, query, pdf_ids, results_count, duration_ms, strategy):
        """Record a query operation."""
        details = {
            "query": query,
            "pdf_ids": pdf_ids,
            "results_count": results_count,
            "strategy": strategy
        }

        self.record_operation(
            operation="query",
            details=details,
            pdf_id=pdf_ids[0] if pdf_ids else None,
            duration_ms=duration_ms
        )

    def record_processing(self, pdf_id, element_count, chunk_count, duration_ms):
        """Record document processing."""
        details = {
            "element_count": element_count,
            "chunk_count": chunk_count
        }

        self.record_operation(
            operation="processing",
            details=details,
            pdf_id=pdf_id,
            duration_ms=duration_ms
        )

    def get_recent_operations(self, limit=100):
        """Get recent operations for dashboard."""
        return list(self.metrics.find().sort("timestamp", -1).limit(limit))

    def get_pdf_metrics(self, pdf_id):
        """Get metrics for a specific PDF."""
        return list(self.metrics.find({"pdf_id": pdf_id}).sort("timestamp", -1))

def initialize_vector_stores(app):
    """Initialize MongoDB and Qdrant connections with enhanced monitoring."""
    from app.chat.vector_stores import get_mongo_store, get_qdrant_store, get_vector_store

    # Import vector store enhancements
    # This will add new methods to the vector store classes
    from app.chat.vector_stores.enhanced_monitoring import enhance_vector_stores

    app.logger.info("Initializing MongoDB connection...")
    mongo_store = get_mongo_store()

    app.logger.info("Initializing Qdrant vector store...")
    qdrant_store = get_qdrant_store()

    app.logger.info("Initializing unified vector store...")
    vector_store = get_vector_store()

    # Apply enhancements to vector stores
    enhance_vector_stores()
    app.logger.info("Applied monitoring enhancements to vector stores")

    # Set stores in app config for access in views
    app.config['MONGO_STORE'] = mongo_store
    app.config['QDRANT_STORE'] = qdrant_store
    app.config['VECTOR_STORE'] = vector_store

    # Verify connections
    try:
        asyncio.run(async_connection_check(app, vector_store))
    except Exception as e:
        app.logger.error(f"Database connection check failed: {str(e)}")

    app.logger.info("Vector stores initialized with monitoring capabilities")

async def async_connection_check(app, vector_store):
    """Check database connections asynchronously."""
    try:
        health = await vector_store.check_health()
        app.logger.info(f"Database health check: {health['status']}")

        if health['status'] != 'ok':
            app.logger.warning(f"Database health check returned status: {health['status']}")
            if 'mongo_status' in health:
                app.logger.warning(f"MongoDB status: {health['mongo_status'].get('status')}")
            if 'qdrant_status' in health:
                app.logger.warning(f"Qdrant status: {health['qdrant_status'].get('status')}")
    except Exception as e:
        app.logger.error(f"Database health check failed: {str(e)}")

def create_app():
    app = Flask(__name__, static_folder="../../client/build")
    app.url_map.strict_slashes = False
    app.config.from_object(Config)

    # Log configuration status
    app.logger.info("MongoDB and Qdrant vector store configuration loaded")
    app.logger.info(f"Using MongoDB at: {os.getenv('MONGODB_URI')}")
    app.logger.info(f"Using Qdrant at: {os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}")

    # Initialize asyncio policy for better Windows support if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        app.logger.info("Set Windows-compatible asyncio event loop policy")

    register_extensions(app)
    register_hooks(app)
    register_blueprints(app)

    # Initialize vector stores with monitoring
    initialize_vector_stores(app)

    # Initialize monitoring
    app.config['RAG_MONITOR'] = RAGMonitor()
    app.logger.info("RAG monitoring initialized")

    if Config.CELERY["broker_url"]:
        celery_init_app(app)

    # Log successful startup with async support
    app.logger.info("PDF RAG application started with async support and enhanced monitoring")

    # Add ContentElement validation during startup
    try:
        from app.chat.utils import validate_content_element_class
        content_element_valid = validate_content_element_class()
        app.logger.info(f"ContentElement validation: {content_element_valid}")
    except Exception as e:
        app.logger.error(f"ContentElement validation failed: {str(e)}")

    return app

def register_extensions(app):
    db.init_app(app)
    migrate.init_app(app, db)
    app.cli.add_command(init_db_command)

    # Configure logging
    if not app.debug:
        # Set up rotating file handler for production
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/pdf_rag.log', maxBytes=10240, backupCount=5)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

    # Always add stdout handler for container environments
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('PDF RAG application startup with LangGraph integration and enhanced monitoring')

def register_blueprints(app):
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(pdf_views.bp)
    app.register_blueprint(conversation_views.bp)
    app.register_blueprint(client_views.bp)
    app.register_blueprint(health_views.bp)

    # Stream views are no longer needed as we handle streaming in conversation_views
    # But keep it registered for backward compatibility
    try:
        app.register_blueprint(stream_views.bp)
    except Exception as e:
        app.logger.warning(f"Could not register stream views: {str(e)}")

def register_hooks(app):
    CORS(app)
    app.before_request(load_logged_in_user)
    app.after_request(add_headers)

    # Enhanced error handling for async errors
    @app.errorhandler(Exception)
    def async_aware_error_handler(error):
        if isinstance(error, asyncio.CancelledError):
            app.logger.error("Async operation was cancelled")
            return {"error": "Operation cancelled"}, 500
        return handle_error(error)
