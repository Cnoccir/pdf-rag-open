"""
Enhanced application initialization with monitoring capabilities.
This patch adds the necessary initializations for health monitoring.
"""

from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from app.web.db import db, init_db_command
# Keep models import at top level - these are needed by other modules
from app.web.db import models
from app.celery import celery_init_app
from app.web.config import Config
from app.web.hooks import load_logged_in_user, handle_error, add_headers
# Import enhanced RAG monitor
from app.monitoring.rag_monitor import RAGMonitor
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

def initialize_vector_stores(app):
    """Initialize MongoDB and Qdrant connections with enhanced monitoring."""
    from app.chat.vector_stores import get_mongo_store, get_qdrant_store, get_vector_store

    # First try to use our enhanced initialization
    try:
        # Initialize vector stores with enhanced initialization
        from app.chat.vector_stores.initialization import initialize_and_verify_stores

        app.logger.info("Initializing vector stores with enhanced monitoring...")
        init_results = initialize_and_verify_stores(max_retries=2, retry_delay=3)

        if init_results["success"]:
            app.logger.info("Vector stores initialized successfully")
        else:
            app.logger.warning("Vector store initialization incomplete:")
            for error in init_results.get("errors", []):
                app.logger.warning(f"- {error}")

    except ImportError:
        # Fall back to basic initialization if enhanced module isn't available
        app.logger.info("Enhanced initialization not available, using basic initialization...")
        app.logger.info("Initializing MongoDB connection...")
        mongo_store = get_mongo_store()

        app.logger.info("Initializing Qdrant vector store...")
        qdrant_store = get_qdrant_store()

        app.logger.info("Initializing unified vector store...")
        vector_store = get_vector_store()

    # For backwards compatibility, try to apply existing enhancements
    try:
        from app.chat.vector_stores.enhanced_monitoring import enhance_vector_stores
        enhance_vector_stores()
        app.logger.info("Applied existing vector store enhancements")
    except ImportError:
        app.logger.info("No additional vector store enhancements found")
    except Exception as e:
        app.logger.warning(f"Error applying vector store enhancements: {str(e)}")

    # Also try to apply our new enhancements if available
    try:
        from app.chat.vector_stores.initialization import enhance_vector_stores as new_enhance
        new_enhance()
        app.logger.info("Applied new vector store enhancements")
    except ImportError:
        pass
    except Exception as e:
        app.logger.warning(f"Error applying new vector store enhancements: {str(e)}")

    # Make sure stores are defined for app config
    mongo_store = get_mongo_store()
    qdrant_store = get_qdrant_store()
    vector_store = get_vector_store()

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
        # Call synchronously because check_health returns a dict.
        health = vector_store.check_health()
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

    # Add version for diagnostics
    app.version = "1.0.0"  # Update with your actual version

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

    # Initialize enhanced monitoring
    app.config['RAG_MONITOR'] = RAGMonitor()
    app.logger.info("Enhanced RAG monitoring initialized")

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
    # Import blueprints inside the function to avoid circular imports
    # This is the critical fix for the circular dependency issue
    from app.web.views import auth_views, pdf_views, client_views
    from app.web.views import conversation_views
    from app.web.views import health_views

    app.register_blueprint(auth_views.bp)
    app.register_blueprint(pdf_views.bp)
    app.register_blueprint(conversation_views.bp)
    app.register_blueprint(client_views.bp)

    # Register health views with enhanced monitoring
    app.register_blueprint(health_views.bp)
    app.logger.info("Health monitoring endpoints registered")

    # Stream views are no longer needed as we handle streaming in conversation_views
    # But keep it registered for backward compatibility
    try:
        from app.web.views import stream_views
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
