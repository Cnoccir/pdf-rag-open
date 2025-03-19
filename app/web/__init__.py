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
    health_views, # Import health_views here
)
# Import the new async wrapper
from app.web.async_wrapper import async_handler, run_async
import os
import logging
from logging.handlers import RotatingFileHandler
from langsmith import Client
import sys
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure LangSmith and LangGraph Studio
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
if os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Configure Neo4j (optional environment variables check)
if not os.getenv("NEO4J_URL"):
    os.environ["NEO4J_URL"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

# Initialize nest_asyncio for better event loop handling
try:
    import nest_asyncio
    nest_asyncio.apply()
    logging.info("nest_asyncio applied successfully")
except ImportError:
    logging.warning("nest_asyncio not available - async nesting may have issues")
except Exception as e:
    logging.warning(f"Error applying nest_asyncio: {str(e)}")

# Initialize LangSmith client for tracing and visualization
langsmith_client = Client()

migrate = Migrate()

def create_app():
    app = Flask(__name__, static_folder="../../client/build")
    app.url_map.strict_slashes = False
    app.config.from_object(Config)

    # Log configuration status
    app.logger.info("Neo4j vector store configuration loaded")
    app.logger.info(f"Using Neo4j at: {os.getenv('NEO4J_URL')}")

    # Set up LangGraph Studio integration if enabled
    if os.getenv("LANGGRAPH_STUDIO_ENABLED", "true").lower() == "true":
        app.logger.info("LangGraph Studio integration enabled")

    # Initialize asyncio policy for better Windows support if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        app.logger.info("Set Windows-compatible asyncio event loop policy")

    register_extensions(app)
    register_hooks(app)
    register_blueprints(app)

    if Config.CELERY["broker_url"]:
        celery_init_app(app)

    # Log successful startup with async support
    app.logger.info("PDF RAG application started with async support")

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
    app.logger.info('PDF RAG application startup with LangGraph integration')

def register_blueprints(app):
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(pdf_views.bp)
    app.register_blueprint(conversation_views.bp)
    app.register_blueprint(client_views.bp)
    app.register_blueprint(health_views.bp) #Register the health view

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
