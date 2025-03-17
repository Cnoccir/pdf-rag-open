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
    score_views,
    client_views,
    conversation_views,
    stream_views,
)
import os
import logging
from logging.handlers import RotatingFileHandler
from langsmith import Client
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
if os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Configure Pinecone
if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_INDEX_NAME"):
    raise ValueError("Missing required Pinecone environment variables")

# Initialize LangSmith client
langsmith_client = Client()

migrate = Migrate()

def create_app():
    app = Flask(__name__, static_folder="../../client/build")
    app.url_map.strict_slashes = False
    app.config.from_object(Config)

    # Log configuration status
    app.logger.info("Pinecone configuration loaded")
    app.logger.info(f"Using Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}")

    register_extensions(app)
    register_hooks(app)
    register_blueprints(app)

    if Config.CELERY["broker_url"]:
        celery_init_app(app)

    return app

def register_extensions(app):
    db.init_app(app)
    migrate.init_app(app, db)
    app.cli.add_command(init_db_command)

    # Add logging for LangSmith
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

def register_blueprints(app):
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(pdf_views.bp)
    app.register_blueprint(score_views.bp)
    app.register_blueprint(conversation_views.bp)
    app.register_blueprint(client_views.bp)
    app.register_blueprint(stream_views.bp)

def register_hooks(app):
    CORS(app)
    app.before_request(load_logged_in_user)
    app.after_request(add_headers)
    app.register_error_handler(Exception, handle_error)
