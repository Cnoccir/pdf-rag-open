#!/usr/bin/env python
"""
Database configuration check script for MongoDB and Qdrant integration.
Use this to verify your database setup before running the full application.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("db_check")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("dotenv not installed, skipping .env file loading")

class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")

def print_status(name, status, details=None):
    """Print a status message with color."""
    if status == "OK":
        status_color = Colors.GREEN
    elif status == "WARNING":
        status_color = Colors.YELLOW
    else:
        status_color = Colors.RED

    print(f"{name.ljust(25)}: {status_color}{status}{Colors.END}")
    if details:
        print(f"  {details}")

async def check_mongodb_connection():
    """Check MongoDB connection."""
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
        mongodb_db = os.environ.get("MONGODB_DB_NAME", "tech_rag")

        print_status("MongoDB URI", "INFO", mongodb_uri)
        print_status("MongoDB DB Name", "INFO", mongodb_db)

        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)

        # Check if the server is responding
        client.admin.command('ping')

        # Check if we can access our database
        db = client[mongodb_db]
        collections = db.list_collection_names()

        print_status("MongoDB Connection", "OK", f"Connected to MongoDB at {mongodb_uri}")
        print_status("MongoDB Collections", "INFO", f"Found {len(collections)} collections in {mongodb_db}")

        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print_status("MongoDB Connection", "ERROR", f"Failed to connect to MongoDB: {str(e)}")
        return False
    except Exception as e:
        print_status("MongoDB Check", "ERROR", f"Unexpected error: {str(e)}")
        return False

async def check_qdrant_connection():
    """Check Qdrant connection."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
        qdrant_collection = os.environ.get("QDRANT_COLLECTION", "content_vectors")

        print_status("Qdrant Host", "INFO", qdrant_host)
        print_status("Qdrant Port", "INFO", str(qdrant_port))
        print_status("Qdrant Collection", "INFO", qdrant_collection)

        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5.0)

        # Check connection with a simple operation
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        print_status("Qdrant Connection", "OK", f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        print_status("Qdrant Collections", "INFO", f"Found {len(collection_names)} collections")

        # Check if our collection exists
        if qdrant_collection in collection_names:
            collection_info = client.get_collection(qdrant_collection)
            print_status("Target Collection", "OK", f"Found collection {qdrant_collection}")

            # Get vector count
            count_result = client.count(qdrant_collection, models.Filter())
            print_status("Vector Count", "INFO", f"Collection has {count_result.count} vectors")
        else:
            print_status("Target Collection", "WARNING", f"Collection {qdrant_collection} not found, but will be created on first use")

        return True
    except Exception as e:
        print_status("Qdrant Connection", "ERROR", f"Failed to connect to Qdrant: {str(e)}")
        return False

async def check_all_databases():
    """Check all database connections."""
    print_header("MongoDB Configuration Check")
    mongodb_ok = await check_mongodb_connection()

    print_header("Qdrant Configuration Check")
    qdrant_ok = await check_qdrant_connection()

    print_header("Summary")
    if mongodb_ok and qdrant_ok:
        print_status("Overall Status", "OK", "All database connections successful")
        print("\nYour database configuration is correct and ready for use with the PDF RAG system.")
    elif mongodb_ok:
        print_status("Overall Status", "WARNING", "MongoDB is working but Qdrant has issues")
        print("\nPlease check your Qdrant configuration before proceeding.")
    elif qdrant_ok:
        print_status("Overall Status", "WARNING", "Qdrant is working but MongoDB has issues")
        print("\nPlease check your MongoDB configuration before proceeding.")
    else:
        print_status("Overall Status", "ERROR", "Both database connections failed")
        print("\nPlease check your database configurations before proceeding.")

    print("\nNote: This script only checks connections. The application will create")
    print("necessary collections and indexes on startup if they don't exist.")

if __name__ == "__main__":
    print_header("Database Configuration Check")
    print(f"Running check at: {datetime.now().isoformat()}")

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print_status("OpenAI API Key", "WARNING", "OPENAI_API_KEY environment variable not set")
    else:
        key = os.environ.get("OPENAI_API_KEY")
        masked_key = key[:4] + '*' * (len(key) - 8) + key[-4:] if len(key) > 8 else '*' * len(key)
        print_status("OpenAI API Key", "OK", f"Found API key: {masked_key}")

    try:
        # Run async checks
        asyncio.run(check_all_databases())
    except Exception as e:
        print_status("Check Error", "ERROR", f"Unexpected error in checker: {str(e)}")
