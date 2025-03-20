"""
Vector store module with unified MongoDB and Qdrant implementation.
This module provides a single point of access to document storage.
"""

import os
import logging
from typing import Optional
import traceback

from app.chat.vector_stores.unified_store import UnifiedVectorStore, get_vector_store
from app.chat.vector_stores.mongo_store import MongoStore, get_mongo_store
from app.chat.vector_stores.qdrant_store import QdrantStore, get_qdrant_store

logger = logging.getLogger(__name__)

# For backward compatibility
TechDocVectorStore = UnifiedVectorStore

__all__ = [
    "UnifiedVectorStore",
    "MongoStore",
    "QdrantStore",
    "TechDocVectorStore",
    "get_vector_store",
    "get_mongo_store",
    "get_qdrant_store"
]
