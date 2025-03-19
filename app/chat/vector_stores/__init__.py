"""
Vector store module with unified Neo4j implementation.
This module provides a single point of access to Neo4j vector storage.
"""

import os
import logging
from typing import Optional

from app.chat.vector_stores.neo4j_store import Neo4jVectorStore

logger = logging.getLogger(__name__)

_vector_store_instance = None

def get_vector_store() -> Neo4jVectorStore:
    """
    Get unified Neo4j vector store instance.
    Uses environment variables for configuration.

    Returns:
        Neo4jVectorStore instance
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        logger.info("Initializing Neo4j vector store")
        try:
            neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

            _vector_store_instance = Neo4jVectorStore(
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password,
                embedding_dimension=embedding_dimension,
                embedding_model="text-embedding-3-small"
            )

            logger.info("Neo4j vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j vector store: {str(e)}")
            raise

    return _vector_store_instance

# Make TechDocVectorStore an alias for Neo4jVectorStore for backward compatibility
TechDocVectorStore = Neo4jVectorStore

__all__ = ["Neo4jVectorStore", "TechDocVectorStore", "get_vector_store"]
